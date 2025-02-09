import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeTracker:
    """Helper to track and display tensor shapes and gradient flow"""

    def __init__(self):
        self.indent_level = 0

    def indent(self):
        self.indent_level += 1

    def dedent(self):
        self.indent_level = max(0, self.indent_level - 1)

    def print(
        self,
        message: str,
        tensor: Optional[torch.Tensor] = None,
        requires_grad: Optional[bool] = None,
    ):
        prefix = "  " * self.indent_level
        print(f"{prefix}{message}")

        if tensor is not None:
            print(f"{prefix}  Shape: {list(tensor.shape)}")
            print(f"{prefix}  Device: {tensor.device}")
            print(f"{prefix}  Requires grad: {tensor.requires_grad}")
            if requires_grad is not None:
                if requires_grad:
                    print(f"{prefix}  ✓ BACKPROP: Gradients will flow through this")
                else:
                    print(f"{prefix}  ✗ NO BACKPROP: Fixed/frozen tensor")


tracker = ShapeTracker()


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_cross: bool = False,
    ):

        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        attn_type = "CROSS-ATTENTION" if is_cross else "SELF-ATTENTION"
        tracker.print(f"\n{'='*60}")
        tracker.print(f"MULTI-HEAD {attn_type}")
        tracker.print(f"{'='*60}")

        tracker.print(f"Query input:", query, requires_grad=True)
        tracker.print(f"Key input:", key, requires_grad=True)
        tracker.print(f"Value input:", value, requires_grad=True)

        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        tracker.print(f"\nAfter linear projections:")
        tracker.indent()
        tracker.print(f"Q:", Q, requires_grad=True)
        tracker.print(f"K:", K, requires_grad=True)
        tracker.print(f"V:", V, requires_grad=True)
        tracker.dedent()

        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)

        tracker.print(f"\nAfter reshape for {self.n_heads} heads:")
        tracker.indent()
        tracker.print(f"Q (per head):", Q)
        tracker.print(f"K (per head):", K)
        tracker.print(f"V (per head):", V)
        tracker.dedent()

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        tracker.print(f"\nAttention scores (Q @ K^T / sqrt(d_k)):", scores)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            tracker.print(f"After masking:", scores)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        tracker.print(f"\nAttention weights (after softmax):", attn_weights)

        # Apply attention
        context = torch.matmul(attn_weights, V)
        tracker.print(f"\nContext (attention @ V):", context)

        # Reshape and project
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_q, self.d_model)
        )
        output = self.W_o(context)

        tracker.print(f"\nFinal output:", output, requires_grad=True)

        return output, attn_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor):
        tracker.print(f"\n{'='*60}")
        tracker.print(f"FEED-FORWARD NETWORK")
        tracker.print(f"{'='*60}")

        tracker.print(f"Input:", x, requires_grad=True)

        # First linear + ReLU
        hidden = F.relu(self.linear1(x))
        tracker.print(f"\nAfter Linear1 + ReLU:", hidden)

        hidden = self.dropout(hidden)
        tracker.print(f"After dropout:", hidden)

        # Second linear
        output = self.linear2(hidden)
        tracker.print(f"\nOutput:", output, requires_grad=True)

        return output


class EncoderLayer(nn.Module):
    """Single encoder layer"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Self-attention with residual
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        tracker.print(f"\nAfter self-attention + residual + norm:", x)

        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        tracker.print(f"\nAfter FFN + residual + norm:", x)

        return x


class DecoderLayer(nn.Module):
    """Single decoder layer"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ):

        # Masked self-attention
        attn_output, _ = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        tracker.print(f"\nAfter masked self-attention + residual + norm:", x)

        # Cross-attention to encoder
        cross_output, _ = self.cross_attn(
            x, encoder_output, encoder_output, src_mask, is_cross=True
        )
        x = self.norm2(x + self.dropout(cross_output))

        tracker.print(f"\nAfter cross-attention + residual + norm:", x)

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        tracker.print(f"\nAfter FFN + residual + norm:", x)

        return x


class EncoderDecoderTransformer(nn.Module):
    """Complete Encoder-Decoder Transformer (like in the paper)"""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 100,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = self.create_positional_encoding(max_len, d_model)

        # Encoder stack
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

        # Decoder stack
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def create_padding_mask(self, seq: torch.Tensor, pad_idx: int = 0):
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def create_causal_mask(self, size: int):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return (mask == 0).unsqueeze(0).unsqueeze(0)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        """Encode source sequence"""

        print("\n" + "█" * 80)
        print("ENCODER STACK")
        print("█" * 80)

        # Embedding + positional encoding
        tracker.print(f"\nSource input tokens:", src)

        x = self.src_embedding(src) * math.sqrt(self.d_model)
        tracker.print(
            f"\nAfter source embedding (* sqrt(d_model)):", x, requires_grad=True
        )

        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        tracker.print(f"\nAfter adding positional encoding:", x, requires_grad=True)

        # Pass through encoder layers
        for i, layer in enumerate(self.encoder_layers):
            print(f"\n{'─'*60}")
            print(f"ENCODER LAYER {i+1}/{self.n_layers}")
            print(f"{'─'*60}")
            tracker.indent()
            x = layer(x, src_mask)
            tracker.dedent()

        tracker.print(f"\nFinal encoder output:", x, requires_grad=True)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ):
        """Decode target sequence"""

        print("\n" + "█" * 80)
        print("DECODER STACK")
        print("█" * 80)

        # Embedding + positional encoding
        tracker.print(f"\nTarget input tokens:", tgt)

        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tracker.print(
            f"\nAfter target embedding (* sqrt(d_model)):", x, requires_grad=True
        )

        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        tracker.print(f"\nAfter adding positional encoding:", x, requires_grad=True)

        # Pass through decoder layers
        for i, layer in enumerate(self.decoder_layers):
            print(f"\n{'─'*60}")
            print(f"DECODER LAYER {i+1}/{self.n_layers}")
            print(f"{'─'*60}")
            tracker.indent()
            x = layer(x, encoder_output, src_mask, tgt_mask)
            tracker.dedent()

        tracker.print(f"\nFinal decoder output:", x, requires_grad=True)
        return x

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """Full forward pass"""

        print("\n" + "▓" * 80)
        print("FULL ENCODER-DECODER TRANSFORMER FORWARD PASS")
        print("▓" * 80)

        # Create masks
        src_mask = self.create_padding_mask(src)
        tgt_mask = self.create_causal_mask(tgt.size(1)).to(tgt.device)

        tracker.print(f"\nSource mask shape:", src_mask)
        tracker.print(f"Target mask shape (causal):", tgt_mask)

        # Encode
        encoder_output = self.encode(src, src_mask)

        # Decode
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary
        print("\n" + "█" * 80)
        print("OUTPUT PROJECTION")
        print("█" * 80)

        output = self.output_projection(decoder_output)
        tracker.print(f"\nFinal output (logits):", output, requires_grad=True)

        return output

    def show_gradient_flow(self):
        """Display which parameters will receive gradients"""

        print("\n" + "▓" * 80)
        print("GRADIENT FLOW ANALYSIS")
        print("▓" * 80)

        print("\nTRAINABLE PARAMETERS (will receive gradients):")
        total_params = 0
        trainable_params = 0

        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"  ✓ {name}: {list(param.shape)}")

        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")


def demo_translation():
    """Demonstrate encoder-decoder for translation"""

    print("=" * 80)
    print("ENCODER-DECODER TRANSFORMER DEMO")
    print("Machine Translation: English -> French")
    print("=" * 80)

    # Model configuration
    src_vocab_size = 100  # English vocab
    tgt_vocab_size = 100  # French vocab
    d_model = 64
    n_heads = 4
    n_layers = 2
    d_ff = 128

    print(f"\nModel Configuration:")
    print(f"  Source vocab size: {src_vocab_size}")
    print(f"  Target vocab size: {tgt_vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Number of layers: {n_layers}")
    print(f"  Feed-forward dimension: {d_ff}")

    # Create model
    model = EncoderDecoderTransformer(
        src_vocab_size, tgt_vocab_size, d_model, n_heads, n_layers, d_ff
    )

    # Example input
    batch_size = 2
    src_seq_len = 5  # "Hello world how are you"
    tgt_seq_len = 4  # "Bonjour monde comment allez"

    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))

    print(f"\nInput shapes:")
    print(
        f"  Source: {list(src.shape)} (batch_size={batch_size}, seq_len={src_seq_len})"
    )
    print(
        f"  Target: {list(tgt.shape)} (batch_size={batch_size}, seq_len={tgt_seq_len})"
    )

    # Forward pass
    output = model(src, tgt)

    print("\n" + "▓" * 80)
    print("BACKPROPAGATION SIMULATION")
    print("▓" * 80)

    # Simulate loss and backward
    loss = F.cross_entropy(output.view(-1, tgt_vocab_size), tgt.view(-1))

    print(f"\nLoss value: {loss.item():.4f}")
    print(f"Loss requires_grad: {loss.requires_grad}")

    print("\nCalling loss.backward() would:")
    print("  1. Compute gradients for all parameters with requires_grad=True")
    print("  2. Gradients flow backward through:")
    print("     - Output projection")
    print("     - All decoder layers (in reverse)")
    print("     - All encoder layers (in reverse)")
    print("     - Embeddings")

    # Show which parameters would get gradients
    model.show_gradient_flow()

    print("\n" + "=" * 80)
    print("INFERENCE MODE (No gradients)")
    print("=" * 80)

    with torch.no_grad():
        print("\nIn torch.no_grad() context:")
        print("  - No gradients computed")
        print("  - Memory efficient")
        print("  - Faster execution")
        output_inference = model(src, tgt[:, :1])  # Teacher forcing with first token
        print(f"\nInference output shape: {list(output_inference.shape)}")
        print(f"Requires grad: {output_inference.requires_grad}")


if __name__ == "__main__":
    demo_translation()
