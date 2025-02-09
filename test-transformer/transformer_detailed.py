import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def print_tensor(
    name: str, tensor: torch.Tensor, show_values: bool = True, indent: int = 0
):
    """Helper to print tensor info with nice formatting"""
    prefix = "  " * indent
    print(f"{prefix}{name}:")
    print(f"{prefix}  Shape: {list(tensor.shape)}")
    print(f"{prefix}  Dtype: {tensor.dtype}")
    if show_values and tensor.numel() <= 50:
        if tensor.dim() == 1:
            print(f"{prefix}  Values: {tensor.tolist()}")
        elif tensor.dim() == 2 and tensor.shape[0] <= 5:
            for i, row in enumerate(tensor):
                if row.numel() <= 10:
                    print(f"{prefix}    [{i}]: {row.tolist()}")
    elif show_values:
        print(f"{prefix}  Values: [Too large to display, {tensor.numel()} elements]")


class MultiHeadAttention:
    """Full multi-head attention implementation"""

    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Initialize weight matrices for all heads at once
        self.W_q = torch.randn(d_model, d_model) * 0.02
        self.W_k = torch.randn(d_model, d_model) * 0.02
        self.W_v = torch.randn(d_model, d_model) * 0.02
        self.W_o = torch.randn(d_model, d_model) * 0.02

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, verbose: bool = True
    ):
        batch_size = 1  # For simplicity
        seq_len = x.shape[0]

        if verbose:
            print("\n" + "=" * 70)
            print("MULTI-HEAD ATTENTION")
            print("=" * 70)
            print_tensor("Input", x)

        # Step 1: Linear projections
        Q = torch.matmul(x, self.W_q)
        K = torch.matmul(x, self.W_k)
        V = torch.matmul(x, self.W_v)

        if verbose:
            print("\n1. Linear Projections (Q, K, V):")
            print_tensor("Q (Query)", Q, show_values=False, indent=1)
            print_tensor("K (Key)", K, show_values=False, indent=1)
            print_tensor("V (Value)", V, show_values=False, indent=1)

        # Step 2: Reshape for multi-head
        Q = Q.view(seq_len, self.n_heads, self.d_k).transpose(0, 1)
        K = K.view(seq_len, self.n_heads, self.d_k).transpose(0, 1)
        V = V.view(seq_len, self.n_heads, self.d_k).transpose(0, 1)

        if verbose:
            print("\n2. Reshape for Multiple Heads:")
            print(f"   Original shape: [{seq_len}, {self.d_model}]")
            print(f"   After reshape: [{self.n_heads}, {seq_len}, {self.d_k}]")
            print(f"   Each head processes {self.d_k}-dim subspace independently")

        # Step 3: Scaled dot-product attention for each head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if verbose:
            print("\n3. Attention Scores (Q @ K^T / sqrt(d_k)):")
            print_tensor("Scores", scores, show_values=False)
            print(f"   First head scores sample:")
            if scores[0].shape[0] <= 5:
                for i in range(min(3, scores[0].shape[0])):
                    print(f"     Token {i}: {scores[0][i].tolist()}")

        # Step 4: Apply mask
        if mask is not None:
            scores = scores + mask
            if verbose:
                print("\n4. Apply Causal Mask (prevent looking ahead):")
                print("   Mask adds -inf to future positions")

        # Step 5: Softmax
        attn_weights = F.softmax(scores, dim=-1)

        if verbose:
            print("\n5. Attention Weights (after softmax):")
            print_tensor("Attention weights", attn_weights, show_values=False)
            print(f"   First head attention pattern:")
            if attn_weights[0].shape[0] <= 5:
                for i in range(min(3, attn_weights[0].shape[0])):
                    weights = attn_weights[0][i].tolist()
                    print(f"     Token {i} attends to: {[f'{w:.3f}' for w in weights]}")

        # Step 6: Apply attention to values
        context = torch.matmul(attn_weights, V)

        if verbose:
            print("\n6. Apply Attention to Values:")
            print_tensor("Context", context, show_values=False)

        # Step 7: Concatenate heads
        context = context.transpose(0, 1).contiguous().view(seq_len, self.d_model)

        if verbose:
            print("\n7. Concatenate Heads:")
            print(f"   Reshape from [{self.n_heads}, {seq_len}, {self.d_k}]")
            print(f"   To [{seq_len}, {self.d_model}]")

        # Step 8: Final linear projection
        output = torch.matmul(context, self.W_o)

        if verbose:
            print("\n8. Final Projection:")
            print_tensor("Output", output, show_values=False)

        return output, attn_weights


class FeedForward:
    """Position-wise feed-forward network"""

    def __init__(self, d_model: int, d_ff: int):
        self.W1 = torch.randn(d_model, d_ff) * 0.02
        self.b1 = torch.zeros(d_ff)
        self.W2 = torch.randn(d_ff, d_model) * 0.02
        self.b2 = torch.zeros(d_model)

    def forward(self, x: torch.Tensor, verbose: bool = True):
        if verbose:
            print("\n" + "=" * 70)
            print("FEED-FORWARD NETWORK")
            print("=" * 70)
            print_tensor("Input", x, show_values=False)

        # First linear + ReLU
        hidden = F.relu(torch.matmul(x, self.W1) + self.b1)

        if verbose:
            print("\n1. First Linear + ReLU:")
            print_tensor("Hidden", hidden, show_values=False)
            print(f"   Expanded from {x.shape[-1]} to {hidden.shape[-1]} dims")

        # Second linear
        output = torch.matmul(hidden, self.W2) + self.b2

        if verbose:
            print("\n2. Second Linear:")
            print_tensor("Output", output, show_values=False)
            print(f"   Projected back to {output.shape[-1]} dims")

        return output


class LayerNorm:
    """Layer normalization"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = torch.ones(d_model)
        self.beta = torch.zeros(d_model)
        self.eps = eps

    def forward(self, x: torch.Tensor, verbose: bool = False):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        output = self.gamma * normalized + self.beta

        if verbose:
            print("\nLayer Normalization:")
            print(f"  Mean: {mean.squeeze().tolist()}")
            print(f"  Std: {std.squeeze().tolist()}")

        return output


class TransformerBlock:
    """A single transformer encoder block"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, verbose: bool = True
    ):
        if verbose:
            print("\n" + "#" * 80)
            print(f"TRANSFORMER BLOCK")
            print("#" * 80)

        # Self-attention with residual connection
        attn_output, attn_weights = self.attention.forward(x, mask, verbose)
        x = self.norm1.forward(x + attn_output, verbose)

        if verbose:
            print("\nAfter Attention + Residual + LayerNorm:")
            print_tensor("x", x, show_values=False)

        # Feed-forward with residual connection
        ffn_output = self.ffn.forward(x, verbose)
        x = self.norm2.forward(x + ffn_output, verbose)

        if verbose:
            print("\nAfter FFN + Residual + LayerNorm:")
            print_tensor("x", x, show_values=False)

        return x, attn_weights


class CompleteTransformer:
    """Complete transformer model with all components"""

    def __init__(self):
        # Vocabulary
        self.vocab = {
            "<pad>": 0,
            "<sos>": 1,
            "<eos>": 2,
            "hi": 3,
            ",": 4,
            "hello": 5,
            "world": 6,
            "!": 7,
            "how": 8,
            "are": 9,
            "you": 10,
        }
        self.id_to_word = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # Model hyperparameters
        self.d_model = 64
        self.n_heads = 4
        self.n_layers = 2
        self.d_ff = 256
        self.max_seq_len = 20

        print("Model Configuration:")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Model dimension (d_model): {self.d_model}")
        print(f"  Number of heads: {self.n_heads}")
        print(f"  Number of layers: {self.n_layers}")
        print(f"  Feed-forward dimension: {self.d_ff}")
        print(f"  Head dimension: {self.d_model // self.n_heads}")

        self.init_model()

    def init_model(self):
        """Initialize all model components"""
        torch.manual_seed(42)

        # Token embeddings
        self.token_embeddings = nn.Embedding(self.vocab_size, self.d_model)
        self.token_embeddings.weight.data.normal_(0, 0.02)

        # Positional encodings (sinusoidal)
        self.pos_encodings = self.create_positional_encodings()

        # Transformer blocks
        self.blocks = [
            TransformerBlock(self.d_model, self.n_heads, self.d_ff)
            for _ in range(self.n_layers)
        ]

        # Output projection
        self.output_projection = torch.randn(self.d_model, self.vocab_size) * 0.02

    def create_positional_encodings(self):
        """Create sinusoidal positional encodings"""
        pe = torch.zeros(self.max_seq_len, self.d_model)
        position = torch.arange(0, self.max_seq_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * -(math.log(10000.0) / self.d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def create_causal_mask(self, seq_len: int):
        """Create causal attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        mask = mask.unsqueeze(0)  # Add head dimension
        return mask

    def tokenize(self, text: str):
        """Convert text to token IDs"""
        tokens = text.lower().replace(",", " ,").split()
        token_ids = [self.vocab.get(token, 0) for token in tokens]
        return torch.tensor(token_ids)

    def embed_and_encode(self, token_ids: torch.Tensor, verbose: bool = True):
        """Embed tokens and add positional encoding"""
        if verbose:
            print("\n" + "=" * 80)
            print("EMBEDDING AND POSITIONAL ENCODING")
            print("=" * 80)
            print(f"Input tokens: {[self.id_to_word[id.item()] for id in token_ids]}")
            print(f"Token IDs: {token_ids.tolist()}")

        # Token embeddings
        embeddings = self.token_embeddings(token_ids)

        if verbose:
            print("\n1. Token Embeddings:")
            print_tensor("Embeddings", embeddings, show_values=False)

        # Add positional encoding
        seq_len = embeddings.shape[0]
        pos_enc = self.pos_encodings[:seq_len]
        x = embeddings + pos_enc

        if verbose:
            print("\n2. Add Positional Encoding:")
            print_tensor("Position encoding", pos_enc, show_values=False)
            print_tensor("Combined (token + position)", x, show_values=False)

        return x

    def decode_logits(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        verbose: bool = True,
    ):
        """Detailed decoding process with multiple strategies"""

        if verbose:
            print("\n" + "=" * 80)
            print("DECODING PROCESS (Logits -> Token)")
            print("=" * 80)
            print("\n1. RAW LOGITS (before any processing):")
            print_tensor("Logits", logits)
            print("\n   Per-token logits:")
            for i, logit in enumerate(logits):
                print(f"     {self.id_to_word[i]:10s}: {logit:.4f}")

        # Apply temperature
        if temperature != 1.0:
            scaled_logits = logits / temperature
            if verbose:
                print(f"\n2. TEMPERATURE SCALING (T={temperature}):")
                print(
                    f"   Effect: {'Sharpens' if temperature < 1 else 'Smooths'} the distribution"
                )
                print_tensor("Scaled logits", scaled_logits)
        else:
            scaled_logits = logits

        # Apply top-k filtering
        if top_k > 0:
            if verbose:
                print(f"\n3. TOP-K FILTERING (k={top_k}):")
            indices_to_remove = (
                scaled_logits < torch.topk(scaled_logits, top_k)[0][..., -1, None]
            )
            scaled_logits[indices_to_remove] = float("-inf")
            if verbose:
                print(f"   Keeping only top {top_k} tokens")
                print_tensor("Filtered logits", scaled_logits)

        # Compute softmax probabilities
        if verbose:
            print("\n4. SOFTMAX COMPUTATION:")
            print("   Formula: exp(logit) / sum(exp(all_logits))")

            # Show exp values
            exp_logits = torch.exp(
                scaled_logits - scaled_logits.max()
            )  # Subtract max for numerical stability
            print("\n   Exp values (numerator):")
            for i, exp_val in enumerate(exp_logits):
                if not torch.isinf(exp_val):
                    print(
                        f"     {self.id_to_word[i]:10s}: exp({scaled_logits[i]:.4f}) = {exp_val:.6f}"
                    )

            print(f"\n   Sum of exp values (denominator): {exp_logits.sum():.6f}")

        probs = F.softmax(scaled_logits, dim=-1)

        if verbose:
            print("\n   Final probabilities (after softmax):")
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            for i in range(min(5, len(sorted_probs))):
                idx = sorted_indices[i].item()
                print(
                    f"     {self.id_to_word[idx]:10s}: {sorted_probs[i]:.6f} ({sorted_probs[i]*100:.2f}%)"
                )

        # Show different decoding strategies
        if verbose:
            print("\n5. TOKEN SELECTION STRATEGIES:")

            # Argmax (greedy)
            argmax_token = torch.argmax(probs).item()
            print(f"\n   a) ARGMAX (Greedy):")
            print(
                f"      Selected: '{self.id_to_word[argmax_token]}' (prob={probs[argmax_token]:.6f})"
            )
            print(f"      This always picks the highest probability token")

            # Top-3 sampling
            print(f"\n   b) TOP-3 CANDIDATES:")
            top3_probs, top3_indices = torch.topk(probs, min(3, len(probs)))
            for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
                print(f"      {i+1}. '{self.id_to_word[idx.item()]}': {prob:.6f}")

            # Sampling
            print(f"\n   c) SAMPLING (Random based on probabilities):")
            if not torch.all(torch.isnan(probs)):
                sampled = torch.multinomial(probs, 1).item()
                print(
                    f"      Could sample: '{self.id_to_word[sampled]}' (prob={probs[sampled]:.6f})"
                )

            print("\n6. FINAL SELECTION:")
            print(f"   Using ARGMAX strategy")
            print(f"   Selected token: '{self.id_to_word[argmax_token]}'")
            print(f"   Token ID: {argmax_token}")

        return argmax_token, probs

    def forward(self, input_text: str, temperature: float = 1.0, verbose: bool = True):
        """Complete forward pass"""

        print("\n" + "█" * 80)
        print(f"COMPLETE TRANSFORMER FORWARD PASS")
        print(f"Input: '{input_text}'")
        print(f"Temperature: {temperature}")
        print("█" * 80)

        # Tokenize
        token_ids = self.tokenize(input_text)

        # Embed and encode
        x = self.embed_and_encode(token_ids, verbose)

        # Create causal mask
        seq_len = x.shape[0]
        mask = self.create_causal_mask(seq_len)

        # Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            if verbose:
                print(f"\n{'='*80}")
                print(f"LAYER {i+1}/{self.n_layers}")
                print(f"{'='*80}")
            x, attn_weights = block.forward(x, mask, verbose)

        # Get last token representation
        last_hidden = x[-1]

        if verbose:
            print("\n" + "=" * 80)
            print("OUTPUT PROJECTION")
            print("=" * 80)
            print_tensor("Last hidden state", last_hidden, show_values=False)

        # Project to vocabulary
        logits = torch.matmul(last_hidden, self.output_projection)

        if verbose:
            print_tensor("Output logits", logits, show_values=False)

        # Decode to token
        next_token_id, probs = self.decode_logits(
            logits, temperature, top_k=5, verbose=verbose
        )

        return next_token_id, probs


def main():
    print("=" * 80)
    print("DETAILED TRANSFORMER INFERENCE")
    print("Understanding each step of 'Attention Is All You Need'")
    print("=" * 80)

    # Create model
    model = CompleteTransformer()

    # Test cases
    test_inputs = [
        ("hello", 0.7),
        ("hi ,", 1.0),
        ("hello world", 0.5),
    ]

    for input_text, temp in test_inputs:
        print("\n" + "▓" * 80)
        print(f"TEST: '{input_text}' with temperature={temp}")
        print("▓" * 80)

        next_token_id, probs = model.forward(input_text, temperature=temp, verbose=True)

        print("\n" + "▓" * 80)
        print(f"RESULT: '{input_text}' -> '{model.id_to_word[next_token_id]}'")
        print("▓" * 80)


if __name__ == "__main__":
    main()
