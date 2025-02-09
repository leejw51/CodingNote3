import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerInferenceDemo:
    """Demonstrates step-by-step transformer inference for translation"""

    def __init__(self):
        # Simple vocabulary for demonstration
        self.src_vocab = {
            "<pad>": 0,
            "<sos>": 1,
            "<eos>": 2,
            "I": 3,
            "love": 4,
            "you": 5,
            "very": 6,
            "much": 7,
        }

        self.tgt_vocab = {
            "<pad>": 0,
            "<sos>": 1,
            "<eos>": 2,
            "Ti": 3,
            "amo": 4,
            "molto": 5,
        }

        self.src_id_to_word = {v: k for k, v in self.src_vocab.items()}
        self.tgt_id_to_word = {v: k for k, v in self.tgt_vocab.items()}

        # Model dimensions (small for demonstration)
        self.d_model = 32
        self.n_heads = 4
        self.d_k = self.d_model // self.n_heads
        self.src_vocab_size = len(self.src_vocab)
        self.tgt_vocab_size = len(self.tgt_vocab)

        # Initialize model components
        self.init_model()

    def init_model(self):
        """Initialize all model weights"""
        torch.manual_seed(42)

        # Embeddings
        self.src_embedding = nn.Embedding(self.src_vocab_size, self.d_model)
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, self.d_model)

        # Positional encoding
        self.pos_encoding = self.create_positional_encoding(20, self.d_model)

        # Simplified transformer weights (for demonstration)
        self.encoder_weights = torch.randn(self.d_model, self.d_model) * 0.1
        self.decoder_weights = torch.randn(self.d_model, self.d_model) * 0.1
        self.cross_attn_weights = torch.randn(self.d_model, self.d_model) * 0.1

        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.tgt_vocab_size)

    def create_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def encode_source(self, src_tokens: List[str]):
        """Encode source sentence (happens once)"""
        print("\n" + "=" * 80)
        print("ENCODER (Processes source sentence ONCE)")
        print("=" * 80)

        # Convert to IDs
        src_ids = torch.tensor([self.src_vocab[tok] for tok in src_tokens])
        print(f"\nSource: {' '.join(src_tokens)}")
        print(f"Token IDs: {src_ids.tolist()}")

        # Embed and add positional encoding
        src_emb = self.src_embedding(src_ids)
        src_emb = src_emb + self.pos_encoding[: len(src_ids)]

        print(f"\nAfter embedding + position:")
        print(
            f"  Shape: {list(src_emb.shape)} [seq_len={len(src_ids)}, d_model={self.d_model}]"
        )

        # Simplified encoder (just a linear transform for demo)
        encoder_output = torch.matmul(src_emb, self.encoder_weights)

        print(f"\nEncoder output:")
        print(f"  Shape: {list(encoder_output.shape)}")
        print(f"  This captures meaning and context of each source word")
        print(f"  Will be used by decoder at EVERY time step via cross-attention")

        return encoder_output, src_ids

    def decode_step(
        self, decoder_input: List[str], encoder_output: torch.Tensor, step_num: int
    ):
        """Single decoder step with detailed output"""

        print("\n" + "█" * 80)
        print(f"INFERENCE TIME STEP = {step_num}")
        print("█" * 80)

        # Convert decoder input to IDs
        tgt_ids = torch.tensor([self.tgt_vocab.get(tok, 0) for tok in decoder_input])

        print(f"\nDecoder Input: {decoder_input}")
        print(f"Token IDs: {tgt_ids.tolist()}")
        print(f"Number of tokens: {len(decoder_input)}")

        # Embed and add positional encoding
        tgt_emb = self.tgt_embedding(tgt_ids)
        tgt_emb = tgt_emb + self.pos_encoding[: len(tgt_ids)]

        print(f"\nAfter embedding + position:")
        print(
            f"  Shape: {list(tgt_emb.shape)} [seq_len={len(tgt_ids)}, d_model={self.d_model}]"
        )

        # Simplified decoder self-attention
        decoder_hidden = torch.matmul(tgt_emb, self.decoder_weights)
        print(f"\nAfter self-attention (masked):")
        print(f"  Shape: {list(decoder_hidden.shape)}")
        print(f"  Each token can only see previous tokens (causal mask)")

        # Cross-attention to encoder
        # In real transformer, this would be attention(Q=decoder, K,V=encoder)
        cross_attn = torch.matmul(decoder_hidden[-1:], self.cross_attn_weights)
        cross_output = (cross_attn + encoder_output.mean(0, keepdim=True)) / 2

        print(f"\nAfter cross-attention to encoder:")
        print(f"  Shape: {list(cross_output.shape)}")
        print(f"  Decoder attends to ALL encoder positions")
        print(f"  Gets information from source sentence")

        # Final decoder output (just last position for next token prediction)
        final_hidden = cross_output[-1]  # Take last position

        print(f"\nFinal hidden state (last position only):")
        print(f"  Shape: {list(final_hidden.shape)} [d_model={self.d_model}]")

        # Project to vocabulary
        print("\n" + "-" * 80)
        print("OUTPUT PROJECTION & SOFTMAX")
        print("-" * 80)

        logits = self.output_projection(final_hidden)

        print(f"\n1. Linear projection to vocabulary:")
        print(f"   Input shape: {list(final_hidden.shape)} [d_model={self.d_model}]")
        print(f"   Weight shape: [{self.d_model}, {self.tgt_vocab_size}]")
        print(
            f"   Output shape: {list(logits.shape)} [vocab_size={self.tgt_vocab_size}]"
        )

        print(f"\n2. Raw logits (before softmax):")
        for i, logit in enumerate(logits):
            word = self.tgt_id_to_word[i]
            print(f"   {word:8s}: {logit:7.3f}")

        # Apply softmax
        print(f"\n3. Softmax computation:")
        print(f"   Formula: exp(logit) / sum(exp(all_logits))")

        # Show exp values
        exp_logits = torch.exp(logits)
        print(f"\n   Exp values:")
        for i, exp_val in enumerate(exp_logits):
            word = self.tgt_id_to_word[i]
            print(f"   {word:8s}: exp({logits[i]:7.3f}) = {exp_val:7.3f}")

        print(f"\n   Sum of exp values: {exp_logits.sum():.3f}")

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)

        print(f"\n4. Final probabilities (after softmax):")
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        for i in range(len(sorted_probs)):
            idx = sorted_indices[i].item()
            word = self.tgt_id_to_word[idx]
            prob = sorted_probs[i].item()
            bar = "█" * int(prob * 50)
            print(f"   {word:8s}: {prob:6.3f} {bar}")

        # Select token
        selected_idx = torch.argmax(probs).item()
        selected_word = self.tgt_id_to_word[selected_idx]

        print(f"\n5. Token selection (argmax):")
        print(f"   Selected: '{selected_word}' (prob={probs[selected_idx]:.3f})")

        if step_num > 1:
            print(f"\n6. Update decoder input:")
            print(f"   Append '{selected_word}' to decoder input for next step")
            print(f"   New decoder input will be: {decoder_input + [selected_word]}")

        return selected_word, probs

    def full_inference(self, src_sentence: str, max_length: int = 10):
        """Complete inference process"""

        print("\n" + "▓" * 80)
        print("COMPLETE TRANSFORMER INFERENCE")
        print(f"Translating: '{src_sentence}'")
        print("▓" * 80)

        # Tokenize source
        src_tokens = ["<sos>"] + src_sentence.split() + ["<eos>"]

        # TRAINING vs INFERENCE difference
        print("\n" + "=" * 80)
        print("TRAINING vs INFERENCE")
        print("=" * 80)
        print("\nTRAINING (Teacher Forcing):")
        print("  - All target tokens provided at once")
        print("  - Parallel computation for all positions")
        print("  - Uses true target for each position")
        print("  - Loss computed on all predictions")

        print("\nINFERENCE (Autoregressive):")
        print("  - Generate one token at a time")
        print("  - Each generated token feeds into next step")
        print("  - Sequential process (can't parallelize)")
        print("  - Stop when <eos> is generated")

        # Encode source (happens ONCE)
        encoder_output, src_ids = self.encode_source(src_tokens)

        # Decoder starts with <sos>
        decoder_input = ["<sos>"]
        generated_tokens = []

        # Generate tokens one by one
        for step in range(1, max_length):
            # Decode one step
            next_token, probs = self.decode_step(decoder_input, encoder_output, step)

            # Add to decoder input for next step
            decoder_input.append(next_token)
            generated_tokens.append(next_token)

            # Stop if we generate <eos>
            if next_token == "<eos>":
                print(f"\n{'='*80}")
                print(f"GENERATION COMPLETE - <eos> token generated")
                print(f"{'='*80}")
                break

            if step < max_length - 1:
                print(f"\n{'='*80}")
                print(f"Proceeding to next time step...")
                print(f"{'='*80}")

        # Final result
        print(f"\n{'▓'*80}")
        print("FINAL TRANSLATION")
        print(f"{'▓'*80}")
        print(f"Source: {' '.join(src_tokens[1:-1])}")
        print(
            f"Target: {' '.join(generated_tokens[:-1] if generated_tokens[-1] == '<eos>' else generated_tokens)}"
        )
        print(f"\nComplete decoder sequence: {' '.join(decoder_input)}")

        return generated_tokens


def demo_with_shapes():
    """Demonstrate with tensor shapes at each step"""

    print("=" * 80)
    print("TRANSFORMER INFERENCE - STEP BY STEP")
    print("Translation Example: English → Italian")
    print("=" * 80)

    # Create model
    model = TransformerInferenceDemo()

    # Translate
    source = "I love you very much"
    translation = model.full_inference(source)

    # Show shape flow summary
    print("\n" + "=" * 80)
    print("TENSOR SHAPE FLOW SUMMARY")
    print("=" * 80)

    print("\nENCODER (once):")
    print("  Input:  [seq_len]           → Token IDs")
    print("  Embed:  [seq_len, d_model]  → After embedding + position")
    print("  Output: [seq_len, d_model]  → Encoder representations")

    print("\nDECODER (each time step):")
    print("  Step 1: [1] → [1, d_model] → [d_model] → [vocab_size] → 'Ti'")
    print("  Step 2: [2] → [2, d_model] → [d_model] → [vocab_size] → 'amo'")
    print("  Step 3: [3] → [3, d_model] → [d_model] → [vocab_size] → 'molto'")
    print("  Step 4: [4] → [4, d_model] → [d_model] → [vocab_size] → '<eos>'")

    print("\nKey insights:")
    print("  1. Encoder runs ONCE, decoder runs MULTIPLE times")
    print("  2. Decoder input grows by one token each step")
    print("  3. Only the LAST position's output is used for prediction")
    print("  4. Softmax converts logits to probabilities")
    print("  5. Argmax selects the most likely token")


if __name__ == "__main__":
    demo_with_shapes()
