import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def print_step(step_name, tensor=None, shape=None):
    """Helper to print each step clearly"""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")
    if tensor is not None:
        if isinstance(tensor, torch.Tensor):
            print(f"Shape: {tensor.shape}")
            print(f"Values:\n{tensor}")
        else:
            print(f"Values: {tensor}")
    elif shape is not None:
        print(f"Shape: {shape}")


class SimpleTransformer:
    def __init__(self):
        # Vocabulary
        self.vocab = {
            "<pad>": 0,
            "<start>": 1,
            "hi": 2,
            ",": 3,
            "hello": 4,
            "world": 5,
            "!": 6,
        }
        self.id_to_word = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # Model dimensions (kept small for clarity)
        self.d_model = 8  # Embedding dimension
        self.n_heads = 2  # Number of attention heads
        self.d_head = self.d_model // self.n_heads  # 4
        self.max_seq_len = 10

        # Initialize model components
        self.init_weights()

    def init_weights(self):
        """Initialize all weight matrices"""
        print_step("INITIALIZING MODEL WEIGHTS")

        # Token embeddings: maps token IDs to vectors
        torch.manual_seed(42)  # For reproducibility
        self.token_embeddings = torch.randn(self.vocab_size, self.d_model) * 0.1
        print(f"Token Embeddings shape: {self.token_embeddings.shape}")
        print(f"  Maps {self.vocab_size} tokens -> {self.d_model}-dim vectors")

        # Positional embeddings: adds position information
        self.pos_embeddings = torch.randn(self.max_seq_len, self.d_model) * 0.1
        print(f"Position Embeddings shape: {self.pos_embeddings.shape}")

        # Attention weights (for single head shown here)
        self.W_query = torch.randn(self.d_model, self.d_model) * 0.1
        self.W_key = torch.randn(self.d_model, self.d_model) * 0.1
        self.W_value = torch.randn(self.d_model, self.d_model) * 0.1
        print(f"Query/Key/Value matrices shape: {self.W_query.shape}")

        # Output projection (to vocabulary)
        self.W_output = torch.randn(self.d_model, self.vocab_size) * 0.1
        print(f"Output projection shape: {self.W_output.shape}")
        print(f"  Maps {self.d_model}-dim vectors -> {self.vocab_size} vocab logits")

    def tokenize(self, text):
        """Convert text to token IDs"""
        tokens = text.lower().replace(",", " ,").split()
        token_ids = [self.vocab.get(token, 0) for token in tokens]
        return torch.tensor(token_ids)

    def embed_tokens(self, token_ids):
        """Step 1: Convert token IDs to embeddings"""
        print_step("1. TOKEN EMBEDDING", token_ids)

        # Look up embeddings for each token
        embeddings = self.token_embeddings[token_ids]
        print(f"\nToken IDs: {token_ids.tolist()}")
        print(f"Words: {[self.id_to_word[id.item()] for id in token_ids]}")
        print(f"\nEmbeddings shape: {embeddings.shape}")
        print(f"Each token -> {self.d_model}-dimensional vector")

        return embeddings

    def add_positional_encoding(self, embeddings):
        """Step 2: Add positional information"""
        print_step("2. ADD POSITIONAL ENCODING")

        seq_len = embeddings.shape[0]
        positions = self.pos_embeddings[:seq_len]

        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Position encodings shape: {positions.shape}")

        embedded = embeddings + positions
        print(f"\nAfter adding positions: {embedded.shape}")
        print("This helps the model know word order!")

        return embedded

    def self_attention(self, x):
        """Step 3: Self-attention mechanism"""
        print_step("3. SELF-ATTENTION MECHANISM")

        seq_len = x.shape[0]

        # Compute Q, K, V
        print("\n3a. Computing Query, Key, Value matrices")
        Q = torch.matmul(x, self.W_query)
        K = torch.matmul(x, self.W_key)
        V = torch.matmul(x, self.W_value)
        print(f"Q shape: {Q.shape}  (each token queries others)")
        print(f"K shape: {K.shape}  (each token provides a key)")
        print(f"V shape: {V.shape}  (each token provides a value)")

        # Compute attention scores
        print("\n3b. Computing attention scores (Q @ K.T)")
        scores = torch.matmul(Q, K.T) / math.sqrt(self.d_model)
        print(f"Attention scores shape: {scores.shape}")
        print(f"Scores (before softmax):\n{scores}")

        # Apply causal mask (can't attend to future tokens)
        print("\n3c. Applying causal mask (no looking ahead!)")
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * -1e9
        scores = scores + mask
        print(f"Masked scores:\n{scores}")

        # Apply softmax to get attention weights
        print("\n3d. Applying softmax to get attention weights")
        attn_weights = F.softmax(scores, dim=-1)
        print(f"Attention weights shape: {attn_weights.shape}")
        print(f"Attention weights (each row sums to 1):\n{attn_weights}")

        # Apply attention to values
        print("\n3e. Applying attention to values")
        attended = torch.matmul(attn_weights, V)
        print(f"Output shape: {attended.shape}")
        print(f"Each token is now a weighted sum of values it can see")

        return attended, attn_weights

    def predict_next_token(self, hidden_states, temperature=1.0):
        """Step 4: Predict next token"""
        print_step("4. PREDICT NEXT TOKEN")

        # Take the last token's hidden state
        last_hidden = hidden_states[-1]
        print(f"Last token hidden state shape: {last_hidden.shape}")

        # Project to vocabulary size
        print("\n4a. Project to vocabulary size")
        logits = torch.matmul(last_hidden, self.W_output)
        print(f"Logits shape: {logits.shape}")
        print(f"Raw logits: {logits}")

        # Apply temperature
        print(f"\n4b. Apply temperature (T={temperature})")
        if temperature != 1.0:
            logits = logits / temperature
            print(f"Logits after temperature: {logits}")

        # Apply softmax to get probabilities
        print("\n4c. Apply softmax to get probabilities")
        probs = F.softmax(logits, dim=-1)
        print(f"Probabilities shape: {probs.shape}")
        print(f"\nProbability distribution:")
        for i, prob in enumerate(probs):
            word = self.id_to_word[i]
            print(f"  {word:10s}: {prob:.4f}")

        # Get most likely token
        print("\n4d. Select token (argmax for greedy, or sample)")
        next_token_id = torch.argmax(probs)
        next_word = self.id_to_word[next_token_id.item()]
        print(f"Selected token ID: {next_token_id} -> '{next_word}'")
        print(f"Probability: {probs[next_token_id]:.4f}")

        return next_token_id, probs

    def inference(self, text, temperature=1.0):
        """Complete inference pipeline"""
        print("\n" + "=" * 80)
        print(f"FULL INFERENCE PIPELINE")
        print(f"Input: '{text}'")
        print(f"Temperature: {temperature}")
        print("=" * 80)

        # Tokenize
        token_ids = self.tokenize(text)

        # Embed tokens
        embeddings = self.embed_tokens(token_ids)

        # Add positional encoding
        embedded = self.add_positional_encoding(embeddings)

        # Self-attention
        attended, attn_weights = self.self_attention(embedded)

        # Predict next token
        next_token, probs = self.predict_next_token(attended, temperature)

        return next_token, probs, attn_weights


def main():
    print("TRANSFORMER INFERENCE DEMO")
    print("Understanding 'Attention Is All You Need'")
    print("-" * 80)

    # Create model
    model = SimpleTransformer()

    # Test with your example
    input_text = "hi, hello"

    print(f"\n{'*'*80}")
    print(f"RUNNING INFERENCE WITH INPUT: '{input_text}'")
    print(f"Expected next word: 'world'")
    print(f"{'*'*80}")

    # Run with different temperatures
    for temp in [1.0, 0.5, 2.0]:
        print(f"\n\n{'#'*80}")
        print(f"TEMPERATURE = {temp}")
        print(f"  Low temp (0.5) = more focused/deterministic")
        print(f"  High temp (2.0) = more random/creative")
        print(f"{'#'*80}")

        next_token, probs, attn = model.inference(input_text, temperature=temp)

    # Visualize attention
    print("\n" + "=" * 80)
    print("ATTENTION VISUALIZATION")
    print("=" * 80)
    print("Shows which tokens each position attends to:")
    print("(Rows = queries, Columns = keys)")
    tokens = input_text.lower().replace(",", " ,").split()
    print(f"\nTokens: {tokens}")
    print("\nAttention matrix:")
    print("       ", "  ".join(f"{t:8s}" for t in tokens))
    for i, row in enumerate(attn):
        print(f"{tokens[i]:8s}", "  ".join(f"{val:.4f}" for val in row[: len(tokens)]))


if __name__ == "__main__":
    main()
