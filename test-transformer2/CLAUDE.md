# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational transformer implementation for English-Italian translation with extensive visualization of encoder-decoder interaction, teacher forcing, and autoregressive generation. Designed for learning, not production.

## Commands

### Setup
```bash
# Install PyTorch and dependencies
pip install torch numpy
```

### Training
```bash
python train.py  # Default: 10 epochs, batch size 1
```

### Inference
```bash
python inference.py  # Runs translation with step-by-step visualization
```

### Visualization & Learning Tools
```bash
python encoder_decoder_flow.py  # Complete encoder-decoder flow explanation
python demo_transformer_flow.py  # Teacher forcing vs autoregressive comparison
python layer_norm_example.py    # Layer normalization demonstration
python softmax_example.py        # Softmax mechanics in attention
python simple_softmax.py         # Vocabulary softmax demonstration
```

## Architecture

### Core Model Flow

The transformer uses an encoder-decoder architecture where:

1. **Encoder** processes source sentence (English) once, creating hidden representations
2. **Decoder** receives:
   - Encoder output via cross-attention (K, V from encoder)
   - Target sequence SHIFTED RIGHT during training (teacher forcing)
3. **Output** predicts the ORIGINAL target sequence

Critical insight: During training, decoder input is `[<sos> ti amo]` to predict `[ti amo <eos>]` - the shift enables learning next-token prediction.

### File Structure

**Core Model Components** (`transformer_model.py`):
- `MultiHeadAttention`: Scaled dot-product attention with multi-head projection
- `PositionalEncoding`: Sinusoidal position embeddings
- `TransformerBlock`: Encoder block with self-attention + FFN
- `DecoderBlock`: Three sub-layers - masked self-attention, cross-attention, feed-forward
- `Encoder`: Stack of TransformerBlocks
- `Decoder`: Stack of DecoderBlocks
- `Transformer`: Complete model orchestrating encoder → decoder → output projection

**Training Pipeline** (`train.py`):
- `create_masks()`: Generates padding and look-ahead masks
- `train_epoch()`: Teacher forcing with parallel prediction, displays actual tokens
- `evaluate()`: Validation loop
- Batch size 1 for educational clarity (line 127)
- Device selection (line 124): "mps" for Apple Silicon, change to "cuda" or "cpu"

**Inference** (`inference.py`):
- `translate()`: Step-by-step autoregressive generation with verbose output
- Shows decoder building sequence one token at a time
- Device selection (line 81)

**Data** (`dataset.py`):
- `TranslationDataset`: 50 hardcoded English-Italian pairs
- `build_vocab()`: Creates vocabulary from training data
- `get_dataloaders()`: Returns train/val/test splits
- Special tokens: `<sos>`, `<eos>`, `<unk>`, `<pad>`

### Attention Mechanisms

Three types of attention in the model:

1. **Encoder Self-Attention**: 
   - Q=K=V from encoder
   - Bidirectional (no masking)
   - Builds contextual representations

2. **Decoder Masked Self-Attention**: 
   - Q=K=V from decoder  
   - Causal mask prevents future token access
   - Maintains autoregressive property

3. **Cross-Attention** (critical for translation):
   - **Query (Q)**: From decoder - "What am I trying to generate?"
   - **Key (K)**: From encoder - "What source information can I search?"
   - **Value (V)**: From encoder - "What information can I retrieve?"
   - Enables decoder to attend to ALL encoder positions

## Model Configuration

Current settings in `train.py`:
- `batch_size`: 1 (line 127)
- `num_epochs`: 10 (line 135)
- `d_model`: 128
- `n_heads`: 4  
- `n_layers`: 2
- `d_ff`: 256
- `learning_rate`: 0.001
- `dropout`: 0.1

## Key Modifications

### Training Parameters
- Line 135 in `train.py`: Adjust `num_epochs` (50+ for reasonable quality)
- Line 127 in `train.py`: Change `batch_size` (currently 1 for clarity)
- Line 132 in `train.py`: Modify learning rate

### Hardware Selection
- Line 124 (`train.py`): Change device from "mps" to "cuda" or "cpu"
- Line 81 (`inference.py`): Match device setting with training

### Verbosity Control
- `transformer_model.py`: Pass `verbose=True` to Transformer constructor
- Internal component printing disabled by default

## Training vs Inference

**Training (Parallel)**:
- Single forward pass processes entire sequence
- Decoder sees correct previous tokens (teacher forcing)
- Input: `<sos> ti amo`, Target: `ti amo <eos>`
- Loss computed on all positions simultaneously

**Inference (Sequential)**:
- Multiple forward passes, one per generated token
- Decoder sees its own predictions (autoregressive)
- Starts with `<sos>`, builds sequence incrementally
- Stops at `<eos>` or max length

## Visualization Output Format

Training displays:
```
🌍 Translation Pair:
   English: 'I love you'
   Italian: 'ti amo'

🔄 Encoder-Decoder Flow:
   1️⃣  Encoder processes: <sos> I love you <eos>
   2️⃣  Decoder receives:
        Input (shifted):  <sos> ti amo
        Target (predict): ti amo <eos>
```

Inference displays step-by-step generation with decoder input/output at each step.

## Dataset Details

50 phrase pairs covering:
- Basic greetings and expressions
- Common verbs and phrases
- Numbers and days
- Questions and responses

Vocabulary sizes:
- English: ~90 tokens
- Italian: ~95 tokens
- Shared special tokens: 4

## Performance Notes

- Model requires 50+ epochs for basic translation quality
- Batch size 1 intentionally limits training speed for educational clarity
- MPS acceleration available for Apple Silicon
- No pre-trained weights; trains from scratch each time