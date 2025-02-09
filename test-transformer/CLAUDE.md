# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains educational implementations of Transformer models demonstrating the "Attention Is All You Need" paper concepts. The codebase includes step-by-step visualizations of transformer inference, encoder-decoder architectures, and detailed attention mechanism demonstrations.

## Development Commands

### Running the Examples

```bash
# Run the encoder-decoder transformer demonstration (translation model)
python encoder_decoder_transformer.py

# Run the detailed transformer with step-by-step visualizations
python transformer_detailed.py

# Run the simple inference demo
python transformer_inference_demo.py

# Run the translation inference step-by-step demo
python transformer_inference_steps.py
```

### Dependencies

The codebase requires:
- Python 3.12+
- PyTorch 2.2+
- NumPy 1.26+

## Architecture

### Key Components

1. **encoder_decoder_transformer.py**: Full encoder-decoder transformer implementation with gradient flow tracking
   - `EncoderDecoderTransformer`: Complete translation model
   - `ShapeTracker`: Utility for visualizing tensor shapes and gradient flow
   - Demonstrates both training and inference modes

2. **transformer_detailed.py**: Detailed transformer implementation with verbose output
   - `CompleteTransformer`: Full transformer with multi-head attention
   - Manual weight initialization (no nn.Module inheritance)
   - Step-by-step visualization of attention computation

3. **transformer_inference_demo.py**: Simple transformer for understanding inference
   - `SimpleTransformer`: Minimal transformer implementation
   - Clear step-by-step inference pipeline
   - Temperature-based sampling demonstration

4. **transformer_inference_steps.py**: Translation model inference demonstration
   - `TransformerInferenceDemo`: Shows autoregressive generation
   - Contrasts training (teacher forcing) vs inference (autoregressive)
   - Visualizes encoder-decoder interaction

### Design Patterns

- Educational focus: All implementations prioritize clarity over performance
- Verbose outputs: Each component prints detailed shape information and intermediate values
- Manual implementations: Some models avoid PyTorch modules to show explicit computations
- Small vocabularies and dimensions for demonstration purposes