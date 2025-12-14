# GPT from Scratch

A code-along implementation of Andrej Karpathy's "GPT from Scratch" tutorial, building a decoder-only transformer model from the ground up.

## Overview

This project implements a character-level GPT model to understand the fundamental architecture behind large language models. By building each component from scratch, it provides deep insights into how modern transformers actually work.

## Key Learnings

Through this hands-on implementation, I gained understanding of:

- **Self-Attention Mechanisms**: How models dynamically weigh the relevance of different tokens in a sequence, enabling context-aware predictions
- **Multi-Head Attention**: The power of parallel attention mechanisms that capture different aspects of relationships between tokens
- **Positional Encodings**: Their crucial role in preserving sequential information in transformer architectures
- **Layer Normalization**: How normalization contributes to training stability in deep networks
- **Residual Connections**: Their importance in facilitating gradient flow through deep architectures
- **Autoregressive Training**: The dynamics of training models to predict the next token in a sequence
- **Transformer Architecture**: A comprehensive understanding of the building blocks that power modern LLMs


## Resources

- [Andrej Karpathy's Video Tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Original "Attention Is All You Need" Paper](https://arxiv.org/abs/1706.03762)
