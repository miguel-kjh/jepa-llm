# JEPA-LLM

This project investigates how to incorporate **Joint-Embedding Predictive Architecture (JEPA)**
mechanisms into language models. JEPA, originally proposed for self-supervised learning in vision,
trains models to predict the *latent representations* of missing or future parts of the input,
rather than reconstructing raw tokens.

The goal is to study whether this latent-prediction objective can improve the representations
and downstream behavior of both:

- **Encoder-only Transformers** (e.g., BERT-style models)
- **Decoder-only Transformers** (e.g., GPT-style autoregressive LLMs)

We will explore different ways of integrating JEPA, either as an auxiliary loss alongside standard
language modeling or as a replacement for it, and evaluate the results on a set of tasks
that are still being defined.

> 🚧 Work in progress: the experimental setup and evaluation tasks are still under development.
