# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research project studying how to incorporate **JEPA (Joint-Embedding Predictive Architecture)** mechanisms into language models — both encoder-only (BERT-style) and decoder-only (GPT-style) Transformers. JEPA trains models to predict *latent representations* of masked/future input rather than raw tokens; here it will be explored as an auxiliary loss alongside language modeling or as a replacement for it. Evaluation tasks are not yet defined.

The repo is at an early scaffold stage: `src/jepa_llm/__init__.py` only contains a placeholder `main()`, there are no dependencies, tests, or lint config yet.

## Tooling

- Python 3.12 (`.python-version`), managed with **uv** (`uv_build` backend, `src/` layout, package `jepa_llm`).
- `uv sync` — create/update the virtualenv
- `uv run jepa-llm` — run the CLI entry point (`jepa_llm:main`)
- `uv add <pkg>` — add a dependency (updates `pyproject.toml` and `uv.lock`)

## Reference material

`wiki/papers/` holds the PDFs of related work the project builds on (LLM-JEPA, BERT-JEPA, DLLM-JEPA, data2vec, JEPA4Rec, JEPA audits for fine-tuning, etc.). Consult them when design decisions depend on how prior work formulates the JEPA objective for text.

## Conventions

- Documentation (README, code, comments) is written in English, even though the maintainer often communicates in Spanish.
