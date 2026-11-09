# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research project studying how to incorporate **JEPA (Joint-Embedding Predictive Architecture)** mechanisms into language models — both encoder-only (BERT-style) and decoder-only (GPT-style) Transformers. JEPA trains models to predict *latent representations* of masked/future input rather than raw tokens; here it will be explored as an auxiliary loss alongside language modeling or as a replacement for it. Evaluation tasks are not yet defined.

The first experiment, LLM-JEPA vs. plain LoRA fine-tuning on NL-RX-Synth (Llama-3.2-1B-Instruct), lives in the `jepa_llm` package:

- `config.py`: `NLRXConfig`, the dataclass with every hyperparameter and the data/output paths.
- `data.py`: loads the pairs and the fixed split (`outputs/llm_jepa_nlrx/split.json` is reused if present).
- `tokenization.py`: `NLRXTokenizer`, which builds the NTP example, the Text/Code views and the `[PRED]` ids.
- `batching.py`: NTP batches and packed JEPA batches with a block-causal 4D mask.
- `model.py`: the LoRA model and the losses.
- `training.py`, `evaluation.py`, `experiment.py`: the training loop, exact-match evaluation, and run orchestration plus saved artifacts.

`notebooks/llm_jepa_nlrx.ipynb` only drives the package. Its saved outputs are from the original run, and the write-up is in `wiki/experiments/`.

## Tooling

- Python 3.12 (`.python-version`), managed with **uv** (`uv_build` backend, `src/` layout, package `jepa_llm`).
- `uv sync` — create/update the virtualenv
- `uv run jepa-llm nlrx --run {baseline,llm_jepa,all}`: train and evaluate the NL-RX experiment. This overwrites `outputs/llm_jepa_nlrx/`, and one run takes about 20 minutes on a 4070.
- `uv run pytest`: run the tests. Tests marked `gpu` are skipped when CUDA or the cached Llama model is missing. `-m "not slow"` skips the training smoke test.
- If `uv add`/`uv remove` breaks `import torch` (`libcudnn.so.9` not found), run `uv sync` with `--reinstall-package` on the `nvidia-*` wheels. Stale cu13 wheels shared the `nvidia/` directory with the cu12 ones.
- `uv add <pkg>` — add a dependency (updates `pyproject.toml` and `uv.lock`)

## Reference material

`wiki/papers/` holds the PDFs of related work the project builds on (LLM-JEPA, BERT-JEPA, DLLM-JEPA, data2vec, JEPA4Rec, JEPA audits for fine-tuning, etc.). Consult them when design decisions depend on how prior work formulates the JEPA objective for text.

## Conventions

- Documentation (README, code, comments) is written in English, even though the maintainer often communicates in Spanish.
