"""Configuration of the LLM-JEPA experiment on NL-RX-Synth."""

from dataclasses import dataclass, field
from pathlib import Path

import torch


def default_root() -> Path:
    """Repository root, whether the code runs from the repo or from `notebooks/`."""
    cwd = Path.cwd()
    return cwd.parent if cwd.name == "notebooks" else cwd


@dataclass
class NLRXConfig:
    root: Path = field(default_factory=default_root)
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    date_string: str = "26 Jul 2024"  # fixed date in the Llama chat template, so prompts do not change day to day

    # One fixed split and one LoRA initialisation shared by both runs (not a seed sweep).
    seed: int = 0
    split_fracs: tuple[float, float, float] = (0.8, 0.1, 0.1)

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    # Training
    lr: float = 2e-5
    batch_size: int = 8
    grad_accum: int = 1  # effective batch size = batch_size * grad_accum
    epochs: int = 4
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    log_every: int = 25
    val_every: int = 250
    diag_examples: int = 300  # first N val examples used for the collapse metrics

    # LLM-JEPA (fixed in advance, no grid search)
    jepa_k: int = 1
    jepa_lambda: float = 1.0

    # Evaluation
    eval_batch_size: int = 64
    max_new_tokens: int = 64

    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    @property
    def data_dir(self) -> Path:
        return self.root / "datasets" / "raw" / "NL-RX-Synth"

    @property
    def out_dir(self) -> Path:
        return self.root / "outputs" / "llm_jepa_nlrx"
