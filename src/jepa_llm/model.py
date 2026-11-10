"""LoRA model and the LLM-JEPA losses.

L = L_NTP(Text -> Code) + lambda * (1 - cos(Pred(Enc(Text)), Enc(Code)))
"""

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from .batching import additive_mask, jepa_batch, ntp_batch
from .config import NLRXConfig
from .tokenization import NLRXTokenizer


def build_model(cfg: NLRXConfig):
    base = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=cfg.dtype).to(cfg.device)
    torch.manual_seed(cfg.seed)  # identical LoRA initialisation for every run
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_targets), task_type="CAUSAL_LM",
    )
    return get_peft_model(base, lora)


def backbone_of(model):
    """Decoder stack (without LM head) of a causal LM, with or without a PEFT wrapper."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model


def packed_hidden_states(backbone, jb: dict) -> torch.Tensor:
    """Last hidden states of a packed JEPA batch, using the block-causal 4D mask."""
    emb = backbone.get_input_embeddings().weight
    return backbone(
        input_ids=jb["input_ids"].to(emb.device),
        position_ids=jb["position_ids"].to(emb.device),
        attention_mask=additive_mask(jb["allowed"], emb.dtype).to(emb.device),
        use_cache=False,
    ).last_hidden_state


POOLINGS = ("last", "mean")


def pool(h: torch.Tensor, jb: dict, pooling: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, d) float32 Pred and Code vectors from the packed hidden states `h` (B, L, d).

    The single place where the loss and the diagnostics read the views, so both see the same vectors.
    "last": last [PRED] token (last Text token if k = 0) and last Code token.
    "mean": mean over `text_mask` (description + [PRED]s) and over `code_mask` (regex).
    """
    if pooling == "last":
        rows = torch.arange(h.shape[0], device=h.device)
        return h[rows, jb["pred_idx"].to(h.device)].float(), h[rows, jb["code_idx"].to(h.device)].float()
    if pooling == "mean":
        def masked_mean(mask):
            m = mask.to(h.device, torch.float32)[..., None]
            return (h.float() * m).sum(dim=1) / m.sum(dim=1)
        return masked_mean(jb["text_mask"]), masked_mean(jb["code_mask"])
    raise ValueError(f"unknown pooling {pooling!r}, expected one of {POOLINGS}")


def jepa_distance(model, jb: dict, pooling: str = "last") -> torch.Tensor:
    # Same LLM as encoder and predictor; LoRA layers are active inside the backbone.
    h = packed_hidden_states(backbone_of(model), jb)
    pred, target = pool(h, jb, pooling)  # no stop-gradient, as in LLM-JEPA
    return (1 - F.cosine_similarity(pred, target, dim=-1)).mean()


def compute_losses(model, batch: list[dict], lam: float, k: int, tok: NLRXTokenizer, pooling: str = "last"):
    """Returns (total loss, detached NTP loss, detached JEPA distance)."""
    device = backbone_of(model).get_input_embeddings().weight.device
    nb = {key: v.to(device) for key, v in ntp_batch(batch, tok.pad_id).items()}
    ntp = model(**nb, use_cache=False).loss
    jb = jepa_batch(batch, k, tok.pred_ids, tok.pad_id)
    if lam > 0:
        dist = jepa_distance(model, jb, pooling)
        total = ntp + lam * dist
    else:
        with torch.no_grad():  # baseline: the distance is only logged
            dist = jepa_distance(model, jb, pooling)
        total = ntp
    return total, ntp.detach(), dist.detach()
