"""Collapse diagnostics for the JEPA embeddings, read through `model.pool` exactly like the loss."""

import torch
import torch.nn.functional as F

from .batching import jepa_batch
from .model import backbone_of, packed_hidden_states, pool
from .tokenization import NLRXTokenizer


@torch.no_grad()
def view_embeddings(
    model, examples: list[dict], tok: NLRXTokenizer, k: int, bs: int = 32, pooling: str = "last"
) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, d) float32 CPU tensors with the Pred and Code vectors, read exactly where `jepa_distance` reads them."""
    was_training = model.training
    model.eval()
    preds, codes = [], []
    for i in range(0, len(examples), bs):
        jb = jepa_batch(examples[i : i + bs], k, tok.pred_ids, tok.pad_id)
        pred, code = pool(packed_hidden_states(backbone_of(model), jb), jb, pooling)
        preds.append(pred.cpu())
        codes.append(code.cpu())
    model.train(was_training)
    return torch.cat(preds), torch.cat(codes)


def effective_rank(x: torch.Tensor, center: bool = True) -> float:
    """Roy & Vetterli (2007): exp of the entropy of the normalised singular values of `x` (rows = samples)."""
    x = x.double()
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    if s.sum() == 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    return torch.exp(-(p * p.log()).sum()).item()


def mean_offdiag_cosine(x: torch.Tensor) -> float:
    """Mean cosine between different rows of `x` (the diagonal is excluded)."""
    n = x.shape[0]
    z = F.normalize(x.double(), dim=-1)
    g = z @ z.T
    return ((g.sum() - g.diagonal().sum()) / (n * (n - 1))).item()


def retrieval_top1(queries: torch.Tensor, keys: torch.Tensor) -> float:
    """Fraction of rows i whose most cosine-similar key is keys[i] (chance level: 1 / N)."""
    sim = F.normalize(queries.double(), dim=-1) @ F.normalize(keys.double(), dim=-1).T
    return (sim.argmax(dim=1) == torch.arange(len(queries))).double().mean().item()


def collapse_metrics(pred: torch.Tensor, code: torch.Tensor) -> dict[str, float]:
    """`cos_pair` and `cos_between` measure the raw geometry, which a shared direction can dominate.

    The centred metrics remove the mean of each view first, so they show whether any pair-specific
    signal is left: `cos_pair_centered` (same pair, centred) and `top1_centered` (Text -> Code retrieval
    among the N examples, centred).
    """
    pred_c = pred.double() - pred.double().mean(dim=0, keepdim=True)
    code_c = code.double() - code.double().mean(dim=0, keepdim=True)
    return {
        "cos_pair": F.cosine_similarity(pred.double(), code.double(), dim=-1).mean().item(),
        "cos_between": mean_offdiag_cosine(code),
        "erank_code": effective_rank(code),
        "erank_text": effective_rank(pred),
        "cos_pair_centered": F.cosine_similarity(pred_c, code_c, dim=-1).mean().item(),
        "top1_centered": retrieval_top1(pred_c, code_c),
    }
