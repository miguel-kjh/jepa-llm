"""Padded NTP batches and packed JEPA batches with a block-causal mask.

The JEPA forward packs `[Text + k·PRED | Code]` into one sequence. The mask is causal inside each block and
blocks never attend to each other; position ids restart at 0 for Code, so each view is encoded exactly as if
it were alone. Padding rows attend only to themselves (avoids fully masked rows).

`text_mask` / `code_mask` mark the content tokens used by mean pooling: the description plus the k `[PRED]`
tokens on the Text side, the regex on the Code side (no headers, system prompt or <|eot_id|>).
"""

import torch


def ntp_batch(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    L = max(len(e["ntp_ids"]) for e in batch)
    input_ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    for i, e in enumerate(batch):
        n = len(e["ntp_ids"])
        input_ids[i, :n] = torch.tensor(e["ntp_ids"])
        labels[i, :n] = torch.tensor(e["ntp_labels"])
        attn[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def jepa_batch(batch: list[dict], k: int, pred_ids: list[int], pad_id: int) -> dict:
    seqs, spans = [], []
    for e in batch:
        a = e["text_view"] + pred_ids[:k]
        b = e["code_view"]
        seqs.append(a + b)
        spans.append((len(a), len(b)))
    B, L = len(batch), max(len(s) for s in seqs)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    position_ids = torch.zeros((B, L), dtype=torch.long)
    allowed = torch.zeros((B, L, L), dtype=torch.bool)
    text_mask = torch.zeros((B, L), dtype=torch.bool)
    code_mask = torch.zeros((B, L), dtype=torch.bool)
    pred_idx, code_idx = [], []
    for i, (e, s, (la, lb)) in enumerate(zip(batch, seqs, spans)):
        n = la + lb
        input_ids[i, :n] = torch.tensor(s)
        position_ids[i, :la] = torch.arange(la)
        position_ids[i, la:n] = torch.arange(lb)
        allowed[i, :la, :la] = torch.tril(torch.ones(la, la, dtype=torch.bool))
        allowed[i, la:n, la:n] = torch.tril(torch.ones(lb, lb, dtype=torch.bool))
        pad = torch.arange(n, L)
        allowed[i, pad, pad] = True
        ts, te = e["text_content"]
        text_mask[i, ts:te] = True
        text_mask[i, len(e["text_view"]) : la] = True  # the k [PRED] tokens
        cs, ce = e["code_content"]
        code_mask[i, la + cs : la + ce] = True
        pred_idx.append(la - 1)   # last [PRED] token (last Text token if k = 0)
        code_idx.append(n - 1)    # last Code token
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "allowed": allowed,
        "pred_idx": torch.tensor(pred_idx),
        "code_idx": torch.tensor(code_idx),
        "text_mask": text_mask,
        "code_mask": code_mask,
        "spans": spans,
    }


def additive_mask(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(B, L, L) boolean mask -> (B, 1, L, L) additive mask (0 where allowed, -inf elsewhere)."""
    mask = torch.zeros(allowed.shape[0], 1, *allowed.shape[1:], dtype=dtype)
    return mask.masked_fill(~allowed[:, None], torch.finfo(dtype).min)
