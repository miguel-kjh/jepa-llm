"""Greedy decoding and exact-match accuracy."""

import torch

from .tokenization import NLRXTokenizer


@torch.no_grad()
def generate_predictions(
    model, data: list[dict], tok: NLRXTokenizer, bs: int = 64, max_new_tokens: int = 64
) -> list[str]:
    model.eval()
    device = model.get_input_embeddings().weight.device
    preds = []
    for i in range(0, len(data), bs):
        chunk = data[i : i + bs]
        L = max(len(e["prompt"]) for e in chunk)
        input_ids = torch.full((len(chunk), L), tok.pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, e in enumerate(chunk):  # left padding for generation
            n = len(e["prompt"])
            input_ids[j, L - n :] = torch.tensor(e["prompt"])
            attn[j, L - n :] = 1
        out = model.generate(
            input_ids=input_ids.to(device), attention_mask=attn.to(device),
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_id, eos_token_id=tok.eot_id,
        )
        preds += [s.strip() for s in tok.batch_decode(out[:, L:], skip_special_tokens=True)]
    return preds


def exact_match(preds: list[str], data: list[dict]) -> float:
    return sum(p == e["regex"] for p, e in zip(preds, data)) / len(data)
