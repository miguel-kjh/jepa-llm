import pytest
import torch
import torch.nn.functional as F

from jepa_llm.batching import ntp_batch
from jepa_llm.config import NLRXConfig
from jepa_llm.model import build_model, compute_losses

from .conftest import requires_gpu_model

pytestmark = [pytest.mark.gpu, requires_gpu_model]


def test_lambda_zero_is_plain_cross_entropy(llama_tok, sample_examples, free_gpu):
    cfg = NLRXConfig()
    model = build_model(cfg)
    batch = sample_examples[:4]

    model.eval()
    with torch.no_grad():
        total, ntp, dist = compute_losses(model, batch, lam=0.0, k=cfg.jepa_k, tok=llama_tok)
        nb = {key: v.to(cfg.device) for key, v in ntp_batch(batch, llama_tok.pad_id).items()}
        logits = model(input_ids=nb["input_ids"], attention_mask=nb["attention_mask"], use_cache=False).logits
        manual = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]), nb["labels"][:, 1:].reshape(-1), ignore_index=-100
        )
    torch.testing.assert_close(total.float(), manual, atol=1e-2, rtol=1e-3)
    assert dist.item() > 0, "the JEPA distance is still measured for the baseline"

    # With λ = 0 the JEPA term must not reach the gradient: it matches the gradient of NTP alone.
    model.train()
    for module in model.modules():  # deterministic forward, so both gradients are comparable
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    total, _, _ = compute_losses(model, batch, lam=0.0, k=cfg.jepa_k, tok=llama_tok)
    g_total = torch.autograd.grad(total, params)
    ntp_only = model(**nb, use_cache=False).loss
    g_ntp = torch.autograd.grad(ntp_only, params)
    for a, b in zip(g_total, g_ntp):
        torch.testing.assert_close(a, b)
    del model, logits
