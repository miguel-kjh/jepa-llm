import pytest
import torch

from jepa_llm.batching import additive_mask, jepa_batch

from .conftest import SAMPLE_PAIRS


@pytest.mark.parametrize("k", [0, 1, 2])
def test_jepa_block_causal_mask(fake_tok, k):
    batch = [fake_tok.encode(s, t) for s, t in SAMPLE_PAIRS[:4]]
    jb = jepa_batch(batch, k, fake_tok.pred_ids, fake_tok.pad_id)
    L = jb["input_ids"].shape[1]
    for i, (la, lb) in enumerate(jb["spans"]):
        al = jb["allowed"][i]
        n = la + lb
        assert not al[la:n, :la].any(), "Code must not attend to Text or [PRED]"
        assert not al[:la, la:n].any(), "Text must not attend to Code"
        assert torch.equal(al[:la, :la], torch.tril(torch.ones(la, la, dtype=torch.bool))), "Text block must be causal"
        assert torch.equal(al[la:n, la:n], torch.tril(torch.ones(lb, lb, dtype=torch.bool))), "Code block must be causal"
        assert torch.equal(al[n:, n:], torch.eye(L - n, dtype=torch.bool)), "padding attends only to itself"
        assert not al[n:, :n].any() and not al[:n, n:].any(), "padding is isolated from the views"
        assert jb["position_ids"][i, la].item() == 0, "Code positions must restart at 0"
        assert jb["pred_idx"][i].item() == la - 1 and jb["code_idx"][i].item() == n - 1
        assert jb["input_ids"][i, :n].tolist() == batch[i]["text_view"] + fake_tok.pred_ids[:k] + batch[i]["code_view"]
        if k > 0:
            assert jb["input_ids"][i, la - 1].item() == fake_tok.pred_ids[k - 1], "last Text-side token must be [PRED]"


def test_additive_mask():
    allowed = torch.tensor([[[True, False], [True, True]]])
    mask = additive_mask(allowed, torch.float32)
    assert mask.shape == (1, 1, 2, 2)
    assert mask[0, 0, 0, 0] == 0 and mask[0, 0, 1, 0] == 0 and mask[0, 0, 1, 1] == 0
    assert mask[0, 0, 0, 1] == torch.finfo(torch.float32).min
