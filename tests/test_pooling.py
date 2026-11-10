"""Mean pooling over the content tokens of each JEPA view (issue #3)."""

import pytest
import torch
import torch.nn.functional as F

from jepa_llm.batching import jepa_batch
from jepa_llm.model import pool

from .conftest import SAMPLE_PAIRS, FakeLlamaTokenizer, requires_gpu_model


def _fake_decode(ids):
    inv = {i: t for t, i in FakeLlamaTokenizer().vocab.items()}
    return "".join(inv[i] if i in inv else chr(i - 1000) for i in ids)


# --- CPU: the masks select exactly the content tokens ------------------------------------------------------------


@pytest.mark.parametrize("k", [0, 1, 2])
def test_masks_decode_to_description_and_regex(fake_tok, k):
    batch = [fake_tok.encode(s, t) for s, t in SAMPLE_PAIRS[:4]]
    jb = jepa_batch(batch, k, fake_tok.pred_ids, fake_tok.pad_id)
    pred_str = "".join(f"<|reserved_special_token_{i}|>" for i in range(k))
    for i, (desc, regex) in enumerate(SAMPLE_PAIRS[:4]):
        ids = jb["input_ids"][i]
        text = _fake_decode(ids[jb["text_mask"][i]].tolist())
        code = _fake_decode(ids[jb["code_mask"][i]].tolist())
        assert text == desc + pred_str, "Text mask = description + the k [PRED] tokens, nothing else"
        assert code == regex, "Code mask = the regex, nothing else"
        for special in ("<|eot_id|>", "<|end_header_id|>", "<|begin_of_text|>"):
            assert special not in text + code
        assert not (jb["text_mask"][i] & jb["code_mask"][i]).any()
        la, lb = jb["spans"][i]
        assert not jb["text_mask"][i, la:].any() and not jb["code_mask"][i, :la].any()
        assert not jb["code_mask"][i, la + lb :].any(), "padding is never pooled"


def test_encode_rejects_content_not_found(fake_tok):
    # A chat template that rewrites the content (here: trims it) must trip the assertion.
    with pytest.raises(AssertionError, match="content tokens not found"):
        fake_tok.content_span(fake_tok.chat_ids([{"role": "user", "content": "abc"}]), "abc ")


def test_last_pooling_reads_pred_and_code_indices(fake_tok):
    batch = [fake_tok.encode(s, t) for s, t in SAMPLE_PAIRS[:4]]
    jb = jepa_batch(batch, 1, fake_tok.pred_ids, fake_tok.pad_id)
    h = torch.randn(*jb["input_ids"].shape, 16, generator=torch.Generator().manual_seed(0)).bfloat16()
    pred, code = pool(h, jb, "last")
    rows = torch.arange(h.shape[0])
    # Exactly what jepa_distance read before pooling was configurable.
    assert torch.equal(pred, h[rows, jb["pred_idx"]].float())
    assert torch.equal(code, h[rows, jb["code_idx"]].float())


def test_mean_pooling_matches_manual_mean(fake_tok):
    batch = [fake_tok.encode(s, t) for s, t in SAMPLE_PAIRS[:4]]
    jb = jepa_batch(batch, 2, fake_tok.pred_ids, fake_tok.pad_id)
    h = torch.randn(*jb["input_ids"].shape, 16, generator=torch.Generator().manual_seed(0))
    pred, code = pool(h, jb, "mean")
    for i in range(len(batch)):
        torch.testing.assert_close(pred[i], h[i, jb["text_mask"][i]].mean(0))
        torch.testing.assert_close(code[i], h[i, jb["code_mask"][i]].mean(0))
    with pytest.raises(ValueError):
        pool(h, jb, "max")


# --- GPU: real tokenizer and model --------------------------------------------------------------------------------


@pytest.mark.gpu
@requires_gpu_model
def test_masks_with_llama_tokenizer(llama_tok, sample_examples):
    jb = jepa_batch(sample_examples, 1, llama_tok.pred_ids, llama_tok.pad_id)
    for i, (desc, regex) in enumerate(SAMPLE_PAIRS):
        ids = jb["input_ids"][i]
        assert ids[jb["text_mask"][i]].tolist() == llama_tok.tok(desc, add_special_tokens=False).input_ids + llama_tok.pred_ids[:1]
        assert ids[jb["code_mask"][i]].tolist() == llama_tok.tok(regex, add_special_tokens=False).input_ids


@pytest.mark.gpu
@requires_gpu_model
@torch.no_grad()
def test_mean_pooled_packed_forward_matches_separate_forwards(llama_tok, sample_examples, free_gpu):
    from transformers import AutoModelForCausalLM

    from .conftest import MODEL_NAME

    k = 1
    ref = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to("cuda").eval()
    batch = sample_examples[:4]
    jb = jepa_batch(batch, k, llama_tok.pred_ids, llama_tok.pad_id)
    from jepa_llm.model import packed_hidden_states

    pred, code = pool(packed_hidden_states(ref.model, jb), jb, "mean")
    for i, e in enumerate(batch):
        a = torch.tensor([e["text_view"] + llama_tok.pred_ids[:k]], device="cuda")
        b = torch.tensor([e["code_view"]], device="cuda")
        ha = ref.model(input_ids=a, use_cache=False).last_hidden_state[0]
        hb = ref.model(input_ids=b, use_cache=False).last_hidden_state[0]
        ts, te = e["text_content"]
        cs, ce = e["code_content"]
        sep_pred = torch.cat([ha[ts:te], ha[len(e["text_view"]) :]]).mean(0)
        torch.testing.assert_close(pred[i], sep_pred, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(code[i], hb[cs:ce].mean(0), atol=1e-4, rtol=1e-4)
    del ref


@pytest.mark.gpu
@requires_gpu_model
def test_last_pooling_distance_is_unchanged(llama_tok, sample_examples, free_gpu):
    from jepa_llm.config import NLRXConfig
    from jepa_llm.model import backbone_of, build_model, jepa_distance, packed_hidden_states

    cfg = NLRXConfig()
    model = build_model(cfg).eval()
    jb = jepa_batch(sample_examples, cfg.jepa_k, llama_tok.pred_ids, llama_tok.pad_id)
    with torch.no_grad():
        dist = jepa_distance(model, jb, "last")
        assert torch.equal(dist, jepa_distance(model, jb)), "'last' is the default"
        # The formula used before issue #3, inlined.
        h = packed_hidden_states(backbone_of(model), jb)
        rows = torch.arange(h.shape[0], device=h.device)
        old_pred = h[rows, jb["pred_idx"].to(h.device)].float()
        old_target = h[rows, jb["code_idx"].to(h.device)].float()
        old = (1 - F.cosine_similarity(old_pred, old_target, dim=-1)).mean()
        assert torch.equal(dist, old)
        assert not torch.isclose(jepa_distance(model, jb, "mean"), dist), "mean pooling reads other vectors"
    del model
