from jepa_llm.batching import ntp_batch

from .conftest import SAMPLE_PAIRS


def test_ntp_loss_only_on_regex(fake_tok):
    batch = [fake_tok.encode(s, t) for s, t in SAMPLE_PAIRS[:4]]
    nb = ntp_batch(batch, fake_tok.pad_id)
    for i, e in enumerate(batch):
        lab = nb["labels"][i]
        assert (lab != -100).sum().item() == e["answer_len"], "loss must count only the regex tokens (+ <|eot_id|>)"
        assert (lab[: len(e["prompt"])] == -100).all(), "prompt must be masked"
        assert (lab[nb["attention_mask"][i] == 0] == -100).all(), "padding must be masked"
        assert (nb["input_ids"][i][nb["attention_mask"][i] == 0] == fake_tok.pad_id).all()


def test_encode_layout(fake_tok):
    e = fake_tok.encode("lines with a vowel", "([AEIOUaeiou])")
    assert e["ntp_ids"][: len(e["prompt"])] == e["prompt"]
    assert e["ntp_ids"][-1] == fake_tok.eot_id
    assert e["answer_len"] == len("([AEIOUaeiou])") + 1  # char-level fake tokenizer + <|eot_id|>
    assert fake_tok.pad_id not in e["text_view"] + e["code_view"]
    assert not set(fake_tok.pred_ids) & set(e["text_view"]), "[PRED] tokens are added only by jepa_batch"
