"""Packed forward ≡ separate forwards.

Done in fp32 on the base model so the tolerance can be tight. If the custom 4D mask were ignored or
mishandled, Enc(Code) would see the Text block and this test would fail.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from jepa_llm.batching import jepa_batch
from jepa_llm.model import packed_hidden_states

from .conftest import MODEL_NAME, requires_gpu_model

pytestmark = [pytest.mark.gpu, requires_gpu_model]


@torch.no_grad()
def test_packed_forward_matches_separate_forwards(llama_tok, sample_examples, free_gpu):
    k = 1
    ref = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to("cuda").eval()
    batch = sample_examples[:4]
    jb = jepa_batch(batch, k, llama_tok.pred_ids, llama_tok.pad_id)
    h = packed_hidden_states(ref.model, jb)
    for i, e in enumerate(batch):
        a = torch.tensor([e["text_view"] + llama_tok.pred_ids[:k]], device="cuda")
        b = torch.tensor([e["code_view"]], device="cuda")
        ha = ref.model(input_ids=a, use_cache=False).last_hidden_state[0, -1]
        hb = ref.model(input_ids=b, use_cache=False).last_hidden_state[0, -1]
        torch.testing.assert_close(h[i, jb["pred_idx"][i]], ha, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(h[i, jb["code_idx"][i]], hb, atol=1e-4, rtol=1e-4)
    # The test can fail: without the block mask, Code would attend to Text.
    plain = ref.model(input_ids=jb["input_ids"][:1].to("cuda"), use_cache=False).last_hidden_state
    leak = (plain[0, jb["code_idx"][0]] - h[0, jb["code_idx"][0]]).abs().max().item()
    assert leak > 1e-2, "a plain causal forward should differ from the packed one"
    del ref
