import math

import pytest

from jepa_llm.config import NLRXConfig
from jepa_llm.training import DIAG_KEYS, train

from .conftest import requires_gpu_model

pytestmark = [pytest.mark.gpu, pytest.mark.slow, requires_gpu_model]


def test_overfit_eight_examples(llama_tok, sample_examples, free_gpu):
    # 20 steps on 8 examples, with a higher LR so the drop is visible.
    cfg = NLRXConfig(val_every=10)
    model, hist = train(
        cfg, llama_tok, "smoke", lam=cfg.jepa_lambda, k=cfg.jepa_k, data=sample_examples,
        lr=1e-4, max_steps=20, log_every=1, val_data=sample_examples,
    )
    assert len(hist["ntp"]) == 20
    assert hist["val_step"] == [10, 20]
    for key in [f"val_{m}" for m in DIAG_KEYS]:
        assert len(hist[key]) == len(hist["val_step"]), f"{key} must be logged at every validation step"
        assert all(math.isfinite(x) for x in hist[key]), f"NaN/inf in {key}"
    assert all(math.isfinite(x) for x in hist["ntp"] + hist["dist"]), "NaN/inf in the losses"
    assert hist["ntp"][-1] < hist["ntp"][0], "NTP loss should go down when overfitting 8 examples"
    assert hist["dist"][-1] < hist["dist"][0], "JEPA distance should go down when it is optimised"
    del model
