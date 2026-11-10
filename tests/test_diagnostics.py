import json
import math

import matplotlib
import pytest
import torch

from jepa_llm.batching import jepa_batch
from jepa_llm.config import NLRXConfig
from jepa_llm.diagnostics import (
    collapse_metrics,
    effective_rank,
    mean_offdiag_cosine,
    retrieval_top1,
    view_embeddings,
)

from .conftest import MODEL_NAME, requires_gpu_model

matplotlib.use("Agg")


# --- CPU: metric definitions -------------------------------------------------------------------------------------


def test_effective_rank_of_known_matrices():
    assert effective_rank(torch.eye(4), center=False) == pytest.approx(4.0)
    rank1 = torch.outer(torch.arange(1.0, 11.0), torch.randn(16))
    assert effective_rank(rank1, center=False) == pytest.approx(1.0, abs=1e-4)
    assert effective_rank(rank1) == pytest.approx(1.0, abs=1e-4), "centring keeps a rank-1 matrix at rank 1"


def test_isotropic_gaussian_is_not_collapsed():
    x = torch.randn(300, 64, generator=torch.Generator().manual_seed(0))
    assert abs(mean_offdiag_cosine(x)) < 0.02
    assert effective_rank(x) > 55


def test_shared_direction_is_collapsed():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, generator=g) + 1e-3 * torch.randn(300, 64, generator=g)
    assert mean_offdiag_cosine(x) > 0.99


def test_diagonal_is_excluded_from_between_cosine():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # with the diagonal the mean would be 0.5
    assert mean_offdiag_cosine(x) == pytest.approx(0.0)


def test_constant_embeddings_do_not_give_nan():
    x = torch.ones(10, 8)  # centred, this is the all-zero matrix
    m = collapse_metrics(x, x)
    assert all(math.isfinite(v) for v in m.values())
    assert m["erank_code"] == 0.0 and m["cos_between"] == pytest.approx(1.0)


def test_collapse_metrics_pair_cosine():
    pred = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    code = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert collapse_metrics(pred, code)["cos_pair"] == pytest.approx(0.5)


def test_retrieval_top1():
    keys = torch.eye(4)
    assert retrieval_top1(keys, keys) == 1.0
    assert retrieval_top1(keys, keys[[1, 0, 2, 3]]) == 0.5


def test_centred_metrics_see_pair_signal_hidden_by_a_shared_direction():
    # Collapsed geometry (huge shared component) with a small pair-specific residual.
    g = torch.Generator().manual_seed(0)
    signal = torch.randn(300, 64, generator=g)
    shared = 100 * torch.randn(64, generator=g)
    pred = shared + 0.1 * (signal + 0.1 * torch.randn(300, 64, generator=g))
    code = shared + 0.1 * signal
    m = collapse_metrics(pred, code)
    assert m["cos_between"] > 0.999, "raw geometry looks fully collapsed"
    assert m["cos_pair_centered"] > 0.9 and m["top1_centered"] > 0.9, "centring recovers the pair signal"
    # Same shared direction, but residuals independent between views: no pair signal left.
    unrelated = collapse_metrics(shared + 0.1 * torch.randn(300, 64, generator=g), code)
    assert abs(unrelated["cos_pair_centered"]) < 0.05 and unrelated["top1_centered"] < 0.05


# --- CPU: plotting histories saved before and after the diagnostics -----------------------------------------------


def _hist(with_diag):
    h = {"step": [1, 2], "ntp": [1.0, 0.5], "dist": [0.5, 0.4], "val_step": [2], "val_ntp": [0.6], "val_dist": [0.4]}
    if with_diag:
        h.update(val_cos_pair=[0.5], val_cos_between=[0.9], val_erank_code=[10.0], val_erank_text=[9.0])
    return h


@pytest.mark.parametrize("with_diag, n_panels", [(False, 2), (True, 3)])
def test_plot_curves_with_old_and_new_histories(tmp_path, with_diag, n_panels):
    from jepa_llm.experiment import plot_curves

    cfg = NLRXConfig(root=tmp_path)
    cfg.out_dir.mkdir(parents=True)
    results = [{"tag": "baseline", "hist": _hist(with_diag)}, {"tag": "llm_jepa", "hist": _hist(False)}]
    fig = plot_curves(cfg, results)
    assert len(fig.axes) == n_panels
    assert (cfg.out_dir / "curves.png").exists()


SAVED_HISTORY = NLRXConfig().out_dir / "llm_jepa" / "history.json"


@pytest.mark.skipif(not SAVED_HISTORY.exists(), reason="no saved history (outputs/ is gitignored)")
def test_plot_curves_with_saved_history(tmp_path):
    from jepa_llm.experiment import plot_curves

    cfg = NLRXConfig(root=tmp_path)
    cfg.out_dir.mkdir(parents=True)
    fig = plot_curves(cfg, [{"tag": "llm_jepa", "hist": json.loads(SAVED_HISTORY.read_text())}])
    assert len(fig.axes) == 2


# --- GPU: the diagnostic reads the same vectors as the loss -------------------------------------------------------


@pytest.mark.gpu
@requires_gpu_model
@pytest.mark.parametrize("pooling", ["last", "mean"])
def test_pair_cosine_matches_jepa_distance(llama_tok, sample_examples, free_gpu, pooling):
    from jepa_llm.model import build_model, jepa_distance

    cfg = NLRXConfig()
    model = build_model(cfg).eval()
    with torch.no_grad():
        jb = jepa_batch(sample_examples, cfg.jepa_k, llama_tok.pred_ids, llama_tok.pad_id)
        dist = jepa_distance(model, jb, pooling)
    # Same batch as the loss: identical padding, so identical bf16 numerics -> tight tolerance.
    pred, code = view_embeddings(model, sample_examples, llama_tok, cfg.jepa_k, bs=len(sample_examples), pooling=pooling)
    assert pred.shape == code.shape == (len(sample_examples), model.config.hidden_size)
    assert 1 - collapse_metrics(pred, code)["cos_pair"] == pytest.approx(dist.item(), abs=1e-5)
    # Chunked: padding length changes, so bf16 rounding differs slightly, but the vectors are the same.
    pred3, code3 = view_embeddings(model, sample_examples, llama_tok, cfg.jepa_k, bs=3, pooling=pooling)
    assert collapse_metrics(pred3, code3)["cos_pair"] == pytest.approx(1 - dist.item(), abs=1e-2)
    assert not model.training, "view_embeddings must restore the previous mode"
    del model


# --- GPU: regression on the saved adapters (numbers from the 2026-11-05 write-up) --------------------------------

SAVED_ADAPTERS = all((NLRXConfig().out_dir / t / "adapter").exists() for t in ("baseline", "llm_jepa"))


@pytest.fixture(scope="module")
def saved_reports():
    from jepa_llm.experiment import diagnose, load_data
    from jepa_llm.tokenization import NLRXTokenizer

    cfg = NLRXConfig()
    if not cfg.data_dir.exists():
        pytest.skip("dataset not in datasets/raw/")
    tok = NLRXTokenizer.from_pretrained(cfg.model_name, cfg.date_string)
    data = load_data(cfg, tok)
    return {tag: diagnose(cfg, tok, data, tag, save=False) for tag in ("baseline", "llm_jepa")}


@pytest.mark.gpu
@requires_gpu_model
@pytest.mark.skipif(not SAVED_ADAPTERS, reason="no saved adapters (outputs/ is gitignored)")
def test_saved_adapters_reproduce_writeup(saved_reports):
    base, jepa = saved_reports["baseline"], saved_reports["llm_jepa"]
    assert base["n_examples"] == jepa["n_examples"] == 300
    assert jepa["cos_between"] > 0.999, "LLM-JEPA collapsed: all Code embeddings share one direction"
    assert 0.90 <= base["cos_between"] <= 0.96
    assert 0.50 <= base["cos_pair"] <= 0.56
    # Once the shared direction is removed, almost no pair-specific signal is left in either run
    # (measured 2026-11-09: centred same-pair cos 0.021 / 0.078, centred top-1 0.010 / 0.017, chance 0.003).
    for r in (base, jepa):
        assert r["cos_pair_centered"] < 0.15 and r["top1_centered"] < 0.05
