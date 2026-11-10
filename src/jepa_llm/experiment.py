"""End-to-end NL-RX-Synth experiment: data, runs, saved artifacts and plots.

Artifacts under `cfg.out_dir` (same layout as the original notebook):
    split.json, results.json, curves.png,
    <tag>/adapter/, <tag>/history.json, <tag>/test_predictions.jsonl, <tag>/collapse.json
    screen/<tag>/history.json  (short screening runs: no adapter, no test predictions)
"""

import gc
import json
from dataclasses import dataclass

import torch

from .config import NLRXConfig
from .data import load_or_make_split, load_pairs
from .diagnostics import collapse_metrics, view_embeddings
from .evaluation import exact_match, generate_predictions
from .tokenization import NLRXTokenizer
from .training import train


@dataclass
class NLRXData:
    pairs: list[tuple[str, str]]
    split: dict[str, list[int]]
    train: list[dict]
    val: list[dict]
    test: list[dict]


def load_data(cfg: NLRXConfig, tok: NLRXTokenizer) -> NLRXData:
    pairs = load_pairs(cfg.data_dir)
    split = load_or_make_split(cfg.out_dir / "split.json", len(pairs), cfg.seed, cfg.split_fracs)
    examples = [tok.encode(s, t) for s, t in pairs]
    return NLRXData(
        pairs=pairs,
        split=split,
        train=[examples[i] for i in split["train"]],
        val=[examples[i] for i in split["val"]],
        test=[examples[i] for i in split["test"]],
    )


def free_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_and_evaluate(cfg: NLRXConfig, tok: NLRXTokenizer, data: NLRXData, tag: str, lam: float, k: int) -> dict:
    """Train one run, decode the test set and save adapter, history and predictions under `out_dir/tag`."""
    model, hist = train(cfg, tok, tag, lam=lam, k=k, data=data.train, val_data=data.val)
    preds = generate_predictions(model, data.test, tok, bs=cfg.eval_batch_size, max_new_tokens=cfg.max_new_tokens)
    acc = exact_match(preds, data.test)
    run_dir = cfg.out_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(run_dir / "adapter")
    (run_dir / "history.json").write_text(json.dumps(hist))
    with open(run_dir / "test_predictions.jsonl", "w") as f:
        for p, e in zip(preds, data.test):
            f.write(json.dumps({"desc": e["desc"], "gold": e["regex"], "pred": p, "correct": p == e["regex"]}) + "\n")
    print(f"[{tag}] test exact match: {acc:.4f}")
    del model
    free_cuda_memory()
    return {"tag": tag, "lambda": lam, "k": k, "test_exact_match": acc, "hist": hist}


def screen(cfg: NLRXConfig, tok: NLRXTokenizer, data: NLRXData, tag: str, lam: float, steps: int) -> dict:
    """Short run for screening: the first `steps` steps of the full run (same LR schedule and data order).

    Only `out_dir/screen/<tag>/history.json` is written; `baseline/` and `llm_jepa/` are never touched.
    """
    model, hist = train(cfg, tok, tag, lam=lam, k=cfg.jepa_k, data=data.train, val_data=data.val, stop_step=steps)
    hist["run"] = {"tag": tag, "lambda": lam, "k": cfg.jepa_k, "pooling": cfg.jepa_pooling, "steps": steps,
                   "val_every": cfg.val_every, "seed": cfg.seed}
    run_dir = cfg.out_dir / "screen" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "history.json").write_text(json.dumps(hist))
    last = {m: hist[f"val_{m}"][-1] for m in ("ntp", "cos_between", "top1_centered", "cos_pair_centered")}
    print(f"[{tag}] step {hist['val_step'][-1]}: " + "  ".join(f"{m} {v:.4f}" for m, v in last.items()), flush=True)
    del model
    free_cuda_memory()
    return hist


def write_results(cfg: NLRXConfig, results: list[dict], n_test: int) -> list[dict]:
    """Save `results.json` (without the histories) and print a Markdown table."""
    summary = [{key: r[key] for key in ("tag", "lambda", "k", "test_exact_match")} for r in results]
    (cfg.out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print("| Run | λ | k | Test exact match |")
    print("|---|---|---|---|")
    for r in summary:
        print(f"| {r['tag']} | {r['lambda']} | {r['k']} | {100 * r['test_exact_match']:.2f} % |")
    print(f"\nn_test = {n_test}; one run per method (no seeds, no CI).")
    return summary


def plot_curves(cfg: NLRXConfig, results: list[dict]):
    """NTP loss and JEPA distance (train and val) for each run; saved to `curves.png`."""
    import matplotlib.pyplot as plt

    with_diag = any("val_cos_between" in r["hist"] for r in results)  # histories saved before issue #2 lack it
    fig, axes = plt.subplots(1, 3 if with_diag else 2, figsize=(18 if with_diag else 12, 4))
    for r, color in zip(results, ["tab:blue", "tab:orange"]):
        h = r["hist"]
        if "val_cos_between" in h:
            axes[2].plot(h["val_step"], h["val_cos_between"], color=color, marker="o", label=f"{r['tag']} val")
        axes[0].plot(h["step"], h["ntp"], color=color, alpha=0.4, label=f"{r['tag']} train")
        axes[0].plot(h["val_step"], h["val_ntp"], color=color, marker="o", label=f"{r['tag']} val")
        axes[1].plot(h["step"], h["dist"], color=color, alpha=0.4, label=f"{r['tag']} train")
        axes[1].plot(h["val_step"], h["val_dist"], color=color, marker="o", label=f"{r['tag']} val")
    axes[0].set(title="NTP loss", xlabel="step", yscale="log")
    axes[1].set(title="JEPA distance 1 - cos(Pred(Enc(Text)), Enc(Code))", xlabel="step")
    if with_diag:
        axes[2].set(title="Collapse: mean cos between Codes of different examples", xlabel="step")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.out_dir / "curves.png", dpi=120)
    return fig


def diagnose(cfg: NLRXConfig, tok: NLRXTokenizer, data: NLRXData, tag: str, save: bool = True) -> dict:
    """Collapse metrics of the saved adapter `out_dir/tag/adapter` on the first `cfg.diag_examples` val examples."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=cfg.dtype).to(cfg.device)
    model = PeftModel.from_pretrained(base, cfg.out_dir / tag / "adapter").eval()
    examples = data.val[: cfg.diag_examples]
    metrics = collapse_metrics(*view_embeddings(model, examples, tok, cfg.jepa_k, pooling=cfg.jepa_pooling))
    report = {"tag": tag, "k": cfg.jepa_k, "pooling": cfg.jepa_pooling, "split": "val", "n_examples": len(examples), **metrics}
    if save:
        (cfg.out_dir / tag / "collapse.json").write_text(json.dumps(report, indent=2))
    del model, base
    free_cuda_memory()
    return report


def print_collapse_table(reports: list[dict]) -> None:
    print(
        "| Run | cos(Pred, Code) same pair | cos between Codes | erank Code / Text "
        "| cos same pair (centred) | Text->Code top-1 (centred) |"
    )
    print("|---|---|---|---|---|---|")
    for r in reports:
        chance = 1 / r["n_examples"]
        print(
            f"| {r['tag']} | {r['cos_pair']:.4f} | {r['cos_between']:.4f} | "
            f"{r['erank_code']:.1f} / {r['erank_text']:.1f} | {r['cos_pair_centered']:.3f} | "
            f"{r['top1_centered']:.3f} (chance {chance:.3f}) |"
        )
