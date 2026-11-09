"""Training loop (AdamW + linear warmup) with periodic validation losses."""

import math
import random
import time

import torch
from transformers import get_linear_schedule_with_warmup

from .config import NLRXConfig
from .model import build_model, compute_losses
from .tokenization import NLRXTokenizer


@torch.no_grad()
def evaluate_losses(model, data: list[dict], k: int, tok: NLRXTokenizer, bs: int = 32) -> tuple[float, float]:
    """Mean NTP loss and mean JEPA distance over `data` (the distance is measured, never optimised)."""
    model.eval()
    ntps, dists = [], []
    for i in range(0, len(data), bs):
        _, ntp, dist = compute_losses(model, data[i : i + bs], lam=0.0, k=k, tok=tok)
        ntps.append(ntp.item())
        dists.append(dist.item())
    model.train()
    return sum(ntps) / len(ntps), sum(dists) / len(dists)


def train(
    cfg: NLRXConfig,
    tok: NLRXTokenizer,
    tag: str,
    lam: float,
    k: int,
    data: list[dict],
    epochs: int | None = None,
    lr: float | None = None,
    max_steps: int | None = None,
    val_data: list[dict] | None = None,
    log_every: int | None = None,
):
    epochs = cfg.epochs if epochs is None else epochs
    lr = cfg.lr if lr is None else lr
    log_every = cfg.log_every if log_every is None else log_every

    model = build_model(cfg)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(data) / (cfg.batch_size * cfg.grad_accum))
    total_steps = max_steps or steps_per_epoch * epochs
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total_steps), total_steps)
    rng = random.Random(cfg.seed)
    hist = {"step": [], "ntp": [], "dist": [], "val_step": [], "val_ntp": [], "val_dist": []}
    step, t0 = 0, time.time()
    run_ntp, run_dist, run_n = 0.0, 0.0, 0
    while step < total_steps:
        order = list(range(len(data)))
        rng.shuffle(order)
        micro = [order[i : i + cfg.batch_size] for i in range(0, len(order), cfg.batch_size)]
        for j in range(0, len(micro), cfg.grad_accum):
            for mb in micro[j : j + cfg.grad_accum]:
                total, ntp, dist = compute_losses(model, [data[i] for i in mb], lam, k, tok)
                (total / cfg.grad_accum).backward()
                run_ntp, run_dist, run_n = run_ntp + ntp.item(), run_dist + dist.item(), run_n + 1
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % log_every == 0 or step == total_steps:
                hist["step"].append(step)
                hist["ntp"].append(run_ntp / run_n)
                hist["dist"].append(run_dist / run_n)
                run_ntp, run_dist, run_n = 0.0, 0.0, 0
            if val_data is not None and (step % cfg.val_every == 0 or step == total_steps):
                v_ntp, v_dist = evaluate_losses(model, val_data, k, tok)
                hist["val_step"].append(step)
                hist["val_ntp"].append(v_ntp)
                hist["val_dist"].append(v_dist)
                print(
                    f"[{tag}] step {step}/{total_steps}  train ntp {hist['ntp'][-1]:.4f}  "
                    f"train dist {hist['dist'][-1]:.4f}  val ntp {v_ntp:.4f}  val dist {v_dist:.4f}  "
                    f"({time.time() - t0:.0f}s)"
                )
            if step >= total_steps:
                break
    return model, hist
