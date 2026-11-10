"""JEPA objectives for language models."""

import argparse

RUNS = ("baseline", "llm_jepa")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="jepa-llm")
    sub = parser.add_subparsers(dest="command", required=True)
    nlrx = sub.add_parser("nlrx", help="LLM-JEPA vs. plain LoRA fine-tuning on NL-RX-Synth")
    nlrx.add_argument("--run", choices=(*RUNS, "all"), default="all", help="which run(s) to train and evaluate")
    nlrx.add_argument("--model", default=None, help="Hugging Face model id (default: Llama-3.2-1B-Instruct)")
    diag = sub.add_parser("nlrx-diagnose", help="collapse metrics of the saved NL-RX adapters")
    diag.add_argument("--run", choices=(*RUNS, "all"), default="all", help="which saved adapter(s) to diagnose")
    diag.add_argument("--model", default=None, help="base model the adapters were trained on")
    scr = sub.add_parser("nlrx-screen", help="short NL-RX run that only writes screen/<tag>/history.json")
    scr.add_argument("--pooling", choices=("last", "mean"), default="last", help="how each JEPA view is pooled")
    scr.add_argument("--lam", type=float, default=1.0, help="JEPA weight (0 = baseline, distance only measured)")
    scr.add_argument("--steps", type=int, default=500, help="stop after this many optimiser steps")
    scr.add_argument("--val-every", type=int, default=100, help="validation and collapse metrics every N steps")
    scr.add_argument("--tag", default=None, help="output folder name (default: lam<λ>_<pooling>)")
    scr.add_argument("--model", default=None, help="Hugging Face model id (default: Llama-3.2-1B-Instruct)")
    args = parser.parse_args(argv)

    # Heavy imports (torch, transformers) only once a command actually runs.
    from .config import NLRXConfig
    from .experiment import (
        diagnose, load_data, plot_curves, print_collapse_table, run_and_evaluate, screen, write_results,
    )
    from .tokenization import NLRXTokenizer

    cfg = NLRXConfig() if args.model is None else NLRXConfig(model_name=args.model)
    if args.command == "nlrx-screen":
        cfg.jepa_pooling, cfg.val_every = args.pooling, args.val_every
    tok = NLRXTokenizer.from_pretrained(cfg.model_name, cfg.date_string)
    data = load_data(cfg, tok)

    if args.command == "nlrx-screen":
        tag = args.tag or f"lam{args.lam:g}_{args.pooling}"
        screen(cfg, tok, data, tag, lam=args.lam, steps=args.steps)
        return

    tags = RUNS if args.run == "all" else (args.run,)

    if args.command == "nlrx-diagnose":
        print_collapse_table([diagnose(cfg, tok, data, tag) for tag in tags])
        return

    lambdas = {"baseline": 0.0, "llm_jepa": cfg.jepa_lambda}
    results = [run_and_evaluate(cfg, tok, data, tag, lam=lambdas[tag], k=cfg.jepa_k) for tag in tags]
    if args.run == "all":
        write_results(cfg, results, n_test=len(data.test))
        import matplotlib

        matplotlib.use("Agg")
        plot_curves(cfg, results)
