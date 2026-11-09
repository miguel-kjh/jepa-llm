"""JEPA objectives for language models."""

import argparse

RUNS = ("baseline", "llm_jepa")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="jepa-llm")
    sub = parser.add_subparsers(dest="command", required=True)
    nlrx = sub.add_parser("nlrx", help="LLM-JEPA vs. plain LoRA fine-tuning on NL-RX-Synth")
    nlrx.add_argument("--run", choices=(*RUNS, "all"), default="all", help="which run(s) to train and evaluate")
    nlrx.add_argument("--model", default=None, help="Hugging Face model id (default: Llama-3.2-1B-Instruct)")
    args = parser.parse_args(argv)

    # Heavy imports (torch, transformers) only once a command actually runs.
    from .config import NLRXConfig
    from .experiment import load_data, plot_curves, run_and_evaluate, write_results
    from .tokenization import NLRXTokenizer

    cfg = NLRXConfig() if args.model is None else NLRXConfig(model_name=args.model)
    tok = NLRXTokenizer.from_pretrained(cfg.model_name, cfg.date_string)
    data = load_data(cfg, tok)
    lambdas = {"baseline": 0.0, "llm_jepa": cfg.jepa_lambda}
    tags = RUNS if args.run == "all" else (args.run,)
    results = [run_and_evaluate(cfg, tok, data, tag, lam=lambdas[tag], k=cfg.jepa_k) for tag in tags]
    if args.run == "all":
        write_results(cfg, results, n_test=len(data.test))
        import matplotlib

        matplotlib.use("Agg")
        plot_curves(cfg, results)
