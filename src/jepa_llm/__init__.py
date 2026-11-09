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
    args = parser.parse_args(argv)

    # Heavy imports (torch, transformers) only once a command actually runs.
    from .config import NLRXConfig
    from .experiment import diagnose, load_data, plot_curves, print_collapse_table, run_and_evaluate, write_results
    from .tokenization import NLRXTokenizer

    cfg = NLRXConfig() if args.model is None else NLRXConfig(model_name=args.model)
    tok = NLRXTokenizer.from_pretrained(cfg.model_name, cfg.date_string)
    data = load_data(cfg, tok)
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
