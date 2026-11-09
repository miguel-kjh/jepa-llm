"""NL-RX-Synth loading and the fixed train/val/test split."""

import json
import random
from pathlib import Path


def load_pairs(data_dir: Path) -> list[tuple[str, str]]:
    """(description, regex) pairs from the parallel files `src.txt` and `targ.txt`."""
    src_lines = (data_dir / "src.txt").read_text(encoding="utf-8").splitlines()
    tgt_lines = (data_dir / "targ.txt").read_text(encoding="utf-8").splitlines()
    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"src.txt and targ.txt must be parallel ({len(src_lines)} vs {len(tgt_lines)} lines)")
    return [(s.strip(), t.strip()) for s, t in zip(src_lines, tgt_lines)]


def make_split(n: int, seed: int, fracs: tuple[float, float, float]) -> dict[str, list[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = int(fracs[0] * n)
    n_val = int(fracs[1] * n)
    return {
        "train": sorted(idx[:n_train]),
        "val": sorted(idx[n_train : n_train + n_val]),
        "test": sorted(idx[n_train + n_val :]),
    }


def load_or_make_split(
    path: Path, n: int, seed: int, fracs: tuple[float, float, float]
) -> dict[str, list[int]]:
    """Load the split from `path` if it exists; otherwise create it and save it there."""
    if path.exists():
        return json.loads(path.read_text())
    split = make_split(n, seed, fracs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split))
    return split
