import json

import pytest

from jepa_llm.config import NLRXConfig
from jepa_llm.data import load_or_make_split, make_split

SAVED_SPLIT = NLRXConfig().out_dir / "split.json"


@pytest.mark.skipif(not SAVED_SPLIT.exists(), reason="no saved split.json (outputs/ is gitignored)")
def test_regenerated_split_matches_saved(tmp_path):
    saved = json.loads(SAVED_SPLIT.read_text())
    n = sum(len(v) for v in saved.values())
    cfg = NLRXConfig()
    path = tmp_path / "split.json"
    assert load_or_make_split(path, n, cfg.seed, cfg.split_fracs) == saved
    assert json.loads(path.read_text()) == saved, "the created split must be saved"


def test_existing_split_is_loaded_not_regenerated(tmp_path):
    path = tmp_path / "split.json"
    custom = {"train": [2], "val": [0], "test": [1]}
    path.write_text(json.dumps(custom))
    assert load_or_make_split(path, 3, seed=0, fracs=(0.8, 0.1, 0.1)) == custom


def test_split_is_a_disjoint_cover():
    split = make_split(1001, seed=0, fracs=(0.8, 0.1, 0.1))
    train, val, test = (set(split[k]) for k in ("train", "val", "test"))
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(range(1001))
    assert (len(train), len(val)) == (800, 100)
