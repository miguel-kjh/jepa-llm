import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from jepa_llm.config import NLRXConfig
from jepa_llm.tokenization import NLRXTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = NLRXConfig().model_name

# First 8 training pairs of the saved split (hard-coded so the GPU tests do not need the dataset).
SAMPLE_PAIRS = [
    ("lines with words with a capital letter and a vowel", "\\b([A-Z])&([AEIOUaeiou])\\b"),
    ("lines containing a letter or a character", ".*([A-Za-z])|(.).*"),
    ("lines with the string 'dog' or the string 'truck', 5 or more times", "((dog)|(truck)){5,}"),
    ("lines with the string 'dog' and the string 'truck' before a number", "((dog)&(truck)).*([0-9]).*"),
    ("lines with words with a number or ending with a character", "\\b([0-9])|((.*)(.))\\b"),
    (
        "lines with a vowel, zero or more times before a lower-case letter and a capital letter",
        "(([AEIOUaeiou])*).*(([a-z])&([A-Z])).*",
    ),
    (
        "lines having either a character, the string 'dog', or a number before the string 'truck'",
        "((.)|(dog)|([0-9])).*(truck).*",
    ),
    (
        "lines with words with either a letter, a capital letter, or a lower-case letter at least once",
        "(\\b([A-Za-z])|([A-Z])|([a-z])\\b)+",
    ),
]


class FakeLlamaTokenizer:
    """Character-level stand-in for the Llama 3 tokenizer: same special tokens and chat layout, no download."""

    SPECIALS = [
        "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
        "<|finetune_right_pad_id|>", *[f"<|reserved_special_token_{i}|>" for i in range(8)],
    ]

    def __init__(self):
        self.vocab = {t: i for i, t in enumerate(self.SPECIALS)}
        self._split = re.compile("(" + "|".join(re.escape(t) for t in self.SPECIALS) + ")")

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, date_string=None):
        text = "<|begin_of_text|>"
        for m in messages:
            text += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
        if add_generation_prompt:
            text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return text

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for piece in self._split.split(text):
            if piece in self.vocab:
                ids.append(self.vocab[piece])
            else:
                ids += [1000 + ord(c) for c in piece]
        return SimpleNamespace(input_ids=ids)


@pytest.fixture
def fake_tok():
    return NLRXTokenizer(FakeLlamaTokenizer(), date_string="26 Jul 2024")


def _model_is_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(MODEL_NAME, "config.json"), str)


requires_gpu_model = pytest.mark.skipif(
    not (torch.cuda.is_available() and _model_is_cached()),
    reason=f"needs CUDA and {MODEL_NAME} in the Hugging Face cache",
)


@pytest.fixture(scope="session")
def llama_tok():
    return NLRXTokenizer.from_pretrained(MODEL_NAME, NLRXConfig().date_string)


@pytest.fixture(scope="session")
def sample_examples(llama_tok):
    return [llama_tok.encode(s, t) for s, t in SAMPLE_PAIRS]


@pytest.fixture
def free_gpu():
    yield
    import gc

    gc.collect()
    torch.cuda.empty_cache()
