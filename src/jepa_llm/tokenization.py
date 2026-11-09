"""Chat-formatted NTP examples, JEPA views and `[PRED]` tokens for Llama 3."""

from transformers import AutoTokenizer

PAD_TOKEN = "<|finetune_right_pad_id|>"
EOT_TOKEN = "<|eot_id|>"
N_PRED_TOKENS = 8


class NLRXTokenizer:
    """Wraps a Llama 3 tokenizer and turns (description, regex) pairs into training examples."""

    def __init__(self, tokenizer, date_string: str):
        self.tok = tokenizer
        self.date_string = date_string
        self.pad_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
        self.eot_id = tokenizer.convert_tokens_to_ids(EOT_TOKEN)
        # [PRED] tokens: reserved special tokens already in the Llama 3 vocabulary (no resizing needed).
        self.pred_ids = [
            tokenizer.convert_tokens_to_ids(f"<|reserved_special_token_{i}|>") for i in range(N_PRED_TOKENS)
        ]

    @classmethod
    def from_pretrained(cls, model_name: str, date_string: str) -> "NLRXTokenizer":
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = PAD_TOKEN
        return cls(tok, date_string)

    def chat_ids(self, messages: list[dict], add_generation_prompt: bool = False) -> list[int]:
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, date_string=self.date_string
        )
        return self.tok(text, add_special_tokens=False).input_ids

    def encode(self, desc: str, regex: str) -> dict:
        prompt = self.chat_ids([{"role": "user", "content": desc}], add_generation_prompt=True)
        answer = self.tok(regex, add_special_tokens=False).input_ids + [self.eot_id]
        return {
            "prompt": prompt,                                   # used for generation
            "ntp_ids": prompt + answer,                         # NTP forward
            "ntp_labels": [-100] * len(prompt) + answer,        # loss only on the regex
            "text_view": self.chat_ids([{"role": "user", "content": desc}]),        # Text view (Enc input)
            "code_view": self.chat_ids([{"role": "assistant", "content": regex}]),  # Code view (target)
            "answer_len": len(answer),
            "desc": desc,
            "regex": regex,
        }

    def decode(self, ids) -> str:
        return self.tok.decode(ids)

    def batch_decode(self, ids, **kwargs) -> list[str]:
        return self.tok.batch_decode(ids, **kwargs)
