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

    def content_span(self, view: list[int], content: str) -> tuple[int, int]:
        """[start, end) of `content` inside a one-message chat view: the tokens right before the final <|eot_id|>.

        Headers, the system prompt with the date and <|eot_id|> are outside the span. The assertion catches
        any tokenizer merge across the header/content boundary.
        """
        content_ids = self.tok(content, add_special_tokens=False).input_ids
        end = len(view) - 1
        start = end - len(content_ids)
        assert view[end] == self.eot_id, "a chat view must end with <|eot_id|>"
        assert start > 0 and view[start:end] == content_ids, f"content tokens not found in the chat view: {content!r}"
        return start, end

    def encode(self, desc: str, regex: str) -> dict:
        prompt = self.chat_ids([{"role": "user", "content": desc}], add_generation_prompt=True)
        answer = self.tok(regex, add_special_tokens=False).input_ids + [self.eot_id]
        text_view = self.chat_ids([{"role": "user", "content": desc}])
        code_view = self.chat_ids([{"role": "assistant", "content": regex}])
        return {
            "prompt": prompt,                                   # used for generation
            "ntp_ids": prompt + answer,                         # NTP forward
            "ntp_labels": [-100] * len(prompt) + answer,        # loss only on the regex
            "text_view": text_view,                             # Text view (Enc input)
            "code_view": code_view,                             # Code view (target)
            "text_content": self.content_span(text_view, desc),  # description tokens, for mean pooling
            "code_content": self.content_span(code_view, regex),  # regex tokens, for mean pooling
            "answer_len": len(answer),
            "desc": desc,
            "regex": regex,
        }

    def decode(self, ids) -> str:
        return self.tok.decode(ids)

    def batch_decode(self, ids, **kwargs) -> list[str]:
        return self.tok.batch_decode(ids, **kwargs)
