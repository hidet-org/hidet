from typing import List
from transformers import AutoTokenizer


# Use the tokenizer from huggingface, for now
class Tokenizer:
    def __init__(self, name):
        self.hf_tokenizer = AutoTokenizer.from_pretrained(name)

    def encode(self, text) -> List[int]:
        return self.hf_tokenizer.encode(text)

    def decode(self, ids) -> str:
        return self.hf_tokenizer.decode(ids)
