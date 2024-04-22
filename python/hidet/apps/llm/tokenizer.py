from typing import List
from transformers import AutoTokenizer
import hidet.option


# Use the tokenizer from huggingface, for now
class Tokenizer:
    def __init__(self, name):
        token = hidet.option.get_option('auth_tokens.for_huggingface')
        self.hf_tokenizer = AutoTokenizer.from_pretrained(name, token=token)

    def encode(self, text) -> List[int]:
        return self.hf_tokenizer.encode(text)

    def decode(self, ids) -> str:
        return self.hf_tokenizer.decode(ids)
