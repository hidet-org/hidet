# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List
import pytest
from transformers import AutoTokenizer

from hidet.testing.tokenizers import Tokenizer


def get_test_texts() -> List[str]:
    """
    Get a list of texts to test tokenization.
    """
    return [
        "Hello, world!",
        "你好，世界！ This 😀 is a test string with emojis 🚀🌟",
        "Invalid UTF-8: \xff\xff\xff \xf0\x28\x8c\xbc, \xc0\xaf, \xf8\xa1\xa1\xa1.",
        "Special tokenization characters: ByteLevel: Ġ, SentencePiece: ▁",
    ]


@pytest.mark.skip(
    'The tokenizer implemented inside hidet is not maintained since '
    'we do not plan to support everything with CompiledApp anymore.'
)
@pytest.mark.parametrize("model", ["huggyllama/llama-7b", "openai-community/gpt2", "facebook/opt-350m"])
@pytest.mark.parametrize("text", get_test_texts())
def test_tokenizer_encode_decode(model: str, text: str):
    hf_tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer = Tokenizer.from_hugging_face(model)

    ids = tokenizer.encode(text)
    assert ids == hf_tokenizer.encode(text)

    decoded = tokenizer.decode(ids)
    assert decoded == hf_tokenizer.decode(ids)
