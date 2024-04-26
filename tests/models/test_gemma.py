import os

import pytest
from transformers import AutoTokenizer

import hidet
from hidet.testing.models.gemma import GemmaForCausalLM


@pytest.mark.skip(reason="This test requires access to the Gemma model on Hugging Face")
def test_gemma():
    os.environ["TOKENIZERS_PARALLELISM"] = 'false'
    model = GemmaForCausalLM().cuda().build()

    tok = AutoTokenizer.from_pretrained("google/gemma-2b")
    text = "Since the beginning of time"

    prompt = tok.encode(text, return_tensors="pt").cuda()
    prompt = hidet.from_torch(prompt)
    output = model.generate(prompt, num_tokens=15)[0].torch()
    ans = tok.decode(output)
    assert ans == "<bos>Since the beginning of time, people have been fascinated by the idea of"
