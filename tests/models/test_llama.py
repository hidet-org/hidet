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
# %%
import pytest
import torch
from torch import nn
from transformers.activations import ACT2FN
from hidet.testing.models.llama import get_compiled_model, generate, convert_model
from hidet.runtime.storage import current_memory_pool
from hidet.testing.torch_utils import Backend


# @pytest.mark.parametrize('device,opt', [('cuda', True)])
@pytest.mark.skip(reason='This test requires a lot of CPU memory > 32GB')
def test_llama(device, opt):
    model, config, tokenizer = get_compiled_model(device=device, opt=opt)

    text = generate('In the beginning was the Word.', model, tokenizer, config, num_tokens=12)
    print(text)
    expected = 'The Word was with God, and the Word was God.'
    assert text == expected

    text = generate(
        "A robot may not injure a human being or, through inaction", model, tokenizer, config, num_tokens=55
    )
    expected = (
        ', allow a human being to come to harm. A robot must obey orders given it by human beings'
        ' except where such orders would conflict with the First Law. A robot must protect its own'
        ' existence as long as such protection does not conflict with the First or Second Laws.'
    )

    print(text)
    assert text == expected

    print(current_memory_pool("cuda"))
    print(current_memory_pool("cpu"))
    print(current_memory_pool("vcuda"))


# @pytest.mark.parametrize('device,opt', [('cuda', True)])
@pytest.mark.skip(reason='This test requires a lot of CPU memory > 32GB, plus you need to sign up for the weights')
def test_llama2(device, opt):
    model, config, tokenizer = get_compiled_model(device=device, opt=opt, name="meta-llama/Llama-2-7b-hf")

    text = generate('In the beginning was the Word.', model, tokenizer, config, num_tokens=12)
    print(text)
    expected = '\nThe Word was with God, and the Word was God'
    assert text == expected

    text = generate(
        "A robot may not injure a human being or, through inaction", model, tokenizer, config, num_tokens=55
    )
    expected = ', allow a human being to come to harm.\nA robot must obey orders given it by human beings except where such orders would conflict with the First Law.\nA robot must protect its own existence as long as such protection does not conflict with the First or Second Law'

    print(text)
    assert text == expected

    print(current_memory_pool("cuda"))
    print(current_memory_pool("cpu"))
    print(current_memory_pool("vcuda"))


@pytest.mark.skip(
    reason='We now focus on the torch.compile API. '
    'The current llama model definition is not compatible huggingface thus disable the test.'
)
def test_model_architecture():

    import torch
    import hidet
    from transformers.models.llama import LlamaForCausalLM as hfLm, LlamaConfig

    config = LlamaConfig(
        **{
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "pad_token_id": 0,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": None,
            "tie_word_embeddings": False,
            "torch_dtype": dtype,
            "use_cache": True,
            "vocab_size": 32000,
        }
    )

    with torch.device("cuda"):
        hf_model = hfLm(config).eval()

    model = convert_model(hf_model, device='cuda', dtype=hidet.float32)

    def build_flow_graph(model, batch_size=1, device='cuda', dtype=dtype):
        config = model.config
        input_ids = hidet.symbol([batch_size, 'seq_len'], dtype=hidet.int32, device=device)
        position_ids = hidet.symbol([batch_size, config.max_position_embeddings], dtype=hidet.int32, device=device)

        y = model(input_ids, position_ids=position_ids, past_key_values=None)  # key_value_cache)
        inputs = [input_ids, position_ids]

        outputs = [y['logits']]
        return hidet.trace_from(outputs, inputs)

    cmodel = build_flow_graph(model, batch_size=1, device='cuda', dtype=hidet.float32)

    x = torch.randint(0, 32000, (1, 512), dtype=torch.int32).cuda()
    pos_ids = torch.arange(0, config.max_position_embeddings, dtype=torch.int32).reshape(1, -1).cuda()
    res1 = hf_model(x)
    res2 = cmodel(hidet.from_torch(x), hidet.from_torch(pos_ids))

    logits1 = res1.logits
    logits2 = res2.torch()
    assert torch.allclose(logits1, logits2, rtol=1e-3, atol=1e-3)


class LlamaMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 4096
        self.intermediate_size = 11008
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        down_proj = self.down_proj(self.gate_proj(x) * self.up_proj(x))
        return down_proj


@pytest.mark.parametrize("dtype", ['bfloat16', 'float16'])
def test_llama_issue419(dtype):
    # hidet.option.cache_dir("./llama")
    # hidet.option.debug_cache_tuning()
    if dtype == 'bfloat16':
        torch_type = torch.bfloat16
    elif dtype == 'float16':
        torch_type = torch.float16
    model = LlamaMLP().cuda().to(torch_type)

    x = torch.randn(1, 143, 4096, dtype=torch_type, device='cuda')
    with torch.inference_mode(True):
        y_true = model(x)
        backend = Backend(backend='hidet', mode='max-autotune', dtype=dtype)
        model = backend.compile(model)
        y_ac = model(x)

    torch.testing.assert_close(y_true, y_ac, rtol=0.01, atol=0.01)
