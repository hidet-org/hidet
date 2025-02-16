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
import hidet
import torch
import pytest
from hidet.testing.models.llama import LlamaConfig, convert_model
from transformers.models.llama import LlamaForCausalLM as hfLm, LlamaConfig as hfConfig

SEARCH_SPACE = 2
NUM_HIDDEN_LAYERS = 4
BATCH_SIZE = 1
PREFILL = 0
SEQ_LEN = 128


@pytest.mark.slow
@pytest.mark.parametrize('device', ['hip', 'cuda'])
@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize('mma', [True, False])
def test_hidet_llama_graph_closeness(device: str, dtype: str, mma: bool):
    if device == 'hip' and not hidet.hip.available():
        pytest.skip('HIP is not available')
    if device == 'cuda' and not hidet.cuda.available():
        pytest.skip('CUDA is not available')

    hidet.option.search_space(SEARCH_SPACE)
    hfconfig = hfConfig(num_hidden_layers=NUM_HIDDEN_LAYERS)
    torch_model = hfLm(hfconfig)
    if dtype == 'float16':
        torch_model = torch_model.half()

    model = convert_model(torch_model, device=device, dtype=dtype)

    def build_flow_graph(model, batch_size=1, device='cuda', dtype='float16', logits=False):
        config = model.config
        input_ids = hidet.symbol([batch_size, "seq_length"], dtype=hidet.int32, device=device)
        position_ids = hidet.symbol([batch_size, config.max_position_embeddings], dtype=hidet.int32, device=device)

        get_sym = lambda: hidet.symbol(
            [batch_size, config.num_key_value_heads, "prev_seq_len", config.hidden_size // config.num_key_value_heads],
            device=device,
            dtype=dtype,
        )
        key_value_cache = [(get_sym(), get_sym()) for _ in range(config.num_hidden_layers)]

        y = model(input_ids, position_ids=position_ids, past_key_values=key_value_cache)
        inputs = [input_ids, position_ids]
        for q, k in key_value_cache:
            inputs.append(q)
            inputs.append(k)

        outputs = [y['new_ids'] if logits == False else y['logits']]
        for q, k in y['past_key_values']:
            outputs.append(q)
            outputs.append(k)

        return hidet.trace_from(outputs, inputs)

    # get logits for testing correctness
    graph = build_flow_graph(model, batch_size=1, device=device, dtype=dtype, logits=True)
    with hidet.graph.PassContext() as ctx:
        if device == 'hip':
            ctx.reduce_hip_compile_mem()
        elif device == 'cuda':
            ctx.reduce_cuda_compile_mem()
        if mma:
            ctx.set_mma('mma')
        flow_graph = hidet.graph.optimize(graph)

    compiled_graph = flow_graph.build(space=2)

    def gen_inputs(config: LlamaConfig, batch_size, input_len, prefill_len, dtype='float32'):
        input_ids = hidet.randint(0, 1000, [batch_size, input_len], dtype=hidet.int32).to(device=device)
        position_ids = hidet.arange(0, config.max_position_embeddings, dtype=hidet.int32).unsqueeze(0).to(device=device)
        make_past = lambda: hidet.zeros(
            [1, config.num_key_value_heads, prefill_len, config.hidden_size // config.num_key_value_heads],
            device=device,
            dtype=dtype,
        )
        past_keys_values = [make_past() for _ in range(config.num_hidden_layers * 2)]

        return [input_ids, position_ids] + past_keys_values

    inputs = gen_inputs(model.config, BATCH_SIZE, SEQ_LEN, PREFILL, dtype=dtype)

    with torch.no_grad():
        torch_model.cuda()
        y1 = torch_model(input_ids=inputs[0].torch())
    y2 = compiled_graph(*inputs)[0].torch().float()

    if dtype == 'float16':
        assert torch.allclose(y1.logits, y2, atol=5e-1, rtol=1e-1)
    elif dtype == 'float32':
        if device == 'cuda':
            # for some reason cuda is less accurate
            assert torch.allclose(y1.logits, y2, atol=1e-1, rtol=1e-1)
        elif device == 'hip':
            assert torch.allclose(y1.logits, y2, atol=1e-3, rtol=1e-3)
    del compiled_graph, torch_model, graph
    torch.cuda.empty_cache()


if __name__ == '__main__':
    pytest.main([__file__])
