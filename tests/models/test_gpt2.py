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
import torch
import transformers
import hidet
import hidet.testing


def generate(model, text, num_hidden_layers, num_heads, head_dim, device, tokens_to_generate=10):
    tokenizer = hidet.testing.models.gpt2.tokenizer()
    input_ids_list: List[int] = tokenizer(text)['input_ids']

    input_ids = hidet.asarray(input_ids_list, dtype=hidet.int32, device=device)
    position_ids = hidet.arange(input_ids.shape[0], dtype=hidet.int32, device=device)
    past_keys = hidet.zeros([num_hidden_layers, num_heads, 0, head_dim], dtype=hidet.float32, device=device)
    past_values = hidet.zeros([num_hidden_layers, num_heads, 0, head_dim], dtype=hidet.float32, device=device)

    output_ids = []
    for _ in range(tokens_to_generate):
        input_ids, position_ids, past_keys, past_values = model(input_ids, position_ids, past_keys, past_values)
        output_ids.append(input_ids[0].item())

    return tokenizer.decode(output_ids)


@pytest.mark.parametrize('device,opt', [('cpu', False), ('cpu', True), ('cuda', False), ('cuda', True)])
def test_gpt2(device: str, opt: bool):
    gpt2_module = hidet.testing.models.gpt2.model(disable_cache=True)

    if device == 'cuda':
        gpt2_module.cuda()

    input_ids = hidet.symbol(['seq_length'], dtype=hidet.int32, device=device)
    position_ids = hidet.symbol(['seq_length'], dtype=hidet.int32, device=device)
    cache_shape = [gpt2_module.num_hidden_layers, gpt2_module.num_heads, 'prev_seq_length', gpt2_module.head_dim]
    past_keys = hidet.symbol(cache_shape, dtype=hidet.float32, device=device)
    past_values = hidet.symbol(cache_shape, dtype=hidet.float32, device=device)

    outputs = gpt2_module(input_ids, position_ids, past_keys, past_values)
    graph = hidet.trace_from(outputs, inputs=[input_ids, position_ids, past_keys, past_values])

    if opt:
        graph = hidet.graph.optimize(graph)

    compiled_model = graph.build()
    compiled_model.save('./outs/compiled.hidet')

    generated_text = generate(
        compiled_model,
        "Alan Turing theorized that computers would one day become",
        gpt2_module.num_hidden_layers,
        gpt2_module.num_heads,
        gpt2_module.head_dim,
        device,
        tokens_to_generate=40,
    )
    expected = (
        ' the most powerful machines on the planet.\n\n'
        'The computer is a machine that can perform complex calculations, and it can '
        'perform these calculations in a way that is very similar to the human brain.\n'
    )
    assert generated_text == expected
