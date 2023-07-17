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
import os
import pickle

import torch
import hidet

import pytest

import hidet.testing
from hidet.testing.models import gpt2


class GPT2LMHead(gpt2.GPT2LMHead):
    def forward(self, input_ids, position_ids, past_keys, past_values):
        # params:
        #   input_ids: int32[seq_length]
        #   position_ids: int32[seq_length]
        #   past_keys: [layers, prev_seq_length, hidden_size]
        #   past_values: [layers, prev_seq_length, hidden_size]
        # return:
        #   logits: dtype[1, vocab_size]
        #   position_ids: int32[1]
        #   updated_keys: [layers, prev_seq_length + seq_length, hidden_size]
        #   updated_values: [layers, prev_seq_length + seq_length, hidden_size]

        # keep logits to calculate perplexity
        hidden_states, position_ids, past_keys, past_values = self.transformer(
            input_ids, position_ids, past_keys, past_values
        )
        logits = self.lm_head(hidden_states)  # [1, vocab_size]
        # we want to keep types consistent, since in the autoregressive case,
        #   the output is fed back into the input of the compiled model
        return logits, position_ids, past_keys, past_values


def model(name='gpt2', disable_cache=False) -> GPT2LMHead:
    cache_path = hidet.utils.cache_file('testing', 'models', 'gpt2_quant', name + '.pkl')
    if os.path.exists(cache_path) and not disable_cache:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    else:
        candidates = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2']
        if name not in candidates:
            raise ValueError(f'got {name}, name should be one of {candidates}')
        m = GPT2LMHead.from_transformers(name)
        with open(cache_path, 'wb') as f:
            pickle.dump(m, f)
        return m


def get_graph(device: str, name='gpt2'):
    gpt2_module = model(name)

    if device == 'cuda':
        gpt2_module.cuda()

    input_ids = hidet.symbol(['seq_length'], dtype=hidet.int32, device=device)
    position_ids = hidet.symbol(['seq_length'], dtype=hidet.int32, device=device)
    cache_shape = [gpt2_module.num_hidden_layers, gpt2_module.num_heads, 0, gpt2_module.head_dim]
    past_keys = hidet.zeros(cache_shape, dtype=hidet.float32, device=device)
    past_values = hidet.zeros(cache_shape, dtype=hidet.float32, device=device)

    outputs = gpt2_module(input_ids, position_ids, past_keys, past_values)
    graph = hidet.trace_from(outputs[0], inputs=[input_ids, position_ids])

    return graph


@pytest.mark.parametrize('model', ['gpt2'])
def test_model_differences(model):
    # Original float32 model
    orig_model = get_graph('cuda', model)
    orig_model = hidet.graph.optimize(orig_model)
    orig_model = orig_model.build()

    input_ids = torch.randint(0, 1024, (1024,), dtype=torch.int32, device='cuda')
    position_ids = torch.arange(0, input_ids.shape[0], dtype=torch.int32, device='cuda')

    orig_logits = orig_model(hidet.from_torch(input_ids), hidet.from_torch(position_ids)).torch()

    # quantize fp16 model to int8
    graph = get_graph('cuda', model)
    with hidet.graph.PassContext() as ctx:
        ctx.set_precision('int8')
        graph = hidet.graph.optimize(graph)

    graph = graph.build()
    new_logits = graph(hidet.from_torch(input_ids), hidet.from_torch(position_ids)).torch()

    assert torch.allclose(orig_logits, new_logits, atol=0.5, rtol=0.5)


if __name__ == "__main__":
    pytest.main([__file__])
