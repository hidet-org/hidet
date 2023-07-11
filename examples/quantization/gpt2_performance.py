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

from hidet.utils.benchmark import benchmark_func

from typing import List
import pytest
import torch
import transformers
import hidet
import hidet.testing


def generate(model, input_ids: hidet.Tensor, num_hidden_layers, num_heads, head_dim, device, tokens_to_generate=10):
    tokenizer = hidet.testing.models.gpt2.tokenizer()    
    position_ids = hidet.arange(input_ids.shape[0], dtype=hidet.int32, device=device)
    past_keys = hidet.zeros([num_hidden_layers, num_heads, 0, head_dim], dtype=hidet.float32, device=device)
    past_values = hidet.zeros([num_hidden_layers, num_heads, 0, head_dim], dtype=hidet.float32, device=device)

    output_ids = []
    for _ in range(tokens_to_generate):
        input_ids, position_ids, past_keys, past_values = model(input_ids, position_ids, past_keys, past_values)
        output_ids.append(input_ids[0].item())

    return tokenizer.decode(output_ids)


def prepare_graph(name: str = 'gpt2', device: str = 'cuda'):
    gpt2_module = hidet.testing.models.gpt2.model(name=name, disable_cache=True)

    if device == 'cuda':
        gpt2_module.cuda()

    input_ids = hidet.symbol(['seq_length'], dtype=hidet.int32, device=device)
    position_ids = hidet.symbol(['seq_length'], dtype=hidet.int32, device=device)
    cache_shape = [gpt2_module.num_hidden_layers, gpt2_module.num_heads, 'prev_seq_length', gpt2_module.head_dim]
    past_keys = hidet.symbol(cache_shape, dtype=hidet.float32, device=device)
    past_values = hidet.symbol(cache_shape, dtype=hidet.float32, device=device)

    outputs = gpt2_module(input_ids, position_ids, past_keys, past_values)
    graph = hidet.trace_from(outputs, inputs=[input_ids, position_ids, past_keys, past_values])

    generate_fn = lambda model, ids, num_tokens: generate(
        model,
        ids,
        gpt2_module.num_hidden_layers,
        gpt2_module.num_heads,
        gpt2_module.head_dim,
        device,
        tokens_to_generate=num_tokens,
    )
    return graph, generate_fn


def benchmark_hidet(model_name='gpt2', space=0, start_tokens=32, num_tokens=100):
    inputs = torch.randint(0, 50257, (start_tokens,)).cuda().to(torch.int32)
    inputs = hidet.from_torch(inputs)

    graph, generate_fn = prepare_graph(model_name, 'cuda')
    graph = hidet.graph.optimize(graph)
    graph = graph.build(space=space)
    orig_latency = benchmark_func(lambda: generate_fn(graph, inputs, num_tokens))

    graph, generate_fn = prepare_graph(model_name, 'cuda')
    graph = hidet.graph.quantize(graph, hidet.graph.quant.default_patterns())
    graph = hidet.graph.optimize(graph)
    graph = graph.build(space=space)
    quant_latency = benchmark_func(lambda: generate_fn(graph, inputs, num_tokens))

    graph, generate_fn = prepare_graph(model_name, 'cuda')
    with hidet.graph.PassContext() as ctx:
        ctx.set_precision('float16')
        graph = hidet.graph.optimize(graph)
    graph = graph.build(space=space)
    fp16_latency = benchmark_func(lambda: generate_fn(graph, inputs, num_tokens))

    graph, generate_fn = prepare_graph(model_name, 'cuda')
    with hidet.graph.PassContext() as ctx:
        ctx.set_precision('float16')
        ctx.add_quantize_pattern(hidet.graph.quant.default_patterns())
        graph = hidet.graph.optimize(graph)
    graph = graph.build(space=space)
    fp16_quant_latency = benchmark_func(lambda: generate_fn(graph, inputs, num_tokens))

    print(f'original f32 latency: {orig_latency}')
    print(f'quantized f32 -> int8 latency: {quant_latency}')
    print(f'f16 latency: {fp16_latency}')
    print(f'quantized f16 -> int8 latency: {fp16_quant_latency}')

benchmark_hidet(model_name='gpt2', space=2, start_tokens=32, num_tokens=100)

