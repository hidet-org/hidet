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
from typing import List

import os
import pickle

from tqdm import tqdm

import torch
import hidet
import torch.nn.functional as F
import datasets

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

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # st()
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # st()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        # correct = (pred == target.view(1, -1).expand_as(pred))
        correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def calculate_perplexity(model, ids, device, cap_length=1024):
    if len(ids) > cap_length:
        ids = ids[:cap_length]
    input_ids = hidet.asarray(ids, dtype=hidet.int32, device=device)
    position_ids = torch.arange(0, input_ids.shape[0], dtype=torch.int32, device=device)

    logits = model(input_ids, hidet.from_torch(position_ids))
    logits: torch.Tensor = logits.torch()
    ids = input_ids.torch().to(torch.int64)
    loss = F.cross_entropy(logits[:-1, :].float(), ids[1:], reduction='mean')
    ppl = torch.exp(loss)

    return ppl


def calculate_accuracy(model, ids, device, topk=(1,), cap_length=1024):
    if len(ids) > cap_length:
        ids = ids[:cap_length]
    input_ids = hidet.asarray(ids, dtype=hidet.int32, device=device)
    position_ids = torch.arange(0, input_ids.shape[0], dtype=torch.int32, device=device)

    logits = model(input_ids, hidet.from_torch(position_ids))
    logits: torch.Tensor = logits.torch()
    ids = input_ids.torch().to(torch.int64)
    acc = accuracy(logits[:-1, :].float(), ids[1:], topk=topk)

    return acc

    
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

def compute_metrics(model, data: List[List[int]], topk=(1, 5, 10)):
    """
    Accepts a model that takes in two hidet tensors of type int,
        first argument are the input ids, second argument are the position ids
        both of shape [seq_length], and returns logits of shape [seq_length, vocab_size]
    """
    max_k = max(topk)
    orig_ppl = 0.0
    orig_acc = [0.0, 0.0, 0.0]
    num_accounted = 0

    for ids in tqdm(data):
        if len(ids) > max_k:
            orig_ppl += float(calculate_perplexity(model, ids, 'cuda'))
            acc = calculate_accuracy(model, ids, 'cuda', topk=topk)
            orig_acc = [x + float(y) for x, y in zip(orig_acc, acc)]
            num_accounted += 1
    orig_ppl /= num_accounted
    orig_acc  = [x / num_accounted for x in orig_acc]
    return orig_ppl, orig_acc


def get_wikitext_test_data() -> List[List[int]]:
    tokenizer = hidet.testing.models.gpt2.tokenizer()
    data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
    data = data['test']
    test_tokenized = data.map(lambda x: tokenizer(x['text']), batched=True)
    tokens = list(filter(lambda x: len(x) > 2, test_tokenized['input_ids']))[:500]
    return tokens


def show_differences(tokens: List[List[int]], model='gpt2'):
    topk = (1, 5, 10)
    format_acc = lambda x: '[' + ', '.join([f'top-{k}: {round(y, 3)}' for k, y in zip(topk, x)]) + ']'
    # Original float32 model
    orig_model = get_graph('cuda', model)
    orig_model = hidet.graph.optimize(orig_model)
    orig_model = orig_model.build()

    orig_model_ppl, orig_model_acc = compute_metrics(orig_model, tokens, topk=topk)
    orig_model_acc = format_acc(orig_model_acc)

    ######################
    # cast model to fp16
    graph = get_graph('cuda', model)        
    with hidet.graph.PassContext() as ctx:
        ctx.set_precision('float16')
        graph = hidet.graph.optimize(graph)
    graph = graph.build()
    fp16_model_ppl, fp16_model_acc = compute_metrics(graph, tokens, topk=topk)
    fp16_model_acc = format_acc(fp16_model_acc)
    #####################

    # quantize fp16 model to int8
    graph = get_graph('cuda', model)
    with hidet.graph.PassContext() as ctx:
        # setting the precision to int8 will first cast the model to float16, then quantize
        # layers according to the default settings
        # Under the default settings, the layers that will be quantized are linear and embedding.

        # More precisely, there is no concept of a 'layer' in the flowgraph, so the quantization
        # actually is applied to certain patterns of operators. For example, the linear 'layer' pattern
        # detects if a matmul has a constant input of len(shape) == 2. If so, the constant will be quantized.
        ctx.set_precision('int8')
        # to customize the quantization settings, use ctx.add_quantize_rules(List[SubgraphRewriteRule]); consult the
        # docs of that function for more info
        graph = hidet.graph.optimize(graph)
    graph = graph.build()
    fused_quant_model_ppl, fused_quant_model_acc = compute_metrics(graph, tokens, topk=topk)
    fused_quant_model_acc = format_acc(fused_quant_model_acc)
    #####################

    print(f'topk: {topk}')
    print(f'original f32 ppl:  {orig_model_ppl}')
    print(f'original f32 acc:  {orig_model_acc}\n')
    print(f'quantized f16 ppl: {fp16_model_ppl}')
    print(f'quantized f16 acc: {fp16_model_acc}\n')
    print(f'quantized f16 -> int8 ppl: {fused_quant_model_ppl}')
    print(f'quantized f16 -> int8 acc: {fused_quant_model_acc}\n')

tokens = get_wikitext_test_data()
show_differences(tokens, 'gpt2')

