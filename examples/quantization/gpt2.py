# %%
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
    

def calculate_perplexity(model, ids, device, cap_length=1024):
    if len(ids) > cap_length:
        ids = ids[:cap_length]
    input_ids = hidet.asarray(ids, dtype=hidet.int32, device=device)
    position_ids = torch.arange(0, input_ids.shape[0], dtype=torch.int32, device=device)

    logits = model(input_ids, hidet.from_torch(position_ids))
    logits: torch.Tensor = logits.torch()
    ids = input_ids.torch().to(torch.int64)
    loss = F.cross_entropy(logits[:-1, :], ids[1:], reduction='mean')
    ppl = torch.exp(loss)

    return ppl


def get_graph(device: str):
    gpt2_module = model()

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


tokenizer = hidet.testing.models.gpt2.tokenizer()
data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
data = data['test']
test_tokenized = data.map(lambda x: tokenizer(x['text']), batched=True)
tokens = list(filter(lambda x: len(x) > 2, test_tokenized['input_ids']))[:250]

orig_model = get_graph('cuda')
orig_model = hidet.graph.optimize(orig_model)
orig_model = orig_model.build()

orig_ppl = 0.0
for ids in tqdm(tokens):
    orig_ppl += float(calculate_perplexity(orig_model, ids, 'cuda'))
orig_ppl /= len(tokens)

quant_model = get_graph('cuda')
quant_model = hidet.graph.quantize(quant_model, hidet.graph.quant.default_quant_patterns())
quant_model = hidet.graph.optimize(quant_model)
quant_model = quant_model.build()

quant_ppl = 0.0
for ids in tqdm(tokens):
    quant_ppl += float(calculate_perplexity(quant_model, ids, 'cuda'))
quant_ppl /= len(tokens)

print(f'original ppl:  {orig_ppl}')
print(f'quantized ppl: {quant_ppl}')

