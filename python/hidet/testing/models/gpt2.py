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
import os
import pickle
import math
from transformers import GPT2Tokenizer
import transformers
import torch
import hidet
import hidet.ir.primitives
from hidet import Tensor
from hidet import nn
from hidet import ops


class GPT2Config:
    def __init__(self):
        self.vocab_size = 50257
        self.max_position_embeddings = 1024
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.intermediate_size = 3072
        self.layer_norm_epsilon = 1e-5
        self.num_heads = 12


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config: GPT2Config = config
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        self.c_attn = nn.LinearTransposed(config.hidden_size, config.hidden_size * 3)
        self.c_proj = nn.LinearTransposed(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: Tensor, last_key, last_value):
        # params:
        #   hidden_states: [seq_length, hidden_size]
        #   last_key: [num_heads, prev_seq_length, head_dim]
        #   last_value: [num_heads, prev_seq_length, head_dim]
        # return:
        #   hidden_states: [seq_length, hidden_size]
        #   key: [num_heads, seq_length, head_dim]
        #   value: [num_heads, seq_length, head_dim]
        seq_length = hidden_states.shape[0]
        prev_seq_length = last_key.shape[1]
        qkv = self.c_attn(hidden_states)  # [seq_length, hidden_size * 3]
        q, k, v = ops.split(qkv, 3, axis=-1)  # [seq_length, hidden_size] * 3
        q = ops.reshape(q, [seq_length, self.num_heads, self.head_dim]).rearrange(
            [[1], [0], [2]]
        )  # [num_heads, seq_length, head_dim]
        k = ops.reshape(k, [seq_length, self.num_heads, self.head_dim]).rearrange(
            [[1], [0], [2]]
        )  # [num_heads, seq_length, head_dim]
        v = ops.reshape(v, [seq_length, self.num_heads, self.head_dim]).rearrange(
            [[1], [0], [2]]
        )  # [num_heads, seq_length, head_dim]

        kk = ops.concat([last_key, k], axis=1)  # [num_heads, prev_seq_length + seq_length, head_dim]
        vv = ops.concat([last_value, v], axis=1)  # [num_heads, prev_seq_length + seq_length, head_dim]

        # [num_heads, seq_length, prev_seq_length + seq_length]
        # like (seq_length = 3, prev_seq_length = 2)
        # 1 1 1
        # 1 1 1 1
        # 1 1 1 1 1
        casual_mask = (
            1
            - hidet.ops.tri(
                n=seq_length,
                m=seq_length + prev_seq_length,
                k=prev_seq_length,
                dtype=hidet.int32,
                device=hidden_states.device,
            )
        ) * hidden_states.dtype.min_value

        # [num_heads, seq_length, prev_seq_length + seq_length]
        attn_weights = ops.matmul(q, ops.transpose(kk, [-1, -2])) / math.sqrt(self.head_dim)

        qk = ops.softmax(attn_weights + casual_mask, axis=-1)  # [num_heads, seq_length, seq_length + prev_seq_length]

        hidden_states = ops.matmul(qk, vv)  # [num_heads, seq_length, head_dim]
        hidden_states = hidden_states.rearrange([[1], [0], [2]]).reshape(
            [seq_length, self.hidden_size]
        )  # [seq_length, hidden_size]
        hidden_states = self.c_proj(hidden_states)  # [seq_length, hidden_size]
        return hidden_states, k, v


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.LinearTransposed(config.hidden_size, config.intermediate_size)
        self.c_proj = nn.LinearTransposed(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        # params:
        #   hidden_states: [seq_length, hidden_size]
        # return:
        #   hidden_states: [seq_length, hidden_size]
        hidden_states = self.c_fc(hidden_states)
        hidden_states = ops.gelu(hidden_states, approximate=True)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, hidden_states, last_key, last_value):
        # params:
        #   hidden_states: [seq_length, hidden_size]
        #   last_key: [num_heads, prev_seq_length, head_dim]
        #   last_value: [num_heads, prev_seq_length, head_dim]
        # return:
        #   hidden_states: [seq_length, hidden_size]
        #   last_key: [num_heads, seq_length, head_dim]
        #   last_value: [num_heads, seq_length, head_dim]
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states, key, value = self.attn(hidden_states, last_key, last_value)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states, key, value


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config: GPT2Config = config
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, position_ids, past_keys, past_values):
        # params:
        #   input_ids: [seq_length]
        #   position_ids: int32[seq_length]
        #   past_keys: [layers, num_heads, prev_seq_length, head_dim]
        #   past_values: [layers, num_heads, prev_seq_length, head_dim]
        # return:
        #   hidden_states: [1, hidden_size]
        #   position_ids: int32[1]
        #   updated_keys: [layers, num_heads, prev_seq_length + seq_length, head_dim]
        #   updated_values: [layers, num_heads, prev_seq_length + seq_length, head_dim]
        inputs_embeds = self.wte(input_ids)  # [seq_length, hidden_size]
        position_embeds = self.wpe(position_ids)  # [seq_length, hidden_size]
        hidden_states = inputs_embeds + position_embeds  # [seq_length, hidden_size]
        cur_keys = []  # layers of [1, num_heads, seq_length, head_dim]
        cur_values = []  # layers of [1, num_heads, seq_length, head_dim]
        for i, block in enumerate(self.h):
            hidden_states, cur_key, cur_value = block(hidden_states, past_keys[i], past_values[i])
            cur_keys.append(cur_key.unsqueeze(0))
            cur_values.append(cur_value.unsqueeze(0))

        hidden_states = self.ln_f(hidden_states)  # [seq_length, hidden_size]

        # [layers, num_heads, prev_seq_length + seq_length, head_dim]]
        updated_cur_keys = ops.concat([past_keys, ops.concat(cur_keys, axis=0)], axis=2)

        # [layers, num_heads, prev_seq_length + seq_length, head_dim]]
        updated_past_values = ops.concat([past_values, ops.concat(cur_values, axis=0)], axis=2)
        # updated_cur_keys = None
        # updated_past_values = None

        position_ids = position_ids[-1:] + 1  # [1]

        return hidden_states[-1:], position_ids, updated_cur_keys, updated_past_values
        # return hidden_states[-1:], position_ids, None, None


class GPT2LMHead(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

    @staticmethod
    def from_transformers(model_name: str):
        assert model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2']

        # load from transformers
        hf_gpt2: torch.nn.Module = transformers.GPT2LMHeadModel.from_pretrained(model_name)
        hf_config = transformers.GPT2Config.from_pretrained(model_name)

        # create config
        config = GPT2Config()
        config.vocab_size = hf_config.vocab_size
        config.hidden_size = hf_config.n_embd
        config.num_hidden_layers = hf_config.n_layer
        config.num_heads = hf_config.n_head
        config.intermediate_size = hf_config.n_inner if hf_config.n_inner else 4 * hf_config.n_embd
        config.max_position_embeddings = hf_config.n_positions
        config.layer_norm_epsilon = hf_config.layer_norm_epsilon

        # create model
        module = GPT2LMHead(config)
        allow_missing = ['lm_head.weight']
        found_tensors = []
        for name, tensor in hf_gpt2.named_parameters():
            pointer = module
            for m_name in name.split('.'):
                pointer = getattr(pointer, m_name)
            if not isinstance(pointer, Tensor):
                raise ValueError('{} is not a tensor'.format(name))
            found_tensors.append(pointer)
            pointer.copy_(hidet.from_torch(tensor))
        module.lm_head.weight = module.transformer.wte.weight

        for name, tensor in module.named_parameters():
            if tensor not in found_tensors and name not in allow_missing:
                raise ValueError(f'not found {name}')
        return module

    def forward(self, input_ids, position_ids, past_keys, past_values):
        # params:
        #   input_ids: int32[seq_length]
        #   position_ids: int32[seq_length]
        #   past_keys: [layers, prev_seq_length, hidden_size]
        #   past_values: [layers, prev_seq_length, hidden_size]
        # return:
        #   input_ids: int32[1]
        #   position_ids: int32[1]
        #   updated_keys: [layers, prev_seq_length + seq_length, hidden_size]
        #   updated_values: [layers, prev_seq_length + seq_length, hidden_size]
        hidden_states, position_ids, past_keys, past_values = self.transformer(
            input_ids, position_ids, past_keys, past_values
        )
        logits = self.lm_head(hidden_states)  # [1, vocab_size]
        updated_input_ids = ops.argmax(logits, dim=-1, keep_dim=False)  # [1]
        # we want to keep types consistent, since in the autoregressive case,
        #   the output is fed back into the input of the compiled model
        updated_input_ids = updated_input_ids.to(input_ids.dtype)
        return updated_input_ids, position_ids, past_keys, past_values


def model(name='gpt2', disable_cache=False) -> GPT2LMHead:
    """
    Get GPT2 model.

    Parameters
    ----------
    name: str
        The size of the model, can be 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl' or 'distilgpt2'.

    disable_cache: bool
        Whether to disable cache for the model.

    Returns
    -------
    ret: GPT2LMHead
        The GPT2 model.
    """
    cache_path = hidet.utils.cache_file('testing', 'models', 'gpt2', name + '.pkl')
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


def tokenizer(name='gpt2') -> GPT2Tokenizer:
    return GPT2Tokenizer.from_pretrained(name)
