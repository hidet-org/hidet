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
from typing import Optional, Tuple, List

from transformers import LlamaConfig

import hidet
from hidet import Tensor
from hidet.ir.type import data_type
from hidet.graph import nn
from hidet.apps.llm.nn.attention import AttentionState, Attention
from hidet.apps.llm.modeling.pretrained import PretrainedModelForCausalLM


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = hidet.ones([hidden_size])
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = (hidden_states.to(hidet.float32) ** 2).mean(-1, keep_dim=True)
        hidden_states = hidden_states * hidet.ops.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [hidet.float16, hidet.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = hidet.asarray(1.0).to(device) / (
            hidet.asarray(base).to(device) ** (hidet.arange(0, dim, 2).float().to(device=device) / dim)
        )
        self.inv_freq = inv_freq

        self.max_seq_len_cached = max_position_embeddings

    def forward(self, x, seq_len):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # return (
        #     self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        #     self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        # )
        t = hidet.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = t[:, None] * self.inv_freq[None, :]

        emb = hidet.ops.concat([freqs, freqs], axis=-1)
        cos_cached = hidet.ops.cos(emb)[None, None, :, :]
        sin_cached = hidet.ops.sin(emb)[None, None, :, :]
        return cos_cached.to(dtype=x.dtype), sin_cached.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return hidet.ops.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze([0, 1])  # [seq_len, dim]
    sin = sin.squeeze([0, 1])  # [seq_len, dim]

    cos = hidet.ops.take(cos, position_ids, 0).unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = hidet.ops.take(sin, position_ids, 0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        if config.hidden_act != "silu":
            raise NotImplementedError("Only silu activation is supported currently in LlamaMLP")
        self.act_fn = hidet.ops.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.self_attn = Attention()

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        if config.rope_scaling is not None:
            raise NotImplementedError("Rotary Scaling is not supported yet.")

        if self.config.rope_scaling is not None:
            raise NotImplementedError('Rope scaling is not supported yet.')
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def forward(
        self, hidden_states: hidet.Tensor, position_ids: hidet.Tensor, attn_state: AttentionState
    ) -> Tuple[hidet.Tensor, Tuple[hidet.Tensor, hidet.Tensor]]:
        bs, seq, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).reshape([bs, seq, self.num_heads, self.head_dim]).transpose(1, 2)
        key_states = (
            self.k_proj(hidden_states).reshape([bs, seq, self.num_key_value_heads, self.head_dim]).transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states).reshape([bs, seq, self.num_key_value_heads, self.head_dim]).transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        attn_output = self.self_attn(query_states, key_states, value_states, attn_state)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape([bs, seq, self.hidden_size])
        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, hidden_states: hidet.Tensor, position_ids: Optional[hidet.Tensor], attn_state: AttentionState
    ) -> Tuple[hidet.Tensor, Optional[Tuple[hidet.Tensor, hidet.Tensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(hidden_states=hidden_states, position_ids=position_ids, attn_state=attn_state)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaModel(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: hidet.Tensor, position_ids: hidet.Tensor, attn_states: List[AttentionState]):
        hidden_states = self.embed_tokens(input_ids)

        # decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(hidden_states, position_ids=position_ids, attn_state=attn_states[idx])

        hidden_states = self.norm(hidden_states)

        return hidden_states


class LlamaForCausalLM(PretrainedModelForCausalLM):
    def __init__(self, config):
        super().__init__()
        self.config: LlamaConfig = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def num_attention_layers(self):
        return self.config.num_hidden_layers

    def num_attention_heads(self):
        return self.config.num_attention_heads

    def attention_head_size(self):
        return self.config.hidden_size // self.config.num_attention_heads

    def embedding(self) -> Tensor:
        return self.lm_head.transposed_weight()  # [hidden_size, vocab_size]

    def dtype(self):
        return data_type(str(self.config.torch_dtype).split('.')[1])

    def forward(self, input_ids: hidet.Tensor, position_ids: hidet.Tensor, attn_states: List[AttentionState]):
        return self.model(input_ids=input_ids, position_ids=position_ids, attn_states=attn_states)
