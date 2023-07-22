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
import math
from typing import Optional, Tuple, List
from collections import OrderedDict

from tqdm import tqdm

import torch
from transformers import LlamaConfig, LlamaTokenizer
from transformers import LlamaForCausalLM as hfLm

import hidet
from hidet.graph import nn


def copy_weights(torch_model, hidet_model):
    found_tensors = []
    for name, tensor in tqdm(list(torch_model.named_parameters()), desc='copying weights'):
        mod = hidet_model
        for m_name in name.split('.'):
            mod = getattr(mod, m_name)

        if not isinstance(mod, hidet.Tensor):
            print(type(mod))
            raise ValueError(f"hidet/hf mismatch at {name}")

        found_tensors.append(mod)
        mod.copy_(hidet.from_torch(tensor).to(mod.dtype, mod.device))
        if mod.shape != tensor.shape:
            print(f"hidet/hf shape mismatch at {name}, hidet: {mod.shape}, torch: {tensor.shape}")

    buffer_names = set(name for name, _ in torch_model.named_buffers())

    for name, tensor in hidet_model.named_parameters():
        if tensor not in found_tensors and name not in buffer_names:
            raise ValueError(f'{name} not copied')


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
        t = hidet.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = t[:, None] * self.inv_freq[None, :]

        emb = hidet.ops.concat([freqs, freqs], axis=-1)
        self.cos_cached = hidet.ops.cos(emb)[None, None, :, :]
        self.sin_cached = hidet.ops.sin(emb)[None, None, :, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len is None:
            seq_len = x.shape[-2]

        # Unfortunately, dynamic shape forbids this
        # if seq_len > self.max_seq_len_cached:
        #     self.max_seq_len_cached = seq_len
        #     t = hidet.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
        #     freqs = t[:, None] * self.inv_freq[None, :]
        #     # Different from paper, but it uses a different permutation in order to obtain the same calculation
        #     emb = hidet.ops.concat((freqs, freqs), axis=-1).to(x.device)
        #     self.cos_cached = hidet.ops.cos(emb)[None, None, :, :]
        #     self.sin_cached = hidet.ops.sin(emb)[None, None, :, :]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


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
        self.pretraining_tp = config.pretraining_tp
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        if config.hidden_act != "silu":
            raise NotImplementedError("Only silu activation is supported currently in LlamaMLP")
        self.act_fn = hidet.ops.silu

    def forward(self, x):
        if self.pretraining_tp > 1:
            # I think this is only for training, so we can skip it
            raise RuntimeError("Pretraining TP > 1 is not supported yet")
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: hidet.Tensor, n_rep: int) -> hidet.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings

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
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def forward(
        self,
        hidden_states: hidet.Tensor,
        attention_mask: Optional[hidet.Tensor] = None,
        position_ids: Optional[hidet.Tensor] = None,
        past_key_value: Optional[Tuple[hidet.Tensor]] = None,
    ) -> Tuple[hidet.Tensor, Tuple[hidet.Tensor, hidet.Tensor]]:
        bsz, q_len, _ = hidden_states.shape

        if self.config.pretraining_tp > 1:
            raise RuntimeError("Pretraining TP > 1 is not supported yet")

        query_states = self.q_proj(hidden_states).reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose(1, 2)
        key_states = self.k_proj(hidden_states).reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose(1, 2)
        value_states = self.v_proj(hidden_states).reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = hidet.ops.concat([past_key_value[0], key_states], axis=2)
            value_states = hidet.ops.concat([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = hidet.ops.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = hidet.ops.softmax(attn_weights, axis=-1).to(query_states.dtype)
        attn_output = hidet.ops.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape([bsz, q_len, self.hidden_size])

        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: hidet.Tensor,
        attention_mask: Optional[hidet.Tensor] = None,
        position_ids: Optional[hidet.Tensor] = None,
        past_key_value: Optional[Tuple[hidet.Tensor]] = None,
    ) -> Tuple[hidet.Tensor, Optional[Tuple[hidet.Tensor, hidet.Tensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value)

        return outputs


def hidet_make_causal_mask(seq_len, dtype, device, past_key_values_length):
    # pylint: disable=protected-access
    x = hidet.ops.tri(
        n=seq_len, m=seq_len + past_key_values_length, k=past_key_values_length, dtype=dtype, device=device
    )
    return (1 - x) * float(dtype.min_value)


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

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: hidet.Tensor,
        position_ids: hidet.Tensor,
        past_key_values: Optional[List[hidet.Tensor]] = None,
        inputs_embeds: Optional[hidet.Tensor] = None,
    ):
        _, seq_length = input_ids.shape
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[-2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = hidet_make_causal_mask(
            seq_length, inputs_embeds.dtype, device=inputs_embeds.device, past_key_values_length=past_key_values_length
        )

        hidden_states = inputs_embeds

        # decoder layers
        next_decoder_cache = tuple()

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value
            )

            hidden_states = layer_outputs[0]

            next_decoder_cache += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        next_cache = next_decoder_cache
        ret = OrderedDict(last_hidden_state=hidden_states, past_key_values=next_cache)
        return ret


class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: hidet.Tensor,
        position_ids: hidet.Tensor,
        past_key_values: List[Tuple[hidet.Tensor]],
        inputs_embeds: hidet.Tensor = None,
    ):
        seq_len = input_ids.shape[-1]
        prev_seq_len = 0
        if past_key_values is not None:
            prev_seq_len = past_key_values[0][0].shape[-2]
        new_seq_len = seq_len + prev_seq_len
        position_ids = position_ids[:, prev_seq_len:new_seq_len]
        outputs = self.model(
            input_ids=input_ids, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds
        )

        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)

        new_ids = hidet.ops.argmax(logits, dim=-1, keep_dim=False)  # [bs, seq_len]

        return OrderedDict(new_ids=new_ids, logits=logits, past_key_values=outputs["past_key_values"])


def convert_model(hf_model: torch.nn.Module, dtype=hidet.float16, device='cuda'):
    config = hf_model.config

    hidet_model = LlamaForCausalLM(config)
    hidet_model.to(device=device, dtype=dtype)

    copy_weights(hf_model, hidet_model)

    return hidet_model


def build_flow_graph(model, batch_size=1, device='cuda', dtype='float16'):
    config = model.config
    input_ids = hidet.symbol([batch_size, "seq_length"], dtype=hidet.int32, device=device)
    position_ids = hidet.symbol([batch_size, config.max_position_embeddings], dtype=hidet.int32, device=device)

    get_sym = lambda: hidet.symbol(
        [batch_size, config.num_attention_heads, "prev_seq_len", config.hidden_size // config.num_attention_heads],
        device=device,
        dtype=dtype,
    )
    key_value_cache = [(get_sym(), get_sym()) for _ in range(config.num_hidden_layers)]

    y = model(input_ids, position_ids=position_ids, past_key_values=key_value_cache)
    inputs = [input_ids, position_ids]
    for q, k in key_value_cache:
        inputs.append(q)
        inputs.append(k)

    outputs = [y['new_ids']]
    for q, k in y['past_key_values']:
        outputs.append(q)
        outputs.append(k)

    return hidet.trace_from(outputs, inputs)


def generate(text: str, model, tokenizer, config, num_tokens=20, device='cuda', dtype='float16'):
    input_ids = tokenizer.encode(text)
    input_ids = hidet.asarray([input_ids]).to(dtype=hidet.int32, device=device)

    position_ids = hidet.arange(0, config.max_position_embeddings, dtype=hidet.int32, device=device).unsqueeze(0)

    make_past = lambda: hidet.zeros(
        [1, config.num_attention_heads, 0, config.hidden_size // config.num_attention_heads], device=device, dtype=dtype
    )
    past_keys_values = [make_past() for _ in range(config.num_hidden_layers * 2)]

    outputs = []
    for _ in range(num_tokens):
        y = model(input_ids, position_ids, *past_keys_values)
        input_ids = y[0][:, -1:].to(dtype=hidet.int32)
        outputs.append(input_ids[0, -1].item())
        past_keys_values = y[1:]

    return tokenizer.decode(outputs)


def generate_torch(input_ids: str, tokenizer, torch_model, num_tokens, device='cuda', dtype=torch.float16):
    torch_model = torch_model.to(device=device, dtype=dtype)
    input_ids = tokenizer.encode(input_ids)
    input_ids = torch.tensor(input_ids).to(device=device).unsqueeze(0)
    config = torch_model.config

    attention_mask = torch.ones([1, config.max_position_embeddings]).to(device=device, dtype=dtype)
    # position_ids = torch.arange(0, config.max_position_embeddings, device='cuda').unsqueeze(0)
    make_past = lambda: torch.zeros(
        [1, config.num_attention_heads, 0, config.hidden_size // config.num_attention_heads]
    ).to(device=device, dtype=dtype)
    key_value_cache = [(make_past(), make_past()) for _ in range(config.num_hidden_layers)]
    outputs = []
    cur_len = input_ids.shape[-1]
    for _ in range(num_tokens):
        y = torch_model(
            input_ids,
            attention_mask=attention_mask[:, :cur_len],
            position_ids=None,
            past_key_values=key_value_cache,
            use_cache=True,
        )
        logits = y['logits']
        new_ids = torch.argmax(logits, -1, keepdim=False)
        new_ids = new_ids[:, -1:]
        outputs.append(new_ids[0, -1].item())
        input_ids = new_ids
        key_value_cache = y['past_key_values']
        cur_len += 1

    return tokenizer.decode(outputs)


def get_compiled_model(name='decapoda-research/llama-7b-hf', device='cuda', opt=False):
    tok = LlamaTokenizer.from_pretrained(name)

    with torch.device("cuda"):  # reduce the time to load the model
        model = hfLm.from_pretrained(name, torch_dtype=torch.float16)

    model.cpu()
    torch.cuda.empty_cache()

    config = model.config

    model: nn.Module = convert_model(model, device=device)

    flow_graph = build_flow_graph(model, device=device)

    if opt:
        with hidet.graph.PassContext() as ctx:
            ctx.reduce_cuda_compile_mem()
            flow_graph = hidet.graph.optimize(flow_graph)

    compiled = flow_graph.build()
    return compiled, config, tok


def demo_usage():
    device = 'cuda'
    opt = False
    model, config, tokenizer = get_compiled_model(device=device, opt=opt)

    text = generate('In the beginning was the Word.', model, tokenizer, config, num_tokens=12)
    assert text == 'The Word was with God, and the Word was God.'

    text = generate(
        "A robot may not injure a human being or, through inaction", model, tokenizer, config, num_tokens=55
    )

    expected = (
        ', allow a human being to come to harm. A robot must obey the orders given it by human beings'
        ' except where such orders would conflict with the First Law. A robot must protect its own'
        ' existence as long as such protection does not conflict with the First or Second Laws'
    )
    assert text == expected


if __name__ == '__main__':
    demo_usage()
