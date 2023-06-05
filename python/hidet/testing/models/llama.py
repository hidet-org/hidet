# %%
import math
from typing import Optional, Tuple, List
from collections import OrderedDict

from tqdm import tqdm

import torch
from transformers import LlamaConfig

import hidet
from hidet.graph import nn


def copy_weights(torch_model, hidet_model):
    found_tensors = []
    for name, tensor in tqdm(list(torch_model.named_parameters()), desc='copying weights'):
        mod = hidet_model
        is_linear = False
        for m_name in name.split('.'):
            mod = getattr(mod, m_name)
            if isinstance(mod, hidet.nn.Linear):
                is_linear = True

        if not isinstance(mod, hidet.Tensor):
            print(type(mod))
            raise ValueError(f"hidet/hf mismatch at {name}")

        found_tensors.append(mod)
        if is_linear:
            # print(f"linear layer {name} found, transposing weight")
            mod.copy_(hidet.from_torch(tensor).to(mod.dtype, mod.device).transpose(0, 1))
        elif list(tensor.shape) == list(mod.shape):
            mod.copy_(hidet.from_torch(tensor).to(mod.dtype, mod.device))
        else:
            raise ValueError(f"{name} is not copied due to shape mismatch")

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

        emb = hidet.ops.concat((freqs, freqs), axis=-1)
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
    return hidet.ops.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]

    cos = hidet.ops.take(cos, position_ids, 0).unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = hidet.ops.take(sin, position_ids, 0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
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
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: hidet.Tensor, seq_len: int, bsz: int):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: hidet.Tensor,
        attention_mask: Optional[hidet.Tensor] = None,
        position_ids: Optional[hidet.Tensor] = None,
        past_key_value: Optional[Tuple[hidet.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[hidet.Tensor, Optional[hidet.Tensor], Optional[Tuple[hidet.Tensor]]]:

        bsz, q_len, _ = hidden_states.shape

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

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = hidet.ops.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            # attn_weights = hidet.ops.max(attn_weights, hidet.asarray(float((attn_weights.dtype.min_value))))

        # upcast attention to fp32
        attn_weights = hidet.ops.softmax(attn_weights, axis=-1).to(query_states.dtype)
        attn_output = hidet.ops.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape([bsz, q_len, self.hidden_size])

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: hidet.Tensor,
        attention_mask: Optional[hidet.Tensor] = None,
        position_ids: Optional[hidet.Tensor] = None,
        past_key_value: Optional[Tuple[hidet.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[hidet.Tensor, Optional[Tuple[hidet.Tensor, hidet.Tensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def hidet_make_causal_mask(seq_len, dtype, device, past_key_values_length):
    # pylint: disable=protected-access
    x = hidet.ops.tri(
        n=seq_len, m=seq_len + past_key_values_length, k=past_key_values_length, dtype=dtype, device=device
    )
    return (1 - x) * float(dtype._min_value)


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

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]

        combined_attention_mask = hidet_make_causal_mask(
            input_shape[1],
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

        return combined_attention_mask

    def forward(
        self,
        input_ids: hidet.Tensor,
        position_ids: hidet.Tensor,
        past_key_values: Optional[List[hidet.Tensor]] = None,
        inputs_embeds: Optional[hidet.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        _, seq_length = input_ids.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[-2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = hidet_make_causal_mask(
            seq_length, inputs_embeds.dtype, device=inputs_embeds.device, past_key_values_length=past_key_values_length
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        ret = OrderedDict(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return ret


class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing

    def forward(
        self,
        input_ids: hidet.Tensor,
        position_ids: hidet.Tensor,
        past_key_values: List[Tuple[hidet.Tensor]],
        inputs_embeds: hidet.Tensor = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        seq_len = input_ids.shape[-1]
        prev_seq_len = 0
        if past_key_values is not None:
            prev_seq_len = past_key_values[0][0].shape[-2]
        new_seq_len = seq_len + prev_seq_len
        position_ids = position_ids[:, prev_seq_len:new_seq_len]
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)

        new_ids = hidet.ops.argmax(logits, dim=-1, keep_dim=False)  # [bs, seq_len]

        return OrderedDict(
            new_ids=new_ids,
            logits=logits,
            past_key_values=outputs["past_key_values"],
            hidden_states=outputs["hidden_states"],
            attentions=outputs["attentions"],
        )


def convert_model(hf_model: torch.nn.Module, dtype=hidet.float16, device='cuda'):
    config = hf_model.config
    # config = LlamaConfig(**hf_config.__dict__)
    orig_set_attr = hidet.nn.Module.__setattr__

    def my_setattr(self, key, value):
        if isinstance(value, hidet.Tensor):
            value = value.to(dtype, device)
        orig_set_attr(self, key, value)

    hidet.nn.Module.__setattr__ = my_setattr
    hidet_model = LlamaForCausalLM(config).to(device=device)
    hidet.nn.Module.__setattr__ = orig_set_attr

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
    key_value_cache = [(get_sym(), get_sym()) for i in range(config.num_hidden_layers)]

    y = model(input_ids, position_ids=position_ids, past_key_values=key_value_cache, use_cache=True)
    inputs = [input_ids, position_ids]
    for (q, k) in key_value_cache:
        inputs.append(q)
        inputs.append(k)

    outputs = [y['new_ids']]
    for (q, k) in y['past_key_values']:
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
        input_ids = y[0][:, -1:]
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
    key_value_cache = [(make_past(), make_past()) for i in range(config.num_hidden_layers)]
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
    from transformers import LlamaTokenizer, LlamaForCausalLM as hfLm

    tok = LlamaTokenizer.from_pretrained(name)

    model = hfLm.from_pretrained(name, torch_dtype=torch.float16)

    config = model.config

    model = convert_model(model, device=device)
    flow_graph = build_flow_graph(model, device=device)

    if opt:
        flow_graph = hidet.graph.optimize(flow_graph)

    compiled = flow_graph.build()
    return compiled, config, tok


def test_llama(device='cuda', opt=False):
    model, config, tokenizer = get_compiled_model(device=device, opt=opt)

    text = generate('In the beginning was the Word.', model, tokenizer, config, num_tokens=12)
    assert text == 'The Word was with God, and the Word was God.'

    text = generate(
        "A robot may not injure a human being or, through inaction", model, tokenizer, config, num_tokens=55
    )
    assert (
        text
        == ', allow a human being to come to harm. A robot must obey orders given it by human beings\
 except where such orders would conflict with the First Law. A robot must protect its own\
 existence as long as such protection does not conflict with the First or Second Laws.'
    )


def failure_case():
    # pylint: disable=unused-variable
    name = 'decapoda-research/llama-7b-hf'
    device = 'cuda'
    from transformers import LlamaTokenizer, LlamaForCausalLM as hfLm

    tok = LlamaTokenizer.from_pretrained(name)

    # model = hfLm.from_pretrained(name, torch_dtype=torch.float16)
    model = hfLm(LlamaConfig(hidden_size=512, intermediate_size=1024, num_hidden_layers=2, num_attention_heads=8))

    config = model.config

    model = convert_model(model, device=device).cuda()
    flow_graph = build_flow_graph(model, device=device)

    flow_graph = hidet.graph.optimize(flow_graph)

    compiled = flow_graph.build()


if __name__ == '__main__':
    hidet.option.parallel_build(False)
    failure_case()
    test_llama(device='cuda', opt=False)
    # test_llama(device='cuda', opt=True)
