# %%
# pylint: skip-file
# since this is just a testing file, remove later

import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List
import math

from collections import OrderedDict

import hidet
import hidet.graph.nn as nn
import torch

from transformers import LlamaConfig

def copy_weights(torch_model, hidet_model):
    found_tensors = []
    for name, tensor in torch_model.named_parameters():
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

def recurse_convert_torch(obj):
    if isinstance(obj, hidet.Tensor):
        return obj.torch()
    elif isinstance(obj, tuple):
        return tuple(recurse_convert_torch(i) for i in obj)
    elif isinstance(obj, list):
        return list(recurse_convert_torch(i) for i in obj)
    elif isinstance(obj, dict):
        return {n: recurse_convert_torch(v) for n, v in obj.items()}
    else:
        return obj
    
def recurse_convert_hidet(obj):
    if isinstance(obj, torch.Tensor):
        return hidet.from_torch(obj.contiguous())
    elif isinstance(obj, tuple):
        return tuple(recurse_convert_hidet(i) for i in obj)
    elif isinstance(obj, list):
        return list(recurse_convert_hidet(i) for i in obj)
    elif isinstance(obj, dict):
        return {n: recurse_convert_hidet(v) for n, v in obj.items()}
    else:
        return obj

def check_equal(torch_cls, hidet_cls, init_args_tch, args, kwargs={}, init_args_hidet=None, return_models=False):
    tch_model = torch_cls(*init_args_tch)
    if init_args_hidet is not None:
        hidet_model = hidet_cls(*init_args_hidet)
    else:
        hidet_model = hidet_cls(*init_args_tch)
        
    copy_weights(tch_model, hidet_model)
    y1 = tch_model(*args, **kwargs)
    y2 = hidet_model(*recurse_convert_hidet(args), **recurse_convert_hidet(kwargs))
    if return_models:
        return tch_model, hidet_model
    else:
        return y1, recurse_convert_torch(y2)

def _expand_mask(mask: hidet.Tensor, dtype, tgt_len=None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype=dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(hidet.dtypes.boolean), float(dtype._min_value))


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

    cos = hidet.ops.take(cos, position_ids, 0).unsqueeze(1) # [bs, 1, seq_len, dim]
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
        self.attn_mask = attention_mask
        self.hidden_states1 = hidden_states
        self.position_ids = position_ids

        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose(1, 2)
        key_states = self.k_proj(hidden_states).reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose(1, 2)
        value_states = self.v_proj(hidden_states).reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose(1, 2)

        self.query_states = query_states
        self.key_states = key_states
        self.value_states = value_states

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, kv_seq_len)
        # self.key_states = key_states
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = hidet.ops.concat([past_key_value[0], key_states], axis=2)
            value_states = hidet.ops.concat([past_key_value[1], value_states], axis=2)


        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = hidet.ops.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        self.attn_weights = attn_weights

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
        self.hidden_states = hidden_states
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
    x = hidet.ops.tri(n=seq_len, m=seq_len + past_key_values_length, k=past_key_values_length, dtype=dtype, device=device)
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
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # use_cache = use_cache if use_cache is not None else self.config.use_cache

        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[-2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = hidet_make_causal_mask(
            seq_length,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
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
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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

        new_ids = hidet.ops.argmax(logits, dim=-1, keep_dim=False) # [bs, seq_len]

        return OrderedDict(
            new_ids=new_ids,
            logits=logits,
            past_key_values=outputs["past_key_values"],
            hidden_states=outputs["hidden_states"],
            attentions=outputs["attentions"],
        )


def get_model(args, hf_config):
    from transformers import AutoModelForCausalLM  # type: ignore[import]

    model_name = args.model
    model_path = args.model_path
    dtype = args.quantization.model_dtype
    max_seq_len = args.max_seq_len

    if model_name.startswith("vicuna-") or model_name.startswith("llama-"):
        config = LlamaConfig(**hf_config, dtype=dtype)
        if max_seq_len != -1:
            config.max_sequence_length = max_seq_len

        hf_model = AutoModelForCausalLM.from_pretrained(model_path)
        # Get a list of parameters in advance, then delete the model to save memory
        param_list = [param for _, param in hf_model.named_parameters()]
        del hf_model

        for i, param in enumerate(param_list):
            param_list[i] = hidet.from_torch(param.detach()).to(config.dtype)

        # construct the new model without duplicating the weights
        orig_set_attr = nn.Module.__setattr__

        def my_setattr(self, key, value):
            # the order of the defined weights are the same
            if isinstance(value, hidet.Tensor):
                value = param_list.pop(0)
            orig_set_attr(self, key, value)

        hidet.nn.Module.__setattr__ = my_setattr
        hidet_model = LlamaForCausalLM(config)
        hidet.nn.Module.__setattr__ = orig_set_attr

        return hidet_model

    raise ValueError(f"Unsupported model: {model_name}")


def small_model_loading_test():
    from transformers import LlamaForCausalLM as hfLlamaModel, LlamaConfig as hfLlamaConfig

    configuration = hfLlamaConfig(
        vocab_size=512, hidden_size=512, intermediate_size=512, num_attention_heads=8, num_hidden_layers=2
    )
    hf_model = hfLlamaModel(configuration)

    ###### use the local LlamaConfig
    config = LlamaConfig(
        vocab_size=512, hidden_size=512, intermediate_size=512, num_attention_heads=8, num_hidden_layers=2
    )
    # Get a list of parameters in advance, then delete the model to save memory
    param_list = [param for _, param in hf_model.named_parameters()]
    del hf_model

    for i, param in enumerate(param_list):
        param_list[i] = hidet.from_torch(param.detach()).to(config.dtype)

    # construct the new model without duplicating the weights
    # since the hidet model is isomorphic to the original torch model
    # this only works if we add .register_buffer(...)
    orig_set_attr = hidet.nn.Module.__setattr__

    def my_setattr(self, key, value):
        # the order of the defined weights are the same
        if isinstance(value, hidet.Tensor):
            value = param_list.pop(0)
        orig_set_attr(self, key, value)

    hidet.nn.Module.__setattr__ = my_setattr
    hidet_model = LlamaForCausalLM(config)
    hidet.nn.Module.__setattr__ = orig_set_attr

    x = hidet.randint(0, 512, shape=[1, 512])
    y = hidet_model(x)

        
def convert_model(hf_model: torch.nn.Module, dtype=hidet.float16, device='cuda'):
    config = hf_model.config
    # config = LlamaConfig(**hf_config.__dict__)
    orig_set_attr = hidet.nn.Module.__setattr__

    def my_setattr(self, key, value):
        if isinstance(value, hidet.Tensor):
            value = value.to(dtype, device)
        orig_set_attr(self, key, value)

    hidet.nn.Module.__setattr__ = my_setattr
    hidet_model = LlamaForCausalLM(config).cuda()
    hidet.nn.Module.__setattr__ = orig_set_attr

    copy_weights(hf_model, hidet_model)
    
    return hidet_model

def build_flow_graph(model, batch_size=1, device='cuda', dtype='float16'):
    config = model.config
    input_ids = hidet.symbol([batch_size, "seq_length"], dtype=hidet.int32, device=device)
    attn_mask = hidet.symbol([batch_size, config.max_position_embeddings], dtype=dtype, device=device) # new_seq_len == seq_length + prev_seq_len
    position_ids = hidet.symbol([batch_size, config.max_position_embeddings], dtype=hidet.int32, device=device)

    get_sym = lambda: hidet.symbol([batch_size, config.num_attention_heads, "prev_seq_len", config.hidden_size // config.num_attention_heads], device=device, dtype=dtype)
    key_value_cache = [
        (get_sym(), get_sym()) for i in range(config.num_hidden_layers)
    ]

    y = model(input_ids, attention_mask=attn_mask, position_ids=position_ids, past_key_values=key_value_cache, use_cache=True)
    inputs = [input_ids, attn_mask, position_ids]
    for (q, k) in key_value_cache: inputs.append(q); inputs.append(k)

    outputs = [y['logits']]
    # for (q, k) in y['past_key_values']: outputs.append(q); outputs.append(k)

    return hidet.trace_from(outputs, inputs)

def generate(text: str, model, tokenizer, config, num_tokens=20, device='cuda', dtype='float16'):
    input_ids = tokenizer.encode(text)
    input_ids = hidet.asarray([input_ids]).to(device=device)
    cur_len = input_ids.shape[0]

    position_ids = hidet.arange(0, config.max_position_embeddings, device=device).unsqueeze(0)

    make_past = lambda: hidet.zeros([1, config.num_attention_heads, 0, config.hidden_size // config.num_attention_heads], device=device, dtype=dtype)
    past_keys_values = [make_past() for _ in range(config.num_hidden_layers * 2)]
    attention_mask = hidet.ones([1, config.max_position_embeddings], device=device, dtype=dtype)

    outputs = []
    for _ in range(num_tokens):
        y = model(input_ids, attention_mask, position_ids, *past_keys_values)
        cur_len += 1
        outputs.append(input_ids[0, -1].item())
        input_ids = y[0][:, -1:]
        past_keys_values = y[1:]
    
    return tokenizer.decode(outputs)

def generate_torch(input_ids: str, tokenizer, torch_model, num_tokens, device='cuda', dtype=torch.float16):
    torch_model = torch_model.to(device=device, dtype=dtype)
    input_ids = tokenizer.encode(input_ids)
    input_ids = torch.tensor(input_ids).to(device=device).unsqueeze(0)
    config = torch_model.config

    attention_mask = torch.ones([1, config.max_position_embeddings]).to(device=device, dtype=dtype)
    # position_ids = torch.arange(0, config.max_position_embeddings, device='cuda').unsqueeze(0)
    make_past = lambda: torch.zeros([1, config.num_attention_heads, 0, config.hidden_size // config.num_attention_heads]).to(device=device, dtype=dtype)
    key_value_cache = [
        (make_past(), make_past()) for i in range(config.num_hidden_layers)
    ]
    outputs = []
    cur_len = input_ids.shape[-1]
    for _ in range(num_tokens):
        y = torch_model(input_ids, attention_mask=attention_mask[:, :cur_len], position_ids=None, past_key_values=key_value_cache, use_cache=True)
        logits = y['logits']
        new_ids = torch.argmax(logits, -1, keepdim=False)
        new_ids = new_ids[:, -1:]
        outputs.append(new_ids[0, -1].item())
        input_ids = new_ids
        key_value_cache = y['past_key_values']
        cur_len += 1
    
    return tokenizer.decode(outputs)

def print_eq(a, b):
    print((a - b.torch()).abs().max())

def get_torch_args(config, seq_len, prev_len):
    ids = torch.randint(0, 1102, [1, seq_len])
    attention_mask = None # torch.ones([1, seq_len + prev_len])
    position_ids = None # torch.arange(0, seq_len + prev_len).unsqueeze(0)
    make_past = lambda: torch.zeros([1, config.num_attention_heads, prev_len, config.hidden_size // config.num_attention_heads])
    past_keys_values = [(make_past(), make_past()) for _ in range(config.num_hidden_layers)]

    args = [ids, attention_mask, position_ids, past_keys_values]
    kwargs = dict(use_cache=True)
    return args, kwargs

def get_hidet_args(config, ids, prev_len):
    ids = hidet.from_torch(ids)
    position_ids = hidet.arange(0, config.max_position_embeddings).unsqueeze(0)
    make_past = lambda: hidet.zeros([1, config.num_attention_heads, prev_len, config.hidden_size // config.num_attention_heads])
    past_keys_values = [(make_past(), make_past()) for _ in range(config.num_hidden_layers)]

    args = [ids, position_ids, past_keys_values]
    kwargs = dict(use_cache=True)
    return args, kwargs

def get_hidet_symbolic_args(config, seq_len, prev_len):
    ids = hidet.symbol([1, seq_len], dtype=hidet.int32)
    position_ids = hidet.symbol([1, config.max_position_embeddings], dtype=hidet.int32)
    make_past = lambda: hidet.symbol([1, config.num_attention_heads, prev_len, config.hidden_size // config.num_attention_heads])
    past_keys_values = [(make_past(), make_past()) for _ in range(config.num_hidden_layers)]

    args = [ids, position_ids, past_keys_values]
    kwargs = dict(use_cache=True)
    return args, kwargs

def flatten(obj):
    total = []
    from typing import Iterable
    def flatten_(obj):
        if isinstance(obj, hidet.Tensor):
            total.append(obj)
        elif isinstance(obj, Iterable):
            for i in obj:
                flatten_(i)
        else:
            raise ValueError(f"unrecognized object with type {type(obj)}")
    flatten_(obj)
    return total


from transformers import LlamaTokenizer, LlamaForCausalLM as hfLm, LlamaConfig as hfConfig
import transformers.models.llama.modeling_llama as OrigImpl

config = hfConfig(hidden_size=512, intermediate_size=1024, num_attention_heads=8, num_hidden_layers=2)
hf_model = OrigImpl.LlamaModel(config)

x = torch.randint(0, 512, [1, 32])
ps_ids = torch.arange(0, 32).unsqueeze(0)
past_vals = [(torch.randn([1, 8, 32, 64]), torch.randn([1, 8, 32, 64])) for i in range(2)]
y = hf_model.forward(x, None, ps_ids, past_vals)

model = LlamaModel(config)
copy_weights(hf_model, model)
xh = hidet.from_torch(x)
ps_ids = hidet.from_torch(ps_ids)
past_vals = [(hidet.from_torch(ps[0]), hidet.from_torch(ps[1])) for ps in past_vals]
args = [xh, ps_ids]
yh = model.forward(*args, past_vals)

print_eq(y['last_hidden_state'], yh['last_hidden_state'])

sym = [hidet.symbol_like(u) for u in args]
sym.append([(hidet.symbol_like(u[0]), hidet.symbol_like(u[1])) for u in past_vals])

syh = model.forward(*sym)

graph = hidet.trace_from(syh['last_hidden_state'], flatten(sym))
cmodel = graph.build()
yh2 = cmodel(*args, *flatten(past_vals))

print_eq(y['last_hidden_state'], yh2)

# %%

model = hfLm(hfConfig(hidden_size=512, intermediate_size=1024, num_attention_heads=8, num_hidden_layers=2))
config = model.config

seq_len = 128
past_len = 12

args, kwargs = get_torch_args(config, seq_len, past_len)
y = model.forward(*args, **kwargs)

hmodel = convert_model(model, dtype=hidet.float32, device='cpu').cpu()

hargs, hkwargs = get_hidet_args(config, args[0], past_len)
y1 = hmodel.forward(*hargs, use_cache=False,)

print_eq(y.logits, y1['logits'])

sargs, skwargs = get_hidet_symbolic_args(config, seq_len, past_len)
y2 = hmodel.forward(*sargs, **skwargs)

flow_graph = hidet.trace_from(y2['logits'], flatten(sargs))
compiled = flow_graph.build()

y3 = compiled(*flatten(hargs))

print_eq(y.logits, y3)

# %%


# %%

print(compiled.weights)

# # %%

# # %%
# l1 = model.model.layers[0].self_attn
# l2 = getattr(hmodel.model.layers, '0').self_attn

# for i in ['attn_mask', 'position_ids', 'query_states', 'key_states', 'value_states', 'attn_weights', 'hidden_states1']:
#     print(i)
#     print_eq(getattr(l1, i), getattr(l2, i))

# # %%
# x = torch.rand([1, 128, 1024])
# l2.q_proj.weight = l2.q_proj.weight.transpose(0, 1)
# print_eq(l1.q_proj(x), l2.q_proj(hidet.from_torch(x)))
# %%
print(hidet.ops.argmax(y3, -1))
# %%
print(torch.argmax(y.logits, -1))
# %%


# # %%
# for i in range(len(y.attentions)):
#     print((y.attentions[i] - y2['attentions'][i].torch()).abs().max())

# # %%
# orig_weights = {n: p for n, p in model.named_parameters()}
# orig_weights.update({n: p for n, p in model.named_buffers()})

# copied_weights = {n: p for n, p in hmodel.named_parameters()}

# for n in orig_weights.keys():
#     assert n in copied_weights, f"{n} not present"
#     # print(n, orig_weights[n].shape, copied_weights[n].shape)
#     if list(orig_weights[n].shape) == list(copied_weights[n].shape):
#         print(n, (orig_weights[n] - copied_weights[n].torch()).abs().max())
#     else:
#         print("alt", n, (orig_weights[n]- copied_weights[n].torch().t()).abs().max())


# %%
from transformers import LlamaTokenizer, LlamaForCausalLM as hfLm, LlamaConfig as hfConfig
tok = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')

# config = hfConfig(hidden_size=1024, intermediate_size=2048, num_attention_heads=8, num_hidden_layers=2, )
model = hfLm.from_pretrained('decapoda-research/llama-7b-hf', torch_dtype=torch.float16)
# model = hfLm(hfConfig(hidden_size=1024, intermediate_size=2048, num_attention_heads=8, num_hidden_layers=2))

config = model.config


# generate_torch("In the beginning was the Word.", tok, model, 12, device='cpu')

# # 'The Word was with God, and the Word was God.'

# generate_torch("A robot may not injure a human being or, through inaction", tok, model, 55, device='cpu')
# # ', allow a human being to come to harm. A robot must obey orders given it by human beings except where such orders would conflict with the First Law. A robot must protect its own existence as long as such protection does not conflict with the First or Second Laws.'

model = convert_model(model)
flow_graph = build_flow_graph(model)
compiled = flow_graph.build()



new = generate('In the beginning was the Word.', compiled, tok, config, num_tokens=12)
print(new)

new = generate("A robot may not injure a human being or, through inaction", compiled, tok, config, num_tokens=55)
print(new)


# %%
