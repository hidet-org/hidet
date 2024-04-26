import math
from typing import Tuple, List, Optional

import hidet

from hidet import Tensor

from hidet.graph import nn, ops
from hidet.runtime import CompiledModule

GemmaConfig = "transformers.GemmaConfig"


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

    def forward(self, x: Tensor, position_ids: Tensor):
        """
        Compute the cosine and sine rotary embedding terms.

        Parameters
        ----------
        x: Tensor
            The input to apply the rotary embedding to later on (shape: [bs, seq_len, head_dim])

        position_ids: Tensor
            The corresponding positions of the given input (shape: [bs, seq_len])

        Returns
        -------
        cos: Tensor
            The cosine term (shape: [bs, seq_len, head_dim])

        sin: Tuple[Tensor, Tensor]
            The sine term (shape: [bs, seq_len, head_dim])
        """
        inv_freq = hidet.asarray(1.0, dtype=hidet.float32, device=x.device) / (
            hidet.asarray(self.base, dtype=hidet.float32, device=x.device)
            ** (hidet.arange(0, self.dim, 2, dtype=hidet.int64, device=x.device).float() / self.dim)
        )
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = ops.concat([freqs, freqs], axis=-1)
        cos = ops.cos(emb)
        sin = ops.sin(emb)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta
        )

    def forward(self, hidden_states: Tensor, position_ids: Tensor, past_kv: Tuple[Tensor, Tensor]):
        """
        Apply scaled dot-product attention to the input.

        Parameters
        ----------
        hidden_states: Tensor
            The input (shape: [bs, seq_len, hidden_size])

        position_ids: Tensor
            The corresponding positions of the given hidden states (shape: [bs, seq_len])

        past_kv: Tuple[Tensor, Tensor]
            A pair of tensors describing the keys and values from previous iterations of this attention block.
            (shape for key/value: [bs, num_key_value_heads, past_seq_len, head_dim])

        Returns
        -------
        output: Tensor
            The output of the attention block (shape: [bs, seq_len, hidden_size])

        present_kv: Tuple[Tensor, Tensor]
            Copy of past_kv updated with the current iteration's keys and values
        """
        bs, seq_len, _ = hidden_states.shape

        # Compute queries/keys/values
        query_states: Tensor = self.q_proj(hidden_states)
        query_states = query_states.reshape([bs, seq_len, self.num_heads, self.head_dim]).transpose(1, 2)
        key_states: Tensor = self.k_proj(hidden_states)
        key_states = key_states.reshape([bs, seq_len, self.num_key_value_heads, self.head_dim]).transpose(1, 2)
        value_states: Tensor = self.v_proj(hidden_states)
        value_states = value_states.reshape([bs, seq_len, self.num_key_value_heads, self.head_dim]).transpose(1, 2)

        # Apply rotary embedding
        def rotate_half(x: Tensor):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return ops.concat([-x2, x1], axis=-1)

        def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
            cos = cos[:, None, :, :]
            sin = sin[:, None, :, :]
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            return q_embed, k_embed

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Use / update KV-cache
        past_k, past_v = past_kv
        key_states = ops.concat([past_k, key_states], axis=2)
        value_states = ops.concat([past_v, value_states], axis=2)
        present_kv = (key_states, value_states)

        # Repeat for multi-query attention
        def repeat_kv(x: Tensor) -> Tensor:
            bs, num_key_value_heads, total_seq_len, head_dim = x.shape
            n = self.num_key_value_groups
            if n == 1:
                return x
            x = x[:, :, None, :, :].expand(bs, num_key_value_heads, n, total_seq_len, head_dim)
            return x.reshape([bs, num_key_value_heads * n, total_seq_len, head_dim])

        key_states = repeat_kv(key_states)
        value_states = repeat_kv(value_states)

        # Mask and compute attention
        causal_mask = (
            1.0 - ops.tri(seq_len, seq_len, dtype=query_states.dtype, device=query_states.device)
        ) * query_states.dtype.min_value
        attn_logits = ops.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # attn_logits: [bs, num_heads, seq_len, (past_seq_len + seq_len)]
        # causal_mask: [seq_len, seq_len]
        # In eager mode this works fine, but when the shape is symbolic we need to broadcast explicitly.
        attn_logits += ops.broadcast(causal_mask, attn_logits.shape)
        attn_output = ops.matmul(ops.softmax(attn_logits, axis=-1), value_states).transpose(1, 2)

        return self.o_proj(attn_output.reshape([bs, seq_len, -1])), present_kv


class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: Tensor):
        """
        Apply the Gemma MLP to the input.

        Parameters
        ----------
        x: Tensor
            The input (shape: [bs, seq_len, hidden_size])

        Returns
        -------
        output: Tensor
            The output (shape: [bs, seq_len, hidden_size])
        """
        return self.down_proj(ops.gelu(self.gate_proj(x), approximate=True) * self.up_proj(x))


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = hidet.ones([dim])

    def _norm(self, x: Tensor):
        y = ops.mean(ops.pow(x, hidet.asarray(2, dtype=x.dtype, device=x.device)), -1, keep_dim=True) + self.eps
        return x * ops.rsqrt(y)

    def forward(self, x: Tensor):
        """
        Apply RMS norm to the input.

        Parameters
        ----------
        x: Tensor
            The input (shape: [bs, seq_len, hidden_size])

        Returns
        -------
        output: Tensor
            The normalized input
        """
        output = self._norm(x.float()) * (1.0 + self.weight.float())
        return output.astype(x.dtype)


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: Tensor, position_ids: Tensor, past_kv: Tuple[Tensor, Tensor]):
        """
        Pass the input hidden states through the decoder layer.

        Parameters
        ----------
        hidden_states: Tensor
            The hidden states (shape: [bs, seq_len, hidden_size])

        position_ids: Tensor
            The corresponding positions of the given hidden states (shape: [bs, seq_len])

        past_kv: Tuple[Tensor, Tensor]
            A pair of tensors describing the keys and values from previous iterations of this decoder layer.
            (shape for key/value: [bs, num_key_value_heads, past_seq_len, head_dim])

        Returns
        -------
        hidden_states: Tensor
            The decoder layer output

        present_kv: Tuple[Tensor, Tensor]
            Copy of past_kv updated with the current iteration's keys and values
        """
        # Input layernorm, self-attention
        h = self.input_layernorm(hidden_states)
        delta, present_kv = self.self_attn(h, position_ids, past_kv)
        hidden_states += delta

        # Post-attention layernorm, MLP
        h = self.post_attention_layernorm(hidden_states)
        delta = self.mlp(h)
        hidden_states += delta

        return hidden_states, present_kv


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([GemmaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: Tensor, position_ids: Tensor, past_kvs: List[Tuple[Tensor, Tensor]]):
        """
        Run the Gemma model.

        Parameters
        ----------
        input_ids: Tensor
            The IDs of the prompt (shape: [bs, seq_len])

        position_ids: Tensor
            The corresponding positions of the given input IDs (shape: [bs, seq_len])

        past_kvs: List[Tuple[Tensor, Tensor]]
            A list of tensor pairs, corresponding to the previous keys and values of each decoder layer,
            respectively. (shape for each key/value: [bs, num_key_value_heads, past_seq_len, head_dim])

        Returns
        -------
        hidden_states: Tensor
            The model output.

        present_kvs: List[Tuple[Tensor, Tensor]]
            Copy of past_kvs updated with the current iteration's keys and values.
        """
        hidden_states = self.embed_tokens(input_ids)
        normalizer = hidet.asarray(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        # Run through each decoder layer, update
        present_kvs = []
        for decoder_layer, past_kv in zip(self.layers, past_kvs):
            decoder_layer: GemmaDecoderLayer
            hidden_states, present_kv = decoder_layer(hidden_states, position_ids=position_ids, past_kv=past_kv)
            present_kvs.append(present_kv)

        hidden_states = self.norm(hidden_states)

        return hidden_states, present_kvs


class GemmaForCausalLM(nn.Module):
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoConfig

        super().__init__()
        self.config: GemmaConfig = AutoConfig.from_pretrained("google/gemma-2b")
        self.model = GemmaModel(config=self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self._compiled: Optional[CompiledModule] = None

        # Copy weights
        target = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
        weights = {k: hidet.from_torch(v) for k, v in target.state_dict().items()}
        self.load_state_dict(weights)

    def forward(self, input_ids: Tensor, position_ids: Tensor, past_kvs: List[Tuple[Tensor, Tensor]]):
        """
        Run the Gemma model.

        Parameters
        ----------
        input_ids: Tensor
            The IDs of the prompt (shape: [bs, seq_len])

        position_ids: Tensor
            The corresponding positions of the given input IDs (shape: [bs, seq_len])

        past_kvs: List[Tuple[Tensor, Tensor]]
            A list of tensor pairs, corresponding to the previous keys and values of each decoder layer,
            respectively (shape for each key/value: [bs, num_key_value_heads, past_seq_len, head_dim])

        Returns
        -------
        logits: Tensor
            The model output (shape: [bs, seq_len, vocab_size])

        present_kvs: List[Tuple[Tensor, Tensor]]
            Copy of past_kvs updated with the current iteration's keys and values
        """
        outputs = self.model(input_ids=input_ids, position_ids=position_ids, past_kvs=past_kvs)
        hidden_states, present_kvs = outputs[0], outputs[1]
        logits = self.lm_head(hidden_states)
        return logits, present_kvs

    def generate(self, input_ids: Tensor, *, num_tokens: int = 20, device: str = "cuda"):
        """
        Generate tokens by greedily selecting the argmax of the returned logits from the model repeatedly.

        Parameters
        ----------
        input_ids: Tensor
            The IDs of the prompt (shape: [bs, seq_len])

        num_tokens: int
            The number of tokens to generate (including input_ids)

        device: str
            The device to run the model on (default: "cuda")

        Returns
        -------
        output_ids: Tensor
            The output IDs (shape: [bs, min(num_tokens, seq_len)])
        """
        bs, seq_len = input_ids.shape

        # The past keys and values are pairs of tensors, each with shape
        # [bs, num_key_value_heads, past_seq_len, head_dim]
        past_kvs = []
        for _ in range(self.config.num_hidden_layers):
            shape = [bs, self.config.num_key_value_heads, 0, self.config.head_dim]
            past_kvs.append((hidet.zeros(shape, device=device), hidet.zeros(shape, device=device)))

        prefilling = True
        output_ids = input_ids[:]
        while output_ids.shape[-1] < num_tokens:
            if prefilling:
                position_ids = hidet.arange(seq_len, device=device)
                position_ids = position_ids.unsqueeze(0).expand(bs, -1)
                prefilling = False
            else:
                _, past_seq_len = output_ids.shape
                position_ids = hidet.arange(past_seq_len - 1, past_seq_len, device=device)
                position_ids = position_ids.unsqueeze(0).expand(bs, -1)

            # Get logits, new KV cache
            if not self._compiled:
                output = self(input_ids, position_ids, past_kvs)
                logits, present_kvs = output
            else:
                input_ids = input_ids.astype(hidet.int32)
                position_ids = position_ids.astype(hidet.int32)
                args = [input_ids, position_ids]
                for past_kv in past_kvs:
                    args.extend(past_kv)
                output = self._compiled(*args)
                logits = output[0]
                present_kvs = [(output[2 * i + 1], output[2 * i + 2]) for i in range(self.config.num_hidden_layers)]
            past_kvs = present_kvs

            # Greedily take token with max logit, use it as the next input
            token = ops.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            input_ids = token
            output_ids = ops.concat([output_ids, token], axis=-1)

        return output_ids

    def build(self, *, device: str = "cuda"):
        """
        Compute the flow graph and build it.

        Parameters
        ----------
        device: str
            The device the compiled model runs on (default: "cuda")
        """
        input_ids = hidet.symbol(["bs", "seq_len"], dtype=hidet.int32, device=device)
        position_ids = hidet.symbol(["bs", "seq_len"], dtype=hidet.int32, device=device)
        past_kvs = []
        for _ in range(self.config.num_hidden_layers):
            shape = ["bs", self.config.num_key_value_heads, "past_seq_len", self.config.head_dim]
            keys = hidet.symbol(shape, device=device, dtype=hidet.float32)
            values = hidet.symbol(shape, device=device, dtype=hidet.float32)
            past_kvs.append((keys, values))

        inputs = [input_ids, position_ids]
        for past_kv in past_kvs:
            inputs.extend(past_kv)

        self._compiled = None
        logits, present_kvs = self(input_ids, position_ids, past_kvs)
        outputs = [logits]
        for present_kv in present_kvs:
            outputs.extend(present_kv)

        graph = hidet.trace_from(outputs, inputs)
        graph = hidet.graph.optimize(graph)
        self._compiled = graph.build()
        return self
