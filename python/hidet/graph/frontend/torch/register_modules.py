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
from __future__ import annotations

from typing import Tuple, Optional

import torch
from hidet.graph import ops
from hidet.graph.tensor import Tensor
from .registry import HidetModule, register_module
from .interpreter import warnings
from . import register_functions as reg_funcs, register_methods as reg_methods
from .dynamo_config import dynamo_config


@register_module(torch.nn.Conv1d)
class HidetConv1d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Conv1d)
        return reg_funcs.conv1d(
            x=x,
            weight=self.param('weight'),
            bias=self.param('bias', optional=True),
            stride=self.mod.stride,
            padding=self.mod.padding,
            dilation=self.mod.dilation,
            groups=self.mod.groups,
        )


@register_module(torch.nn.ConvTranspose1d)
class HidetConvTranspose1d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.ConvTranspose1d)
        return reg_funcs.conv1d_transpose(
            x=x,
            weight=self.param('weight'),
            bias=self.param('bias', optional=True),
            stride=self.mod.stride,
            padding=self.mod.padding,
            output_padding=self.mod.output_padding,
            groups=self.mod.groups,
            dilation=self.mod.dilation,
        )


@register_module(torch.nn.Conv2d)
class HidetConv2d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Conv2d)
        return reg_funcs.conv2d(
            x=x,
            weight=self.param('weight'),
            bias=self.param('bias', optional=True),
            stride=self.mod.stride,
            padding=self.mod.padding,
            dilation=self.mod.dilation,
            groups=self.mod.groups,
        )


@register_module(torch.nn.ConvTranspose2d)
class HidetConvTranspose2d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.ConvTranspose2d)
        return reg_funcs.conv2d_transpose(
            x=x,
            weight=self.param('weight'),
            bias=self.param('bias', optional=True),
            stride=self.mod.stride,
            padding=self.mod.padding,
            output_padding=self.mod.output_padding,
            groups=self.mod.groups,
            dilation=self.mod.dilation,
        )


@register_module(torch.nn.Conv3d)
class HidetConv3d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Conv3d)
        return reg_funcs.conv3d(
            x=x,
            weight=self.param('weight'),
            bias=self.param('bias', optional=True),
            stride=self.mod.stride,
            padding=self.mod.padding,
            dilation=self.mod.dilation,
            groups=self.mod.groups,
        )


@register_module(torch.nn.ConvTranspose3d)
class HidetConvTranspose3d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.ConvTranspose3d)
        return reg_funcs.conv3d_transpose(
            x=x,
            weight=self.param('weight'),
            bias=self.param('bias', optional=True),
            stride=self.mod.stride,
            padding=self.mod.padding,
            output_padding=self.mod.output_padding,
            groups=self.mod.groups,
            dilation=self.mod.dilation,
        )


@register_module(torch.nn.AdaptiveAvgPool2d)
class HidetAdaptiveAvgPool2d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.AdaptiveAvgPool2d)
        return reg_funcs.adaptive_avg_pool2d(x, self.mod.output_size)


@register_module(torch.nn.AdaptiveAvgPool3d)
class HidetAdaptiveAvgPool3d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.AdaptiveAvgPool3d)
        return reg_funcs.adaptive_avg_pool3d(x, self.mod.output_size)


@register_module(torch.nn.ReLU)
class HidetReLU(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.ReLU)
        return reg_funcs.relu(x, self.mod.inplace)


@register_module(torch.nn.LeakyReLU)
class HidetLeakyReLU(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.LeakyReLU)
        return reg_funcs.leaky_relu(x, self.mod.negative_slope, self.mod.inplace)


@register_module(torch.nn.MaxPool2d)
class HidetMaxPool2d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.MaxPool2d)
        return reg_funcs.max_pool2d(
            x=x,
            kernel_size=self.mod.kernel_size,
            stride=self.mod.stride,
            padding=self.mod.padding,
            dilation=self.mod.dilation,
            ceil_mode=self.mod.ceil_mode,
            return_indices=self.mod.return_indices,
        )


@register_module(torch.nn.MaxPool3d)
class HidetMaxPool3d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.MaxPool3d)
        return reg_funcs.max_pool3d(
            x=x,
            kernel_size=self.mod.kernel_size,
            stride=self.mod.stride,
            padding=self.mod.padding,
            dilation=self.mod.dilation,
            ceil_mode=self.mod.ceil_mode,
            return_indices=self.mod.return_indices,
        )


@register_module(torch.nn.ZeroPad2d)
class HidetZeroPad2d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.ZeroPad2d)
        return reg_funcs.torch_pad(x=x, pad=self.mod.padding, mode='constant', value=0.0)


@register_module(torch.nn.Linear)
class HidetLinear(HidetModule):
    def __init__(self, torch_module: torch.nn.Module):
        super().__init__(torch_module)
        steal = dynamo_config['steal_weights']
        self.transposed_weight = ops.transpose(self.param('weight', steal=steal), [1, 0])
        self.torch_params['weight'] = None
        self.hidet_params['weight'] = None
        torch.cuda.empty_cache()

    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Linear)
        return reg_funcs.linear(
            x=x, weight=self.transposed_weight, bias=self.param('bias', optional=True), weight_is_transposed=True
        )


@register_module(torch.nn.BatchNorm2d)
@register_module(torch.nn.BatchNorm3d)
class HidetBatchNorm2d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
        return reg_funcs.batch_norm(
            x=x,
            running_mean=self.param('running_mean'),
            running_var=self.param('running_var'),
            weight=self.param('weight'),
            bias=self.param('bias'),
            training=self.mod.training,
            momentum=self.mod.momentum,
            eps=self.mod.eps,
        )


@register_module(torch.nn.Dropout)
@register_module(torch.nn.Dropout1d)
@register_module(torch.nn.Dropout2d)
@register_module(torch.nn.Dropout3d)
class HidetDropout2d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d))
        return reg_funcs.dropout(x, self.mod.p, self.mod.training, self.mod.inplace)


@register_module(torch.nn.LayerNorm)
class HidetLayerNorm(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.LayerNorm)
        return reg_funcs.layer_norm(
            x=x,
            normalized_shape=self.mod.normalized_shape,
            weight=self.param('weight'),
            bias=self.param('bias', optional=True),
            eps=self.mod.eps,
        )


@register_module(torch.nn.GroupNorm)
class HidetGroupNorm(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.GroupNorm)
        return reg_funcs.group_norm(
            x=x,
            num_groups=self.mod.num_groups,
            num_channels=self.mod.num_channels,
            weight=self.param('weight'),
            bias=self.param('bias'),
            eps=self.mod.eps,
        )


@register_module(torch.nn.Tanh)
class HidetTanh(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Tanh)
        return reg_funcs.tanh(x)


@register_module(torch.nn.Hardtanh)
class HidetHardtanh(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Hardtanh)
        return reg_funcs.hardtanh(x, self.mod.min_val, self.mod.max_val)


@register_module(torch.nn.Embedding)
class HidetEmbedding(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Embedding)
        return reg_funcs.embedding(
            x=x,
            weight=self.param('weight'),
            padding_idx=self.mod.padding_idx,
            max_norm=self.mod.max_norm,
            norm_type=self.mod.norm_type,
            scale_grad_by_freq=self.mod.scale_grad_by_freq,
            sparse=self.mod.sparse,
        )


@register_module(torch.nn.EmbeddingBag)
class HidetEmbeddingBag(HidetModule):
    def __call__(
        self, input: Tensor, offsets: Optional[Tensor] = None, per_sample_weights: Optional[Tensor] = None
    ) -> Tensor:
        assert isinstance(self.mod, torch.nn.EmbeddingBag)
        return reg_funcs.torch_embedding_bag(
            input=input,
            weight=self.param('weight'),
            offsets=offsets,
            max_norm=self.mod.max_norm,
            norm_type=self.mod.norm_type,
            scale_grad_by_freq=self.mod.scale_grad_by_freq,
            mode=self.mod.mode,
            sparse=self.mod.sparse,
            per_sample_weights=per_sample_weights,
            include_last_offset=self.mod.include_last_offset,
            padding_idx=self.mod.padding_idx,
        )


@register_module(torch.nn.ReLU6)
class HidetReLU6(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.ReLU6)
        return reg_funcs.relu6(x, self.mod.inplace)


@register_module(torch.nn.Sigmoid)
class HidetSigmoid(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Sigmoid)
        return reg_funcs.sigmoid(x)


@register_module(torch.nn.Hardsigmoid)
class HidetHardsigmoid(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Hardsigmoid)
        return reg_funcs.hardsigmoid(x, self.mod.inplace)


@register_module(torch.nn.AvgPool2d)
class HidetAvgPool2d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.AvgPool2d)
        return reg_funcs.avg_pool2d(
            x=x,
            kernel_size=self.mod.kernel_size,
            stride=self.mod.stride,
            padding=self.mod.padding,
            ceil_mode=self.mod.ceil_mode,
            count_include_pad=self.mod.count_include_pad,
            divisor_override=self.mod.divisor_override,
        )


@register_module(torch.nn.Flatten)
class HidetFlatten(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Flatten)
        return reg_funcs.flatten(x, self.mod.start_dim, self.mod.end_dim)


@register_module(torch.nn.Hardswish)
class HidetHardswish(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Hardswish)
        return reg_funcs.hardswish(x, self.mod.inplace)


@register_module(torch.nn.GELU)
class HidetGELU(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.GELU)
        return reg_funcs.gelu(x, self.mod.approximate)


@register_module(torch.nn.SiLU)
class HidetSiLU(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.SiLU)
        return reg_funcs.silu(x, self.mod.inplace)


@register_module(torch.nn.Softmax)
class HidetSoftmax(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Softmax)
        return reg_funcs.softmax(x, self.mod.dim)


@register_module(torch.nn.Softmin)
class HidetSoftmin(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Softmin)
        return reg_funcs.softmin(x, self.mod.dim)


@register_module(torch.nn.Softplus)
class HidetSoftplus(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Softplus)
        return reg_funcs.softplus(x, self.mod.beta, self.mod.threshold)


@register_module(torch.nn.Softsign)
class HidetSoftsign(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Softsign)
        return reg_funcs.softsign(x)


@register_module(torch.nn.Softshrink)
class HidetSoftshrink(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Softshrink)
        return reg_funcs.softshrink(x, self.mod.lambd)


@register_module(torch.nn.Tanhshrink)
class HidetTanhshrink(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Tanhshrink)
        return reg_funcs.tanhshrink(x)


@register_module(torch.nn.Hardshrink)
class HidetHardshrink(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Hardshrink)
        return reg_funcs.hardshrink(x, self.mod.lambd)


@register_module(torch.nn.CELU)
class HidetCELU(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.CELU)
        return reg_funcs.celu(x, self.mod.alpha)


@register_module(torch.nn.LogSigmoid)
class HidetLogSigmoid(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.LogSigmoid)
        return reg_funcs.logsigmoid(x)


@register_module(torch.nn.Mish)
class HidetMish(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Mish)
        return reg_funcs.mish(x, self.mod.inplace)


@register_module(torch.nn.Identity)
class HidetIdentity(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Identity)
        return x


@register_module(torch.nn.Upsample)
class HidetUpsample(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Upsample)
        return reg_funcs.interpolate(
            x,
            size=self.mod.size,
            scale_factor=self.mod.scale_factor,
            mode=self.mod.mode,
            align_corners=self.mod.align_corners,
            recompute_scale_factor=self.mod.recompute_scale_factor,
        )


@register_module(torch.nn.MultiheadAttention)
class HidetMultiheadAttention(HidetModule):
    def __init__(self, torch_module: torch.nn.Module):
        super().__init__(torch_module)
        steal = dynamo_config['steal_weights']
        self.in_proj_weight_transposed = ops.transpose(self.param('in_proj_weight', steal=steal), [1, 0])
        self.out_proj_weight_transposed = ops.transpose(self.param('out_proj.weight', steal=steal), [1, 0])
        self.torch_params['in_proj_weight'] = None
        self.torch_params['out_proj.weight'] = None
        self.hidet_params['in_proj_weight'] = None
        self.hidet_params['out_proj.weight'] = None

        self.num_heads = self.mod.num_heads
        self.head_dim = self.mod.head_dim
        torch.cuda.empty_cache()

    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        assert isinstance(self.mod, torch.nn.MultiheadAttention)
        # pylint: disable=protected-access
        supported = (
            self.mod._qkv_same_embed_dim
            and self.mod.bias_k is None
            and self.mod.bias_v is None
            and not self.mod.add_zero_attn
            and key_padding_mask is None
        )
        # pylint: enable=protected-access
        if not supported:
            raise NotImplementedError(
                f"""
                HidetMultiheadAttention got: kdim={self.mod.kdim}, vdim={self.mod.vdim}, embed_dim={self.mod.embed_dim},
                self.mod.bias_k = {self.mod.bias_k}, self.mod.bias_v = {self.mod.bias_v},
                add_zero_attn={self.mod.add_zero_attn},
                batch_first={self.mod.batch_first}, key_padding_mask={key_padding_mask},
                need_weights={need_weights}, average_attn_weights={average_attn_weights}, is_causal={is_causal}.
                currently only supports kdim=vdim=embed_dim, add_bias_kv=False, add_zero_attn=False,
                batch_first=True, forward(key_padding_mask=None, need_weights=False).
                """
            )

        if need_weights:
            warnings.warn_once(
                """HidetMultiheadAttention: had need_weights=True, but
            currently need_weights will be treated as False, as it forces a much slower computation of SDPA,
            and can likely be turned off in most production scenarios."""
            )

        wq, wk, wv = ops.split(self.in_proj_weight_transposed, parts_or_sections=3, axis=1)
        query = ops.matmul(query, wq)
        key = ops.matmul(key, wk)
        value = ops.matmul(value, wv)
        if self.mod.in_proj_bias is not None:
            bq, bk, bv = ops.split(self.param('in_proj_bias'), parts_or_sections=3, axis=0)
            query = ops.add(query, bq)
            key = ops.add(key, bk)
            value = ops.add(value, bv)

        assert (
            self.mod.bias_k is None and self.mod.bias_v is None
        ), "HidetMultiheadAttention currently does not support bias_k and bias_v."

        if not self.mod.batch_first:
            return self._forward_not_batch_first(query, key, value, attn_mask, is_causal)

        else:
            return self._forward_batch_first(query, key, value, attn_mask, is_causal)

    def _forward_not_batch_first(self, query: Tensor, key: Tensor, value: Tensor, attn_mask=True, is_causal=False):
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        # Preparing attention mask
        if attn_mask is not None:
            # ensure attn_mask is 3D
            if len(attn_mask.shape) == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but it should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif len(attn_mask.shape) == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.size}, but it should be {correct_3d_size}."
                    )
            else:
                raise RuntimeError(f"attn_mask's dimensionality is {len(attn_mask.shape)}, but it should be 2 or 3.")

            if attn_mask.shape[0] == 1 and len(attn_mask.shape) == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = reg_methods.tensor_view(
                    attn_mask, bsz, self.num_heads, attn_mask.size / (bsz * self.num_heads * src_len), src_len
                )

        q = reg_methods.tensor_view(query, tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = reg_methods.tensor_view(key, src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = reg_methods.tensor_view(value, value.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # the new source seq length
        src_len = k.shape[1]

        q = reg_methods.tensor_view(q, bsz, self.num_heads, tgt_len, self.head_dim)
        k = reg_methods.tensor_view(k, bsz, self.num_heads, src_len, self.head_dim)
        v = reg_methods.tensor_view(v, bsz, self.num_heads, src_len, self.head_dim)

        attn_output = reg_funcs.scaled_dot_product_attention(q, k, v, attn_mask, is_causal)
        attn_output = reg_funcs.permute(attn_output, 2, 0, 1, 3)
        attn_output = reg_methods.tensor_view(attn_output, bsz * tgt_len, embed_dim)

        attn_output = ops.matmul(attn_output, self.out_proj_weight_transposed)
        if self.mod.out_proj.bias is not None:
            attn_output = ops.add(attn_output, self.param('out_proj.bias'))
        attn_output = reg_methods.tensor_view(attn_output, tgt_len, bsz, attn_output.shape[1])
        return attn_output, None

    def _forward_batch_first(self, query: Tensor, key: Tensor, value: Tensor, attn_mask=None, is_causal=False):
        split_head_dims = [query.shape[0], query.shape[1], self.num_heads, query.shape[2] // self.num_heads]
        query = ops.transpose(query.reshape(split_head_dims), [0, 2, 1, 3])
        key = ops.transpose(key.reshape(split_head_dims), [0, 2, 1, 3])
        value = ops.transpose(value.reshape(split_head_dims), [0, 2, 1, 3])

        # fmha
        out = reg_funcs.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=self.mod.dropout, is_causal=is_causal
        )

        # Output feed forward
        merge_head_dims = [out.shape[0], out.shape[2], self.mod.embed_dim]
        out = ops.transpose(out, [0, 2, 1, 3]).reshape(merge_head_dims)
        out = ops.matmul(out, self.out_proj_weight_transposed)
        if self.mod.out_proj.bias is not None:
            out = ops.add(out, self.param('out_proj.bias'))

        return out, None


@register_module(torch.nn.TransformerEncoderLayer)
class HidetTransformerEncoderLayer(HidetModule):
    def __init__(self, torch_module: torch.nn.Module):
        super().__init__(torch_module)

        self.self_attn = HidetMultiheadAttention(self.mod.self_attn)

        self.linear1 = HidetLinear(self.mod.linear1)
        self.dropout = HidetDropout2d(self.mod.dropout)

        self.linear2 = HidetLinear(self.mod.linear2)

        self.norm_first = self.mod.norm_first
        self.norm1 = HidetLayerNorm(self.mod.norm1)
        self.norm2 = HidetLayerNorm(self.mod.norm2)

        self.dropout1 = HidetDropout2d(self.mod.dropout1)
        self.dropout2 = HidetDropout2d(self.mod.dropout2)

        from hidet.graph.frontend.torch.interpreter import Registry

        mod_activation = self.mod.activation
        if mod_activation.__class__ in Registry.registered_modules:
            self.activation = Registry.registered_modules[mod_activation.__class__](mod_activation)
        elif mod_activation in Registry.registered_functions:
            self.activation = Registry.registered_functions[mod_activation]
        else:
            import torchmultimodal

            # torchmultimodal.modules.layers.activation.SiLU is encountered
            # while compiling the model torch_multimodal_clip from TorchBench
            if isinstance(mod_activation, torchmultimodal.modules.layers.activation.SiLU):
                self.activation = lambda x: reg_funcs.sigmoid(1.702 * x) * x
            else:
                raise NotImplementedError(
                    f"HidetTransformerEncoder: activation function {mod_activation} is not supported."
                )

    def supported(self):
        # pylint: disable=protected-access
        return (
            self.mod.self_attn._qkv_same_embed_dim
            and self.mod.self_attn.bias_k is None
            and self.mod.self_attn.bias_v is None
            and not self.mod.self_attn.add_zero_attn
        )
        # pylint: enable=protected-access

    def print_info(self):
        # pylint: disable=protected-access
        info_str = f"""
        self_attn._qkv_same_embed_dim = {self.mod.self_attn._qkv_same_embed_dim},\n
        self_attn.bias_k = {self.mod.self_attn.bias_k},\n
        self_attn.bias_v = {self.mod.self_attn.bias_v},\n
        self_attn.add_zero_attn = {self.mod.self_attn.add_zero_attn},\n
        self_attn.batch_first = {self.mod.self_attn.batch_first},\n
        """
        # pylint: enable=protected-access
        return info_str

    def __call__(self, src: Tensor, src_mask=None, src_key_padding_mask=None, is_causal: bool = False) -> Tensor:
        assert isinstance(self.mod, torch.nn.TransformerEncoderLayer)

        if src_key_padding_mask is not None:
            raise NotImplementedError(
                f"""HidetTransformerEncoderLayer currently only supports src_key_padding_mask=None,
                but got src_key_padding_mask={src_key_padding_mask}."""
            )

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask=src_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask=src_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x: Tensor, attn_mask: Tensor, is_causal: bool) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


@register_module(torch.nn.TransformerEncoder)
class HidetTransformerEncoder(HidetModule):
    def __init__(self, torch_module: torch.nn.Module):
        super().__init__(torch_module)
        self.layers = [HidetTransformerEncoderLayer(layer) for layer in self.mod.layers]
        self.num_layers = self.mod.num_layers
        assert self.num_layers == len(self.layers)
        self.norm = HidetLayerNorm(self.mod.norm) if self.mod.norm is not None else None
        self.mask_check = self.mod.mask_check

    def __call__(self, src: Tensor, mask=None, src_key_padding_mask=None, is_causal=None) -> Tensor:
        self_first_layer = self.layers[0]
        if not isinstance(self_first_layer, HidetTransformerEncoderLayer):
            raise NotImplementedError(
                f"""Hidet Transformer Encoder currently only HidetTransformerEncoderLayer,
        but got {self_first_layer.__class__}."""
            )

        if not self_first_layer.supported():
            raise NotImplementedError(
                f"""Hidet Transformer Encoder currently only supports self_attn with
                kdim=vdim=embed_dim, add_bias_kv=False, add_zero_attn=False,
                batch_first=True, forward(src_key_padding_mask=None).
                \n But we got:
                \n {self_first_layer.print_info()}.
                """
            )

        if mask is not None and mask.device != src.device:
            mask = ops.transfer(mask, src.device)

        if not (src_key_padding_mask is None and all(layer.supported() for layer in self.layers)):
            raise NotImplementedError(
                f"""Hidet Transformer Encoder currently only supports self_attn with
                kdim=vdim=embed_dim, add_bias_kv=False, add_zero_attn=False,
                batch_first=True, forward(src_key_padding_mask=None),
                but we got src_key_padding_mask={src_key_padding_mask} and is_causal={is_causal},
                """
            )

        output = src

        batch_first = self.layers[0].mod.self_attn.batch_first
        src_size = len(src.shape)
        if src_size == 2:
            seq_len = src.shape[0]
        else:
            seq_len = src.shape[1 if batch_first else 0]
        is_causal = self._detect_is_causal_mask(mask, is_causal, seq_len)

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
        if self.norm is not None:
            output = self.norm(output)
        return output

    def _detect_is_causal_mask(self, mask: Optional[Tensor], is_causal, sz: Optional[int] = None):
        make_causal = is_causal is True
        if is_causal is None and mask is not None:
            sz = mask.shape[-2] if sz is None else sz
            causal_mask = ops.triu(ops.full((sz, sz), float('-inf'), dtype=mask.dtype, device=mask.device), diagonal=1)
            if mask.shape == causal_mask.shape:
                make_causal = bool(ops.all(ops.equal(mask, causal_mask)))
            else:
                make_causal = False
        return make_causal
