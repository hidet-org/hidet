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
import torch
from hidet.graph import ops
from hidet.graph.tensor import Tensor
from .interpreter import HidetModule, register_module
from . import register_functions as regs
from .dynamo_config import dynamo_config


@register_module(torch.nn.Conv1d)
class HidetConv1d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Conv1d)
        return regs.conv1d(
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
        return regs.conv1d_transpose(
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
        return regs.conv2d(
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
        return regs.conv2d_transpose(
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
        return regs.conv3d(
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
        return regs.conv3d_transpose(
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
        return regs.adaptive_avg_pool2d(x, self.mod.output_size)


@register_module(torch.nn.AdaptiveAvgPool3d)
class HidetAdaptiveAvgPool3d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.AdaptiveAvgPool3d)
        return regs.adaptive_avg_pool3d(x, self.mod.output_size)


@register_module(torch.nn.ReLU)
class HidetReLU(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.ReLU)
        return regs.relu(x, self.mod.inplace)


@register_module(torch.nn.MaxPool2d)
class HidetMaxPool2d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.MaxPool2d)
        return regs.max_pool2d(
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
        return regs.max_pool3d(
            x=x,
            kernel_size=self.mod.kernel_size,
            stride=self.mod.stride,
            padding=self.mod.padding,
            dilation=self.mod.dilation,
            ceil_mode=self.mod.ceil_mode,
            return_indices=self.mod.return_indices,
        )


@register_module(torch.nn.Linear)
class HidetLinear(HidetModule):
    def __init__(self, torch_module: torch.nn.Module):
        super().__init__(torch_module)
        steal = dynamo_config['steal_weights']
        self.transposed_weight = ops.transpose(self.param('weight', steal=steal), [1, 0])

    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Linear)
        return regs.linear(
            x=x, weight=self.transposed_weight, bias=self.param('bias', optional=True), weight_is_transposed=True
        )


@register_module(torch.nn.BatchNorm2d)
@register_module(torch.nn.BatchNorm3d)
class HidetBatchNorm2d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
        return regs.batch_norm(
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
        return regs.dropout(x, self.mod.p, self.mod.training, self.mod.inplace)


@register_module(torch.nn.LayerNorm)
class HidetLayerNorm(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.LayerNorm)
        return regs.layer_norm(
            x=x,
            normalized_shape=self.mod.normalized_shape,
            weight=self.param('weight'),
            bias=self.param('bias'),
            eps=self.mod.eps,
        )


@register_module(torch.nn.GroupNorm)
class HidetGroupNorm(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.GroupNorm)
        return regs.group_norm(
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
        return regs.tanh(x)


@register_module(torch.nn.Hardtanh)
class HidetHardtanh(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Hardtanh)
        return regs.hardtanh(x, self.mod.min_val, self.mod.max_val)


@register_module(torch.nn.Embedding)
class HidetEmbedding(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Embedding)
        return regs.embedding(
            x=x,
            weight=self.param('weight'),
            padding_idx=self.mod.padding_idx,
            max_norm=self.mod.max_norm,
            norm_type=self.mod.norm_type,
            scale_grad_by_freq=self.mod.scale_grad_by_freq,
            sparse=self.mod.sparse,
        )


@register_module(torch.nn.ReLU6)
class HidetReLU6(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.ReLU6)
        return regs.relu6(x, self.mod.inplace)


@register_module(torch.nn.Sigmoid)
class HidetSigmoid(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Sigmoid)
        return regs.sigmoid(x)


@register_module(torch.nn.Hardsigmoid)
class HidetHardsigmoid(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Hardsigmoid)
        return regs.hardsigmoid(x, self.mod.inplace)


@register_module(torch.nn.AvgPool2d)
class HidetAvgPool2d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.AvgPool2d)
        return regs.avg_pool2d(
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
        return regs.flatten(x, self.mod.start_dim, self.mod.end_dim)


@register_module(torch.nn.Hardswish)
class HidetHardswish(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Hardswish)
        return regs.hardswish(x, self.mod.inplace)


@register_module(torch.nn.GELU)
class HidetGELU(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.GELU)
        return regs.gelu(x, self.mod.approximate)


@register_module(torch.nn.SiLU)
class HidetSiLU(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.SiLU)
        return regs.silu(x, self.mod.inplace)


@register_module(torch.nn.Softmax)
class HidetSoftmax(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Softmax)
        return regs.softmax(x, self.mod.dim)


@register_module(torch.nn.Softmin)
class HidetSoftmin(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Softmin)
        return regs.softmin(x, self.mod.dim)


@register_module(torch.nn.Softplus)
class HidetSoftplus(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Softplus)
        return regs.softplus(x, self.mod.beta, self.mod.threshold)


@register_module(torch.nn.Softsign)
class HidetSoftsign(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Softsign)
        return regs.softsign(x)


@register_module(torch.nn.Softshrink)
class HidetSoftshrink(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Softshrink)
        return regs.softshrink(x, self.mod.lambd)


@register_module(torch.nn.Tanhshrink)
class HidetTanhshrink(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Tanhshrink)
        return regs.tanhshrink(x)


@register_module(torch.nn.Hardshrink)
class HidetHardshrink(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Hardshrink)
        return regs.hardshrink(x, self.mod.lambd)


@register_module(torch.nn.CELU)
class HidetCELU(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.CELU)
        return regs.celu(x, self.mod.alpha)


@register_module(torch.nn.LogSigmoid)
class HidetLogSigmoid(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.LogSigmoid)
        return regs.logsigmoid(x)


@register_module(torch.nn.Mish)
class HidetMish(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Mish)
        return regs.mish(x, self.mod.inplace)


@register_module(torch.nn.Identity)
class HidetIdentity(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Identity)
        return x


@register_module(torch.nn.Upsample)
class HidetUpsample(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Upsample)
        return regs.interpolate(
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
    ) -> Tensor:
        assert isinstance(self.mod, torch.nn.MultiheadAttention)
        supported = (
            self.mod._qkv_same_embed_dim
            and self.mod.bias_k is None
            and self.mod.bias_v is None
            and not self.mod.add_zero_attn
            and self.mod.batch_first
            and key_padding_mask is None
            and not need_weights
        )
        if not supported:
            raise NotImplementedError(
                "Hidet Multihead Attention currently only supports "
                "kdim=vdim=embed_dim, add_bias_kv=False, add_zero_attn=False, "
                "batch_first=True, forward(key_padding_mask=None, need_weights=False)."
            )

        # Input feed forward
        wq, wk, wv = ops.split(self.in_proj_weight_transposed, parts_or_sections=3, axis=1)
        query = ops.matmul(query, wq)
        key = ops.matmul(key, wk)
        value = ops.matmul(value, wv)
        if self.mod.in_proj_bias is not None:
            bq, bk, bv = ops.split(self.param('in_proj_bias'), parts_or_sections=3, axis=0)
            query = ops.add(query, bq)
            key = ops.add(key, bk)
            value = ops.add(value, bv)

        # Split heads
        split_head_dims = [query.shape[0], query.shape[1], self.mod.num_heads, query.shape[2] // self.mod.num_heads]
        query = ops.transpose(query.reshape(split_head_dims), [0, 2, 1, 3])
        key = ops.transpose(key.reshape(split_head_dims), [0, 2, 1, 3])
        value = ops.transpose(value.reshape(split_head_dims), [0, 2, 1, 3])

        # fmha
        out = regs.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=self.mod.dropout, is_causal=is_causal
        )

        # Output feed forward
        merge_head_dims = [out.shape[0], out.shape[2], self.mod.embed_dim]
        out = ops.transpose(out, [0, 2, 1, 3]).reshape(merge_head_dims)
        out = ops.matmul(out, self.out_proj_weight_transposed)
        if self.mod.out_proj.bias is not None:
            out = ops.add(out, self.param('out_proj.bias'))
        return out
