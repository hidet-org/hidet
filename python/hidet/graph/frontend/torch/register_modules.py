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
from hidet.graph.tensor import Tensor
from .interpreter import HidetModule, register_module
from . import register_functions as regs


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


@register_module(torch.nn.AdaptiveAvgPool2d)
class HidetAdaptiveAvgPool2d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.AdaptiveAvgPool2d)
        return regs.adaptive_avg_pool2d(x, self.mod.output_size)


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
        from hidet import ops

        self.transposed_weight = ops.transpose(self.param('weight'), [1, 0])

    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Linear)
        return regs.linear(x=x, weight=self.transposed_weight, bias=self.param('bias'))


@register_module(torch.nn.BatchNorm2d)
class HidetBatchNorm2d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.BatchNorm2d)
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


@register_module(torch.nn.Tanh)
class HidetTanh(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Tanh)
        return regs.tanh(x)


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
    def __call__(self, x=Tensor) -> Tensor:
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

# to implement identity
@register_module(torch.nn.Identity)
class HidetIdentity(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.Identity)
        return regs.identity(x)


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
            affine=self.mod.affine,
        )

# to implement multihead_attention 
@register_module(torch.nn.MultiheadAttention)
class HidetMultiheadAttention(torch.nn.MultiheadAttention):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.MultiheadAttention)
        return regs.multihead_attention(
            x=x,
            embed_dim=self.mod.embed_dim,
            num_heads=self.mod.num_heads,
            dropout=self.mod.dropout,
            bias=self.param('bias'),
            add_bias_attn=False,
            add_zero_attn=self.mod.add_zero_attn,
            kdim=self.mod.kdim,
            vdim=self.mod.vdim,
            batch_first=self.mod.batch_first,
        )

# to implement conv_transpose2d
@register_module(torch.nn.ConvTranspose2d)
class HidetConvTranspose2d(HidetModule):
    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(self.mod, torch.nn.ConvTranspose2d)
        return regs.conv_transpose2d(
            x=x,
            in_channels=self.mod.in_channels,
            out_channels=self.mod.out_channels,
            kernel_size=self.mod.kernel_size,
            stride=self.mod.stride,
            padding=self.mod.padding,
            output_padding=self.mod.output_padding,
            groups=self.mod.groups,
            bias=self.param("bias"),
            dilation=self.mod.dilation,
        )
