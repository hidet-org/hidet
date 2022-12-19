from __future__ import annotations
import torch
from hidet.graph.tensor import Tensor
from .interpreter import HidetModule, register_module, warnings
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


@register_module(torch.nn.Linear)
class HidetLinear(HidetModule):
    def __init__(self, torch_module: torch.nn.Module):
        super().__init__(torch_module)
        self.transposed_weight = self.param('weight').transpose([1, 0])

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
        if self.mod.training:
            warnings.warn_once(
                'hidet: Dropout/1D/2D/3D in training mode is not supported. Treating as in inference mode.'
            )
        return x


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
            sparse=self.mod.sparse
        )
