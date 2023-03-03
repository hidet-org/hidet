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
"""
Import onnx model to hidet.

Please refers to https://github.com/onnx/onnx/blob/main/docs/Operators.md for operator definition.
Please refers to https://github.com/onnx/onnx/blob/main/onnx/onnx.proto for proto structure of onnx format.
"""
# pylint: disable=unused-argument
from typing import List, Union, Optional, Dict, Callable, Type, Sequence, Set
from collections import defaultdict
import warnings
import os
import logging
import numpy as np
import onnx
import onnx.numpy_helper
import onnx.external_data_helper
import hidet
from hidet.graph.modules import nn
from hidet.graph import ops
from hidet.graph.tensor import Tensor, from_numpy, randn
from . import utils

log = logging.getLogger(__name__)


class OnnxOperator:
    def __init__(self, node, op_sets: List[int]):
        """
        Parameters
        ----------
        node: onnx.NodeProto
        """
        self.node: onnx.NodeProto = node
        self.op_sets: List[int] = op_sets
        self.input_names: List[str] = [name for name in node.input]
        self.output_names: List[str] = [name for name in node.output]
        self.attrs = {}
        for attr in node.attribute:
            if attr.type == 1:  # float
                v = attr.f
            elif attr.type == 2:  # int
                v = attr.i
            elif attr.type == 3:  # string
                v = attr.s.decode('utf-8')
            elif attr.type == 4:  # tensor
                v = from_numpy(onnx.numpy_helper.to_array(tensor=attr.t)).cuda()
            elif attr.type == 5:  # graph
                v = attr.g
            elif attr.type == 6:  # floats
                v = list(attr.floats)
            elif attr.type == 7:  # ints
                v = list(attr.ints)
            elif attr.type == 8:  # strings
                v = [s.decode('utf-8') for s in attr.strings]
            else:
                raise ValueError('Can not recognize type id {} of attribute {}'.format(attr.type, attr.name))
            self.attrs[attr.name] = v

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        opset = self.resolve_opset(self.op_sets)
        run_func: Callable[[List[Tensor]], List[Tensor]] = getattr(self, 'run_v{}'.format(opset))
        outs = run_func(inputs)
        return outs

    def resolve_opset(self, op_sets: List[int]) -> int:
        for op_set in op_sets:
            try_op_set = op_set
            while try_op_set >= 1:
                if self.implemented(try_op_set):
                    return try_op_set
                try_op_set -= 1
        raise NotImplementedError(
            'Can not resolve opset for operator {} given opsets {}.'.format(self.node.op_type, op_sets)
        )

    def implemented(self, opset: int):
        func_name = 'run_v{}'.format(opset)
        this_func = getattr(self, func_name)
        base_func = getattr(OnnxOperator, func_name)
        return this_func.__func__ is not base_func

    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        return NotImplemented

    def run_v2(self, inputs: List[Tensor]) -> List[Tensor]:
        return NotImplemented

    def run_v3(self, inputs: List[Tensor]) -> List[Tensor]:
        return NotImplemented

    def run_v4(self, inputs: List[Tensor]) -> List[Tensor]:
        return NotImplemented

    def run_v5(self, inputs: List[Tensor]) -> List[Tensor]:
        return NotImplemented

    def run_v6(self, inputs: List[Tensor]) -> List[Tensor]:
        return NotImplemented

    def run_v7(self, inputs: List[Tensor]) -> List[Tensor]:
        return NotImplemented

    def run_v8(self, inputs: List[Tensor]) -> List[Tensor]:
        return NotImplemented

    def run_v9(self, inputs: List[Tensor]) -> List[Tensor]:
        return NotImplemented

    def run_v10(self, inputs: List[Tensor]) -> List[Tensor]:
        return NotImplemented

    def run_v11(self, inputs: List[Tensor]) -> List[Tensor]:
        return NotImplemented

    def run_v12(self, inputs: List[Tensor]) -> List[Tensor]:
        return NotImplemented

    def run_v13(self, inputs: List[Tensor]) -> List[Tensor]:
        return NotImplemented

    def run_v14(self, inputs: List[Tensor]) -> List[Tensor]:
        return NotImplemented

    def run_v15(self, inputs: List[Tensor]) -> List[Tensor]:
        return NotImplemented

    def run_v16(self, inputs: List[Tensor]) -> List[Tensor]:
        return NotImplemented

    def run_v17(self, inputs: List[Tensor]) -> List[Tensor]:
        return NotImplemented

    def run_v18(self, inputs: List[Tensor]) -> List[Tensor]:
        return NotImplemented

    @staticmethod
    def tensor2list(tensor: Tensor) -> Union[List, int, float]:
        ret = tensor.cpu().numpy().tolist()
        assert isinstance(ret, (list, int, float))
        return ret

    @staticmethod
    def tensor2scalar(tensor: Tensor) -> Union[int, float]:
        value = OnnxOperator.tensor2list(tensor)
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return value[0]
            else:
                raise ValueError('Expect a scalar, got {}'.format(value))
        else:
            assert isinstance(value, (int, float))
            return value

    @staticmethod
    def optional_inputs(inputs: List[Tensor], requires: List[bool]) -> List[Union[Tensor, None]]:
        diff = len(requires) - len(inputs)
        assert diff >= 0, 'Onnx get {} inputs but expect at most {}.'.format(len(inputs), len(requires))
        ret: List[Union[Tensor, None]] = []
        ret += inputs
        ret += [None for _ in range(diff)]
        for i, (t, r) in enumerate(zip(ret, requires)):
            if t is None and r:
                raise ValueError('The {}th input is required.'.format(i))
        return ret


dispatch_table: Dict[str, Type[OnnxOperator]] = {}


def register_onnx_operator(cls: Type[OnnxOperator]):
    if not issubclass(cls, OnnxOperator):
        raise ValueError('Can only register a sub-class of OnnxOperator as an onnx operator.')
    cls_name = cls.__name__
    if not cls_name.startswith('Onnx'):
        raise ValueError(
            'Please name the class as OnnxOPNAME such as OnnxConv and OnnxAdd,'
            ' where OPNAME is the same as the operator name used by ONNX. Got {}'.format(cls_name)
        )
    dispatch_table[cls_name[4:]] = cls
    return cls


@register_onnx_operator
class OnnxConv(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        groups = self.attrs.get('group', 1)
        if len(inputs) == 2:
            x, w = inputs
            bias = None
        else:
            x, w, bias = inputs
        if len(x.shape) == 4:
            dilations = self.attrs.get('dilations', [1, 1])
            padding = self.attrs.get('pads', [0, 0, 0, 0])
            strides = self.attrs.get('strides', [1, 1])
            x = ops.pad(x, ops.utils.normalize_padding(padding))
            output = ops.conv2d(x, w, stride=strides, dilations=dilations, groups=groups)
            if bias is not None:
                bias = ops.unsqueeze(bias, [0, 2, 3])
                output = output + bias
        elif len(x.shape) == 5:
            dilations = self.attrs.get('dilations', [1, 1, 1])
            padding = self.attrs.get('pads', [0, 0, 0, 0, 0, 0])
            strides = self.attrs.get('strides', [1, 1, 1])
            x = ops.pad(x, ops.utils.normalize_padding(padding, dim=3))
            output = ops.conv3d(x, w, stride=strides, dilations=dilations, groups=groups)
            if bias is not None:
                bias = ops.unsqueeze(bias, [0, 2, 3, 4])
                output = output + bias
        else:
            raise NotImplementedError('Currently only support 2D and 3D convolution, got x {}.'.format(x.shape))
        return [output]

    def run_v11(self, inputs: List[Tensor]) -> List[Tensor]:
        return self.run_v1(inputs)


@register_onnx_operator
class OnnxBatchNormalization(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        epsilon: float = self.attrs.get('epsilon', 1e-5)
        # for inference, we can ignore this momentum attribute
        momentum: float = self.attrs.get('momentum', 0.9)  # pylint: disable=unused-variable
        training_mode: int = self.attrs.get('training_mode', 0)
        assert training_mode == 0, 'BatchNorm in training mode occurs, currently, hidet does not support training.'

        x, scale, bias, running_mean, running_var = inputs
        if len(x.shape) == 1:
            y = (x - running_mean) * (running_var + epsilon).rsqrt()
            return [y * scale + bias]
        else:
            unsqueeze_dims = [dim for dim in range(len(x.shape)) if dim != 1]
            y = ops.batch_norm_infer(x, running_mean=running_mean, running_var=running_var, epsilon=epsilon, axis=1)
            return [y * scale.unsqueeze(unsqueeze_dims) + bias.unsqueeze(unsqueeze_dims)]


@register_onnx_operator
class OnnxRelu(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.relu(inputs[0])]


@register_onnx_operator
class OnnxSin(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.sin(inputs[0])]


@register_onnx_operator
class OnnxCos(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.cos(inputs[0])]


@register_onnx_operator
class OnnxPow(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        x, y = inputs
        return [ops.pow(x, y)]


@register_onnx_operator
class OnnxDiv(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        x, y = inputs
        return [ops.divide(x, y)]


@register_onnx_operator
class OnnxSqrt(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.sqrt(inputs[0])]


@register_onnx_operator
class OnnxErf(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.erf(inputs[0])]


@register_onnx_operator
class OnnxTanh(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.tanh(inputs[0])]


@register_onnx_operator
class OnnxMaxPool(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        kernel_size = list(self.attrs.get('kernel_shape'))
        x = inputs[0]
        if len(x.shape) == 4:
            padding = list(self.attrs.get('pads', [0, 0, 0, 0]))
            strides = list(self.attrs.get('strides', [1, 1]))
            return [ops.max_pool2d(inputs[0], kernel_size, strides, padding)]
        elif len(x.shape) == 5:
            padding = list(self.attrs.get('pads', [0, 0, 0, 0, 0, 0]))
            strides = list(self.attrs.get('strides', [1, 1, 1]))
            return [ops.max_pool3d(inputs[0], kernel_size, strides, padding)]
        else:
            raise NotImplementedError('Currently only support 2d and 3d max pooling')


@register_onnx_operator
class OnnxReduceMean(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        dims = self.attrs.get('axes')
        keep_dim = self.attrs.get('keepdims', 1) == 1
        return [ops.mean(inputs[0], dims, keep_dim)]


@register_onnx_operator
class OnnxSqueeze(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        dims = self.attrs.get('axes', None)
        data = inputs[0]
        if dims is None:
            # squeeze all dimensions with extent 1
            dims = [i for i, dim in enumerate(data.shape) if dim == 1]
        else:
            dims = list(dims)
        return [ops.squeeze(inputs[0], dims)]

    def run_v13(self, inputs: List[Tensor]) -> List[Tensor]:
        data, axes = inputs
        dims = self.tensor2list(axes)
        return [ops.squeeze(data, dims)]


@register_onnx_operator
class OnnxAdd(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [inputs[0] + inputs[1]]


@register_onnx_operator
class OnnxSub(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [inputs[0] - inputs[1]]


@register_onnx_operator
class OnnxMul(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [inputs[0] * inputs[1]]


@register_onnx_operator
class OnnxMatMul(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        a, b = inputs
        return [ops.matmul(a, b)]


@register_onnx_operator
class OnnxSoftmax(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        axis = self.attrs.get('axis', 1)
        return [ops.softmax(inputs[0], axis)]

    def run_v13(self, inputs: List[Tensor]) -> List[Tensor]:
        axis = self.attrs.get('axis', -1)
        return [ops.softmax(inputs[0], axis)]


@register_onnx_operator
class OnnxGlobalAveragePool(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        (x,) = inputs
        dims = list(range(2, len(x.shape)))
        return [ops.mean(x, dims=dims, keep_dim=True)]


@register_onnx_operator
class OnnxFlatten(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        axis = self.attrs.get('axis', 1)
        x = inputs[0]
        rank = len(x.shape)
        axis = (axis + rank) % rank
        dims = list(range(rank))
        return [ops.rearrange(x, plan=[dims[:axis], dims[axis:]])]


@register_onnx_operator
class OnnxUnsqueeze(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        axes = self.attrs['axes']  # in [-output_rank, output_rank - 1]
        x = inputs[0]
        rank = len(x.shape) + len(axes)
        axes = [(axis + rank) % rank for axis in axes]
        return [ops.unsqueeze(x, axes)]

    def run_v13(self, inputs: List[Tensor]) -> List[Tensor]:
        x, axes = inputs
        axes = self.tensor2list(axes)
        rank = len(x.shape) + len(axes)
        axes = [(axis + rank) % rank for axis in axes]
        return [ops.unsqueeze(x, axes)]


@register_onnx_operator
class OnnxReshape(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        allow_zero = self.attrs.get('allowzero', 0)  # pylint: disable=unused-variable
        x, shape = inputs
        shape = self.tensor2list(shape)
        return [ops.reshape(x, shape)]


@register_onnx_operator
class OnnxTranspose(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        perm = self.attrs.get('perm', None)
        x = inputs[0]
        perm = perm if perm else list(reversed(range(len(x.shape))))
        return [ops.transpose(x, perm)]


@register_onnx_operator
class OnnxConcat(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        axis = self.attrs.get('axis')
        return [ops.concat(inputs, axis)]


@register_onnx_operator
class OnnxArgMax(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        axis = self.attrs.get('axis', 0)
        keepdims = self.attrs.get('keepdims', True)
        select_last_index = self.attrs.get('select_last_index', False)
        if select_last_index:
            raise NotImplementedError()
        return [ops.argmax(inputs[0], dim=axis, keep_dim=keepdims)]


@register_onnx_operator
class OnnxGemm(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        alpha = self.attrs.get('alpha', 1.0)
        beta = self.attrs.get('beta', 0.0)
        trans_a = self.attrs.get('transA', 0)
        trans_b = self.attrs.get('transB', 0)

        a, b = inputs[:2]
        c = inputs[2] if len(inputs) > 2 else None
        if trans_a == 1:
            a = ops.rearrange(a, plan=[[1], [0]])
        if trans_b == 1:
            b = ops.rearrange(b, plan=[[1], [0]])
        assert a.shape[1] == b.shape[0]
        d = ops.matmul(a, b)
        if alpha != 1.0:
            d = d * alpha
        if c is not None and beta != 0.0:
            d = d + c * beta
        return [d]


@register_onnx_operator
class OnnxCast(OnnxOperator):
    code2dtype = {
        1: 'float32',
        2: 'uint8',
        3: 'int8',
        4: 'uint16',
        5: 'int16',
        6: 'int32',
        7: 'int64',
        8: 'string',
        9: 'bool',
        10: 'float16',
        11: 'float64',
        12: 'uint32',
        13: 'uint64',
        14: 'complex64',
        15: 'complex128',
        16: 'bfloat16',
    }

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        to = self.attrs.get('to')
        x = inputs[0]
        dtype = self.code2dtype[to]
        return [ops.cast(x, dtype)]


@register_onnx_operator
class OnnxShape(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        start = self.attrs.get('start', 0)
        end: Optional[int] = self.attrs.get('end', None)

        x = inputs[0]
        rank = len(x.shape)
        start = start + rank if start < 0 else start
        if end is not None:
            end = end + rank if end < 0 else end
        else:
            end = rank
        start = max(min(start, rank), 0)
        end = max(min(end, rank), 0)
        return [hidet.asarray(x.shape[start:end]).cuda()]


@register_onnx_operator
class OnnxConstant(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        value: Optional[Tensor] = self.attrs.get('value')
        if value is None:
            raise NotImplementedError('Currently, only support Tensor constant in onnx importer')
        assert len(inputs) == 0
        return [value]


@register_onnx_operator
class OnnxGather(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        axis = self.attrs.get('axis', 0)
        data, indices = inputs
        return [ops.take(data, indices, axis)]


@register_onnx_operator
class OnnxSlice(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        data = inputs[0]
        starts = self.attrs['starts']
        ends = self.attrs['ends']
        axes = self.attrs.get('axes', list(range(len(starts))))
        ends = [min(end, data.shape[i]) for i, end in zip(axes, ends)]
        return [ops.strided_slice(data, starts, ends, axes)]

    def run_v10(self, inputs: List[Tensor]) -> List[Tensor]:
        data, starts, ends = inputs[:3]
        axes = inputs[3] if len(inputs) > 3 else None
        steps = inputs[4] if len(inputs) > 4 else None
        starts = self.tensor2list(starts)
        ends = self.tensor2list(ends)
        axes = self.tensor2list(axes) if axes is not None else None
        steps = self.tensor2list(steps) if steps is not None else None
        ends = [min(end, data.shape[i]) for i, end in zip(axes, ends)]
        return [ops.strided_slice(data, starts, ends, axes, steps)]


@register_onnx_operator
class OnnxSigmoid(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.sigmoid(inputs[0])]


@register_onnx_operator
class OnnxInstanceNormalization(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        epsilon = self.attrs.get('epsilon', 1e-5)

        x, scale, bias = inputs
        rank = len(x.shape)
        dims = [0] + list(range(2, rank))
        scale = ops.unsqueeze(scale, dims)  # [1, C, D1, ...]
        bias = ops.unsqueeze(bias, dims)  # [1, C, D1, ...]
        return [ops.instance_norm(x, epsilon) * scale + bias]


@register_onnx_operator
class OnnxConstantOfShape(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        value = self.attrs.get('value')
        if value is None:
            value = hidet.zeros([1], dtype='float32')

        shape = inputs[0].cpu().numpy().tolist()
        assert all(v >= 0 for v in shape)
        return [ops.broadcast(value, shape)]


@register_onnx_operator
class OnnxPad(OnnxOperator):
    def run_v2(self, inputs: List[Tensor]) -> List[Tensor]:
        data = inputs[0]
        mode = self.attrs.get('mode', 'constant')
        pads = self.attrs.get('pads')
        value = self.attrs.get('value', 0.0)
        return [ops.pad(data, pads, mode, value)]

    def run_v13(self, inputs: List[Tensor]) -> List[Tensor]:
        mode = self.attrs.get('mode', 'constant')
        data, pads = inputs[:2]
        value = self.tensor2list(inputs[2]) if len(inputs) > 2 else 0.0
        pads = self.tensor2list(pads)
        return [ops.pad(data, pads, mode, value)]


@register_onnx_operator
class OnnxResize(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        coordinate_transformation_mode = self.attrs.get('coordinate_transformation_mode', 'half_pixel')
        cubic_coeff_a = self.attrs.get('cubic_coeff_a', -0.75)
        exclude_outside = self.attrs.get('exclude_outside', 0)
        extrapolation_value = self.attrs.get('extrapolation_value', 0.0)
        mode = self.attrs.get('mode', 'nearest')
        nearest_mode = self.attrs.get('nearest_mode', 'round_prefer_floor')

        x, roi, scales, sizes = self.optional_inputs(inputs, requires=[True, False, False, False])
        if roi is not None:
            roi = self.tensor2list(roi)
        target_size = None
        if scales is not None and scales.size > 0:
            scales = self.tensor2list(scales)
            assert len(x.shape) == len(scales)
            target_size = [int(a * b) for a, b in zip(x.shape, scales)]
        elif sizes is not None and sizes.size > 0:
            sizes = self.tensor2list(sizes)
            target_size = [int(v) for v in sizes]
        else:
            raise ValueError('Resize operator in onnx must give either scales or sizes.')
        if len(x.shape) == 4:
            if not (target_size[0] == x.shape[0] and target_size[1] == x.shape[1]):
                raise ValueError('Unsupported resize on batch and channel dimension.')
            return [
                ops.resize2d(
                    x,
                    target_size[2:],
                    mode,
                    coordinate_transformation_mode,
                    nearest_mode,
                    roi,
                    cubic_coeff_a,
                    exclude_outside,
                    extrapolation_value,
                )
            ]
        else:
            raise NotImplementedError('Current only support 2d resize, got x {}.'.format(x.shape))


@register_onnx_operator
class OnnxExpand(OnnxOperator):
    def run_v8(self, inputs: List[Tensor]) -> List[Tensor]:
        data, new_shape = inputs
        new_shape = self.tensor2list(new_shape)
        new_shape = hidet.graph.ops.definitions.arithmetic.broadcast_shape(data.shape, new_shape)
        return [ops.broadcast(data, new_shape)]


@register_onnx_operator
class OnnxRange(OnnxOperator):
    def run_v11(self, inputs: List[Tensor]) -> List[Tensor]:
        start, limit, delta = [self.tensor2list(t) for t in inputs]
        array = np.arange(start=start, stop=limit, step=delta)
        array = hidet.asarray(array).cuda().astype(dtype=inputs[0].dtype)
        return [array]


@register_onnx_operator
class OnnxTile(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        data, repeats = inputs
        repeats = self.tensor2list(repeats)
        return [ops.tile(data, repeats)]


@register_onnx_operator
class OnnxAveragePool(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        auto_pad = self.attrs.get('auto_pad', 'NOTSET')
        ceil_mode = self.attrs.get('ceil_mode', 0)
        count_include_pad = self.attrs.get('count_include_pad', 0)
        kernel_shape = self.attrs.get('kernel_shape')
        if auto_pad != 'NOTSET' or ceil_mode != 0 or count_include_pad != 0:
            raise NotImplementedError(self)

        x = inputs[0]
        if len(x.shape) == 4:
            pads = list(self.attrs.get('pads', [0, 0, 0, 0]))
            strides = list(self.attrs.get('strides', [1, 1]))
            x = ops.avg_pool2d(x, kernel_shape, strides, pads)
        elif len(x.shape) == 5:
            pads = list(self.attrs.get('pads', [0, 0, 0, 0, 0, 0]))
            strides = list(self.attrs.get('strides', [1, 1, 1]))
            x = ops.avg_pool3d(x, kernel_shape, strides, pads)
        else:
            raise NotImplementedError('Currently only support 2d and 3d avg pooling')
        return [x]


@register_onnx_operator
class OnnxClip(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        (x,) = inputs
        min_value = self.attrs.get('min', None)
        max_value = self.attrs.get('max', None)
        x = ops.clip(x, min_value, max_value)
        return [x]

    def run_v11(self, inputs: List[Tensor]) -> List[Tensor]:
        data, min_value, max_value = self.optional_inputs(inputs, requires=[True, False, False])
        if min_value is not None:
            min_value = self.tensor2scalar(min_value)
        if max_value is not None:
            max_value = self.tensor2scalar(max_value)
        return [ops.clip(data, min_value, max_value)]


@register_onnx_operator
class OnnxEqual(OnnxOperator):
    def run_v11(self, inputs: List[Tensor]) -> List[Tensor]:
        a, b = inputs
        return [ops.equal(a, b)]


@register_onnx_operator
class OnnxLess(OnnxOperator):
    def run_v9(self, inputs: List[Tensor]) -> List[Tensor]:
        a, b = inputs
        return [ops.less(a, b)]


@register_onnx_operator
class OnnxGreater(OnnxOperator):
    def run_v7(self, inputs: List[Tensor]) -> List[Tensor]:
        a, b = inputs
        return [ops.greater(a, b)]


@register_onnx_operator
class OnnxGreaterOrEqual(OnnxOperator):
    def run_v12(self, inputs: List[Tensor]) -> List[Tensor]:
        a, b = inputs
        return [ops.greater_equal(a, b)]


@register_onnx_operator
class OnnxLessOrEqual(OnnxOperator):
    def run_v12(self, inputs: List[Tensor]) -> List[Tensor]:
        a, b = inputs
        return [ops.less_equal(a, b)]


@register_onnx_operator
class OnnxWhere(OnnxOperator):
    def run_v9(self, inputs: List[Tensor]) -> List[Tensor]:
        cond, a, b = inputs
        return [ops.where(cond, a, b)]


@register_onnx_operator
class OnnxSplit(OnnxOperator):
    def run_v2(self, inputs: List[Tensor]) -> List[Tensor]:
        axis = self.attrs.get('axis', 0)
        parts = self.attrs['split']
        data = inputs[0]
        return ops.split(data, axis, parts)

    def run_v13(self, inputs: List[Tensor]) -> List[Tensor]:
        data = inputs[0]
        axis = self.attrs.get('axis', 0)
        if len(inputs) == 1:
            num_outputs = len(self.output_names)
            extent = data.shape[axis]
            if extent % num_outputs != 0:
                raise ValueError(
                    'Can not split tensor with shape {} on axis {} into {} parts evenly.'.format(
                        data.shape, axis, num_outputs
                    )
                )
            parts = [extent // num_outputs] * num_outputs
        elif len(inputs) == 2:
            parts = self.tensor2list(inputs[1])
        else:
            raise ValueError(
                'Expect the input of Split operator have 1 or 2 inputs, but got {} inputs. See:\n'.format(len(inputs))
                + 'https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split'
            )
        return ops.split(data, axis, parts)


@register_onnx_operator
class OnnxReduceSum(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        axes = self.attrs['axes']
        keepdims = self.attrs.get('keepdims', True)
        data = inputs[0]
        return [ops.sum(data, dims=axes, keep_dim=keepdims)]

    def run_v13(self, inputs: List[Tensor]) -> List[Tensor]:
        keepdims = self.attrs.get('keepdims', True)
        noop_with_emtpy_axes = self.attrs.get('noop_with_empty_axes', False)
        data = inputs[0]
        if len(inputs) == 1:
            if noop_with_emtpy_axes:
                axes = []
            else:
                axes = list(range(len(data.shape)))
        else:
            axes = self.tensor2list(inputs[1])
        return [ops.sum(data, dims=axes, keep_dim=keepdims)]


@register_onnx_operator
class OnnxReduceMin(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        axes = self.attrs['axes']
        keepdims = self.attrs.get('keepdims', True)
        data = inputs[0]
        return [ops.min(data, dims=axes, keep_dim=keepdims)]


@register_onnx_operator
class OnnxReduceMax(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        axes = self.attrs['axes']
        keepdims = self.attrs.get('keepdims', True)
        data = inputs[0]
        return [ops.max(data, dims=axes, keep_dim=keepdims)]


@register_onnx_operator
class OnnxMax(OnnxOperator):
    def run_v6(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.maximum(*inputs)]

    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError()


@register_onnx_operator
class OnnxMin(OnnxOperator):
    def run_v6(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.minimum(*inputs)]


@register_onnx_operator
class OnnxReciprocal(OnnxOperator):
    def run_v6(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.reciprocal(inputs[0])]


@register_onnx_operator
class OnnxExp(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.exp(inputs[0])]


@register_onnx_operator
class OnnxLog(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.log(inputs[0])]


@register_onnx_operator
class OnnxNeg(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.negative(inputs[0])]


@register_onnx_operator
class OnnxIf(OnnxOperator):
    def __init__(self, node, op_sets: List[int]):
        super().__init__(node, op_sets)
        self.env_tensors: Dict[str, Tensor] = {}

    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        cond = inputs[0]
        if cond.storage is None:
            raise ValueError(
                'Hidet currently does not support dynamic control flow in computation graph'
                ' (If operator with condition that depends on non-const input).'
            )

        cond = cond.numpy().flatten()
        if cond.size > 1:
            raise ValueError('Condition in If operator can only have a single element.')
        if np.all(cond):
            graph = OnnxGraph(self.attrs['then_branch'], self.op_sets, self.env_tensors)
        else:
            graph = OnnxGraph(self.attrs['else_branch'], self.op_sets, self.env_tensors)
        return graph(*inputs[1:])


@register_onnx_operator
class OnnxNot(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.logical_not(inputs[0])]


@register_onnx_operator
class OnnxCumSum(OnnxOperator):
    def run_v11(self, inputs: List[Tensor]) -> List[Tensor]:
        x, axis = inputs
        axis = self.tensor2list(axis)
        exclusive = self.attrs.get('exclusive', False)
        reverse = self.attrs.get('reverse', False)
        return [ops.cumsum(x, axis, exclusive=exclusive, reverse=reverse)]


@register_onnx_operator
class OnnxIdentity(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return inputs


@register_onnx_operator
class OnnxPyFunc(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        warnings.warn(
            'PyFunc operator in ONNX model encountered, dummy output is returned. '
            'If dummy output are used, there will be errors.'
        )
        return [randn([1]) for name in self.output_names]


@register_onnx_operator
class OnnxLeakyRelu(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        alpha = self.attrs.get('alpha', 0.01)
        return [ops.leaky_relu(inputs[0], alpha)]


@register_onnx_operator
class OnnxConvTranspose(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        from hidet.graph.ops.definitions.utils import normalize_stride

        data, weight = inputs[:2]
        if len(data.shape) != 4:
            raise ValueError('Currently, only support 2D ConvTranspose.')
        auto_pad: str = self.attrs.get('auto_pad', 'NOTSET')
        dilations: Union[int, List[int]] = self.attrs.get('dilations', 1)
        group: int = self.attrs.get('group', 1)
        output_padding: Union[int, List[int]] = self.attrs.get('output_padding', 0)
        output_shape: Optional[List[int]] = self.attrs.get('output_shape', None)
        pads: Union[int, List[int]] = self.attrs.get('pads', 0)
        strides: int = self.attrs.get('strides', 1)

        if auto_pad != 'NOTSET':
            raise NotImplementedError('auto_pad {} is not supported yet.'.format(auto_pad))
        if output_shape is not None:
            raise NotImplementedError('output_shape is not supported yet.')
        if isinstance(dilations, int):
            dilations = [dilations] * 2
        if any(d != 1 for d in dilations):
            raise NotImplementedError('dilations {} is not supported yet.'.format(dilations))

        output_padding = normalize_stride(output_padding)

        if len(pads) == 4 and any(p < 0 for p in pads[2:]):
            # sometimes upstream framework may export onnx model with negative pads
            # this is a workaround to fix it
            # remove this when upstream framework fix their bug
            for i, p in enumerate(pads[2:]):
                if p < 0:
                    pads[2 + i] = 0
                    output_padding[i] += -p

        output = ops.conv2d_transpose(
            data, weight, stride=strides, padding=pads, groups=group, output_padding=output_padding
        )
        if len(inputs) > 2:
            bias: Tensor = inputs[2]  # 1D tensor added on channel axis
            output = output + ops.unsqueeze(bias, [0, 2, 3])
        return [output]


@register_onnx_operator
class OnnxPRelu(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.prelu(inputs[0], inputs[1])]


@register_onnx_operator
class OnnxAbs(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.abs(inputs[0])]


@register_onnx_operator
class OnnxAnd(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.logical_and(inputs[0], inputs[1])]


@register_onnx_operator
class OnnxBitShift(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        direction = self.attrs.get('direction', 'RIGHT')
        if direction == 'RIGHT':
            return [ops.bitwise_right_shift(inputs[0], inputs[1])]
        else:
            return [ops.bitwise_left_shift(inputs[0], inputs[1])]


@register_onnx_operator
class OnnxBitwiseAnd(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.bitwise_and(inputs[0], inputs[1])]


@register_onnx_operator
class OnnxBitwiseNot(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.bitwise_invert(inputs[0])]


@register_onnx_operator
class OnnxBitwiseOr(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.bitwise_or(inputs[0], inputs[1])]


@register_onnx_operator
class OnnxBitwiseXor(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.bitwise_xor(inputs[0], inputs[1])]


@register_onnx_operator
class OnnxCeil(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.ceil(inputs[0])]


@register_onnx_operator
class OnnxReduceL2(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        axes: Optional[List[int]] = self.attrs.get('axes', None)
        keepdims: int = self.attrs.get('keepdims', 1)
        assert len(inputs) == 1
        data: Tensor = inputs[0]
        rank = len(data.shape)
        if axes is None:
            axes = list(range(rank))
        axes: List[int] = [ops.utils.normalize_dim(axis, rank) for axis in axes]
        return [ops.sqrt(ops.sum(ops.square(data), axes, keep_dim=bool(keepdims)))]

    def run_v18(self, inputs: List[Tensor]) -> List[Tensor]:
        keepdims: int = self.attrs.get('keepdims', 1)
        noop_with_empty_axes: int = self.attrs.get('noop_with_empty_axes', 0)
        data, axes_tensor = self.optional_inputs(inputs, requires=[True, False])
        if axes_tensor is None:
            if noop_with_empty_axes:
                return [data]
            else:
                axes: List[int] = list(range(len(data.shape)))
        else:
            axes: List[int] = self.tensor2list(axes_tensor)
        return [ops.sqrt(ops.sum(ops.square(data), axes, keep_dim=bool(keepdims)))]


def dispatch(node, op_sets: List[int]) -> OnnxOperator:
    op_type = node.op_type
    if op_type not in dispatch_table:
        raise NotImplementedError(
            "Operator '{}' (in opset {}) from onnx has not been supported yet.".format(op_type, op_sets)
        )
    op = dispatch_table[op_type](node, op_sets)
    return op


def dispatch_operators(nodes: Sequence[onnx.NodeProto], op_sets: List[int]) -> List[OnnxOperator]:
    dispatched: List[OnnxOperator] = []
    unsupported: Set[str] = set()

    for node in nodes:
        op_type: str = node.op_type
        if op_type not in dispatch_table:
            unsupported.add(op_type)
        else:
            op_cls: Type[OnnxOperator] = dispatch_table[op_type]
            dispatched.append(op_cls(node, op_sets))
    if len(unsupported) > 0:
        raise NotImplementedError("Operator(s) {} from onnx have not been supported yet.".format(list(unsupported)))
    return dispatched


def run_trt(node: OnnxOperator, inputs: List[Tensor]) -> List[Tensor]:
    # pylint: disable=no-member
    import onnxruntime

    hidet_outputs = node.run(inputs)
    inputs_value_info = [
        onnx.helper.make_value_info(
            name=name,
            type_proto=onnx.helper.make_tensor_type_proto(
                elem_type=utils.dtype_to_onnx(tensor.dtype), shape=tensor.shape
            ),
        )
        for name, tensor in zip(node.input_names, inputs)
    ]
    outputs_value_info = [
        onnx.helper.make_value_info(
            name=name,
            type_proto=onnx.helper.make_tensor_type_proto(
                elem_type=utils.dtype_to_onnx(tensor.dtype), shape=tensor.shape
            ),
        )
        for name, tensor in zip(node.output_names, hidet_outputs)
    ]
    graph = onnx.helper.make_graph(nodes=[node.node], name='test', inputs=inputs_value_info, outputs=outputs_value_info)
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", opset) for opset in node.op_sets])
    # print(model)
    onnx.checker.check_model(model)
    # serialized_model = onnx._serialize(model)
    serialized_model = model.SerializeToString()
    session = onnxruntime.InferenceSession(serialized_model, providers=['CPUExecutionProvider'])
    outputs = session.run(
        node.output_names, input_feed={name: tensor.cpu().numpy() for name, tensor in zip(node.input_names, inputs)}
    )
    return [hidet.asarray(output).cuda() for output in outputs]


class OnnxGraph(nn.Module):
    def __init__(self, graph: onnx.GraphProto, op_sets: List[int], env_tensors: Optional[Dict[str, Tensor]] = None):
        super().__init__()
        self.op_sets = op_sets
        self.name: str = graph.name
        for param in graph.initializer:
            numpy_array = onnx.numpy_helper.to_array(tensor=param)
            self.parameters[param.name] = from_numpy(numpy_array).cuda()
        self.input_names: List[str] = [input.name for input in graph.input if input.name not in self.parameters]
        self.output_names: List[str] = [output.name for output in graph.output]
        self.operators: List[OnnxOperator] = dispatch_operators(graph.node, op_sets)
        # self.operators: List[OnnxOperator] = [dispatch(node, op_sets=self.op_sets) for node in graph.node]
        self.env_tensors: Dict[str, Tensor] = env_tensors if env_tensors else {}
        self.usage_count: Dict[str, int] = self.count_usage()

    def forward(self, *args):
        name2tensor = {"": None}
        if self.env_tensors:
            name2tensor.update(self.env_tensors)
        assert len(args) == len(self.input_names)
        # parameters
        for name, param in self.parameters.items():
            name2tensor[name] = param
        # inputs
        for name, inp in zip(self.input_names, args):
            name2tensor[name] = inp
        # run nodes

        log.info('start to interpret onnx graph')

        usage_count = self.usage_count.copy()
        for operator in self.operators:
            for name in operator.input_names:
                if name not in name2tensor:
                    raise ValueError('Tensor "{}" is used before produce.'.format(name))
            inputs = [name2tensor[name] for name in operator.input_names]
            if isinstance(operator, OnnxIf):
                operator.env_tensors = name2tensor
            outputs = operator.run(inputs)
            if not isinstance(outputs, (tuple, list)):
                raise ValueError(
                    'Operator "{}" should return a sequence of tensors, got {}.'.format(
                        operator.node.op_type, type(outputs)
                    )
                )

            check = False
            if check:
                outputs_trt = run_trt(operator, inputs)
                for a, b in zip(outputs, outputs_trt):
                    try:
                        np.testing.assert_allclose(a.cpu().numpy(), b.cpu().numpy(), atol=1e-3, rtol=1e-3)
                    except AssertionError as e:
                        print('Operator check failed: {:>20}'.format(operator.node.name))
                        # print('{}'.format(', '.join(out.signature() for out in outputs)))
                        raise e

            assert len(outputs) == len(operator.output_names)
            for name, tensor in zip(operator.output_names, outputs):
                name2tensor[name] = tensor
                # print('{:>50} {}'.format(name, tensor.signature()))
            for name in operator.input_names:
                if name not in self.env_tensors:
                    usage_count[name] -= 1
                    if usage_count[name] == 0:
                        # free memory
                        del name2tensor[name]

        # put outputs
        results = [name2tensor[name] for name in self.output_names]

        log.info('finish to interpret onnx graph')

        return results

    def count_usage(self):
        usage_count = defaultdict(int)
        for op in self.operators:
            for input_name in op.input_names:
                usage_count[input_name] += 1
        for graph_output_name in self.output_names:
            usage_count[graph_output_name] += 1
        # todo: add the usage of sub graphs
        return usage_count


class OnnxModule(nn.Module):
    """Loaded ONNX model.

    Parameters
    ----------
    model: onnx.ModelProto
        The onnx model to load, in the protobuf format.

    Attributes
    ----------
    op_sets: List[int]
        The operator sets used by the loaded model.

    input_names: List[str]
        The input names of the loaded onnx model.

    output_names: List[str]
        The output names of the loaded onnx model.
    """

    def __init__(self, model: onnx.ModelProto):
        super().__init__()
        op_sets = []
        for opset_import in model.opset_import:
            if opset_import.domain not in ['', 'ai.onnx', 'ai.onnx.ml']:
                # we currently only support standard onnx operator domain
                raise ValueError(
                    'Onnx model imports unknown operator domain: {}, we currently '
                    'only support standard onnx operator set.'.format(repr(opset_import.domain))
                )
            op_sets.append(int(opset_import.version))
        self.op_sets: List[int] = list(reversed(sorted(op_sets)))
        self.graph: OnnxGraph = OnnxGraph(model.graph, op_sets=self.op_sets)
        self.input_names: List[str] = self.graph.input_names
        self.output_names: List[str] = self.graph.output_names

    def forward(self, *args):
        """Run the onnx model with given inputs.

        Parameters
        ----------
        args: Sequence[hidet.Tensor]
            The input tensors. The number and order of the input tensors should match the
            OnnxModule.input_names attributes.

        Returns
        -------
        ret: Union[hidet.Tensor, List[hidet.Tensor]]
            The output tensor(s). If there are 2 or more tensors returned,
            a list of tensors are return with the order of OnnxModule.output_names.
            If there is only one tensor is returned, the single tensor is directly returned (instead of a list).
        """
        results = self.graph(*args)
        if len(results) == 1:
            return results[0]
        else:
            return results

    def dict_forward(self, feed_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        args = []
        for name in self.input_names:
            if name not in feed_dict:
                raise ValueError('Missing input: {}'.format(name))
            args.append(feed_dict[name])
        outputs = self.graph(*args)
        output_dict = {name: value for name, value in zip(self.output_names, outputs)}
        return output_dict


def from_onnx(model: Union[str, 'onnx.ModelProto']) -> OnnxModule:
    """
    Load an onnx model to hidet.graph.nn.Module.

    Parameters
    ----------
    model: Union[str, onnx.ModelProto]
        The path or model proto of given onnx model.

    Returns
    -------
    ret: OnnxModule
        The loaded model.
    """
    if isinstance(model, str):
        model = os.path.expanduser(model)
        model = onnx.load_model(model, load_external_data=False)
    try:
        onnx.checker.check_model(model, full_check=True)
    except ValueError:
        # ignore 'ValueError: Message onnx.ModelProto exceeds maximum protobuf size of 2GB'
        pass
    except onnx.onnx_cpp2py_export.checker.ValidationError:  # pylint: disable=c-extension-no-member
        warnings.warn('The onnx model has not pass the onnx checker.')
    return OnnxModule(model)
