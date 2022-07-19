from typing import List, Union, Sequence, Optional, Dict, Callable
from collections import defaultdict
import os
import numpy as np
import hidet
from hidet.common import HidetNotImplementedError
from hidet.tos.modules import nn
from hidet.tos import ops
from hidet.tos.tensor import Tensor, from_numpy, randn
from hidet.utils import line_profile, prod

"""
Please refers to https://github.com/onnx/onnx/blob/main/docs/Operators.md for operator definition when adding new operators.
Please refers to https://github.com/onnx/onnx/blob/main/onnx/onnx.proto for proto structure of onnx format.
"""


class OnnxOperator:
    def __init__(self, node, op_sets: List[int]):
        """
        Parameters
        ----------
        node: onnx.NodeProto
        """
        import onnx.numpy_helper
        self.node = node
        self.op_sets = op_sets
        self.input_names = [name for name in node.input]
        self.output_names = [name for name in node.output]
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
        raise NotImplementedError('Can not resolve opset for operator {} given opsets {}.'.format(self.node.op_type, op_sets))

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

    @staticmethod
    def tensor2list(tensor: Tensor) -> Union[List, int, float]:
        return tensor.cpu().numpy().tolist()

    @staticmethod
    def optional_inputs(inputs: List[Tensor], requires: List[bool]) -> List[Union[Tensor, None]]:
        diff = len(requires) - len(inputs)
        assert diff >= 0, 'Onnx get {} inputs but expect at most {}.'.format(len(inputs), len(requires))
        ret: List[Union[Tensor, None]] = []
        ret += inputs
        ret += [None for _ in range(diff)]
        for i, (t, r) in enumerate(zip(ret, requires)):
            if t is None and r:
                raise 'The {}th input is required.'.format(i)
        return ret


class OnnxConv(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        padding = self.attrs.get('pads', [0, 0, 0, 0])
        strides = self.attrs.get('strides', [1, 1])
        groups = self.attrs.get('group', 1)
        if len(inputs) == 2:
            x, w = inputs
            bias = None
        else:
            x, w, bias = inputs
        x = ops.pad(x, ops.utils.normalize_padding(padding))
        output = ops.conv2d(x, w, stride=strides, groups=groups)
        if bias is not None:
            bias = ops.unsqueeze(bias, [0, 2, 3])
            output = output + bias
        return [output]

    def run_v11(self, inputs: List[Tensor]) -> List[Tensor]:
        return self.run_v1(inputs)


class OnnxBatchNormalization(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        epsilon: float = self.attrs.get('epsilon', 1e-5)
        momentum: float = self.attrs.get('momentum', 0.9)
        training_mode: int = self.attrs.get('training_mode', 0)
        assert training_mode == 0, 'BatchNorm in training mode occurs, currently, hidet does not support training.'

        x, scale, bias, running_mean, running_var = inputs
        y = ops.batch_norm_infer(x, running_mean=running_mean, running_var=running_var, epsilon=epsilon, axis=1)
        return [y * scale.unsqueeze([0, 2, 3]) + bias.unsqueeze([0, 2, 3])]


class OnnxRelu(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.relu(inputs[0])]


class OnnxSin(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.sin(inputs[0])]


class OnnxCos(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.cos(inputs[0])]


class OnnxPow(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        x, y = inputs
        return [ops.pow(x, y)]


class OnnxDiv(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        x, y = inputs
        return [ops.divide(x, y)]


class OnnxSqrt(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.sqrt(inputs[0])]


class OnnxErf(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.erf(inputs[0])]


class OnnxTanh(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.tanh(inputs[0])]


class OnnxMaxPool(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        kernel_size = list(self.attrs.get('kernel_shape'))
        padding = list(self.attrs.get('pads', [0, 0, 0, 0]))
        strides = list(self.attrs.get('strides'))
        return [ops.max_pool2d(inputs[0], kernel_size, strides, padding)]


class OnnxReduceMean(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        dims = self.attrs.get('axes')
        keep_dim = self.attrs.get('keepdims', 1) == 1
        return [ops.reduce_mean(inputs[0], dims, keep_dim)]


class OnnxSqueezeOp(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        dims = self.attrs.get('axes', None)
        data = inputs[0]
        if dims is None:
            # squeeze all dimensions with extent 1
            dims = [i for i, dim in enumerate(data.shape) if dim == 1]
        else:
            dims = list(dims)
        return [ops.squeeze(inputs[0], dims)]


class OnnxAdd(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [inputs[0] + inputs[1]]


class OnnxSub(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [inputs[0] - inputs[1]]


class OnnxMul(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [inputs[0] * inputs[1]]


class OnnxMatMul(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        a, b = inputs
        assert len(a.shape) >= 2 and len(b.shape) >= 2
        if len(a.shape) == 2 and len(b.shape) == 2:
            return [ops.matmul(a, b)]
        else:
            prefix_shape = hidet.tos.ops.definitions.arithmatic.broadcast_shape(a.shape[:-2], b.shape[:-2])
            a = ops.broadcast(a, prefix_shape + a.shape[-2:])
            b = ops.broadcast(b, prefix_shape + b.shape[-2:])
            a = ops.flatten(a, end_dim=-2)  # [B, M, K]
            b = ops.flatten(b, end_dim=-2)  # [B, K, N]
            c = ops.matmul(a, b)  # [B, M, N]
            c_expect_shape = prefix_shape + [a.shape[-2], b.shape[-1]]
            c = c.reshape(c_expect_shape)
            return [c]


class OnnxSoftmax(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        axis = self.attrs.get('axis')
        return [ops.softmax(inputs[0], axis)]


class OnnxGlobalAveragePool(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        x, = inputs
        n, c, h, w = x.shape
        return [ops.avg_pool2d(x, kernel=(h, w), stride=(1, 1), padding=(0, 0))]


class OnnxFlatten(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        axis = self.attrs.get('axis', 1)
        x = inputs[0]
        rank = len(x.shape)
        axis = (axis + rank) % rank
        dims = list(range(rank))
        return [ops.rearrange(x, plan=[dims[:axis], dims[axis:]])]


class OnnxUnsqueeze(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        axes = self.attrs['axes']   # in [-output_rank, output_rank - 1]
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


class OnnxReshape(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        allow_zero = self.attrs.get('allowzero', 0)
        x, shape = inputs
        shape = self.tensor2list(shape)
        return [ops.reshape(x, shape)]


class OnnxTranspose(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        perm = self.attrs.get('perm', None)
        x = inputs[0]
        perm = perm if perm else list(reversed(range(len(x.shape))))
        return [ops.transpose(x, perm)]


class OnnxConcat(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        axis = self.attrs.get('axis')
        return [ops.concat(inputs, axis)]


class OnnxArgMax(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return inputs
        # raise NotImplementedError('ArgMax')


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
        if c and beta != 0.0:
            d = d + c * beta
        return [d]


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
        11: 'double',
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
        return [hidet.array(x.shape[start:end]).cuda()]


class OnnxConstant(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        value: Optional[Tensor] = self.attrs.get('value')
        if value is None:
            raise NotImplementedError('Currently, only support Tensor constant in onnx importer')
        assert len(inputs) == 0
        return [value]


class OnnxGather(OnnxOperator):

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        axis = self.attrs.get('axis', 0)
        data, indices = inputs
        return [ops.take(data, indices, axis)]


class OnnxSlice(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        data, starts, ends = inputs[:3]
        axes = inputs[3] if len(inputs) > 3 else None
        steps = inputs[4] if len(inputs) > 4 else None
        starts = self.tensor2list(starts)
        ends = self.tensor2list(ends)
        axes = self.tensor2list(axes) if axes else None
        steps = self.tensor2list(steps) if steps else None
        return [ops.strided_slice(data, starts, ends, axes, steps)]


class OnnxSigmoid(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.sigmoid(inputs[0])]


class OnnxInstanceNormalization(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        epsilon = self.attrs.get('epsilon', 1e-5)

        x, scale, bias = inputs
        rank = len(x.shape)
        dims = [0] + list(range(2, rank))
        scale = ops.unsqueeze(scale, dims)  # [1, C, D1, ...]
        bias = ops.unsqueeze(bias, dims)  # [1, C, D1, ...]
        return [ops.instance_norm(x, epsilon) * scale + bias]


class OnnxConstantOfShape(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        value = self.attrs.get('value')
        if value is None:
            value = hidet.zeros([1], dtype='float32')

        shape = inputs[0].cpu().numpy().tolist()
        assert all(v >= 0 for v in shape)
        return [ops.broadcast(value, shape)]


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
        if scales is not None:
            scales = self.tensor2list(scales)
            assert len(x.shape) == len(scales)
            target_size = [int(a * b) for a, b in zip(x.shape, scales)]
        if sizes is not None:
            sizes = self.tensor2list(sizes)
            target_size = [int(v) for v in sizes]
        if target_size is None:
            raise ValueError('Resize operator in onnx must give either scales or sizes.')
        if len(x.shape) == 4:
            if not (target_size[0] == x.shape[0] and target_size[1] == x.shape[1]):
                raise ValueError('Unsupported resize on batch and channel dimension.')
            return [ops.resize2d(x, target_size[2:], mode, coordinate_transformation_mode, nearest_mode,
                                 roi, cubic_coeff_a, exclude_outside, extrapolation_value)]
        else:
            raise NotImplementedError('Current only support 2d resize, got x {}.'.format(x.shape))


class OnnxExpand(OnnxOperator):
    def run_v8(self, inputs: List[Tensor]) -> List[Tensor]:
        data, new_shape = inputs
        new_shape = self.tensor2list(new_shape)
        new_shape = hidet.tos.ops.definitions.arithmatic.broadcast_shape(data.shape, new_shape)
        return [ops.broadcast(data, new_shape)]


class OnnxRange(OnnxOperator):
    def run_v11(self, inputs: List[Tensor]) -> List[Tensor]:
        start, limit, delta = [self.tensor2list(t) for t in inputs]
        array = np.arange(start=start, stop=limit, step=delta)
        array = hidet.array(array).cuda().cast(dtype=inputs[0].dtype)
        return [array]


class OnnxTile(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        data, repeats = inputs
        repeats = self.tensor2list(repeats)
        return [ops.tile(data, repeats)]


class OnnxAveragePool(OnnxOperator):

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        auto_pad = self.attrs.get('auto_pad', 'NOTSET')
        ceil_mode = self.attrs.get('ceil_mode', 0)
        count_include_pad = self.attrs.get('count_include_pad', 0)
        kernel_shape = self.attrs.get('kernel_shape')
        pads = self.attrs.get('pads')
        strides = self.attrs.get('strides')
        if auto_pad != 'NOTSET' or ceil_mode != 0 or count_include_pad != 0:
            raise NotImplementedError(self)

        x = inputs[0]
        if len(x.shape) != 4:
            raise NotImplementedError('Currently only support 2-d avg pooling')
        x = ops.avg_pool2d(x, kernel_shape, strides, pads)
        return [x]


class OnnxClip(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError()

    def run_v6(self, inputs: List[Tensor]) -> List[Tensor]:
        x = inputs[0]
        min_value = self.attrs.get('min', None)
        max_value = self.attrs.get('max', None)
        x = ops.clip(x, min_value, max_value)
        return [x]

    def run_v11(self, inputs: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError()

    def run_v12(self, inputs: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError()


class OnnxEqual(OnnxOperator):
    def run_v11(self, inputs: List[Tensor]) -> List[Tensor]:
        a, b = inputs
        return [ops.equal(a, b)]


class OnnxLess(OnnxOperator):
    def run_v9(self, inputs: List[Tensor]) -> List[Tensor]:
        a, b = inputs
        return [ops.less(a, b)]


class OnnxWhere(OnnxOperator):
    def run_v9(self, inputs: List[Tensor]) -> List[Tensor]:
        cond, a, b = inputs
        return [ops.where(cond, a, b)]


class OnnxSplit(OnnxOperator):
    def run_v2(self, inputs: List[Tensor]) -> List[Tensor]:
        axis = self.attrs.get('axis', 0)
        parts = self.attrs['split']
        data = inputs[0]
        return ops.split(data, axis, parts)

    def run_v13(self, inputs: List[Tensor]) -> List[Tensor]:
        axis = self.attrs.get('axis', 0)
        data, parts = inputs
        parts = self.tensor2list(parts)
        return ops.split(data, axis, parts)


class OnnxReduceSum(OnnxOperator):
    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        axes = self.attrs['axes']
        keepdims = self.attrs.get('keepdims', True)
        data = inputs[0]
        return [ops.reduce_sum(data, dims=axes, keep_dim=keepdims)]

    def run_v11(self, inputs: List[Tensor]) -> List[Tensor]:
        return self.run_v1(inputs)

    def run_v13(self, inputs: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError()


class OnnxMax(OnnxOperator):
    def run_v13(self, inputs: List[Tensor]) -> List[Tensor]:
        return self.run_v6(inputs)

    def run_v12(self, inputs: List[Tensor]) -> List[Tensor]:
        return self.run_v6(inputs)

    def run_v8(self, inputs: List[Tensor]) -> List[Tensor]:
        return self.run_v6(inputs)

    def run_v6(self, inputs: List[Tensor]) -> List[Tensor]:
        pass

    def run_v1(self, inputs: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError()



def dispatch(node, op_sets: List[int]) -> OnnxOperator:
    dispatch_table = {
        'Conv': OnnxConv,
        'Relu': OnnxRelu,
        'Pow': OnnxPow,
        'Div': OnnxDiv,
        'Sqrt': OnnxSqrt,
        'Erf': OnnxErf,
        'Tanh': OnnxTanh,
        'MaxPool': OnnxMaxPool,
        'ReduceMean': OnnxReduceMean,
        'Squeeze': OnnxSqueezeOp,
        'Add': OnnxAdd,
        'Sub': OnnxSub,
        'Mul': OnnxMul,
        'MatMul': OnnxMatMul,
        'Softmax': OnnxSoftmax,
        'ArgMax': OnnxArgMax,
        'BatchNormalization': OnnxBatchNormalization,
        'GlobalAveragePool': OnnxGlobalAveragePool,
        'Flatten': OnnxFlatten,
        'Unsqueeze': OnnxUnsqueeze,
        'Concat': OnnxConcat,
        'Cast': OnnxCast,
        'Constant': OnnxConstant,
        'Reshape': OnnxReshape,
        'Shape': OnnxShape,
        'Gemm': OnnxGemm,
        'Gather': OnnxGather,
        'Slice': OnnxSlice,
        'Transpose': OnnxTranspose,
        'Sin': OnnxSin,
        'Cos': OnnxCos,
        'Sigmoid': OnnxSigmoid,
        'InstanceNormalization': OnnxInstanceNormalization,
        'ConstantOfShape': OnnxConstantOfShape,
        'Pad': OnnxPad,
        'Resize': OnnxResize,
        'Expand': OnnxExpand,
        'Range': OnnxRange,
        'Tile': OnnxTile,
        'AveragePool': OnnxAveragePool,
        'Clip': OnnxClip,
        'Equal': OnnxEqual,
        'Less': OnnxLess,
        'Where': OnnxWhere,
        'Split': OnnxSplit,
        'ReduceSum': OnnxReduceSum,
    }
    op_type = node.op_type
    if op_type not in dispatch_table:
        raise NotImplementedError("Operator '{}' (in opset {}) from onnx has not been supported yet.".format(op_type, op_sets))
    op = dispatch_table[op_type](node, op_sets)
    return op


def run_trt(node: OnnxOperator, inputs: List[Tensor]) -> List[Tensor]:
    import onnx
    from onnx.helper import make_value_info, make_tensor_type_proto
    from onnx import TensorProto
    import onnxruntime
    hidet_outputs = node.run(inputs)
    dtype_map = {
        'float32': TensorProto.FLOAT,
        'int64': TensorProto.INT64,
        'bool': TensorProto.BOOL
    }
    inputs_value_info = [
        make_value_info(
            name=name,
            type_proto=make_tensor_type_proto(
                elem_type=dtype_map[tensor.dtype],
                shape=tensor.shape
            )
        ) for name, tensor in zip(node.input_names, inputs)
    ]
    outputs_value_info = [
        make_value_info(
            name=name,
            type_proto=make_tensor_type_proto(
                elem_type=dtype_map[tensor.dtype],
                shape=tensor.shape
            )
        ) for name, tensor in zip(node.output_names, hidet_outputs)
    ]
    graph = onnx.helper.make_graph(
        nodes=[node.node],
        name='test',
        inputs=inputs_value_info,
        outputs=outputs_value_info
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", opset) for opset in node.op_sets])
    # print(model)
    onnx.checker.check_model(model)
    # serialized_model = onnx._serialize(model)
    serialized_model = model.SerializeToString()
    session = onnxruntime.InferenceSession(serialized_model, providers=['CPUExecutionProvider'])
    outputs = session.run(node.output_names, input_feed={
        name: tensor.cpu().numpy() for name, tensor in zip(node.input_names, inputs)
    })
    return [hidet.array(output).cuda() for output in outputs]


class OnnxModule(nn.Module):
    def __init__(self, model):
        """
        Parameters
        ----------
        model: onnx.ModelProto
        """
        super().__init__()
        import onnx.numpy_helper
        import onnx.external_data_helper
        graph = model.graph

        # check operator set
        self.op_sets = []
        for opset_import in model.opset_import:
            if opset_import.domain not in ['', 'ai.onnx', 'ai.onnx.ml']:
                # we currently only support standard onnx operator domain
                raise ValueError('Onnx model imports unknown operator domain: {}'.format(repr(opset_import.domain)))
            self.op_sets.append(int(opset_import.version))
        self.op_sets = list(reversed(sorted(self.op_sets)))

        self.name: str = graph.name
        self.model = model
        for param in graph.initializer:
            numpy_array = onnx.numpy_helper.to_array(tensor=param)
            self.parameters[param.name] = from_numpy(numpy_array).cuda()
        self.input_names: List[str] = [input.name for input in graph.input if input.name not in self.parameters]
        self.output_names: List[str] = [output.name for output in graph.output]

        self.operators: List[OnnxOperator] = []
        self.operators: List[OnnxOperator] = [dispatch(node, op_sets=self.op_sets) for node in graph.node]
        self.usage_count: Dict[str, int] = self.count_usage()

    def forward(self, *args):
        name2tensor = {}
        assert len(args) == len(self.input_names)
        # parameters
        for name, param in self.parameters.items():
            name2tensor[name] = param
        # inputs
        for name, input in zip(self.input_names, args):
            name2tensor[name] = input
        # run nodes
        usage_count = self.usage_count.copy()
        for operator in self.operators:
            inputs = [name2tensor[name] for name in operator.input_names]
            outputs = operator.run(inputs)

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
            for name in operator.input_names:
                usage_count[name] -= 1
                if usage_count[name] == 0:
                    # free memory
                    del name2tensor[name]
        # put outputs
        results = [name2tensor[name] for name in self.output_names]
        if len(results) == 1:
            return results[0]
        else:
            return results

    def count_usage(self):
        usage_count = defaultdict(int)
        for op in self.operators:
            for input_name in op.input_names:
                usage_count[input_name] += 1
        for graph_output_name in self.output_names:
            usage_count[graph_output_name] += 1
        return usage_count


def from_onnx(model: Union[str, 'onnx.ModelProto']) -> OnnxModule:
    """
    Load an onnx model to hidet.tos.nn.Module.

    Parameters
    ----------
    model: Union[str, onnx.ModelProto]
        The path or model proto of given onnx model.

    Returns
    -------
    ret: OnnxModule
        The loaded model.
    """
    import onnx
    if isinstance(model, str):
        model = os.path.expanduser(model)
        model = onnx.load_model(model, load_external_data=False)
    try:
        onnx.checker.check_model(model, full_check=True)
    except ValueError:
        # ignore 'ValueError: Message onnx.ModelProto exceeds maximum protobuf size of 2GB'
        pass
    return OnnxModule(model)
