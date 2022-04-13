from typing import List, Union, Sequence, Optional
import os
import numpy as np
import hidet
from hidet.tos import nn
from hidet.tos import operators as ops
from hidet.tos.tensor import Tensor, from_numpy, randn
from hidet.utils import line_profile, prod

"""
Please refers to https://github.com/onnx/onnx/blob/main/docs/Operators.md when adding new operators.
"""


class OnnxOperator:
    def __init__(self, node):
        """
        Parameters
        ----------
        node: onnx.NodeProto
        """
        import onnx.numpy_helper
        self.node = node
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
        raise NotImplementedError()

    @staticmethod
    def tensor2list(tensor: Tensor) -> List:
        return tensor.cpu().numpy().tolist()


class OnnxConv(OnnxOperator):
    def __init__(self, node):
        super().__init__(node)
        self.padding = self.attrs.get('pads', [0, 0, 0, 0])
        self.strides = self.attrs.get('strides')

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        output = ops.conv2d(inputs[0], inputs[1], self.padding, self.strides)
        if len(inputs) > 2:
            assert len(inputs) == 3
            bias = ops.unsqueeze(inputs[2], [0, 2, 3])
            output = output + bias
        return [output]


class OnnxBatchNormalization(OnnxOperator):
    def __init__(self, node):
        super().__init__(node)
        self.epsilon: float = self.attrs.get('epsilon', 1e-5)
        self.momentum: float = self.attrs.get('momentum', 0.9)
        self.training_mode: int = self.attrs.get('training_mode', 0)
        assert self.training_mode == 0, 'BatchNorm in training mode occurs, currently, hidet does not support training.'

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        x, scale, bias, running_mean, running_var = inputs
        y = ops.batch_norm_infer(x, running_mean=running_mean, running_var=running_var, epsilon=self.epsilon, axis=1)
        return [y * scale.unsqueeze([0, 2, 3]) + bias.unsqueeze([0, 2, 3])]


class OnnxRelu(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.relu(inputs[0])]


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
    def __init__(self, node):
        super().__init__(node)
        self.kernel_size = list(self.attrs.get('kernel_shape'))
        self.padding = list(self.attrs.get('pads', [0, 0, 0, 0]))
        self.strides = list(self.attrs.get('strides'))

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.max_pool2d(inputs[0], self.kernel_size, self.strides, self.padding)]


class OnnxReduceMean(OnnxOperator):
    def __init__(self, node):
        super().__init__(node)
        self.dims = self.attrs.get('axes')
        self.keep_dim = self.attrs.get('keepdims', 1) == 1

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.reduce_mean(inputs[0], self.dims, self.keep_dim)]


class OnnxSqueezeOp(OnnxOperator):
    def __init__(self, node):
        super().__init__(node)
        self.dims = list(self.attrs.get('axes'))

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.squeeze(inputs[0], self.dims)]


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
            if self.can_squeeze(a.shape) and self.can_squeeze(b.shape):
                assert isinstance(a, Tensor)
                c_rank = max(len(a.shape), len(b.shape))
                a = a.squeeze(range(len(a.shape) - 2)) if len(a.shape) > 2 else a
                b = b.squeeze(range(len(b.shape) - 2)) if len(b.shape) > 2 else b
                c = ops.matmul(a, b)
                if c_rank > len(c.shape):
                    c = c.unsqueeze(dims=range(c_rank - len(c.shape)))
                return [c]
            else:
                prefix_shape = hidet.tos.operators.basic.arithmatic.broadcast_shape(a.shape[:-2], b.shape[:-2])
                c_shape = prefix_shape + [a.shape[-2]] + [b.shape[-1]]
                print('fake batched matmul')
                return [hidet.empty(c_shape)]
                # raise NotImplementedError('Matmul with shapes {} and {}'.format(a.shape, b.shape))

    @staticmethod
    def can_squeeze(shape: List[int]) -> bool:
        return len(shape) == 2 or (len(shape) > 2 and prod(shape[:-2]) == 1)


class OnnxSoftmax(OnnxOperator):
    def __init__(self, node):
        super().__init__(node)
        self.axis = self.attrs.get('axis')

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        # print('fake softmax')
        # return [randn(inputs[0].shape)]
        return [ops.softmax(inputs[0], self.axis)]


class OnnxGlobalAveragePool(OnnxOperator):
    def __init__(self, node):
        super().__init__(node)

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        x, = inputs
        n, c, h, w = x.shape
        return [ops.avg_pool2d(x, kernel=(h, w), stride=(1, 1), padding=(0, 0))]


class OnnxFlatten(OnnxOperator):
    def __init__(self, node):
        super().__init__(node)
        self.axis = self.attrs.get('axis', 1)

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        x = inputs[0]
        rank = len(x.shape)
        axis = (self.axis + rank) % rank
        dims = list(range(rank))
        return [ops.rearrange(x, plan=[dims[:axis], dims[axis:]])]


class OnnxUnsqueeze(OnnxOperator):
    def __init__(self, node):
        super().__init__(node)
        self.axes = self.attrs.get('axes')

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.unsqueeze(inputs[0], self.axes)]


class OnnxReshape(OnnxOperator):
    def __init__(self, node):
        super().__init__(node)
        self.allow_zero = self.attrs.get('allowzero', 0)

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        x, shape = inputs
        shape = self.tensor2list(shape)
        return [ops.reshape(x, shape)]


class OnnxTranspose(OnnxOperator):
    def __init__(self, node):
        super().__init__(node)
        self.perm = self.attrs.get('perm', None)

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        x = inputs[0]
        perm = self.perm if self.perm else list(reversed(range(len(x.shape))))
        return [ops.transpose(x, perm)]


class OnnxConcat(OnnxOperator):
    def __init__(self, node):
        super().__init__(node)
        self.axis = self.attrs.get('axis')

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.concat(inputs, self.axis)]


class OnnxArgMax(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError('ArgMax')


class OnnxGemm(OnnxOperator):
    def __init__(self, node):
        super().__init__(node)
        self.alpha = self.attrs.get('alpha', 1.0)
        self.beta = self.attrs.get('beta', 0.0)
        self.trans_a = self.attrs.get('transA', 0)
        self.trans_b = self.attrs.get('transB', 0)

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        a, b = inputs[:2]
        c = inputs[2] if len(inputs) > 2 else None

        if self.trans_a == 1:
            a = ops.rearrange(a, plan=[[1], [0]])
        if self.trans_b == 1:
            b = ops.rearrange(b, plan=[[1], [0]])
        assert a.shape[1] == b.shape[0]
        d = ops.matmul(a, b)
        if self.alpha != 1.0:
            d = d * self.alpha
        if c and self.beta != 0.0:
            d = d + c * self.beta
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

    def __init__(self, node):
        super().__init__(node)
        self.to = self.attrs.get('to')

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        x = inputs[0]
        dtype = self.code2dtype[self.to]
        return [ops.cast(x, dtype)]


class OnnxShape(OnnxOperator):
    def __init__(self, node):
        super().__init__(node)
        self.start = self.attrs.get('start', 0)
        self.end: Optional[int] = self.attrs.get('end', None)

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        x = inputs[0]
        rank = len(x.shape)
        start = self.start + rank if self.start < 0 else self.start
        if self.end is not None:
            end = self.end + rank if self.end < 0 else self.end
        else:
            end = rank
        start = max(min(start, rank), 0)
        end = max(min(end, rank), 0)
        return [hidet.array(x.shape[start:end]).cuda()]


class OnnxConstant(OnnxOperator):
    def __init__(self, node):
        super().__init__(node)
        self.value = self.attrs.get('value')
        if self.value is None:
            raise NotImplementedError('Currently, only support Tensor constant in onnx importer')

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        assert len(inputs) == 0
        return [self.value]


class OnnxGather(OnnxOperator):
    def __init__(self, node):
        super().__init__(node)
        self.axis = self.attrs.get('axis', 0)

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        data, indices = inputs
        return [ops.take(data, indices, self.axis)]


class OnnxSlice(OnnxOperator):
    def __init__(self, node):
        super().__init__(node)

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        data, starts, ends = inputs[:3]
        axes = inputs[3] if len(inputs) > 3 else None
        steps = inputs[4] if len(inputs) > 4 else None
        starts = self.tensor2list(starts)
        ends = self.tensor2list(ends)
        axes = self.tensor2list(axes) if axes else None
        steps = self.tensor2list(steps) if steps else None
        return [ops.strided_slice(data, starts, ends, axes, steps)]


def dispatch(node) -> OnnxOperator:
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
    }
    op_type = node.op_type
    if op_type not in dispatch_table:
        raise NotImplementedError("Operator '{}' from onnx has not been supported yet.".format(op_type))
    return dispatch_table[op_type](node)


class OnnxModule(nn.Module):
    def __init__(self, model):
        """
        Parameters
        ----------
        model: onnx.ModelProto
        """
        super().__init__()
        import onnx.numpy_helper
        graph = model.graph
        self.name = graph.name
        self.graph = graph
        for param in graph.initializer:
            numpy_array = onnx.numpy_helper.to_array(tensor=param)
            self.parameters[param.name] = from_numpy(numpy_array).cuda()
        self.input_names = [input.name for input in graph.input if input.name not in self.parameters]
        self.output_names = [output.name for output in graph.output]
        self.operators = [dispatch(node) for node in graph.node]

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
        for operator in self.operators:
            inputs = [name2tensor[name] for name in operator.input_names]
            print('{:>20}: '.format(operator.node.name), end='')
            outputs = operator.run(inputs)
            print('{}'.format(', '.join(out.signature() for out in outputs)))
            assert len(outputs) == len(operator.output_names)
            for name, tensor in zip(operator.output_names, outputs):
                name2tensor[name] = tensor
        # put outputs
        results = [name2tensor[name] for name in self.output_names]
        if len(results) == 1:
            return results[0]
        else:
            return results


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
        model = onnx.load_model(model)
    return OnnxModule(model)
