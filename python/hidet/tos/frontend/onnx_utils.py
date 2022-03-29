from typing import List, Union
import onnx
from onnx import numpy_helper
from hidet.tos import nn, ops
from hidet.tos.tensor import Tensor, from_numpy, randn


class OnnxOperator:
    def __init__(self, node: onnx.NodeProto):
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
            elif attr.type == 6:  # floats
                v = attr.floats
            elif attr.type == 7:  # ints
                v = attr.ints
            elif attr.type == 8:  # strings
                v = [s.decode('utf-8') for s in attr.strings]
            else:
                raise ValueError('Can not recognize type id {} of attribute {}'.format(attr.type, attr.name))
            self.attrs[attr.name] = v

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError()


class OnnxConv(OnnxOperator):
    def __init__(self, node: onnx.NodeProto):
        super().__init__(node)
        padding = self.attrs.get('pads', [0, 0, 0, 0])
        self.padding = [max(padding[0], padding[2]), max(padding[1], padding[3])]
        self.strides = [v for v in self.attrs.get('strides')]

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        output = ops.conv2d(inputs[0], inputs[1], self.padding, self.strides)
        if len(inputs) > 2:
            assert len(inputs) == 3
            bias = ops.unsqueeze(inputs[2], [0, 2, 3])
            output = output + bias
        print(output.shape)
        return [output]


class OnnxRelu(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.relu(inputs[0])]


class OnnxMaxPool(OnnxOperator):
    def __init__(self, node: onnx.NodeProto):
        super().__init__(node)
        padding = self.attrs.get('pads')
        self.kernel_size = list(self.attrs.get('kernel_shape'))
        self.padding = [max(padding[0], padding[2]), max(padding[1], padding[3])]
        self.strides = list(self.attrs.get('strides'))

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.max_pool2d(inputs[0], self.kernel_size, self.strides, self.padding)]


class OnnxReduceMean(OnnxOperator):
    def __init__(self, node: onnx.NodeProto):
        super().__init__(node)
        self.dims = list(self.attrs.get('axes'))
        self.keep_dim = self.attrs.get('keepdims') == 1

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.reduce_mean(inputs[0], self.dims, self.keep_dim)]


class OnnxSqueezeOp(OnnxOperator):
    def __init__(self, node: onnx.NodeProto):
        super().__init__(node)
        self.dims = list(self.attrs.get('axes'))

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.squeeze(inputs[0], self.dims)]


class OnnxAdd(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [inputs[0] + inputs[1]]


class OnnxMatMul(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.matmul(inputs[0], inputs[1])]


class OnnxSoftmax(OnnxOperator):
    def __init__(self, node: onnx.NodeProto):
        super().__init__(node)
        self.axis = self.attrs.get('axis')

    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        return [ops.softmax(inputs[0], self.axis)]


class OnnxArgMax(OnnxOperator):
    def run(self, inputs: List[Tensor]) -> List[Tensor]:
        # todo: support this op
        return inputs


def dispatch(node: onnx.NodeProto) -> OnnxOperator:
    dispatch_table = {
        'Conv': OnnxConv,
        'Relu': OnnxRelu,
        'MaxPool': OnnxMaxPool,
        'ReduceMean': OnnxReduceMean,
        'Squeeze': OnnxSqueezeOp,
        'Add': OnnxAdd,
        'MatMul': OnnxMatMul,
        'Softmax': OnnxSoftmax,
        'ArgMax': OnnxArgMax,
    }
    op_type = node.op_type
    if op_type not in dispatch_table:
        raise NotImplementedError("Operator '{}' from onnx has not been supported yet.".format(op_type))
    return dispatch_table[op_type](node)


class OnnxModule(nn.Module):
    def __init__(self, model: onnx.ModelProto):
        super().__init__()
        graph = model.graph
        self.name = graph.name
        self.graph = graph
        for param in graph.initializer:
            numpy_array = onnx.numpy_helper.to_array(tensor=param)
            self.parameters[param.name] = from_numpy(numpy_array).cuda()
        self.input_names = [input.name for input in graph.input]
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
            outputs = operator.run(inputs)
            assert len(outputs) == len(operator.output_names)
            for name, tensor in zip(operator.output_names, outputs):
                name2tensor[name] = tensor
        # put outputs
        results = [name2tensor[name] for name in self.output_names]
        if len(results) == 1:
            return results[0]
        else:
            return results


def from_onnx(model: Union[str, onnx.ModelProto]) -> nn.Module:
    if isinstance(model, str):
        model = onnx.load_model(model)
    return OnnxModule(model)


def main():
    model = onnx.load_model('/home/yaoyao/model_zoo/resnet50_v1.onnx')
    # model = onnx.load_model('/home/yaoyao/model_zoo/resnet50-v2-7.onnx')
    x = randn([1, 3, 224, 224])
    module = OnnxModule(model)
    y = module(x)
    print(module)
    print(y[0])
    print(y[1])
    # y = module(x)
    # print(type(model))
    # print(dir(model))


if __name__ == '__main__':
    main()
