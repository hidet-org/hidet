from typing import List
import onnx
from onnx import numpy_helper
from hidet.tos import nn, ops
from hidet.tos.tensor import Tensor, from_numpy, randn
import tvm.relay.frontend.onnx


class OnnxModule(nn.Module):
    def __init__(self, model: onnx.ModelProto):
        super().__init__()
        graph = model.graph
        self.name = graph.name
        self.graph = graph
        for param in graph.initializer:
            numpy_array = numpy_helper.to_array(tensor=param)
            self.parameters[param.name] = from_numpy(numpy_array).cuda()
        self.input_names = [input.name for input in graph.input]
        self.output_names = [output.name for output in graph.output]
        self.nodes = [node for node in graph.node]

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
        for node in self.nodes:
            inputs = [name2tensor[name] for name in node.input]
            outputs = self.run_onnx_op(inputs, node)
            assert len(outputs) == len(node.output)
            for name, tensor in zip(node.output, outputs):
                name2tensor[name] = tensor
        # put outputs
        results = [name2tensor[name] for name in self.output_names]
        if len(results) == 1:
            return results[0]
        else:
            return results

    @staticmethod
    def run_onnx_op(inputs: List[Tensor], node: onnx.NodeProto) -> List[Tensor]:
        op_type = node.op_type
        attr = node.attribute
        if op_type == 'Conv':
            assert node.auto_pad == 'NOTSET'
            padding = node.pads
            assert len(padding) == 4 and padding[0] == padding[2] and padding[1] == padding[3]
            padding = [padding[0], padding[1]]
            stride = node.strides
            y = ops.conv2d(inputs[0], inputs[1], padding, stride)
            if len(inputs) == 3:
                y = y + inputs[3]
            return y
        else:
            raise NotImplementedError('Operator {} from onnx has not been supported yet.'.format(op_type))


def main():
    model = onnx.load_model('/home/yaoyao/model_zoo/resnet50_v1.onnx')
    x = randn([1, 3, 224, 224])
    module = OnnxModule(model)
    y = module(x)
    print(type(model))
    print(dir(model))


if __name__ == '__main__':
    main()
