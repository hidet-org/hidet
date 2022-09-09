from typing import Any, Union, Sequence, List

import torch
from torch import autograd
from hidet import ops, jit
from hidet.graph.tensor import from_torch, Tensor
from torch.autograd.function import FunctionCtx, Function


@jit(opt=True)
def conv2d(data: Tensor,
           weight: Tensor,
           strides: Union[int, Sequence[int]],
           padding: Union[int, Sequence[int]],
           groups: int = 1):
    data = ops.conv_pad(data, padding)
    return ops.conv2d(data, weight, strides, groups)


class Conv2d(autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx,
                data: torch.Tensor,
                weight: torch.Tensor,
                strides, padding, groups=1):
        # view the torch tensor as hidet tensor
        data = from_torch(data)
        weight = from_torch(weight)
        # create the flow graph
        flow_graph = conv2d.flow_graph_for(data, weight, strides, padding, groups)
        # the symbolic hidet output tensors
        outputs: List[Tensor] = flow_graph.dummy_outputs()
        # the torch tensor output
        torch_outputs: List[torch.Tensor] = [output.torch() for output in outputs]
        # view the torch output tensors as hidet tensors
        shared_outputs: List[Tensor] = [from_torch(output) for output in torch_outputs]
        # run the flow graph with hidet view tensors (the actual memory allocated by torch)
        flow_graph.pure_forward(inputs=[data, weight], outputs=shared_outputs)
        assert len(torch_outputs) == 1
        return torch_outputs[0]

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor):
        raise NotImplementedError()


if __name__ == '__main__':
    data: torch.Tensor = torch.randn([1, 3, 12, 12]).cuda()
    weight = torch.randn([1, 3, 3, 3]).cuda()
    data.requires_grad = True
    weight.requires_grad = True
    output: torch.Tensor = Conv2d.apply(data, weight, 1, 0)
    s = output.sum()
    print(data)
    print(weight)
    print(output)
    s.backward()
