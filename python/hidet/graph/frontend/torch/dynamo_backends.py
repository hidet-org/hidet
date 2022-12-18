# pylint: disable=no-name-in-module
from typing import List, Callable, Sequence
import logging
import torch
import hidet.option
from hidet.graph.ir.flow_graph import FlowGraph
from hidet.graph.transforms import PassContext, optimize
from .utils import serialize_output, deserialize_output


logger = logging.getLogger(__name__)

"""
The schedule search space for the operator kernel tuning
Candidates are: 0, 1, 2
 - 0: Use the default schedule, without tuning
 - 1: Tune the schedule in a small search space, usually takes less than one minute to tune a kernel
 - 2: Tune the schedule in a large search space, achieves the best performance, but takes longer time to tune a kernel
"""
search_space: int = 2

"""
Parallelization on k dimension of the matrix multiplication
Candidates are: 'default', 'disabled', 'search'
 - 'default':
    Default parallelization strategy. A heuristic strategy is used to decide whether to parallelize on k dimension and
    the size of split factor
 - 'disabled':
    Disable parallelization on k dimension
 - 'search':
    Search for the best parallelization strategy. The most expensive option but usually achieves the best performance.
"""
parallel_k: str = 'default'

"""
Whether to use float16 data type
"""
use_fp16 = False
use_fp16_reduction = False

"""
Whether to use cuda graph
"""
use_cuda_graph = True


def generate_executor(flow_graph: FlowGraph) -> Callable:
    from hidet.runtime import CudaGraph

    with PassContext() as ctx:
        if use_fp16:
            ctx.set_precision('float16')
        if use_fp16 and use_fp16_reduction:
            ctx.set_reduce_precision('float16')
        ctx.set_parallel_k(disabled=(parallel_k == 'disabled'), search=(parallel_k == 'search'))
        graph_opt: FlowGraph = optimize(flow_graph)

    if use_cuda_graph:
        with hidet.option.context():
            hidet.option.search_space(search_space)
            cuda_graph: CudaGraph = graph_opt.cuda_graph()

        def run(*inputs: torch.Tensor):
            hidet_inputs: List[hidet.Tensor] = [hidet.from_torch(tensor) for tensor in inputs]
            cuda_graph.set_input_tensors(hidet_inputs)
            cuda_graph.run()
            hidet_outputs: List[hidet.Tensor] = cuda_graph.get_output_tensors()
            torch_outputs: List[torch.Tensor] = [tensor.torch() for tensor in hidet_outputs]
            return torch_outputs

    else:
        with hidet.option.context():
            hidet.option.search_space(search_space)
            dummy_inputs = flow_graph.dummy_inputs()
            graph_opt(*dummy_inputs)

        def run(*inputs: torch.Tensor):
            hidet_inputs: List[hidet.Tensor] = [hidet.from_torch(tensor) for tensor in inputs]
            hidet_outputs: List[hidet.Tensor] = graph_opt(*hidet_inputs)
            torch_outputs: List[torch.Tensor] = [tensor.torch() for tensor in hidet_outputs]
            return torch_outputs

    return run


def onnx2hidet_backend(subgraph):
    from torch._dynamo.optimizations.subgraph import SubGraph

    assert isinstance(subgraph, SubGraph)
    from hidet.runtime import CudaGraph
    from hidet.graph import nn

    if not subgraph.is_cuda:
        # fallback to the default backend
        logger.warning('onnx2hidet: fallback to the default backend as the subgraph is not on CUDA')
        return subgraph.model

    onnx_module: nn.Module = hidet.graph.frontend.from_onnx(subgraph.onnx_filename)
    example_inputs: List[hidet.Tensor] = [hidet.from_torch(tensor) for tensor in subgraph.example_inputs]
    symbolic_inputs: List[hidet.Tensor] = [hidet.symbol_like(tensor) for tensor in example_inputs]
    symbolic_outputs = onnx_module(*symbolic_inputs)
    flow_graph: FlowGraph = hidet.trace_from(symbolic_outputs, inputs=symbolic_inputs)
    with hidet.graph.PassContext() as ctx:
        if use_fp16:
            ctx.set_precision('float16')
        if use_fp16 and use_fp16_reduction:
            ctx.set_reduce_precision('float16')
        ctx.set_parallel_k(disabled=(parallel_k == 'disabled'), search=(parallel_k == 'search'))
        graph_opt: FlowGraph = hidet.graph.optimize(flow_graph)

    if use_cuda_graph:
        with hidet.option.context():
            hidet.option.search_space(search_space)
            cuda_graph: CudaGraph = graph_opt.cuda_graph()

        def run(*inputs: torch.Tensor):
            hidet_inputs: List[hidet.Tensor] = [hidet.from_torch(tensor) for tensor in inputs]
            cuda_graph.set_input_tensors(hidet_inputs)
            cuda_graph.run()
            hidet_outputs: List[hidet.Tensor] = cuda_graph.get_output_tensors()
            torch_outputs: List[torch.Tensor] = [tensor.torch() for tensor in hidet_outputs]
            return torch_outputs

    else:
        with hidet.option.context():
            hidet.option.search_space(search_space)
            graph_opt(*example_inputs)

        def run(*inputs: torch.Tensor):
            hidet_inputs: List[hidet.Tensor] = [hidet.from_torch(tensor) for tensor in inputs]
            hidet_outputs: List[hidet.Tensor] = graph_opt(*hidet_inputs)
            torch_outputs: List[torch.Tensor] = [tensor.torch() for tensor in hidet_outputs]
            return torch_outputs

    return subgraph.wrap_returns(run)


def hidet_backend(subgraph):
    from hidet import Tensor
    from torch._dynamo.optimizations.subgraph import SubGraph
    from .interpreter import ImportedTorchModule
    from .utils import symbol_like_torch

    assert isinstance(subgraph, SubGraph)

    symbolic_inputs: List[Tensor] = []
    for example_input in subgraph.example_inputs:
        if isinstance(example_input, torch.Tensor):
            symbolic_inputs.append(symbol_like_torch(example_input))
        else:
            raise ValueError('hidet_backend: only support torch.Tensor as example input')

    assert isinstance(subgraph.model, torch.fx.GraphModule)
    graph_module: torch.fx.GraphModule = subgraph.model
    imported_module: ImportedTorchModule = hidet.frontend.from_torch(graph_module)
    output = imported_module(*symbolic_inputs)
    output_format, output_tensors = serialize_output(output)
    flow_graph: FlowGraph = hidet.trace_from(output_tensors, inputs=symbolic_inputs)
    executor = generate_executor(flow_graph)

    def wrapper(*args: Tensor):
        outputs: Sequence[torch.Tensor] = executor(*args)
        ret = deserialize_output(output_format, outputs)
        return ret[0]

    return wrapper


def register_dynamo_backends():
    from torch._dynamo.optimizations.backends import create_backend

    onnx2hidet_backend.__name__ = 'onnx2hidet'
    create_backend(onnx2hidet_backend)

    hidet_backend.__name__ = 'hidet'
    create_backend(hidet_backend)
