# pylint: disable=no-name-in-module
from typing import List, Callable, Sequence
import logging
import torch
import hidet.option
from hidet.ir.type import data_type
from hidet.graph.ir.flow_graph import FlowGraph
from hidet.graph.transforms import PassContext, optimize
from .utils import serialize_output, deserialize_output


logger = logging.getLogger(__name__)


class DynamoConfig:
    def __init__(self):
        self._search_space: int = 0
        self._parallel_k: str = 'default'
        self._use_fp16: bool = False
        self._use_fp16_reduction: bool = False
        self._use_cuda_graph: bool = True
        self._print_input_graph: bool = False
        self._correctness_report: bool = False

    def __getitem__(self, item: str):
        assert isinstance(item, str)
        return getattr(self, f"_{item}")

    def search_space(self, level: int = 2):
        """
        The schedule search space for the operator kernel tuning
        Candidates are: 0, 1, 2
         - 0: Use the default schedule, without tuning.
         - 1: Tune the schedule in a small search space. Usually takes less than one minute to tune a kernel.
         - 2: Tune the schedule in a large search space. Usually achieves the best performance, but takes longer time.
        """
        self._search_space = level
        return self

    def parallel_k(self, strategy="default"):
        """
        Parallelization on k dimension of the matrix multiplication
        Candidates are: 'default', 'disabled', 'search'
         - 'default':
            Default parallelization strategy. A heuristic strategy is used to decide whether to parallelize on k
            dimension and the size of split factor
         - 'disabled':
            Disable parallelization on k dimension
         - 'search':
            Search for the best parallelization strategy. Takes more time but usually achieves the best performance.
        """
        self._parallel_k = strategy

    def use_fp16(self, flag=True):
        """
        Whether to use float16 data type
        """
        self._use_fp16 = flag
        return self

    def use_fp16_reduction(self, flag=True):
        """
        Whether to use float16 data type for reduction
        """
        self._use_fp16_reduction = flag
        return self

    def use_cuda_graph(self, flag=True):
        """
        Whether to use cuda graph
        """
        self._use_cuda_graph = flag
        return self

    def print_input_graph(self, flag=True):
        """
        Whether to print the input graph
        """
        self._print_input_graph = flag
        return self

    def correctness_report(self, flag=True):
        """
        Whether to check correctness and print report error
        """
        self._correctness_report = flag
        return self


dynamo_config = DynamoConfig()


def generate_executor(flow_graph: FlowGraph) -> Callable:
    from hidet.runtime import CudaGraph

    use_fp16 = dynamo_config['use_fp16']
    use_fp16_reduction = dynamo_config['use_fp16_reduction']
    use_cuda_graph = dynamo_config['use_cuda_graph']
    search_space = dynamo_config['search_space']
    parallel_k = dynamo_config['parallel_k']

    with PassContext() as ctx:
        if use_fp16:
            ctx.set_precision('float16')
        if use_fp16 and use_fp16_reduction:
            ctx.set_reduce_precision('float16')
        ctx.set_parallel_k(disabled=(parallel_k == 'disabled'), search=(parallel_k == 'search'))
        logger.info('start to optimize the flow graph')
        graph_opt: FlowGraph = optimize(flow_graph)
        logger.info('finish optimizing the flow graph')

    logger.info('schedule search space: %d', search_space)

    has_cpu_tensor = any(tensor.device == 'cpu' for tensor in graph_opt.inputs + graph_opt.outputs)
    has_cuda_tensor = any(tensor.device == 'cuda' for tensor in graph_opt.inputs + graph_opt.outputs)

    if has_cpu_tensor and has_cuda_tensor:
        raise RuntimeError('the flow graph contains both CPU and CUDA tensors, currently not supported by hidet')

    if use_cuda_graph and not has_cpu_tensor:
        with hidet.option.context():
            hidet.option.search_space(search_space)
            logger.info('start to generate the cuda graph')
            cuda_graph: CudaGraph = graph_opt.cuda_graph()
            logger.info('finish generating the cuda graph')

        def run(*inputs: torch.Tensor):
            hidet_inputs: List[hidet.Tensor] = [hidet.from_torch(tensor) for tensor in inputs]
            cuda_graph.set_input_tensors(hidet_inputs)
            cuda_graph.run()
            hidet_outputs: List[hidet.Tensor] = cuda_graph.get_output_tensors()
            torch_outputs: List[torch.Tensor] = [tensor.torch() for tensor in hidet_outputs]
            return torch_outputs

    else:
        logger.info('start to generate the executor without cuda graph')
        with hidet.option.context():
            hidet.option.search_space(search_space)
            dummy_inputs = flow_graph.dummy_inputs()
            graph_opt(*dummy_inputs)
        logger.info('finish generating the executor without cuda graph')

        def run(*inputs: torch.Tensor):
            hidet_inputs: List[hidet.Tensor] = [hidet.from_torch(tensor) for tensor in inputs]
            hidet_outputs: List[hidet.Tensor] = graph_opt(*hidet_inputs)
            torch_outputs: List[torch.Tensor] = [tensor.torch() for tensor in hidet_outputs]
            return torch_outputs

    return run


def onnx2hidet_backend(subgraph):
    from torch._dynamo.optimizations.subgraph import SubGraph

    assert isinstance(subgraph, SubGraph)
    from hidet.graph import nn

    if not subgraph.is_cuda:
        # fallback to the default backend
        logger.warning('fallback to the default backend as the subgraph is not on CUDA')
        return subgraph.model

    onnx_module: nn.Module = hidet.graph.frontend.from_onnx(subgraph.onnx_filename)
    example_inputs: List[hidet.Tensor] = [hidet.from_torch(tensor) for tensor in subgraph.example_inputs]
    symbolic_inputs: List[hidet.Tensor] = [hidet.symbol_like(tensor) for tensor in example_inputs]
    symbolic_outputs = onnx_module(*symbolic_inputs)
    flow_graph: FlowGraph = hidet.trace_from(symbolic_outputs, inputs=symbolic_inputs)
    return subgraph.wrap_returns(generate_executor(flow_graph))


def hidet_backend(subgraph):
    from hidet import Tensor
    from torch._dynamo.optimizations.subgraph import SubGraph
    from .interpreter import Interpreter
    from .utils import symbol_like_torch

    assert isinstance(subgraph, SubGraph)

    logger.info('received a subgraph with %d nodes to optimize', len(subgraph.model.graph.nodes))
    logger.debug('graph: %s', subgraph.model.graph)

    if dynamo_config['print_input_graph']:
        subgraph.model.graph.print_tabular()

    # get the interpreter for the subgraph
    assert isinstance(subgraph.model, torch.fx.GraphModule)
    graph_module: torch.fx.GraphModule = subgraph.model
    interpreter: Interpreter = hidet.frontend.from_torch(graph_module)

    # prepare dummy and symbolic inputs for correctness and flow graph construction
    symbolic_inputs: List[Tensor] = []  # for flow graph construction
    for example_input in subgraph.example_inputs:
        if isinstance(example_input, torch.Tensor):
            symbolic_input = symbol_like_torch(example_input)
            symbolic_inputs.append(symbolic_input)
        else:
            raise ValueError('hidet_backend: only support torch.Tensor as example input')

    if dynamo_config['correctness_report']:
        # check correctness using random inputs
        logger.info('start to check correctness')
        dummy_inputs: List[Tensor] = []  # for correctness check
        for symbolic_input in symbolic_inputs:
            if data_type(symbolic_input.dtype).is_integer():
                dummy_input = hidet.zeros_like(symbolic_input)
            else:
                dummy_input = hidet.randn_like(symbolic_input)
            dummy_inputs.append(dummy_input)
        report: str = interpreter.forward_with_check(*dummy_inputs)
        logger.info('finish checking correctness')
        print(report)

    logger.info('hidet: symbolic inputs: ')
    for symbolic_input in symbolic_inputs:
        logger.info('hidet:   %s', symbolic_input.signature())

    # symbolic run to get flow graph
    output = interpreter(*symbolic_inputs)
    output_format, output_tensors = serialize_output(output)
    flow_graph: FlowGraph = hidet.trace_from(output_tensors, inputs=symbolic_inputs)

    executor = generate_executor(flow_graph)

    def wrapper(*args: Tensor):
        outputs: Sequence[torch.Tensor] = executor(*args)
        ret = deserialize_output(output_format, outputs)
        return ret

    logger.info('finish generating the executor')

    return wrapper


def register_dynamo_backends():
    from torch._dynamo.optimizations.backends import create_backend

    onnx2hidet_backend.__name__ = 'onnx2hidet'
    create_backend(onnx2hidet_backend)

    hidet_backend.__name__ = 'hidet'
    create_backend(hidet_backend)
