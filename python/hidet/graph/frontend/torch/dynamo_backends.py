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
# pylint: disable=no-name-in-module
from typing import List, Callable, Sequence, Union
import logging
import torch
import hidet.option
from hidet.ir.type import DataType
from hidet.ir.expr import SymbolVar
from hidet.graph.flow_graph import FlowGraph
from hidet.graph.transforms import PassContext, optimize
from hidet.runtime import CompiledGraph
from hidet.cuda.graph import CudaGraphCreationError
from hidet.ir import dtypes
from .utils import serialize_output, deserialize_output, resolve_save_dir_multigraph
from .dynamo_config import dynamo_config


logger = logging.getLogger(__name__)


def generate_executor(flow_graph: FlowGraph) -> Callable:
    use_fp16 = dynamo_config['use_fp16']
    use_fp16_reduction = dynamo_config['use_fp16_reduction']
    use_cuda_graph = dynamo_config['use_cuda_graph']
    use_attention = dynamo_config['use_attention']
    search_space = dynamo_config['search_space']
    parallel_k = dynamo_config['parallel_k']
    tensor_core = dynamo_config['use_tensor_core']
    save_dir = dynamo_config['dump_graph_ir']

    with PassContext() as ctx:
        if use_fp16:
            ctx.set_precision('float16')
        if use_fp16 and use_fp16_reduction:
            ctx.set_reduce_precision('float16')
        ctx.set_use_attention(use_attention)
        if save_dir:
            graph_dir = resolve_save_dir_multigraph(save_dir)
            ctx.save_graph_instrument(graph_dir)
        if tensor_core:
            ctx.set_mma('mma' if tensor_core else 'simt')
        ctx.set_parallel_k(disabled=(parallel_k == 'disabled'), search=(parallel_k == 'search'))
        logger.info('start to optimize the flow graph')
        graph_opt: FlowGraph = optimize(flow_graph)
        logger.info('finish optimizing the flow graph')

    logger.info('schedule search space: %d', search_space)

    def preprocess_inputs(inputs: Sequence[torch.Tensor]) -> List[hidet.Tensor]:
        torch_inputs: List[torch.Tensor] = []
        for x in inputs:
            if not x.is_contiguous():
                # warnings.warn_once('Hidet received a non-contiguous torch input tensor, converting it to contiguous')
                x = x.contiguous()
            torch_inputs.append(x)
        hidet_inputs: List[hidet.Tensor] = [hidet.from_torch(tensor) for tensor in torch_inputs]
        return hidet_inputs

    logger.info('start to build the optimized computation graph')
    cgraph: CompiledGraph = graph_opt.build(space=search_space)
    logger.info('finish building computation graph')

    if use_cuda_graph:
        try:
            runner = cgraph.cuda_graph()
        except CudaGraphCreationError:
            runner = cgraph
    else:
        runner = cgraph

    def run(*inputs: torch.Tensor):
        hidet_inputs = preprocess_inputs(inputs)
        hidet_outputs: List[hidet.Tensor] = runner.run_async(hidet_inputs)
        torch_outputs: List[torch.Tensor] = [tensor.torch() for tensor in hidet_outputs]
        return torch_outputs

    return run


def hidet_backend(graph_module, example_inputs):
    from hidet import Tensor
    from .interpreter import Interpreter
    from .utils import symbol_like_torch

    assert isinstance(graph_module, torch.fx.GraphModule)

    logger.info('received a subgraph with %d nodes to optimize', len(graph_module.graph.nodes))
    logger.debug('graph: %s', graph_module.graph)

    if dynamo_config['print_input_graph']:
        graph_module.print_readable()
        print('---')
        graph_module.graph.print_tabular()

    # get the interpreter for the subgraph
    interpreter: Interpreter = hidet.frontend.from_torch(graph_module)

    # prepare dummy and symbolic inputs for correctness and flow graph construction
    inputs: List[Union[Tensor, SymbolVar, int, bool, float]] = []  # for flow graph construction
    for example_input in example_inputs:
        if isinstance(example_input, torch.Tensor):
            symbolic_input = symbol_like_torch(example_input)
            inputs.append(symbolic_input)
        elif isinstance(example_input, (int, bool, float)):
            inputs.append(example_input)
        elif isinstance(example_input, torch.SymInt):
            from torch.fx.experimental.symbolic_shapes import SymNode

            node: SymNode = example_input.node
            try:
                inputs.append(node.pytype(example_input))
            except RuntimeError:
                # is a symbolic scalar input
                pytype2dtype = {int: dtypes.int32, float: dtypes.float32, bool: dtypes.boolean}
                inputs.append(hidet.symbol_var(name=str(example_input), dtype=pytype2dtype[node.pytype]))
        else:
            raise ValueError(f'hidet_backend: unexpected example input {example_input}, type {type(example_input)}')

    if dynamo_config['correctness_report']:
        # check correctness using random inputs
        def wrapper(*args):
            report, output = interpreter.forward_with_check(*args)
            logger.info('finish checking correctness')
            print(report)
            return output

        return wrapper
    else:
        logger.info('hidet:   inputs: ')
        for arg in inputs:
            if isinstance(arg, hidet.Tensor):
                logger.info('hidet:   %s', arg.signature())
            else:
                logger.info('hidet:   %s', arg)

        # symbolic run to get flow graph
        output = interpreter(*inputs)
        output_format, output_tensors = serialize_output(output)
        input_tensors = [x for x in inputs if isinstance(x, hidet.Tensor)]
        flow_graph: FlowGraph = hidet.trace_from(output_tensors, inputs=input_tensors)

        executor = generate_executor(flow_graph)

        def wrapper(*args: Tensor):
            tensor_args = []
            for param, arg in zip(inputs, args):
                if isinstance(param, Tensor):
                    tensor_args.append(arg)
                elif isinstance(param, SymbolVar):
                    dtype = param.type
                    assert isinstance(dtype, DataType)
                    if dtype.name == 'int32':
                        from hidet.ffi import runtime_api

                        runtime_api.set_symbol_value(param.name, int(arg))
                    else:
                        raise ValueError(
                            f'hidet_backend: unsupported symbolic dtype {dtype}. We only support int32 now.'
                        )
                else:
                    # ignore constant
                    pass
            outputs: Sequence[torch.Tensor] = executor(*tensor_args)
            ret = deserialize_output(output_format, outputs)
            return ret

        logger.info('finish generating the executor')

        return wrapper
