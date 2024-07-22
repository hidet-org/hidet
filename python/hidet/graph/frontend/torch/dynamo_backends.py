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
from typing import List, Sequence, Union
import logging
import torch
import hidet.option
from hidet import Tensor
from hidet.ir.type import DataType
from hidet.ir.expr import SymbolVar
from hidet.runtime import CompiledGraph
from hidet.graph.flow_graph import FlowGraph
from hidet.graph.transforms import PassContext, optimize
from hidet.cuda.graph import CudaGraphCreationError
from hidet.ffi import runtime_api
from .dynamo_config import dynamo_config
from .interpreter import Interpreter
from .utils import serialize_output, deserialize_output, resolve_save_dir_multigraph
from .utils import symbol_like_torch

logger = logging.getLogger(__name__)

# NOTES ABOUT DYNAMIC SHAPE.
# From pytorch we got two argument:
#   - fxgraph
#   - example_inputs
# In case when we are requested to create dynamic shape, `example_inputs` contain info
# about used symbols only (all symbols are presented in `example_input` as element of list).
# But in `example_inputs` there is no information about what dimentions of input tensors
# should be symbolic and correspondence between symbol and dimention.
# These info is presented in fxgraph. Every input corresponds fxgraph node.
# in `fx_node.meta['example_value']` stored `FakeTensor` that contain all symbols in its shape.
# We use this data to determinate shapes of the inputs.
def get_flow_graph(interpreter: Interpreter, example_inputs):
    inputs: List[Union[Tensor, SymbolVar]] = []  # for flow graph construction
    for fxgraph_node, example_input in zip(interpreter.graph.nodes, example_inputs):
        if isinstance(example_input, torch.Tensor):
            fake_input = fxgraph_node.meta['example_value']
            symbolic_input = symbol_like_torch(fake_input)
            inputs.append(symbolic_input)
        elif isinstance(example_input, int):
            inputs.append(example_input)
        elif isinstance(example_input, torch.SymInt):
            assert fxgraph_node.op == 'placeholder' and fxgraph_node.type is torch.SymInt
            name = fxgraph_node.name
            var = hidet.symbol_var(name)
            inputs.append(var)
        else:
            raise ValueError(f"hidet_backend: unexpected example input {example_input}, type {type(example_input)}")

    logger.info('hidet:   inputs: ')
    for arg in inputs:
        if isinstance(arg, hidet.Tensor):
            logger.info('hidet:   %s', arg.signature())
        else:
            logger.info('hidet:   %s', arg)

    output = interpreter(*inputs)
    output_format, output_tensors = serialize_output(output)
    input_tensors = [x for x in inputs if isinstance(x, hidet.Tensor)]

    return hidet.trace_from(output_tensors, inputs=input_tensors), inputs, output_format


def get_compiled_graph(flow_graph: FlowGraph):
    use_fp16 = dynamo_config['use_fp16']
    use_fp16_reduction = dynamo_config['use_fp16_reduction']
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
    logger.info('start to build the optimized computation graph')
    cgraph: CompiledGraph = graph_opt.build(space=search_space)
    logger.info('finish building computation graph')
    return cgraph


def preprocess_inputs(inputs: Sequence[torch.Tensor]) -> List[hidet.Tensor]:
    torch_inputs: List[torch.Tensor] = []
    for x in inputs:
        if not x.is_contiguous():
            # warnings.warn_once('Hidet received a non-contiguous torch input tensor, converting it to contiguous')
            x = x.contiguous()
        torch_inputs.append(x)
    hidet_inputs: List[hidet.Tensor] = [hidet.from_torch(tensor) for tensor in torch_inputs]
    return hidet_inputs


class HidetCompiledModel:
    def __init__(self, cgraph: CompiledGraph, inputs, output_format):
        super().__init__()
        self.inputs = inputs
        self.output_format = output_format
        self.cgraph_configured = False
        self.cgraph = cgraph

    def configure_cgraph(self):
        if dynamo_config['use_cuda_graph']:
            try:
                self.cgraph = self.cgraph.cuda_graph()
            except CudaGraphCreationError:
                pass  # Leave cgraph as is

    def __call__(self, *args):
        if not self.cgraph_configured:
            self.configure_cgraph()
            self.cgraph_configured = True

        tensor_args = []
        for param, arg in zip(self.inputs, args):
            if isinstance(param, Tensor):
                tensor_args.append(arg)
            elif isinstance(param, SymbolVar):
                dtype = param.type
                assert isinstance(dtype, DataType)
                if dtype.name == 'int32':
                    runtime_api.set_symbol_value(param.name, int(arg))
                else:
                    raise ValueError(f'hidet_backend: unsupported symbolic dtype {dtype}. We only support int32 now.')
            else:
                # ignore constant
                pass

        hidet_inputs = preprocess_inputs(tensor_args)
        hidet_outputs: List[hidet.Tensor] = self.cgraph.run_async(hidet_inputs)
        outputs: Sequence[torch.Tensor] = [tensor.torch() for tensor in hidet_outputs]
        return deserialize_output(self.output_format, outputs)


def hidet_backend(graph_module, example_inputs):
    assert isinstance(graph_module, torch.fx.GraphModule)

    logger.info('received a subgraph with %d nodes to optimize', len(graph_module.graph.nodes))
    logger.debug('graph: %s', graph_module.graph)

    if dynamo_config['print_input_graph']:
        graph_module.print_readable()
        print('---')
        graph_module.graph.print_tabular()

    # get the interpreter for the subgraph
    interpreter: Interpreter = hidet.frontend.from_torch(graph_module)

    if dynamo_config['correctness_report']:
        # check correctness using random inputs
        def wrapper(*args):
            report, output = interpreter.forward_with_check(*args)
            logger.info('finish checking correctness')
            print(report)
            return output

        return wrapper

    flow_graph, inputs, output_format = get_flow_graph(interpreter, example_inputs)

    cgraph = get_compiled_graph(flow_graph)

    return HidetCompiledModel(cgraph, inputs, output_format)
