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
from .utils import serialize_output, deserialize_output, resolve_save_dir_multigraph, tensor_from_torch
from .utils import symbol_like_torch
from .registry import allow_in_graph_registered_funcs_only


logger = logging.getLogger(__name__)


# TODO: after search_space=1 will be tuned switch search_space from 0 to 1
def process_options(kwargs):
    # Default options for case mode is not passed to torch.compile()
    hidet.option.search_space(0)
    hidet.torch.dynamo_config.use_cuda_graph(False)

    if 'mode' in kwargs:
        mode = kwargs['mode']
        if mode == 'max-autotune':
            hidet.option.search_space(2)
            hidet.torch.dynamo_config.use_cuda_graph(True)
        elif mode == 'max-autotune-no-cudagraphs':
            hidet.option.search_space(2)
            hidet.torch.dynamo_config.use_cuda_graph(False)
        elif mode == 'reduce-overhead':
            hidet.option.search_space(0)
            hidet.torch.dynamo_config.use_cuda_graph(True)
        elif mode == 'default':
            hidet.option.search_space(0)
            hidet.torch.dynamo_config.use_cuda_graph(False)
        else:
            raise ValueError(f'hidet_backend: unknown torch.compile mode={mode}')

    for option, value in kwargs.get('options', {}).items():
        if not hidet.option.is_option_exist(option):
            raise ValueError(f'hidet_backend: unknown torch.compile option={option}')
        hidet.option.set_option(option, value)


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
    traceable_input_ids = []
    for idx, (fxgraph_node, example_input) in enumerate(zip(interpreter.graph.nodes, example_inputs)):
        if isinstance(example_input, torch.Tensor):
            tensor_dict = fxgraph_node.meta.get('tensor_dict', {})
            if len(tensor_dict) == 0:
                traceable_input_ids.append(idx)
                fake_input = fxgraph_node.meta['example_value']
                symbolic_input = symbol_like_torch(fake_input)
                inputs.append(symbolic_input)
            elif tensor_dict.get('_dynamo_static_input_type', None) == 'unguarded':
                # Usually, such tensors are weight tensors passed as inputs
                inputs.append(tensor_from_torch(example_input))
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

    with hidet.option.context():
        hidet.option.execution_mode('symbolic')
        output = interpreter(*inputs)
        output_format, output_tensors = serialize_output(output)
        input_tensors = []
        weight_tensors = []
        for idx, x in enumerate(inputs):
            if not isinstance(x, hidet.Tensor):
                continue
            if idx in traceable_input_ids:
                input_tensors.append(x)
            else:
                weight_tensors.append(x)

    return (
        hidet.trace_from(output_tensors, inputs=input_tensors, weight_tensors=weight_tensors),
        inputs,
        traceable_input_ids,
        output_format,
    )


def get_compiled_graph(flow_graph: FlowGraph):
    save_dir = dynamo_config['dump_graph_ir']
    with PassContext() as ctx:
        if save_dir:
            graph_dir = resolve_save_dir_multigraph(save_dir)
            ctx.save_graph_instrument(graph_dir)
        ctx.allow_source_graph_removal(True)
        logger.info('start to optimize the flow graph')
        graph_opt: FlowGraph = optimize(flow_graph)
        logger.info('finish optimizing the flow graph')

    logger.info('schedule search space: %d', hidet.option.get_search_space())
    logger.info('start to build the optimized computation graph')
    cgraph: CompiledGraph = graph_opt.build(space=hidet.option.get_search_space())
    logger.info('finish building computation graph')
    return cgraph


def preprocess_inputs(inputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    for i, x in enumerate(inputs):
        if not x.is_contiguous():
            inputs[i] = x.contiguous()
    return inputs


class HidetCompiledModel:
    def __init__(self, cgraph: CompiledGraph, inputs, nonconstant_input_ids, output_format):
        '''
        Torch (>=2.5) compile treats all weights as inputs. Hidet, on the other hand,
        treats weights as constant tensors. Actual inputs selected from all inputs
        provided by torch using `nonconstant_input_ids`.
        '''
        super().__init__()
        self.inputs = inputs
        self.nonconstant_input_ids = nonconstant_input_ids
        self.output_format = output_format
        self.cgraph_configured = False
        self.cgraph = cgraph

    def configure_cgraph(self):
        if dynamo_config['use_cuda_graph']:
            try:
                inputs_for_cgraph = [self.inputs[idx] for idx in self.nonconstant_input_ids]
                self.cgraph = self.cgraph.cuda_graph(*inputs_for_cgraph)
            except CudaGraphCreationError:
                pass  # Leave cgraph as is

    def __call__(self, *args):
        if not self.cgraph_configured:
            self.configure_cgraph()
            self.cgraph_configured = True

        tensor_args = []
        for idx, (param, arg) in enumerate(zip(self.inputs, args)):
            if isinstance(param, Tensor) and idx in self.nonconstant_input_ids:
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
        # Inherited cuda stream from torch
        runtime_api.set_current_stream(torch.cuda.current_stream().cuda_stream)
        # Prepare inputs
        tensor_args = preprocess_inputs(tensor_args)
        # Run graph/model
        outputs = self.cgraph.run_async(tensor_args, output_to_torch_tensor=True)
        outputs: Sequence[torch.Tensor] = [
            tensor.torch() if isinstance(tensor, hidet.Tensor) else tensor for tensor in outputs
        ]

        return deserialize_output(self.output_format, outputs)


def hidet_backend(graph_module, example_inputs, **kwargs):
    assert isinstance(graph_module, torch.fx.GraphModule)

    logger.info('received a subgraph with %d nodes to optimize', len(graph_module.graph.nodes))
    logger.debug('graph: %s', graph_module.graph)

    with hidet.option.context():
        # Process options passed to torch.compile
        process_options(kwargs)

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

        flow_graph, inputs, traceable_input_ids, output_format = get_flow_graph(interpreter, example_inputs)
        del interpreter
        cgraph = get_compiled_graph(flow_graph)
        return HidetCompiledModel(cgraph, inputs, traceable_input_ids, output_format)


allow_in_graph_registered_funcs_only()
