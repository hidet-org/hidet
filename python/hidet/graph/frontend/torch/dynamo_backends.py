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
from typing import List, Sequence
import logging
import torch
import hidet.option
from hidet.runtime import CompiledGraph
from hidet.graph.flow_graph import FlowGraph
from hidet.graph.transforms import PassContext, optimize
from hidet.cuda.graph import CudaGraphCreationError
from .dynamo_config import dynamo_config
from .interpreter import Interpreter
from .utils import serialize_output, deserialize_output, resolve_save_dir_multigraph
from .utils import convert_compilation_input, convert_runtime_input
from .registry import allow_in_graph_registered_funcs_only
from .flow_graph_cache import flow_graph_cache_load, flow_graph_cache_save


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


def get_flow_graph(interpreter: Interpreter, example_inputs):
    input_format, hidet_inputs, flatten_hidet_inputs = convert_compilation_input(example_inputs, interpreter.graph)
    logger.info('hidet:   inputs: ')
    for arg in hidet_inputs:
        if isinstance(arg, hidet.Tensor):
            logger.info('hidet:   %s', arg.signature())
        else:
            logger.info('hidet:   %s', arg)

    with hidet.option.context():
        hidet.option.execution_mode('symbolic')
        output = interpreter(*hidet_inputs)
        output_format, output_tensors = serialize_output(output)

    return (hidet.trace_from(output_tensors, inputs=flatten_hidet_inputs), input_format, output_format)


def get_compiled_graph(flow_graph: FlowGraph, kwargs):
    # check on-disk cache first before compiling FlowGraph into CompiledGraph
    # and acquire the hash key for saving the FlowGraph after compiling
    cached_compiled_graph, flowgraph_key = flow_graph_cache_load(flow_graph, kwargs)
    if cached_compiled_graph is not None:
        return cached_compiled_graph
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
    flow_graph_cache_save(flowgraph_key, cgraph)
    return cgraph


def preprocess_inputs(inputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    for i, x in enumerate(inputs):
        if not x.is_contiguous():
            inputs[i] = x.contiguous()
    return inputs


class HidetCompiledModel:
    def __init__(self, cgraph: CompiledGraph, input_format, output_format):
        '''
        Torch (>=2.5) compile treats all weights as inputs. Hidet, on the other hand,
        treats weights as constant tensors. Actual inputs selected from all inputs
        provided by torch using `nonconstant_input_ids`.
        '''
        super().__init__()
        self.input_format = input_format
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

        tensor_args = preprocess_inputs(convert_runtime_input(self.input_format, args))
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

        flow_graph, input_format, output_format = get_flow_graph(interpreter, example_inputs)
        del interpreter
        cgraph = get_compiled_graph(flow_graph, kwargs)
        return HidetCompiledModel(cgraph, input_format, output_format)


allow_in_graph_registered_funcs_only()
