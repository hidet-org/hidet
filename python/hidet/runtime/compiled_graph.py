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
from typing import List, Optional, Tuple, Dict, Any, Callable
import zipfile
import os
import json
from dataclasses import dataclass
import warnings
import tempfile

from tabulate import tabulate
import numpy

import hidet
from hidet.ffi.utils import Array, ctypes_func_pointer
from hidet.ir.type import void_p, data_type
from hidet.ir.dtypes import i32, i64
from hidet.runtime.device import Device
from hidet.runtime.compiled_module import CompiledModule
from hidet.runtime.compiled_task import CompiledTask, TensorSignature, _check_inputs
from hidet.runtime.storage import Storage
from hidet.ffi import runtime_api
from hidet.utils.py import prod, median
from hidet.utils.trace_utils import TraceEventEmitter

ModelExecutionHook = Callable[[int, List['Tensor'], List['Tensor']], None]


class ExternalStorage(Storage):
    def __init__(self, device: str, addr: int, num_bytes: int):
        super().__init__(Device(device), addr, num_bytes, lambda x: x)


@dataclass
class GraphMetaData:
    inputs: List[TensorSignature]
    outputs: List[TensorSignature]
    hidet_version: str
    num_kernels: int
    graph_hash: str
    share_map: Dict[int, int]


@dataclass
class GraphExecutionInstruction:
    task_idx: int
    inputs: List[int]
    outputs: List[int]
    free: List[int]


@dataclass
class GraphExecution:
    weights_index: List[int]
    inputs_index: List[int]
    instructions: List[GraphExecutionInstruction]
    outputs_index: List[int]
    tensor_device: List[str]


class CompiledGraph:
    """
    A compiled graph that can be directly called in Python.

    This class should not be instantiated directly. Instead, use :func:`load_compiled_graph` to load a compiled graph
    from disk, or build a compiled graph from :class:`FlowGraph` using :func:`hidet.drivers.build_flow_graph`.

    Parameters
    ----------
    meta: GraphMetaData
        The meta-data of the graph.

    graph_module: CompiledModule
        The graph compiled module that contains execution logic of the computation graph.

    weights: List[hidet.Tensor]
        The weights of the graph.

    compiled_tasks: List[CompiledTask]
        The compiled tasks of the graph that correspond to the operators in the computation graph.

    graph_execution: GraphExecution
        The execution plan of the graph (the order and connections of the compiled tasks).

    graph_string: str
        The string representation of the computation graph.
    """

    def __init__(
        self,
        meta: GraphMetaData,
        graph_module: CompiledModule,
        weights,
        compiled_tasks: List[CompiledTask],
        graph_execution: GraphExecution,
        graph_string: str,
    ):
        from hidet.graph.tensor import Tensor

        # graph module functions
        self._init = graph_module['init']
        self._get_output_shape = graph_module['get_output_shape']
        self._set_workspace = graph_module['set_workspace']
        self._get_workspace_size = graph_module['get_workspace_size']
        self._launch = graph_module['launch']

        # graph assets
        self.meta: GraphMetaData = meta
        self.graph_module: CompiledModule = graph_module
        self.weights: List[Tensor] = weights
        self.compiled_tasks: List[CompiledTask] = compiled_tasks
        self.graph_execution: GraphExecution = graph_execution
        self.graph_string: str = graph_string

        # derived properties (will be initialized in _init_compiled_graph at the end of this constructor)
        self.dynamic_dims: List[Tuple[str, Tuple[int, int]]] = []  # [(name, (tensor_index, dim_index))]
        self.is_dynamic: bool = False

        # runtime state
        self.working_dir: str = hidet.utils.cache_file('graphs', self.meta.graph_hash)
        self.dispatch_table_path = hidet.utils.cache_file('graphs', self.meta.graph_hash, 'dispatch_table.txt')
        self.dispatch_table: Dict[Tuple[int, ...], Array] = {}
        self.cuda_workspace: Optional[Storage] = None
        self.cpu_workspace: Optional[Storage] = None

        if len(self.weights) == len(graph_execution.weights_index):
            # the weights are already loaded, initialize the graph directly
            self._init_compiled_graph()

    def __str__(self):
        """
        Get the basic information of this compiled graph.

        Returns
        -------
        ret: str
            The human readable basic information.
        """
        rows = []
        for i, sig in enumerate(self.meta.inputs):
            dtype = data_type(sig.dtype)
            if i == 0:
                head = 'input'
            else:
                head = ''
            rows.append([head, dtype.short_name + str(sig.shape)])
        for i, sig in enumerate(self.meta.outputs):
            dtype = data_type(sig.dtype)
            if i == 0:
                head = 'output'
            else:
                head = ''
            rows.append([head, dtype.short_name + str(sig.shape)])
        weight_size = sum(w.nbytes for w in self.weights)
        rows.append(['weights', '{:.3f} GiB'.format(weight_size / 1024 / 1024 / 1024)])
        rows.append(['parameters', '{}'.format(sum(prod(x.shape) for x in self.weights))])

        return tabulate(rows, colalign=('right', 'left'), tablefmt='simple')

    def __call__(self, *args):
        """
        Run the model asynchronously with the given inputs.

        Parameters
        ----------
        args: Sequence[hidet.Tensor]
            The input tensors.

        Returns
        -------
        ret: Union[hidet.Tensor, List[hidet.Tensor]]
            The output tensor(s).
        """
        outs = self.run_async(args)
        if len(outs) == 1:
            return outs[0]
        else:
            return outs

    def _init_compiled_graph(self):
        # initialize the derived properties
        for tensor_index, sig in enumerate(self.meta.inputs):
            for dim_index, dim in enumerate(sig.shape):
                if isinstance(dim, str) and dim not in [v for v, _ in self.dynamic_dims]:
                    self.dynamic_dims.append((dim, (tensor_index, dim_index)))
        if len(self.dynamic_dims) > 0 or any(isinstance(dim, str) for sig in self.meta.outputs for dim in sig.shape):
            self.is_dynamic = True
        else:
            self.is_dynamic = False

        # initialize weights
        weights_buffer = Array(void_p, len(self.weights))
        for i in range(len(self.weights)):
            weights_buffer[i] = self.weights[i].storage.addr
        self._init(len(self.weights), weights_buffer)

        # load the dispatch table
        if os.path.exists(self.dispatch_table_path):
            with open(self.dispatch_table_path, 'r') as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    if idx == 0:
                        continue  # skip the header line
                    items = line.split()
                    if len(items) == 0:
                        continue  # skip empty lines
                    if len(items) != len(self.dynamic_dims) + len(self.compiled_tasks):
                        raise RuntimeError('Invalid dispatch table')
                    items = [int(item) for item in items]
                    symbol_dims = items[: len(self.dynamic_dims)]
                    schedule_indices = items[len(self.dynamic_dims) :]
                    kernel_array = Array(void_p, len(self.compiled_tasks))
                    for task_idx, (compiled_task, sch_idx) in enumerate(zip(self.compiled_tasks, schedule_indices)):
                        if not 0 <= sch_idx < len(compiled_task.candidates):
                            raise RuntimeError(
                                'Invalid schedule index {} for compiled task at {}'.format(
                                    sch_idx, compiled_task.task_dir
                                )
                            )
                        kernel_array[task_idx] = ctypes_func_pointer(compiled_task.candidates[sch_idx].ctypes_func)
                    self.dispatch_table[tuple(symbol_dims)] = kernel_array

    def _update_symbol_table(self, symbol_dims: Tuple[int, ...], best_candidates: List[int]):
        kernel_array = Array(void_p, len(self.compiled_tasks))
        for task_idx, best_candidate in enumerate(best_candidates):
            kernel_array[task_idx] = ctypes_func_pointer(
                self.compiled_tasks[task_idx].candidates[best_candidate].ctypes_func
            )
        self.dispatch_table[symbol_dims] = kernel_array

        if not os.path.exists(self.dispatch_table_path):
            with open(self.dispatch_table_path, 'w') as f:
                symbol_names = [name for name, _ in self.dynamic_dims]
                f.write(' '.join(symbol_names))
                f.write('\n')
        with open(self.dispatch_table_path, 'a') as f:
            f.write(' '.join(str(x) for x in symbol_dims))
            f.write(' ')
            f.write(' '.join(str(x) for x in best_candidates))
            f.write('\n')

    def _update_symbol_dims(self, inputs) -> Tuple[int, ...]:
        symbol_dims = []
        for name, (tensor_index, dim_index) in self.dynamic_dims:
            symbol_dims.append(inputs[tensor_index].shape[dim_index])
            runtime_api.set_symbol_value(name, symbol_dims[-1])
        return tuple(symbol_dims)

    def _create_outputs(self, inputs):
        from hidet.graph.tensor import empty, Tensor

        outputs = []
        exec_idx_to_output_idx: Dict[int, int] = {}
        for output_index, (exec_idx, sig) in enumerate(zip(self.graph_execution.outputs_index, self.meta.outputs)):
            if exec_idx in self.graph_execution.inputs_index:
                # the graph directly returns an input tensor
                outputs.append(inputs[self.graph_execution.inputs_index.index(exec_idx)])
            elif exec_idx in self.graph_execution.weights_index:
                # the graph directly returns a weight tensor
                outputs.append(self.weights[self.graph_execution.weights_index.index(exec_idx)])
            elif exec_idx in exec_idx_to_output_idx:
                # the graph returns the same tensor multiple times
                outputs.append(outputs[exec_idx_to_output_idx[exec_idx]])
            else:
                # get the shape of output tensor
                if self.is_dynamic:
                    shape_buffer = Array(i32, len(sig.shape))
                    self._get_output_shape(output_index, shape_buffer)
                    shape = list(shape_buffer)
                else:
                    shape = sig.shape

                if output_index not in self.meta.share_map:
                    # create the output tensor
                    outputs.append(empty(shape=shape, dtype=sig.dtype, device=sig.device))
                else:
                    # this output tensor shares the storage with one input tensor, reuse the storage
                    input_tensor: Tensor = inputs[self.meta.share_map[output_index]]
                    outputs.append(
                        Tensor(shape=shape, dtype=sig.dtype, device=sig.device, storage=input_tensor.storage)
                    )

                # record the exec_idx of this output tensor, in case the graph returns the same tensor multiple times
                exec_idx_to_output_idx[exec_idx] = output_index

        return outputs

    def _prepare_workspace(self):
        buffer = Array(i64, 2)
        self._get_workspace_size(buffer)
        required_cpu_workspace, required_cuda_workspace = list(buffer)

        if self.cpu_workspace is None or self.cpu_workspace.num_bytes < required_cpu_workspace:
            self.cpu_workspace = Storage.new('cpu', required_cpu_workspace)
            self._set_workspace(0, self.cpu_workspace.addr)

        if self.cuda_workspace is None or self.cuda_workspace.num_bytes < required_cuda_workspace:
            self.cuda_workspace = Storage.new('cuda', required_cuda_workspace)
            self._set_workspace(1, self.cuda_workspace.addr)

    def _run_fast_path(self, inputs, symbol_dims: Tuple[int, ...]):
        # create output tensors
        outputs = self._create_outputs(inputs)

        # prepare workspace
        self._prepare_workspace()

        # run the kernels
        kernel_array = self.dispatch_table[symbol_dims]
        self._launch(*inputs, *outputs, kernel_array)

        return outputs

    def _run_slow_path(self, inputs, symbol_dims: Tuple[int, ...]):
        """Interpret the graph execution"""
        from hidet.graph.tensor import Tensor

        index2tensor: Dict[int, Tensor] = {}
        exe = self.graph_execution
        for i in range(len(inputs)):
            index2tensor[exe.inputs_index[i]] = inputs[i]
        for i in range(len(self.weights)):
            index2tensor[exe.weights_index[i]] = self.weights[i]

        best_candidates = [-1 for _ in range(len(self.compiled_tasks))]
        trace_emitter = TraceEventEmitter({'graph': self.graph_string})
        for inst in exe.instructions:
            # prepare inputs and kernel
            node_inputs = [index2tensor[i] for i in inst.inputs]
            node_kernel: CompiledTask = self.compiled_tasks[inst.task_idx]

            # run the kernel
            node_outputs = node_kernel.run_async(node_inputs)

            # record outputs
            for i, output_index in enumerate(inst.outputs):
                index2tensor[output_index] = node_outputs[i]

            # record best candidate for this kernel
            best_candidates[inst.task_idx] = node_kernel.pick_best_candidate(node_inputs, node_outputs)

            # record trace events
            trace_emitter.append(
                name=node_kernel.meta_data.name,
                duration_us=int(median(node_kernel.profile(*node_inputs, *node_outputs)) * 1000),
                args={
                    'name': node_kernel.meta_data.name,
                    'inputs': ['{}{}'.format(x.dtype, x.shape) for x in node_kernel.meta_data.inputs],
                    'outputs': ['{}{}'.format(x.dtype, x.shape) for x in node_kernel.meta_data.outputs],
                },
            )

            # free tensors that are no longer needed
            for idx in inst.free:
                del index2tensor[idx]

        outputs = [index2tensor[i] for i in exe.outputs_index]

        # update the dispatch table
        self._update_symbol_table(symbol_dims, best_candidates)

        # save the trace
        trace_filename = 'trace{}.json'.format('_'.join(str(x) for x in symbol_dims))
        with open(os.path.join(self.working_dir, trace_filename), 'w') as f:
            trace_emitter.save(f)

        return outputs

    def set_weights(self, weights):
        """
        Set the weights of the model.

        When the weights exist in the model file, the user does not need to set the weights manually.
        However, when the weights are not saved in the model file, the user needs to set the weights manually before
        running the model.

        Parameters
        ----------
        weights: List[hidet.Tensor]
            The weights to set.
        """
        from hidet.runtime.device import instantiate_device

        if len(self.weights) == len(self.graph_execution.weights_index):
            raise RuntimeError('The weights are already set.')
        if len(weights) != len(self.graph_execution.weights_index):
            raise ValueError('Expect {} weights, got {}.'.format(len(self.graph_execution.weights_index), len(weights)))
        if any(not isinstance(w, hidet.Tensor) for w in weights):
            raise ValueError('Expect all weights to be hidet.Tensor, got {}'.format([type(w) for w in weights]))
        for idx, weight in enumerate(weights):
            expected_device = instantiate_device(
                self.graph_execution.tensor_device[self.graph_execution.weights_index[idx]]
            )
            if expected_device != weight.device:
                raise ValueError(
                    'Expect weight {} to be on device {}, got {}.'.format(idx, expected_device, weight.device)
                )
        self.weights = weights
        self._init_compiled_graph()

    def run_async(self, inputs):
        """
        Run the model asynchronously.

        Parameters
        ----------
        inputs: Sequence[hidet.Tensor]
            The input tensors.

        Returns
        -------
        ret: List[hidet.Tensor]
            The output tensors.
        """
        if hidet.option.get_runtime_check():
            _check_inputs(self.meta.inputs, inputs)
        if len(self.weights) != len(self.graph_execution.weights_index):
            raise RuntimeError('Please set the weights before running the model with compiled_graph.set_weights(...).')

        symbol_dims = self._update_symbol_dims(inputs)

        if symbol_dims in self.dispatch_table:
            return self._run_fast_path(inputs, symbol_dims)
        else:
            return self._run_slow_path(inputs, symbol_dims)

    def cuda_graph(self):
        """
        Create a CUDA graph for this compiled graph.

        Returns
        -------
        cuda_graph: hidet.cuda.graph.CudaGraph
            The CUDA graph.
        """
        from hidet.cuda.graph import CudaGraph, CudaGraphCreationError
        from hidet.graph.tensor import Tensor, randn, zeros, empty

        for x in self.meta.inputs + self.meta.outputs:
            if x.device == 'cpu':
                raise CudaGraphCreationError(f'Cannot create CUDA graph for a model with CPU inputs:\n {x}')
            for d in x.shape:
                if not isinstance(d, int):
                    raise CudaGraphCreationError(f'Cannot create CUDA graph for a model with dynamic inputs:\n {x}')
        if any(device == 'cpu' for device in self.graph_execution.tensor_device):
            raise CudaGraphCreationError('Cannot create CUDA graph for a model with CPU tensors.')
        for ctask in self.compiled_tasks:
            if len(ctask.meta_data.symbols) > 0:
                raise CudaGraphCreationError('Cannot create CUDA graph for a model with dynamic symbols.')

        def f_create_inputs() -> List[Tensor]:
            with hidet.option.context():
                hidet.option.imperative(True)
                dummy_inputs = []
                for meta_input in self.meta.inputs:
                    dtype = hidet.ir.data_type(meta_input.dtype)
                    if dtype.is_float():
                        inp = randn(shape=meta_input.shape, dtype=dtype, device=meta_input.device)
                    elif dtype.is_integer():
                        inp = zeros(shape=meta_input.shape, dtype=dtype, device=meta_input.device)
                    else:
                        warnings.warn('Creating dummy input with "empty" for data type {}'.format(dtype))
                        inp = empty(shape=meta_input.shape, dtype=dtype, device=meta_input.device)
                    dummy_inputs.append(inp)

                return dummy_inputs

        def f_run(inputs: List[Tensor]) -> List[Tensor]:
            return self.run_async(inputs)

        # clear the workspace to avoid the storage being captured by the CUDA graph.
        self.cuda_workspace = None

        return CudaGraph(f_create_inputs, f_run, ref_objs=[self])

    def save(self, path: str, save_dispatch_table: bool = False):
        """
        Save the compiled graph to disk.

        See Also
        --------
        load_compiled_graph

        Parameters
        ----------
        path: str
            The path to save the compiled graph. By convention, the path should end with '.hidet'.

        save_dispatch_table:
            Whether to save the dispatch table to disk. See `save_compiled_graph` for details.
        """
        save_compiled_graph(self, path, save_dispatch_table)


def save_compiled_graph(model: CompiledGraph, file: str, save_dispatch_table: bool = False, save_weights: bool = True):
    """
    Save the compiled graph to disk.

    Parameters
    ----------
    model: CompiledGraph
        The compiled graph to save.

    file: str
        The path to save the compiled graph. By convention, the path should end with '.hidet'.

    save_dispatch_table:
        Whether to save the dispatch table to disk.

        When we run the model that contains alternative kernels for the same operator, we will pick the best kernel
        by benchmarking all the alternatives. The dispatch table is used to record the best kernel for the given
        input shapes. If the dispatch table is not saved, we will benchmark all the alternatives again when we load
        the model next time.

        Default: False

    save_weights:
        Whether to save the weights to disk. If False, the weights will not be saved, and the users can save the
        weights separately. This is useful when we want to save the weights separately.

        Default: True
    """
    from hidet.utils.dataclass import asdict

    dirname = os.path.dirname(file)
    os.makedirs(dirname, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=dirname, delete=False) as temp_file:
        temp_path = temp_file.name

        with zipfile.ZipFile(temp_path, 'w') as zf:

            def _save_under(dir_path: str, dir_in_zip: str, exclude: Optional[List[str]] = None):
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_in_zip = os.path.join(dir_in_zip, os.path.relpath(file_path, dir_path))
                        with zf.open(file_in_zip, 'w') as f1:
                            if exclude and file in exclude:
                                continue
                            with open(file_path, 'rb') as f2:
                                f1.write(f2.read())

            # meta info
            with zf.open('meta.json', 'w') as f:
                meta_bytes = json.dumps(asdict(model.meta), indent=4).encode('utf-8')
                f.write(meta_bytes)

            # save the modules
            _save_under(model.graph_module.module_dir, 'graph_module/')

            # save weights
            if save_weights:
                # zip.open(..., force_zip64=True) is required for >4GB weights
                with zf.open('weights.npz', 'w', force_zip64=True) as f:
                    numpy.savez(f, *[weight.cpu().numpy() for weight in model.weights])

            # save the kernels (i.e., compiled tasks)
            for i, compiled_task in enumerate(model.compiled_tasks):
                _save_under(compiled_task.task_dir, 'kernels/{}/'.format(i))

            # save graph execution
            with zf.open('graph_execution.json', 'w') as f:
                ge_bytes = json.dumps(asdict(model.graph_execution), indent=4).encode('utf-8')
                f.write(ge_bytes)

            # save dispatch table file
            if save_dispatch_table and os.path.exists(model.dispatch_table_path):
                with zf.open('dispatch_table.txt', 'w') as f:
                    with open(model.dispatch_table_path, 'rb') as f2:
                        f.write(f2.read())

            # save graph string
            with zf.open('graph_string.txt', 'w') as f:
                f.write(model.graph_string.encode('utf-8'))

    os.rename(temp_path, file)


def load_compiled_graph(path: str) -> CompiledGraph:
    """
    Load a compiled graph from disk.

    The compiled graph is saved with zip format. The path can be either a single file to the zip file, or a directory
    that contains the contents of the zip file.

    Parameters
    ----------
    path: str
        The path to load the compiled graph (can be either a single file or a directory).

    Returns
    -------
    ret: CompiledGraph
        The loaded compiled graph.
    """
    from hidet.utils.dataclass import from_dict

    if os.path.isfile(path):
        with zipfile.ZipFile(path, 'r') as zf:
            # load meta data
            with zf.open('meta.json', 'r') as f:
                meta_data: GraphMetaData = from_dict(GraphMetaData, json.load(f))

            # extract all files except weights
            files_to_extract: List[str] = zf.namelist()
            if 'weights.npz' in files_to_extract:
                files_to_extract.remove('weights.npz')
            cache_dir = hidet.utils.cache_dir('graphs', meta_data.graph_hash)
            if not os.path.exists(os.path.join(cache_dir, 'graph_string.txt')):
                # only extract files if the graph_string.txt is not in the cache
                # here 'graph_string.txt' is just the last file we usually save to disk, we use it as a flag
                # to indicate whether the graph is already in the cache
                zf.extractall(cache_dir, files_to_extract)

            graph_path = cache_dir
    else:
        graph_path = path

    # load meta data
    with open(os.path.join(graph_path, 'meta.json'), 'r') as f:
        meta_data: GraphMetaData = from_dict(GraphMetaData, json.load(f))

    # load graph execution
    with open(os.path.join(graph_path, 'graph_execution.json'), 'r') as f:
        graph_execution: GraphExecution = from_dict(GraphExecution, json.load(f))

    # load weights if it exists
    weights = []

    def load_weights_from_npz(npz: zipfile.ZipFile):
        for weight_idx, name in enumerate(npz.namelist()):
            with npz.open(name, 'r') as npy_file:
                npy_file: Any  # used to suppress type checker warning
                device = graph_execution.tensor_device[graph_execution.weights_index[weight_idx]]
                weights.append(hidet.asarray(numpy.load(npy_file), device=device))

    if os.path.exists(os.path.join(graph_path, 'weights.npz')):
        with zipfile.ZipFile(os.path.join(graph_path, 'weights.npz'), 'r') as npz:
            load_weights_from_npz(npz)
    elif os.path.isfile(path):
        with zipfile.ZipFile(path, 'r') as zf:
            if 'weights.npz' in zf.namelist():
                # weights are loaded directly from the zip file to memory
                # avoid extracting the weights to disk and then loading them from disk
                with zf.open('weights.npz', 'r') as f:
                    with zipfile.ZipFile(f, 'r') as npz:
                        load_weights_from_npz(npz)

    # load kernels (i.e., compiled tasks)
    num_kernels = meta_data.num_kernels
    compiled_tasks = [CompiledTask(task_dir=os.path.join(graph_path, 'kernels', str(i))) for i in range(num_kernels)]

    # load graph module
    graph_module = CompiledModule(module_dir=os.path.join(graph_path, 'graph_module'))

    # load graph string
    with open(os.path.join(graph_path, 'graph_string.txt'), 'r') as f:
        graph_string = f.read()

    # construct the compiled graph
    ret = CompiledGraph(meta_data, graph_module, weights, compiled_tasks, graph_execution, graph_string)

    return ret
