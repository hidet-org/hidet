from typing import List, Optional, Tuple, Dict, Any, Union, Callable, Sequence
import functools
import zipfile
import os
import json
import numpy
from dataclasses import dataclass
from tabulate import tabulate

import hidet
from hidet.ffi.utils import Array, ctypes_func_pointer
from hidet.ir.type import void_p, data_type
from hidet.ir.dtypes import i32, i64
from hidet.runtime.device import Device
from hidet.runtime.compiled_module import CompiledModule
from hidet.runtime.compiled_task import CompiledTask, TensorSignature
from hidet.runtime.storage import Storage
from hidet.ffi import runtime_api
from hidet.utils import prod


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


class CompiledGraph:
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
        self._register_hook = graph_module['register_hook']
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

        # derived properties
        self.dynamic_dims: List[Tuple[str, Tuple[int, int]]] = []  # [(name, (tensor_index, dim_index))]
        for tensor_index, sig in enumerate(self.meta.inputs):
            for dim_index, dim in enumerate(sig[1:]):
                if isinstance(dim, str) and dim not in [v for v, _ in self.dynamic_dims]:
                    self.dynamic_dims.append((dim, (tensor_index, dim_index)))
        self.is_dynamic: bool = len(self.dynamic_dims) > 0

        # runtime state
        self._registered_hook: Optional[Any] = None
        self.dispatch_table_path = hidet.utils.cache_file('graphs', self.meta.graph_hash, 'dispatch_table.txt')
        self.dispatch_table: Dict[Tuple[int, ...], Array] = {}
        self.cuda_workspace: Optional[Storage] = None
        self.cpu_workspace: Optional[Storage] = None

        self._init_compiled_graph()

    def __str__(self):
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
        outs = self.run_async(args)
        if len(outs) == 1:
            return outs[0]
        else:
            return outs

    def _init_compiled_graph(self):
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
                    schedule_indices = items[len(self.dynamic_dims):]
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

    def _create_outputs(self):
        from hidet.graph.tensor import empty

        outputs = []
        if self.is_dynamic:
            for output_index, sig in enumerate(self.meta.outputs):
                shape_buffer = Array(i32, len(sig.shape))
                self._get_output_shape(output_index, shape_buffer)
                outputs.append(empty(shape=list(shape_buffer), dtype=sig.dtype, device=sig.device))
        else:
            for sig in self.meta.outputs:
                outputs.append(empty(shape=sig.shape, dtype=sig.dtype, device=sig.device))
        return outputs

    def _prepare_workspace(self):
        buffer = Array(i64, 2)
        self._get_workspace_size(buffer)
        required_cpu_workspace, required_cuda_workspace = list(buffer)

        if self.cpu_workspace is None or self.cpu_workspace.num_bytes < required_cpu_workspace:
            self.workspace = Storage.new('cpu', required_cpu_workspace)
            self._set_workspace(0, self.workspace.addr)

        if self.cuda_workspace is None or self.cuda_workspace.num_bytes < required_cuda_workspace:
            self.workspace = Storage.new('cuda', required_cpu_workspace)
            self._set_workspace(1, self.workspace.addr)

    def _run_fast_path(self, inputs, symbol_dims: Tuple[int, ...]):
        # create output tensors
        outputs = self._create_outputs()

        # prepare workspace
        self._prepare_workspace()

        # run the kernels
        kernel_array = self.dispatch_table[symbol_dims]
        self._launch(*inputs, *outputs, kernel_array)

        return outputs

    def _run_slow_path(self, inputs, symbol_dims: Tuple[int, ...]):
        from hidet.graph.tensor import Tensor

        index2tensor: Dict[int, Tensor] = {}
        exe = self.graph_execution
        for i in range(len(inputs)):
            index2tensor[exe.inputs_index[i]] = inputs[i]
        for i in range(len(self.weights)):
            index2tensor[exe.weights_index[i]] = self.weights[i]
        best_candidates = [-1 for _ in range(len(self.compiled_tasks))]
        for inst in exe.instructions:
            node_inputs = [index2tensor[i] for i in inst.inputs]
            node_kernel: CompiledTask = self.compiled_tasks[inst.task_idx]
            node_outputs = node_kernel.run_async(node_inputs)
            for i, output_index in enumerate(inst.outputs):
                index2tensor[output_index] = node_outputs[i]

            best_candidates[inst.task_idx] = node_kernel.pick_best_candidate(node_inputs, node_outputs)

            for idx in inst.free:
                del index2tensor[idx]

        outputs = [index2tensor[i] for i in exe.outputs_index]

        self._update_symbol_table(symbol_dims, best_candidates)

        return outputs

    def register_hook(self, hook: ModelExecutionHook):
        """
        Register a hook that will be called after each operator run.

        Parameters
        ----------
        hook: ModelExecutionHook
            The hook to be registered. It has the following signature:
                hook(idx: int, inputs: List[Tensor], outputs: List[Tensor) -> None
            The hook will be called after the idx-th (0-base) operator finishes execution, with the inputs and outputs
            of the operator as arguments. When the whole model finishes execution, the hook will be called with
            idx=len(operators). There is at most one hook can be registered at any time. Registering a new hook will
            overwrite the previous one.
        """
        from ctypes import CFUNCTYPE, c_uint64, POINTER, cast, c_void_p

        def hook_wrapper(args: Sequence[int], device, hook: ModelExecutionHook):
            from hidet.graph.tensor import Tensor
            from ctypes import c_char_p

            cur = 0

            def get():
                nonlocal cur
                cur += 1
                return args[cur - 1]

            idx: int = get()
            num_inputs: int = get()
            num_outputs: int = get()
            params: List[Tensor] = []
            for _ in range(num_inputs + num_outputs):
                dtype: str = c_char_p(get()).value.decode('utf-8')
                rank: int = get()
                shape: List[int] = [get() for _ in range(rank)]
                pointer: int = get()
                storage = ExternalStorage('cpu', pointer, prod(shape) * data_type(dtype).nbytes)
                params.append(Tensor(shape, dtype, device, storage))
            hook(idx, params[:num_inputs], params[num_inputs:])

        self._registered_hook = CFUNCTYPE(None, POINTER(c_uint64))(
            functools.partial(hook_wrapper, device=self.device, hook=hook)
        )
        self._register_hook(cast(self._registered_hook, c_void_p))

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
        symbol_dims = self._update_symbol_dims(inputs)

        if symbol_dims in self.dispatch_table:
            return self._run_fast_path(inputs, symbol_dims)
        else:
            return self._run_slow_path(inputs, symbol_dims)

    def save(self, path: str):
        save_compiled_graph(self, path)


def load_compiled_graph(path: str) -> CompiledGraph:
    with zipfile.ZipFile(path, 'r') as zf:
        files_to_extract: List[str] = zf.namelist()
        files_to_extract.remove('weights.npz')  # weights are loaded directly from the zip file

        # load meta data
        with zf.open('meta.json', 'r') as f:
            meta_data = GraphMetaData.from_state_dict(json.load(f))

        # load weights as numpy arrays
        with zf.open('weights.npz', 'r') as f:
            with zipfile.ZipFile(f, 'r') as npz:
                weights = []
                for name in npz.namelist():
                    with npz.open(name, 'r') as npy_file:
                        npy_file: Any  # used to suppress type checker warning
                        weights.append(hidet.asarray(numpy.load(npy_file), device=meta_data.device))

        # extract all files except weights
        cache_dir = hidet.utils.cache_dir('graphs', meta_data.graph_hash)

        if not os.path.exists(os.path.join(cache_dir, 'graph_module/lib.so')):
            # only extract files if the compiled graph is not in the cache
            zf.extractall(cache_dir, files_to_extract)

    # load kernels (i.e., compiled tasks)
    num_kernels = meta_data.num_kernels
    compiled_tasks = [CompiledTask(task_dir=os.path.join(cache_dir, 'kernels', str(i))) for i in range(num_kernels)]

    # load graph module
    graph_module = CompiledModule(module_dir=os.path.join(cache_dir, 'graph_module'))

    # load graph execution
    with open(os.path.join(cache_dir, 'graph_execution.json'), 'r') as f:
        graph_execution = GraphExecution.from_state_dict(json.load(f))

    # load graph string
    with open(os.path.join(cache_dir, 'graph_string.txt'), 'r') as f:
        graph_string = f.read()

    # construct the compiled graph
    ret = CompiledGraph(meta_data, graph_module, weights, compiled_tasks, graph_execution, graph_string)

    return ret


def save_compiled_graph(model: CompiledGraph, path: str):
    with zipfile.ZipFile(path, 'w') as zf:

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
            meta_bytes = json.dumps(model.meta.state_dict(), indent=4).encode('utf-8')
            f.write(meta_bytes)

        # save the modules
        _save_under(model.graph_module.module_dir, 'graph_module/')

        # save weights
        with zf.open('weights.npz', 'w', force_zip64=True) as f:  # force_zip64 is required for >4GB weights
            numpy.savez(f, *[weight.cpu().numpy() for weight in model.weights])

        # save the kernels (i.e., compiled tasks)
        for i, compiled_task in enumerate(model.compiled_tasks):
            _save_under(compiled_task.task_dir, 'kernels/{}/'.format(i))

        # save graph execution
        with zf.open('graph_execution.json', 'w') as f:
            ge_bytes = json.dumps(model.graph_execution.state_dict(), indent=4).encode('utf-8')
            f.write(ge_bytes)

        # save graph string
        with zf.open('graph_string.txt', 'w') as f:
            f.write(model.graph_string.encode('utf-8'))
