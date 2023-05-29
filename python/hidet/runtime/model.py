from typing import List, Optional, Tuple, Dict, Any, Union, Callable, Sequence
import tempfile
import functools
import zipfile
import os
import json
import numpy
from tabulate import tabulate

import hidet
from hidet.ffi.utils import Array, ctypes_func_pointer
from hidet.ir.type import void_p, data_type
from hidet.ir.dtypes import i32
from hidet.runtime.device import Device
from hidet.runtime.module import CompiledModule
from hidet.runtime.storage import Storage
from hidet.ffi import runtime_api
from hidet.utils import prod

ModelExecutionHook = Callable[[int, List['Tensor'], List['Tensor']], None]


class ExternalStorage(Storage):
    def __init__(self, device: str, addr: int, num_bytes: int):
        super().__init__(Device(device), addr, num_bytes, lambda x: x)


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


class ModelMetaData:
    def __init__(self, input_signatures, output_signatures, device, hidet_version, num_kernels):
        # the signature will in form
        # [type, dim1, dim2, ..., dimN] where dim_i can be a number or a string
        # like [['float32', 1, 3, 224, 224]] or [['int32', 'seq']]
        self.input_signatures: List[Union[str, int]] = input_signatures
        self.output_signatures: List[Union[str, int]] = output_signatures
        self.device: str = device
        self.hidet_version: str = hidet_version
        self.num_kernels: int = num_kernels

        self.dynamic_dims: List[Tuple[str, Tuple[int, int]]] = []  # [(name, (tensor_index, dim_index))]
        for tensor_index, sig in enumerate(self.input_signatures):
            for dim_index, dim in enumerate(sig[1:]):
                if isinstance(dim, str):
                    self.dynamic_dims.append((dim, (tensor_index, dim_index)))

    def export_state(self) -> Dict[str, Any]:
        return {
            'inputs': self.input_signatures,
            'outputs': self.output_signatures,
            'device': self.device,
            'hidet_version': self.hidet_version,
            'num_kernels': self.num_kernels,
        }

    @staticmethod
    def from_state(state):
        return ModelMetaData(
            input_signatures=state['inputs'],
            output_signatures=state['outputs'],
            device=state['device'],
            hidet_version=state['hidet_version'],
            num_kernels=state['num_kernels'],
        )


class CompiledModel:
    def __init__(self, meta_data: ModelMetaData, graph_module: CompiledModule, weights, kernels: List[CompiledModule]):
        from hidet.graph.tensor import Tensor

        self._init = graph_module['init']
        self._register_hook = graph_module['register_hook']
        self._get_output_shape = graph_module['get_output_shape']
        self._set_workspace = graph_module['set_workspace']
        self._get_workspace_size = graph_module['get_workspace_size']
        self._launch = graph_module['launch']

        self._registered_hook: Optional[Any] = None

        self.meta_data: ModelMetaData = meta_data
        self.graph_module: CompiledModule = graph_module
        self.weights: List[Tensor] = weights
        self.kernels: List[CompiledModule] = kernels
        self.workspace: Optional[Storage] = None
        self.is_dynamic: bool = len(self.meta_data.dynamic_dims) > 0
        self.device: Device = self.weights[0].device if len(self.weights) > 0 else Device('cpu')

        self._init_weights_and_kernels()

    def __str__(self):
        rows = []
        for i in range(len(self.meta_data.input_signatures)):
            dtype, shape = self.meta_data.input_signatures[i][0], self.meta_data.input_signatures[i][1:]
            dtype = data_type(dtype)
            if i == 0:
                head = 'input'
            else:
                head = ''
            rows.append([head, dtype.short_name + str(shape)])
        for i in range(len(self.meta_data.output_signatures)):
            dtype, shape = self.meta_data.output_signatures[i][0], self.meta_data.output_signatures[i][1:]
            dtype = data_type(dtype)
            if i == 0:
                head = 'output'
            else:
                head = ''
            rows.append([head, dtype.short_name + str(shape)])
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

    def _init_weights_and_kernels(self):
        weights_buffer = Array(void_p, len(self.weights))
        kernels_buffer = Array(void_p, len(self.kernels))
        for i in range(len(self.weights)):
            weights_buffer[i] = self.weights[i].storage.addr
        for i in range(len(self.kernels)):
            kernels_buffer[i] = ctypes_func_pointer(self.kernels[i]['launch'].ctypes_func)
        self._init(len(self.kernels), kernels_buffer, len(self.weights), weights_buffer)

    def _create_outputs(self, inputs):
        """
        Create the output tensors.

        Parameters
        ----------
        inputs: Sequence[hidet.Tensor]
            The input tensors.

        Returns
        -------
        ret: List[hidet.Tensor]
            The output tensors.
        """
        from hidet.graph.tensor import empty

        dtypes = []
        shapes = []

        if self.is_dynamic:
            # set the dynamic dims
            for name, (tensor_index, dim_index) in self.meta_data.dynamic_dims:
                runtime_api.set_symbol_value(name, inputs[tensor_index].shape[dim_index])

            # get the output shapes for this input size
            for output_index, sig in enumerate(self.meta_data.output_signatures):
                shape_buffer = Array(i32, len(sig) - 1)
                self._get_output_shape(output_index, shape_buffer)
                dtypes.append(sig[0])
                shapes.append(list(shape_buffer))
        else:
            for sig in self.meta_data.output_signatures:
                dtypes.append(sig[0])
                shapes.append(sig[1:])
        return [empty(shape, dtype, device=self.device) for shape, dtype in zip(shapes, dtypes)]

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
        outputs = self._create_outputs(inputs)
        required_workspace_size = self._get_workspace_size()
        if self.workspace is None or self.workspace.num_bytes < required_workspace_size:
            self.workspace = Storage.new(self.meta_data.device, required_workspace_size)
            self._set_workspace(self.workspace.addr)
        self._launch(*inputs, *outputs)
        return outputs

    def save(self, path: str):
        save_model(self, path)


def load_model(path: str) -> CompiledModel:
    temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with

    with zipfile.ZipFile(path, 'r') as zf:
        files_to_extract: List[str] = zf.namelist()
        files_to_extract.remove('weights.npz')  # weights are loaded directly from the zip file
        zf.extractall(temp_dir.name, files_to_extract)

        with zf.open('weights.npz', 'r') as f:
            with zipfile.ZipFile(f, 'r') as npz:
                weights = []
                for name in npz.namelist():
                    with npz.open(name, 'r') as npy_file:
                        weights.append(hidet.asarray(numpy.load(npy_file)))

    with open(os.path.join(temp_dir.name, 'meta_data.json'), 'r') as f:
        meta_data_state = json.load(f)
        meta_data = ModelMetaData.from_state(meta_data_state)

    num_kernels = meta_data.num_kernels
    kernel_modules = [
        CompiledModule(module_dir=os.path.join(temp_dir.name, 'kernels', str(i))) for i in range(num_kernels)
    ]
    graph_module = CompiledModule(module_dir=os.path.join(temp_dir.name, 'graph_module'))
    weights = [hidet.asarray(weights[i]).to(device=meta_data.device) for i in range(len(weights))]
    ret = CompiledModel(meta_data, graph_module, weights, kernel_modules)

    # prevent the temp dir from being deleted before the model is deleted
    ret._temp_dir = temp_dir  # pylint: disable=attribute-defined-outside-init, protected-access
    return ret


def save_model(model: CompiledModel, path: str):
    def _save_files_under(dir_path: str, zf: zipfile.ZipFile, dir_in_zip: str):
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_in_zip = os.path.join(dir_in_zip, os.path.relpath(file_path, dir_path))
                with zf.open(file_in_zip, 'w') as f:
                    with open(file_path, 'rb') as f2:
                        f.write(f2.read())

    meta_data = model.meta_data.export_state()
    with zipfile.ZipFile(path, 'w') as zf:
        # meta info
        with zf.open('meta_data.json', 'w') as f:
            meta_bytes = json.dumps(meta_data, indent=4).encode('utf-8')
            f.write(meta_bytes)

        # save the kernels
        for i, kernel_module in enumerate(model.kernels):
            _save_files_under(kernel_module.module_dir, zf, 'kernels/{}/'.format(i))

        # save the modules
        _save_files_under(model.graph_module.module_dir, zf, 'graph_module/')

        # save weights
        with zf.open('weights.npz', 'w', force_zip64=True) as f:  # force_zip64 is required for >4GB weights
            numpy.savez(f, *[weight.cpu().numpy() for weight in model.weights])
