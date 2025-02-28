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
from typing import List, Dict, Tuple, Union, Optional, Iterable
from dataclasses import dataclass
import os
import json
from collections import namedtuple
from hidet.runtime.compiled_module import CompiledModule, CompiledFunction, load_compiled_module
from hidet.ir.dtypes import i32
from hidet.ffi.array import Array
from hidet.runtime.dispatch_table import DispatchTable, IntervalsDispachTable, PointsDispachTable


@dataclass
class TensorSignature:
    device: str
    dtype: str
    shape: List[Union[str, int]]


@dataclass
class TaskMetaData:
    name: str
    symbols: List[str]
    inputs: List[TensorSignature]
    outputs: List[TensorSignature]
    share_map: Dict[int, int]
    target: str
    num_candidates: int
    hidet_version: str


class CompiledTask:
    """
    A compiled task is a special kind of compiled module that implements a computation task.

    A compiled task is a compiled module with the following conventions:

    1. The compiled module contains functions named `launch_0`, `launch_1`, ..., `launch_N-1`, where N is the number of
       candidates for the task.
    2. There are two shape-related functions `get_input_shape` and `get_output_shape` that return the shape of inputs
       and outputs respectively.

    When a compiled task is called, the input arguments should be consistent with the input signature of the task.
    The compiled task will pick the best candidate based on the input shapes and dispatch the computation to the
    corresponding candidate. The output tensors will be created and passed to the candidate function as arguments.
    When the candidate finishes the execution, the output tensors will be returned.

    This class is not intended to be instantiated by users directly. Instead, users should use the
    :func:`load_compiled_task` function to load a compiled task from the given directory, or use
    :func:`hidet.drivers.build_task` to build a compiled task from a task definition.

    Parameters
    ----------
    task_dir: str
        The directory of the compiled task.
    """

    def __init__(self, task_dir: str):
        self.task_dir: str = task_dir
        self.meta_data: TaskMetaData = self._load_meta_data()
        self.task_module: CompiledModule = load_compiled_module(task_dir)
        self.candidates: List[CompiledFunction] = [
            self.task_module['launch_{}'.format(i)] for i in range(self.meta_data.num_candidates)
        ]
        self.dispatch_table: DispatchTable = self.construct_dispatch_table()

        self._get_input_shape = self.task_module['get_input_shape']
        self._get_output_shape = self.task_module['get_output_shape']

    def __call__(self, *args):
        """
        Run the compiled task with the given arguments.

        Parameters
        ----------
        args: a sequence of input tensors or scalars
            The input arguments. They should be consistent with the input signature of the task.

        Returns
        -------
        A sequence of output tensors:
            The output tensors. They are created by the task and passed to the candidate function as arguments.
            When the candidate finishes the execution, the output tensors will be returned.
        """
        outs = self.run_async(args)
        if len(outs) == 1:
            return outs[0]
        else:
            return outs

    def _load_meta_data(self) -> TaskMetaData:
        from hidet.utils.dataclass import from_dict

        meta_data_path = os.path.join(self.task_dir, 'meta.json')
        with open(meta_data_path, 'r') as f:
            return from_dict(TaskMetaData, json.load(f))

    def _load_compiled_modules(self) -> List[CompiledModule]:
        compiled_modules = []
        candidates_dir = os.path.join(self.task_dir, 'candidates')
        if not os.path.exists(candidates_dir) or not os.path.isdir(candidates_dir):
            raise RuntimeError(f'Cannot find candidates dir: {candidates_dir}')
        for module_dir in os.listdir(candidates_dir):
            if not os.path.isdir(module_dir):
                continue
            compiled_modules.append(CompiledModule(module_dir))
        if len(compiled_modules) == 0:
            raise RuntimeError(f'No compiled module found in {candidates_dir}')
        return compiled_modules

    def use_dynamic(self) -> bool:
        return len(self.meta_data.symbols) != 0 and len(self.candidates) > 1

    def construct_dispatch_table(self) -> DispatchTable:
        if self.use_dynamic():
            input_shapes = [inp.shape for inp in self.meta_data.inputs]
            output_shapes = [out.shape for out in self.meta_data.outputs]
            try:
                return IntervalsDispachTable(
                    candidates=self.candidates,
                    input_shapes=input_shapes,
                    output_shapes=output_shapes,
                    task_dir=self.task_dir,
                    symbols=self.meta_data.symbols,
                    name=self.meta_data.name,
                )
            except NotImplementedError:
                pass
        return PointsDispachTable(
            candidates=self.candidates, task_dir=self.task_dir, symbols=self.meta_data.symbols, name=self.meta_data.name
        )

    def create_outputs(self, inputs):
        import hidet

        outputs = []

        for idx, sig in enumerate(self.meta_data.outputs):
            shape_buffer = Array(i32, len(sig.shape))
            self._get_output_shape(idx, shape_buffer)
            shape: List[int] = list(shape_buffer)
            if idx not in self.meta_data.share_map:
                outputs.append(hidet.empty(shape, sig.dtype, sig.device))
            else:
                shared_tensor = inputs[self.meta_data.share_map[idx]]
                if not isinstance(shared_tensor, hidet.Tensor):
                    import torch

                    assert isinstance(shared_tensor, torch.Tensor), "Unknown tensor type"
                    tensor_dtype = getattr(torch, sig.dtype)

                    # we need to turn the tensor into a view with the graph output's shape & dtype
                    input_tensor = shared_tensor.view(*shape).view(tensor_dtype)
                else:
                    input_tensor = hidet.Tensor(
                        shape=shape, dtype=sig.dtype, device=sig.device, storage=shared_tensor.storage
                    )

                outputs.append(input_tensor)
        return outputs

    def pick_best_candidate(self, inputs, outputs) -> int:
        """
        Pick the best candidate kernel based on cached dispatch table or by benchmarking.
        Once determined, the chosen candidate is recorded in the dispatch table for future use.
        """
        return self.dispatch_table.pick_best_candidate(inputs, outputs)

    def run_async(self, inputs):
        """
        Run the compiled task with the given arguments.

        Parameters
        ----------
        inputs: a sequence of input tensors or scalars
            The input arguments. They should be consistent with the input signature of the task.

        Returns
        -------
        A sequence of output tensors:
            The output tensors. They are created by the task and passed to the candidate function as arguments.
            When the candidate finishes the execution, the output tensors will be returned.
        """
        from hidet import option

        if option.get_runtime_check():
            _check_inputs(self.meta_data.inputs, inputs)

        outputs = self.create_outputs(inputs)

        candidate = self.candidates[self.pick_best_candidate(inputs, outputs)]
        candidate(*inputs, *outputs)

        return outputs

    def profile(self, *args, warmup=1, number=2, repeat=10):
        """
        Run the compiled task with the given arguments and profile the execution time.

        Parameters
        ----------
        args: a sequence of input tensors or scalars
            The input arguments. They should be consistent with the input signature of the task.

        warmup: int
            The number of warmup runs.

        number: int
            The number of runs for each measurement.

        repeat: int
            The number of measurements.

        Returns
        -------
        latency: List[float]
            The measured latency in milliseconds. The length of the list is equal to `repeat`.
        """

        num_inputs = len(self.meta_data.inputs)
        inputs = args[:num_inputs]
        outputs = args[num_inputs:]

        # For operators like scatter_add_, if we run it multiple times on the same input & output tensors,
        # the input and output tensors will be wrong as they will be wrongly updated multiple times.
        # to avoid this, make a clone of the output tensors if they share the memory with some input tensors.
        if len(self.meta_data.share_map) > 0:
            from hidet import Tensor

            outputs = list(outputs)
            inputs = list(inputs)
            for output_idx in self.meta_data.share_map:
                original_output = outputs[output_idx]
                if isinstance(original_output, Tensor):
                    outputs[output_idx] = original_output.copy()
                else:
                    outputs[output_idx] = original_output.clone()
                args = inputs + outputs

        candidate = self.candidates[self.pick_best_candidate(inputs, outputs)]
        return candidate.profile(*args, warmup=warmup, number=number, repeat=repeat)


def load_compiled_task(compiled_task_dir: str) -> CompiledTask:
    """
    Load a compiled task from the given directory.

    Parameters
    ----------
    compiled_task_dir: str
        The directory of the compiled task.

    Returns
    -------
    ret: CompiledTask
        The loaded compiled task.
    """
    return CompiledTask(compiled_task_dir)


CompiledTaskKey = namedtuple('CompiledTaskKey', ['device', 'space', 'task_str'])


class CompiledTaskCache:
    def __init__(self):
        self.cached: Dict[Tuple[str, int, str], CompiledTask] = {}

    def contains(self, device_type: str, space: int, task_str: str) -> bool:
        key = CompiledTaskKey(device_type, space, task_str)
        return key in self.cached

    def get(self, device_type: str, space: int, task_str: str) -> Optional[CompiledTask]:
        key = CompiledTaskKey(device_type, space, task_str)
        return self.cached.get(key) if key in self.cached else None

    def add(self, device_type: str, space: int, task_str: str, compiled_task: CompiledTask):
        key = CompiledTaskKey(device_type, space, task_str)
        self.cached[key] = compiled_task


compiled_task_cache = CompiledTaskCache()


def _check_inputs(traced_inputs: Iterable[TensorSignature], inputs):
    from hidet.ir import data_type
    from hidet.graph.frontend.torch.utils import dtype_to_torch
    from torch import Tensor as TorchTensor

    symbol_map = {}
    for i, (traced, new) in enumerate(zip(traced_inputs, inputs)):
        if isinstance(new, TorchTensor):
            traced_dev_kind = traced.device.partition(':')[0]
            new_device_target = 'cuda' if new.device.type in ['cuda', 'vcuda'] else 'cpu'
            if traced_dev_kind != new_device_target:
                raise RuntimeError(
                    f"device mismatch at arg {i} between original: {traced.device} and new: {new.device.kind}"
                )
            if dtype_to_torch(data_type(traced.dtype)) != new.dtype:
                raise RuntimeError(f"dtype mismatch at arg {i} between original: {traced.dtype} and new: {new.dtype}")
            traced_shape = traced.shape
            concrete_shape = new.shape
            if len(traced_shape) != len(concrete_shape):
                raise RuntimeError(
                    f"Rank of input {i} not equal to original. ({len(concrete_shape)} vs. {len(traced_shape)})"
                )
            for j, (orig_shape, new_shape) in enumerate(zip(traced_shape, concrete_shape)):
                if isinstance(orig_shape, int) and orig_shape != new_shape:
                    raise RuntimeError(
                        f'shape mismatch at dimension {j}, original: \
                                        {orig_shape} vs. new: {new_shape}'
                    )
                elif orig_shape not in symbol_map:
                    symbol_map[orig_shape] = new_shape
                elif symbol_map[orig_shape] != new_shape:
                    raise RuntimeError(
                        f"There exists multiple instances of the same symbol {orig_shape}\
                        with different values in inputs (ex: {symbol_map[orig_shape]} and {new_shape})"
                    )
        else:
            traced_dev_kind = traced.device.partition(':')[0]
            if traced_dev_kind != new.device.target:
                raise RuntimeError(
                    f"device mismatch at arg {i} between original: {traced.device} and new: {new.device.kind}"
                )
            if data_type(traced.dtype) != new.dtype:
                raise RuntimeError(f"dtype mismatch at arg {i} between original: {traced.dtype} and new: {new.dtype}")
            traced_shape = traced.shape
            concrete_shape = new.shape
            if len(traced_shape) != len(concrete_shape):
                raise RuntimeError(
                    f"Rank of input {i} not equal to original. ({len(concrete_shape)} vs. {len(traced_shape)})"
                )
            for j, (orig_shape, new_shape) in enumerate(zip(traced_shape, concrete_shape)):
                if isinstance(orig_shape, int) and orig_shape != new_shape:
                    raise RuntimeError(
                        f'shape mismatch at dimension {j}, original: \
                                        {orig_shape} vs. new: {new_shape}'
                    )
                elif orig_shape not in symbol_map:
                    symbol_map[orig_shape] = new_shape
                elif symbol_map[orig_shape] != new_shape:
                    raise RuntimeError(
                        f"There exists multiple instances of the same symbol {orig_shape}\
                        with different values in inputs (ex: {symbol_map[orig_shape]} and {new_shape})"
                    )
