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
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass
import os
import json
import time
from collections import namedtuple
from hidet.runtime.compiled_module import CompiledModule, CompiledFunction, load_compiled_module
from hidet.ir.dtypes import i32
from hidet.ffi import runtime_api
from hidet.ffi.utils import Array


@dataclass
class TensorSignature:
    device: str
    dtype: str
    shape: List[Union[str, int]]


@dataclass
class TaskMetaData:
    symbols: List[str]
    inputs: List[TensorSignature]
    outputs: List[TensorSignature]
    target: str
    num_candidates: int
    hidet_version: str


class CompiledTask:
    def __init__(self, task_dir: str):
        self.task_dir: str = task_dir
        self.meta_data: TaskMetaData = self._load_meta_data()
        self.task_module: CompiledModule = load_compiled_module(task_dir)
        self.candidates: List[CompiledFunction] = [
            self.task_module['launch_{}'.format(i)] for i in range(self.meta_data.num_candidates)
        ]
        self.dispatch_table_path = os.path.join(self.task_dir, 'dispatch_table.txt')
        self.dispatch_table: Dict[Tuple[int, ...], int] = self._load_dispatch_table()

        self._get_input_shape = self.task_module['get_input_shape']
        self._get_output_shape = self.task_module['get_output_shape']

    def __call__(self, *args):
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

    def _load_dispatch_table(self):
        if not os.path.exists(self.dispatch_table_path):
            return {}
        dispatch_table = {}
        with open(self.dispatch_table_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue
                items = line.split()
                if len(items) == 0:
                    continue
                if len(items) != len(self.meta_data.symbols) + 1:
                    os.remove(self.dispatch_table_path)
                    raise RuntimeError(f'Invalid dispatch table: {self.dispatch_table_path}')
                key = tuple(int(item) for item in items[:-1])
                value = int(items[-1])
                dispatch_table[key] = value
        return dispatch_table

    def _get_symbol_values(self) -> Tuple[int, ...]:
        return tuple(runtime_api.get_symbol_value(symbol) for symbol in self.meta_data.symbols)

    def create_outputs(self):
        import hidet

        outputs = []

        for idx, sig in enumerate(self.meta_data.outputs):
            shape_buffer = Array(i32, len(sig.shape))
            self._get_output_shape(idx, shape_buffer)
            shape: List[int] = list(shape_buffer)
            outputs.append(hidet.empty(shape, sig.dtype, sig.device))
        return outputs

    def pick_best_candidate(self, inputs, outputs) -> int:
        import hidet

        key = self._get_symbol_values()
        if key not in self.dispatch_table:
            warmup, number, repeat = hidet.option.get_bench_config()
            latencies = []
            for candidate in self.candidates:
                for _ in range(warmup):
                    candidate(*inputs, *outputs)
                candidate_latency = 0.0
                for _ in range(repeat):
                    hidet.cuda.synchronize()
                    t1 = time.time()
                    for _ in range(number):
                        candidate(*inputs, *outputs)
                    hidet.cuda.synchronize()
                    t2 = time.time()
                    candidate_latency += (t2 - t1) / number
                latencies.append(candidate_latency / repeat)
            self.dispatch_table[key] = latencies.index(min(latencies))

            if not os.path.exists(self.dispatch_table_path):
                with open(self.dispatch_table_path, 'w') as f:
                    f.write(' '.join(self.meta_data.symbols) + '\n')
            with open(self.dispatch_table_path, 'a') as f:
                f.write(' '.join([str(v) for v in key]) + ' ' + str(self.dispatch_table[key]) + '\n')

        candidate_index = self.dispatch_table[key]
        if candidate_index >= len(self.candidates):
            raise RuntimeError(f'Invalid candidate index: {candidate_index}')
        return candidate_index

    def run_async(self, inputs):
        from hidet import option, ir

        if option.runtime_check():
            symbol_map = {}
            for i, (traced, new) in enumerate(zip(self.meta_data.inputs, inputs)):
                if ir.data_type(traced.dtype) != new.dtype:
                    raise RuntimeError(
                        f"dtype mismatch at arg {i} between original: {traced.dtype} and new: {new.dtype}"
                    )
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

        outputs = self.create_outputs()
        candidate = self.candidates[self.pick_best_candidate(inputs, outputs)]
        candidate(*inputs, *outputs)
        return outputs


def load_compiled_task(compiled_task_dir: str) -> CompiledTask:
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
