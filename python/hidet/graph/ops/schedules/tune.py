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
from typing import Union, Sequence, TypeVar, Any, Dict, List, Optional
import os
import itertools
from tqdm import tqdm
import numpy as np
from hidet.ir.func import IRModule
from hidet.ir.task import Task
import hidet.option
from hidet.utils import prod
from .resolve import dummy_inputs_from_task

Choice = TypeVar('Choice')


class ScheduleError(Exception):
    pass


class TuningSpace:
    MAX_SPACE_SIZE = 12000

    def __init__(self):
        self.spaces: Dict[int, Dict[str, Any]] = {}
        self.existing_names: List[str] = []

    def iterate_space(self, level: int):
        # when given level is not defined, down to lower level
        while level > 0 and level not in self.spaces:
            level -= 1
        if level == 0 and level not in self.spaces:
            yield {}
            return

        sub_keys = list(self.spaces[level].keys())
        sub_spaces = list(self.spaces[level].values())
        space_size = prod([len(s) for s in sub_spaces])
        if space_size > self.MAX_SPACE_SIZE:
            raise ValueError(
                f'The search space has {space_size} schedules, '
                f'which is larger than the predefined limit {self.MAX_SPACE_SIZE}. '
                f'Please consider to reduce the search space.'
            )
        for values in itertools.product(*sub_spaces):
            kwargs = {}
            for key, value in zip(sub_keys, values):
                if ',' in key:
                    for name, v in zip(key.split(','), value):
                        kwargs[name] = v
                else:
                    kwargs[key] = value
            yield kwargs

    def add_sub_space(self, level: int, names: str, choices: Sequence[Union[Choice, Sequence[Choice]]]):
        if level not in self.spaces:
            self.spaces[level] = {}
        names: List[str] = [name.strip() for name in names.split(',')]
        for name in names:
            if name in self.existing_names:
                raise ValueError(f'Subspace {name} is already added.')
        if len(names) > 1:
            for choice in choices:
                if not hasattr(choice, '__len__'):
                    raise ValueError(f'When multiple names are given, choices must be iterable, got {type(choice)}')
                if len(choice) != len(names):
                    raise ValueError(f'Number of choices {len(choice)} does not match number of names {len(names)}.')
        self.spaces[level][",".join(names)] = choices


def space(level: int, names: str, choices: Sequence[Union[Choice, Sequence[Choice]]]):
    def wrapper(func):
        if not hasattr(func, '_tuning_space'):
            # attach tuning space when the first time of this function is called
            setattr(func, '_tuning_space', TuningSpace())
        tuning_space: TuningSpace = getattr(func, '_tuning_space')
        tuning_space.add_sub_space(level, names, choices)
        return func

    return wrapper


def _generate_summary(kwargs_list: List[Dict[str, Any]], latencies: List[float]) -> str:
    # sort by latency
    indices, kwargs_list, latencies = zip(
        *sorted(zip(range(len(latencies)), kwargs_list, latencies), key=lambda x: x[-1])
    )

    # generate summary column by column
    columns = []
    columns.append(['index'] + [str(v) for v in indices])
    keys = reversed(kwargs_list[0].keys())
    for key in keys:
        columns.append([key] + [str(kwargs[key]) for kwargs in kwargs_list])
    columns.append(['latency'] + [f'{v:.6f}' for v in latencies])
    column_widths = [max([len(str(v)) for v in column]) + 2 for column in columns]
    justified_columns = []
    for column, width in zip(columns, column_widths):
        justified_columns.append([v.ljust(width) for v in column])
    summary = '\n'.join(''.join(row_items) for row_items in zip(*justified_columns))

    return summary


def tune(template_func, task: Task, target_device: str, working_dir: str) -> IRModule:
    from hidet.driver import build_ir_module_batch
    from hidet.runtime import CompiledFunction

    # get ir modules to tune
    if hasattr(template_func, '_tuning_space'):
        tuning_space: TuningSpace = getattr(template_func, '_tuning_space')
        # iterate space and instantiate schedules into tensor programs
        kwargs_list = list(tuning_space.iterate_space(hidet.option.get_search_space()))
    else:
        raise ValueError(
            'No tuning space is attached to the template function.\n'
            'Please use @tune.space to decorate the template function to define the search space.'
        )

    ir_modules = []
    ir_modules_kwargs = []
    for kwargs in kwargs_list:
        try:
            ir_modules.append(template_func(**kwargs))
            ir_modules_kwargs.append(kwargs)
        except ScheduleError:
            # the schedule is invalid, skip it
            continue

    if len(ir_modules) == 0:
        raise ValueError('No valid schedule is found.')
    elif len(ir_modules) == 1:
        # do not need to tune
        return ir_modules[0]

    # build ir modules into compiled functions
    compiled_funcs: List[Optional[CompiledFunction]] = build_ir_module_batch(
        ir_modules, func_name=task.name, output_dir=os.path.join(working_dir, 'tuning'), parallel=True, verbose=True
    )
    assert len(compiled_funcs) == len(ir_modules)
    if any(f is None for f in compiled_funcs):
        raise ValueError('All ir modules failed to build.')

    # benchmark
    dummy_inputs = dummy_inputs_from_task(task, target_device=target_device)
    latencies = []
    warmup, number, repeat = hidet.option.get_option('bench_config')
    for compiled_func in tqdm(compiled_funcs, desc='Benchmarking', total=len(ir_modules), ncols=80):
        if compiled_func:
            repeat_latency = compiled_func.profile(*dummy_inputs, warmup=warmup, number=number, repeat=repeat)
            latency = float(np.median(repeat_latency))
        else:
            # this ir module failed in building, skip
            latency = 1e30
        latencies.append(latency)

    # generate summary
    summary = _generate_summary(ir_modules_kwargs, latencies)
    with open(os.path.join(working_dir, 'tuning_summary.txt'), 'w') as f:
        f.write(summary)

    # select the best schedule and return
    return ir_modules[np.argmin(latencies)]


def check(condition: bool, message: str = ""):
    if not condition:
        raise ScheduleError(message)
