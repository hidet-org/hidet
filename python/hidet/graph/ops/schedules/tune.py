from typing import Union, Sequence, TypeVar, Any, Dict, List, Tuple, Set, Optional
import os
import itertools
import inspect
from tqdm import tqdm
import numpy as np
from hidet.ir.func import IRModule
from hidet.ir.task import Task
import hidet.option
from hidet.utils import prod, strict_zip
from .resolve import dummy_inputs_from_task

Choice = TypeVar('Choice')


class TuningSpace:
    MAX_SPACE_SIZE = 10000

    def __init__(self):
        self.spaces: Dict[int, Dict[str, Any]] = {}
        self.existing_names: List[str] = []

    def iterate_space(self, level: int):
        sub_keys = list(self.spaces[level].keys())
        sub_spaces = list(self.spaces[level].values())
        space_size = prod([len(s) for s in sub_spaces])
        if space_size > self.MAX_SPACE_SIZE:
            raise ValueError(
                f'The search space has {space_size} schedules, '
                f'which is too large. Please consider to reduce the search space.'
            )
        for values in itertools.product(sub_spaces):
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


def tune(template_func, task: Task, target_device: str, working_dir: str) -> IRModule:
    from hidet.backend import BuildInstance, batch_build_ir_modules
    from hidet.runtime import CompiledFunction

    # get ir modules to tune
    if hasattr(template_func, '_tuning_space'):
        tuning_space: TuningSpace = getattr(template_func, '_tuning_space')
        # iterate space and instantiate schedules into tensor programs
        kwargs_list = list(tuning_space.iterate_space(hidet.option.get_search_space()))
    else:
        kwargs_list = [{}]

    ir_modules = []
    for kwargs in kwargs_list:
        ir_modules.append(template_func(**kwargs))

    if len(ir_modules) == 1:
        return ir_modules[0]

    # build ir modules into compiled functions
    build_instances = [
        BuildInstance(
            ir_module=ir_module,
            output_dir=os.path.join(working_dir, 'resolve', str(idx)),
            keep_ir=False,
            nvcc_keep=False,
            verbose=False,
        )
        for idx, ir_module in enumerate(ir_modules)
    ]
    compiled_funcs: List[Optional[CompiledFunction]] = batch_build_ir_modules(
        build_instances, parallel=hidet.option.get_parallel_build(), verbose=True
    )
    if any([f is None for f in compiled_funcs]):
        raise ValueError('All ir modules failed to build.')

    # benchmark
    dummy_inputs = dummy_inputs_from_task(task, target_device=target_device)
    latencies = []
    warmup, number, repeat = hidet.option.get_option('bench_config')
    for ir_module, compiled_func in tqdm(
        strict_zip(ir_modules, compiled_funcs), desc='Benchmarking', total=len(ir_modules)
    ):
        if compiled_func:
            repeat_latency = compiled_func.profile(*dummy_inputs, warmup=warmup, number=number, repeat=repeat)
            latency = float(np.median(repeat_latency))
        else:
            # this ir module failed in building, skip
            latency = 1e30
        latencies.append(latency)

    # generate summary
    columns = []
    columns.append(['index'] + [str(v) for v in range(len(ir_modules))])
    keys = list(kwargs_list[0].keys())
    for key in keys:
        columns.append([key] + [str(kwargs[key]) for kwargs in kwargs_list])
    columns.append(['latency'] + [str(v) for v in latencies])
    column_widths = [max([len(str(v)) for v in column]) + 2 for column in columns]
    justified_columns = []
    for column, width in zip(columns, column_widths):
        justified_columns.append(''.join([v.ljust(width) for v in column]))
    summary = '\n'.join(''.join(row_items) for row_items in zip(*justified_columns))
    with open(os.path.join(working_dir, 'tune_summary.txt'), 'w') as f:
        f.write(summary)

    # select the best schedule and return
    return ir_modules[np.argmin(latencies)]

