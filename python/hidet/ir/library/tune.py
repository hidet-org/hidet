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
import itertools
import warnings
from typing import Union, Sequence, TypeVar, Any, Dict, List
from tqdm import tqdm

import hidet.option
from hidet.ir.module import IRModule
from hidet.utils import prod
from hidet.utils.multiprocess import parallel_imap_2ndlevel


Choice = TypeVar('Choice')


class ScheduleError(Exception):
    pass


class TuningSpace:
    MAX_SPACE_SIZE = 1200000

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

        sub_spaces = [s(level) if callable(s) else s for s in sub_spaces]
        sub_spaces = [list(s) for s in sub_spaces]

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

    def add_sub_space(self, level: int, name_choice_dict: Dict[str, Sequence[Union[Choice, Sequence[Choice],]]]):
        if level in self.spaces:
            raise ValueError(f'Level {level} is already defined.')
        if level == 0:
            raise ValueError('Level 0 is reserved for the default space. Use the default arguments to define it.')

        self.spaces[level] = {}
        for names, choices in name_choice_dict.items():
            if not callable(choices):
                choices = list(choices)
            names = [name.strip() for name in names.split(',')]
            for name in names:
                if name in self.existing_names:
                    raise ValueError(f'Subspace {name} is already added.')
            if len(names) > 1:
                for choice in choices:
                    if not hasattr(choice, '__len__'):
                        raise ValueError(f'When multiple names are given, choices must be iterable, got {type(choice)}')
                    if len(choice) != len(names):
                        raise ValueError(
                            f'Number of choices {len(choice)} does not match number of names {len(names)}.'
                        )
            self.spaces[level][",".join(names)] = choices


class MetricCollectContext:
    current = None

    def __init__(self):
        self.metrics: Dict[str, Union[str, int, float]] = {}

    def __enter__(self):
        MetricCollectContext.current = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        MetricCollectContext.current = None


def space(level: int, /, **subspaces: Sequence[Choice]):
    def wrapper(func):
        if not hasattr(func, 'tuning_space'):
            # attach tuning space when the first time of this function is called
            setattr(func, 'tuning_space', TuningSpace())
        tuning_space: TuningSpace = getattr(func, 'tuning_space')
        if 'combinations' in subspaces:
            combinations = subspaces.pop('combinations')
            subspaces.update(combinations)
        tuning_space.add_sub_space(level, subspaces)
        return func

    return wrapper


def extract_ir_modules(template_func) -> List[IRModule]:
    def _extract_ir_modules(kwargs):
        with MetricCollectContext() as metric_ctx:
            try:
                ir_module = template_func(**kwargs)
            except ScheduleError:
                # the schedule is invalid, skip it
                return None
        kwargs.update(metric_ctx.metrics)
        setattr(ir_module, '_tuning_kwargs', kwargs)  # workaround to pass kwargs to the tune function
        return ir_module

    MAX_VALID_SPACE_SIZE = 2000
    # get ir modules to tune
    if hasattr(template_func, 'tuning_space'):
        tuning_space: TuningSpace = getattr(template_func, 'tuning_space')
        # iterate space and instantiate schedules into tensor programs
        kwargs_list = list(tuning_space.iterate_space(hidet.option.get_search_space()))
    else:
        raise ValueError(
            'No tuning space is attached to the template function.\n'
            'Please use @tune.space to decorate the template function to define the search space.'
        )

    # Generate IR for all set of params
    from hidet.drivers.utils import lazy_initialize_cuda

    lazy_initialize_cuda()
    ir_modules = list(
        tqdm(
            parallel_imap_2ndlevel(_extract_ir_modules, kwargs_list),
            desc='Generating Hidet IR',
            total=len(kwargs_list),
            ncols=80,
        )
    )
    ir_modules = [i for i in ir_modules if i is not None]

    # Too many schedules
    if len(ir_modules) > MAX_VALID_SPACE_SIZE:
        warnings.warn(
            f'The tune space has {len(ir_modules)} valid schedules, '
            f'which is larger than the predefined limit {MAX_VALID_SPACE_SIZE}. '
            f'Please consider to reduce the search space.'
        )

    return ir_modules


def check(condition: bool, message: str = ""):
    if not condition:
        raise ScheduleError(message)


def metric(name: str, value: Union[float, int, str]):
    if MetricCollectContext.current is None:
        return
    ctx: MetricCollectContext = MetricCollectContext.current
    if name in ctx.metrics:
        warnings.warn(f'Metric {name} is already defined, the value will be overwritten.')
    ctx.metrics[name] = value
