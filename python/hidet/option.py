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
from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable, Iterable, Tuple
import os


class OptionRegistry:
    registered_options: Dict[str, OptionRegistry] = {}

    def __init__(
        self,
        name: str,
        type_hint: str,
        description: str,
        default_value: Any,
        normalizer: Optional[Callable[[Any], Any]] = None,
        choices: Optional[Iterable[Any]] = None,
        checker: Optional[Callable[[Any], bool]] = None,
    ):
        self.name = name
        self.type_hint = type_hint
        self.description = description
        self.default_value = default_value
        self.normalizer = normalizer
        self.choices = choices
        self.checker = checker

    @staticmethod
    def register_option(
        name: str,
        type_hint: str,
        description: str,
        defalut_value: Any,
        normalizer: Optional[Callable[[Any], Any]] = None,
        choices: Optional[Iterable[Any]] = None,
        checker: Optional[Callable[[Any], bool]] = None,
    ):
        registered_options = OptionRegistry.registered_options
        if name in registered_options:
            raise KeyError(f'Option {name} has already been registered.')
        registered_options[name] = OptionRegistry(
            name, type_hint, description, defalut_value, normalizer, choices, checker
        )
        return OptionRegistry


register_option = OptionRegistry.register_option


def register_hidet_options():
    from hidet.utils import git_utils

    register_option(
        name='bench_config',
        type_hint='Tuple[int, int, int]',
        description='The (warmup, number, repeat) parameters for benchmarking. '
        'The benchmarking will run warmup + number * repeat times.',
        defalut_value=(3, 10, 3),
    ).register_option(
        name='search_space',  #
        type_hint='int',
        description='The search space level.',
        defalut_value=0,
        choices=[0, 1, 2],
    ).register_option(
        name='cache_operator',
        type_hint='bool',
        description='Whether to enable operator cache on disk.',
        defalut_value=True,
        choices=[True, False],
    ).register_option(
        name='cache_dir',
        type_hint='path',
        description='The directory to store the cache.',
        defalut_value=os.path.abspath(
            os.path.join(git_utils.git_repo_root(), '.hidet_cache')  # developer mode
            if git_utils.in_git_repo()
            else os.path.join(os.path.expanduser('~'), '.hidet', 'cache')  # user mode
        ),
        normalizer=os.path.abspath,
    ).register_option(
        name='parallel_build',
        type_hint='bool',
        defalut_value=True,
        description='Whether to build operators in parallel.',
        choices=[True, False],
    ).register_option(
        name='save_lower_ir',
        type_hint='bool',
        defalut_value=False,
        description='Whether to save the IR when lower an IRModule to the operator cache.',
        choices=[True, False],
    )


register_hidet_options()


class OptionContext:
    """
    The option context.
    """

    stack: List[OptionContext] = []

    def __init__(self):
        self.options: Dict[str, Any] = {}

    def __str__(self):
        pass

    def __enter__(self):
        """
        Enter the option context.

        Returns
        -------
        ret: OptionContext
            The option context itself.
        """
        OptionContext.stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the option context.
        """
        OptionContext.stack.pop()

    @staticmethod
    def current() -> OptionContext:
        return OptionContext.stack[-1]

    def set_option(self, name: str, value: Any):
        if name not in OptionRegistry.registered_options:
            raise KeyError(f'Option {name} has not been registered.')
        registry = OptionRegistry.registered_options[name]
        if registry.normalizer is not None:
            value = registry.normalizer(value)
        if registry.checker is not None:
            if not registry.checker(value):
                raise ValueError(f'Invalid value for option {name}: {value}')
        if registry.choices is not None:
            if value not in registry.choices:
                raise ValueError(f'Invalid value for option {name}: {value}, choices {registry.choices}')
        self.options[name] = value

    def get_option(self, name: str) -> Any:
        for ctx in reversed(OptionContext.stack):
            if name in ctx.options:
                return ctx.options[name]
        if name not in OptionRegistry.registered_options:
            raise KeyError(f'Option {name} has not been registered.')
        registry = OptionRegistry.registered_options[name]
        return registry.default_value


OptionContext.stack.append(OptionContext())


def dump_options() -> Dict[str, Any]:
    """
    Dump the options in option context stack.

    Returns
    -------
    ret: Dict[str, Any]
        The dumped options.
    """
    return {'option_context_stack': OptionContext.stack, 'registered_options': OptionRegistry.registered_options}


def restore_options(dumped_options: Dict[str, Any]):
    """
    Restore the options from dumped options.

    Parameters
    ----------
    dumped_options: Dict[str, Any]
        The dumped options.
    """
    OptionContext.stack = dumped_options['option_context_stack']
    OptionRegistry.registered_options = dumped_options['registered_options']


def current_context() -> OptionContext:
    """
    Get the current option context.

    To get the value of an option in the current context:

    .. code-block:: python

        ctx = hidet.option.current_context()
        cache_dir: str = ctx.get_option('cache_dir')
        cache_operator: bool = ctx.get_option('cache_operator')
        ...

    Returns
    -------
    ctx: OptionContext
        The current option context.
    """
    return OptionContext.current()


def context() -> OptionContext:
    """
    Create a new option context.

    To set options in the new context, use the ``with`` statement:

    .. code-block:: python

        with hidet.option.context() as ctx:
            hidet.option.cache_dir('./new_cache_dir')               # set predefined option
            hidet.option.set_option('other_option', 'other_value')  # set a custom option
            ...

    Returns
    -------
    ctx: OptionContext
        The new option context.
    """
    return OptionContext()


def set_option(name: str, value: Any):
    """
    Set the value of an option in current option context.

    The option must be registered before setting via :py:func:`hidet.option.register_option`.

    Parameters
    ----------
    name: str
        The name of the option.
    value: Any
        The value of the option.
    """
    OptionContext.current().set_option(name, value)


def get_option(name: str) -> Any:
    """
    Get the value of an option in current option context.

    Parameters
    ----------
    name: str
        The name of the option.

    Returns
    -------
    ret: Any
        The value of the option.
    """
    return OptionContext.current().get_option(name)


def bench_config(warmup: int = 1, number: int = 5, repeat: int = 5):
    """
    Set the benchmark config of operator tuning.

    To profile a schedule, hidet will run the following code:

    .. code-block:: python

        for i in range(warmup):
            run()
        latency = []
        for i in range(repeat):
            synchronize device
            t1 = time()
            for j in range(number):
                run()
            synchronize device
            t2 = time()
            latency.append((t2 - t1) / number)
        return median of latency

    Thus, there will be total ``warmup + number * repeat`` times of execution.

    Parameters
    ----------
    warmup: int
        The number of warmup runs.
    number: int
        The number of runs in a repeat.
    repeat: int
        The number of repeats.
    """
    OptionContext.current().set_option('bench_config', (warmup, number, repeat))


def get_bench_config() -> Tuple[int, int, int]:
    """
    Get the benchmark config of operator tuning.

    Returns
    -------
    ret: Tuple[int, int, int]
        The benchmark config.
    """
    return OptionContext.current().get_option('bench_config')


def search_space(space: int):
    """
    Set the schedule search space of tunable operator.

    Some operators can be tuned in hidet to achieve the best performance, such as matrix multiplication.

    During tuning, different operator schedules will be tried and profiled to get the best one.

    We call the space of the tried operator schedule `schedule space`. There is a trade-off between the
    tuning time and the operator execution time. If we try more schedules, the tuning process would take
    longer time, and we are likely to find better schedule.

    This function allows user to set the space level that controls the search space we tried.

    By convention, we have space level

    - 0 for schedule space contains only a single schedule.
    - 1 for schedule space contains tens of schedules so that the tuning time will be less than 1 minute.
    - 2 for arbitrary large space.

    Usage

    .. code-block:: python

        hidet.search_space(2)

    After calling above function, all subsequent compilation would use space level 2, until we call this
    function again with another space level.

    Parameters
    ----------
    space: int
        The space level to use. Candidates: 0, 1, and 2.
    """
    OptionContext.current().set_option('search_space', space)


def get_search_space() -> int:
    """
    Get the schedule search space of tunable operator.

    Returns
    -------
    ret: int
        The schedule space level.
    """
    return OptionContext.current().get_option('search_space')


def cache_operator(enabled: bool = True):
    """
    Whether to cache compiled operator on disk.

    By default, hidet would cache all compiled operator and reuse whenever possible.

    If user wants to disable the cache, run

    .. code-block:: python

        hidet.option.cache_operator(False)

    Parameters
    ----------
    enabled: bool
        Whether to cache the compiled operator.
    """
    OptionContext.current().set_option('cache_operator', enabled)


def get_cache_operator() -> bool:
    """
    Get the option value of whether to cache compiled operator on disk.

    Returns
    -------
    ret: bool
        Whether to cache the compiled operator.
    """
    return OptionContext.current().get_option('cache_operator')


def cache_dir(new_dir: str):
    """
    Set the directory to store the cache.

    The default cache directory:

    - If the hidet code is in a git repo, the cache will be stored in the repo root:
      ``hidet-repo/.hidet_cache``.
    - Otherwise, the cache will be stored in the user home directory: ``~/.hidet/cache``.

    Parameters
    ----------
    new_dir: str
        The new directory to store the cache.
    """
    OptionContext.current().set_option('cache_dir', new_dir)


def get_cache_dir() -> str:
    """
    Get the directory to store the cache.

    Returns
    -------
    ret: str
        The directory to store the cache.
    """
    return OptionContext.current().get_option('cache_dir')


def parallel_build(enabled: bool = True):
    """
    Whether to build operators in parallel.

    Parameters
    ----------
    enabled: bool
        Whether to build operators in parallel.
    """
    OptionContext.current().set_option('parallel_build', enabled)


def get_parallel_build() -> bool:
    """
    Get the option value of whether to build operators in parallel.

    Returns
    -------
    ret: bool
        Whether to build operators in parallel.
    """
    return OptionContext.current().get_option('parallel_build')


def save_lower_ir(enabled: bool = True):
    """
    Whether to save the lower IR.

    Parameters
    ----------
    enabled: bool
        Whether to save the lower IR.
    """
    OptionContext.current().set_option('save_lower_ir', enabled)


def get_save_lower_ir() -> bool:
    """
    Get the option value of whether to save the lower IR.

    Returns
    -------
    ret: bool
        Whether to save the lower IR.
    """
    return OptionContext.current().get_option('save_lower_ir')
