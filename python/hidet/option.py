from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional, Callable, Iterable
import os

class OptionRegistry:
    """
    A registry for registered option.
    """
    registered_options: Dict[str, OptionRegistry] = {}

    def __init__(
            self,
            name: str,
            type_hint: str,
            description: str,
            default_value: Any,
            choices: Optional[Iterable[Any]] = None,
            normalizer: Optional[Callable[[Any], Any]] = None,
            checker: Optional[Callable[[Any], bool]] = None
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
            checker: Optional[Callable[[Any], bool]] = None
    ):
        registered_options = OptionRegistry.registered_options
        if name in registered_options:
            raise KeyError(f'Option {name} has already been registered.')
        registered_options[name] = OptionRegistry(name, type_hint, description, defalut_value, normalizer, choices, checker)
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
        checker=lambda x: isinstance(x, tuple) and len(x) == 3 and all(v >= 0 for v in x)
    ).register_option(
        name='search_space',
        type_hint='int',
        description='The search space level.',
        defalut_value=0,
        choices=[0, 1, 2]
    ).register_option(
        name='cache_operator',
        type_hint='bool',
        description='Whether to enable operator cache on disk.',
        defalut_value=True,
        choices=[True, False]
    ).register_option(
        name='cache_dir',
        type_hint='path',
        description='The directory to store the cache.',
        defalut_value=os.path.abspath(
            os.path.join(git_utils.get_git_repo_root(), '.hidet_cache')  # developer mode
            if git_utils.in_git_repo() else
            os.path.join(os.path.expanduser('~'), '.hidet', 'cache')     # user mode
        ),
        normalizer=lambda x: os.path.abspath(x)
    ).register_option(
        name='parallel_build',
        type_hint='bool',
        defalut_value=True,
        description='Whether to build operators in parallel.',
        choices=[True, False]
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
        OptionContext.stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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


def current_context() -> OptionContext:
    """
    Get the current option context.

    To get the value of an option in the current context::

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

    To set options in the new context, use the ``with`` statement::

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
    """Set the benchmark config of operator tuning.

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

def search_space(space: int):
    """Set the schedule space level of tunable operator.

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

        hidet.space_level(2)

    After calling above function, all subsequent compilation would use space level 2, until we call this
    function again with another space level.

    Parameters
    ----------
    space: int
        The space level to use. Candidates: 0, 1, and 2.
    """
    OptionContext.current().set_option('search_space', space)

def cache_operator(enabled: bool = True):
    """
    Whether to cache compiled operator on disk.

    By default, hidet would cache all compiled operator and reuse whenever possible.

    If user wants to disable the cache, run

    .. code-block:: python

        hidet.cache_operator(False)

    Parameters
    ----------
    enabled: bool
        Whether to cache the compiled operator.
    """
    OptionContext.current().set_option('cache_operator', enabled)

def cache_dir(new_dir: str):
    """
    Set the directory to store the cache.

    - If the hidet source code is in a git repository, the cache will be stored in the root directory of the git
      repository.
    - Otherwise, the cache will be stored in ``~/.hidet/cache``.

    Parameters
    ----------
    new_dir: str
        The new directory to store the cache.
    """
    OptionContext.current().set_option('cache_dir', new_dir)

def parallel_build(enabled: bool = True):
    """
    Whether to build operators in parallel.

    Parameters
    ----------
    enabled: bool
        Whether to build operators in parallel.
    """
    OptionContext.current().set_option('parallel_build', enabled)
