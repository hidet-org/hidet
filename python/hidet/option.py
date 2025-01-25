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
from typing import Dict, Any, List, Optional, Callable, Iterable, Tuple, Union
import warnings
import os
import subprocess
import tomlkit


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


def create_toml_doc() -> tomlkit.TOMLDocument:
    def nest_flattened_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        new_dict = {}
        for k, v in d.items():
            if '.' in k:
                prefix, suffix = k.split('.', 1)
                if prefix not in new_dict:
                    new_dict[prefix] = {suffix: v}
                else:
                    new_dict[prefix][suffix] = v
            else:
                new_dict[k] = v
        for k, v in new_dict.items():
            if isinstance(v, dict):
                new_dict[k] = nest_flattened_dict(v)
        return new_dict

    def gen_doc(d: Dict[str, Any], toml_doc: tomlkit.TOMLDocument):
        for k, v in d.items():
            if isinstance(v, dict):
                table = tomlkit.table()
                gen_doc(v, table)
                toml_doc.add(k, table)
            elif isinstance(v, OptionRegistry):
                toml_doc.add(tomlkit.comment(v.description))
                if v.choices is not None:
                    toml_doc.add(tomlkit.comment(f'  choices: {v.choices}'))
                if isinstance(v.default_value, (bool, int, float, str)):
                    toml_doc.add(k, v.default_value)
                elif isinstance(v.default_value, Tuple):
                    # represent tuples are toml arrays, do not allow python lists are default values to avoid ambiguity
                    val = list(v.default_value)
                    arr = tomlkit.array()
                    arr.extend(val)
                    toml_doc.add(k, arr)
                else:
                    raise ValueError(f'Invalid type of default value for option {k}: {type(v.default_value)}')
                toml_doc.add(tomlkit.nl())
            else:
                raise ValueError(f'Invalid type of default value for option {k}: {type(v)}')

    fd = nest_flattened_dict(OptionRegistry.registered_options)
    doc = tomlkit.document()
    gen_doc(fd, doc)
    return doc


def _load_config(config_file_path: str):
    def collapse_nested_dict(d: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool, Tuple]]:
        # {"cuda": {"arch": "hopper", "cc": [9, 0]}} -> {"cuda.arch": 90, "cuda.cc": (9, 0)}
        ret = {}
        for k, v in d.items():
            if isinstance(v, dict):
                v = collapse_nested_dict(v)
                for k1, v1 in v.items():
                    ret[f'{k}.{k1}'] = v1
                continue
            if isinstance(v, list):
                v = tuple(v)
            ret[k] = v
        return ret

    with open(config_file_path, 'r') as f:
        config_doc = tomlkit.parse(f.read())
    for k, v in collapse_nested_dict(config_doc).items():
        if k not in OptionRegistry.registered_options:
            continue  # ignore the config that can not be recognized
        OptionRegistry.registered_options[k].default_value = v


def _write_default_config(config_file_path: str, config_doc: tomlkit.TOMLDocument):
    with open(config_file_path, 'w') as f:
        tomlkit.dump(config_doc, f)


def register_option(
    name: str,
    type_hint: str,
    description: str,
    default_value: Any,
    normalizer: Optional[Callable[[Any], Any]] = None,
    choices: Optional[Iterable[Any]] = None,
    checker: Optional[Callable[[Any], bool]] = None,
):
    registered_options = OptionRegistry.registered_options
    if name in registered_options:
        raise KeyError(f'Option {name} has already been registered.')
    registered_options[name] = OptionRegistry(name, type_hint, description, default_value, normalizer, choices, checker)


def register_hidet_options():
    from hidet.utils import git_utils

    register_option(
        name='bench_config',
        type_hint='Tuple[int, int, int]',
        description='The (warmup, number, repeat) parameters for benchmarking. '
        'The benchmarking will run warmup + number * repeat times.',
        default_value=(3, 10, 3),
    )
    register_option(
        name='search_space',  #
        type_hint='int',
        description='The search space level.',
        default_value=0,
        choices=[0, 1, 2],
    )
    register_option(
        name='cache_operator',
        type_hint='bool',
        description='Whether to enable operator cache on disk.',
        default_value=True,
        choices=[True, False],
    )
    register_option(
        name='cache_dir',
        type_hint='path',
        description='The directory to store the cache.',
        default_value=os.path.abspath(
            os.path.join(git_utils.git_repo_root(), '.hidet_cache')  # developer mode
            if git_utils.in_git_repo()
            else os.path.join(os.path.expanduser('~'), '.cache', 'hidet')  # user mode
        ),
        normalizer=os.path.abspath,
    )
    register_option(
        name='parallel_build',
        type_hint='bool',
        default_value=True,
        description='Whether to build operators in parallel.',
        choices=[True, False],
    )
    register_option(
        name='parallel_tune',
        type_hint='int, float',
        default_value=(-1, 1.5, 32),
        description='Deprecated option. Please use the `num_local_workers` option instead.',
    )
    register_option(
        name='fix_gpu_frequency_for_tuning',
        type_hint='bool',
        default_value=False,
        description='Whether to fix the GPU frequency during tuning to avoid frequency throttling.',
        choices=[True, False],
    )
    register_option(
        name='num_local_workers',
        type_hint='int',
        default_value=os.cpu_count(),
        description='Number of local worker processes to use for parallel compilation/tuning',
    )
    register_option(
        name='save_lower_ir',
        type_hint='bool',
        default_value=False,
        description='Whether to save the IR when lower an IRModule to the operator cache.',
        choices=[True, False],
    )
    register_option(
        name='debug_cache_tuning',
        type_hint='bool',
        default_value=False,
        description='Whether to cache the generated kernels during tuning.',
        choices=[True, False],
    )
    register_option(
        name='debug_enable_var_id',
        type_hint='bool',
        default_value=False,
        description='Assign a variable id to each variable in the IR. If set to false, all variable IDs will be 0',
        choices=[True, False],
    )
    register_option(
        name='debug_show_var_id',
        type_hint='bool',
        default_value=False,
        description='Whether to show the variable id in the IR.\
                     Hint: all variable ids will be 0 unless the debug_enable_var_id option is set to True.',
        choices=[True, False],
    )
    register_option(
        name='debug_strict_broadcast_check',
        type_hint='bool',
        default_value=False,
        description=(
            'Whether to enforce equality of shapes in symbolic broadcasts.'
            ' If set to True, the symbolic equivalence checker is used to prove correctness of broadcasts,'
            ' so broadcasting shapes [n] to [m] will raise ValueError. If set to False, broadcasting between'
            ' shapes [n] and [m] will proceed assuming n == m.'
        ),
    )
    register_option(
        name='runtime_check',
        type_hint='bool',
        default_value=False,
        description='Whether to check shapes of compiled graph and tasks during execution.',
        choices=[True, False],
    )
    register_option(
        name='debug_show_verbose_flow_graph',
        type_hint='bool',
        default_value=False,
        description='Whether to show the verbose flow graph.',
        choices=[True, False],
    )
    register_option(
        name='compile_server.addr',
        type_hint='str',
        default_value='localhost',
        description='The address of the compile server. Can be an IP address or a domain name.',
    )
    register_option(
        name='compile_server.port', type_hint='int', default_value=3281, description='The port of the compile server.'
    )
    register_option(
        name='compile_server.enabled',
        type_hint='bool',
        default_value=False,
        description='Whether to enable the compile server.',
        choices=[True, False],
    )
    register_option(
        name='compile_server.username',
        type_hint='str',
        default_value='admin',
        description='The user name to access the compile server.',
    )
    register_option(
        name='compile_server.password',
        type_hint='str',
        default_value='admin_password',
        description='The password to access the compile server.',
    )
    register_option(
        name='compile_server.repo_url',
        type_hint='str',
        default_value='https://github.com/hidet-org/hidet',
        description='The URL of the repository that the remote server will use.',
    )
    register_option(
        name='compile_server.repo_version',
        type_hint='str',
        default_value='main',
        description='The version (e.g., branch, commit, or tag) that the remote server will use.',
    )
    # TODO: this value should be requested from the compilation server
    register_option(
        name='compile_server.num_workers',
        type_hint='int',
        default_value=96,
        description='The number of worker processes of the compile server.',
    )
    register_option(
        name='cuda.arch',
        type_hint='str',
        default_value='auto',
        description='The CUDA architecture to compile the kernels for (e.g., "sm_70"). "auto" for auto-detect.',
    )
    register_option(
        name='cuda.cpu_arch',
        type_hint='str',
        default_value='auto',
        description='The CPU architecture to compile the host code for (e.g., "x86-64"). "auto" for auto-detect.',
    )
    register_option(
        name='cpu.arch',
        type_hint='str',
        default_value='auto',
        description='The CPU architecture to compile the kernels for (e.g., "x86-64"). "auto" for auto-detect.',
    )
    register_option(
        name='execution_mode',
        type_hint='str',
        default_value='interpreter',
        description="Use 'symbolic', 'interpreter', or 'compilation' mode for run() function in Operator allowed",
        choices=['symbolic', 'interpreter', 'compilation'],
    )
    register_option(
        name='parallel_k',
        type_hint='Union[str, int]',
        default_value='disabled',
        description='Parallelization on k dimension of the matrix multiplication.',
    )
    register_option(
        name='hexcute_matmul',
        type_hint='str',
        default_value='enable',
        description="Whether to enable the hexcute matmul kernels. The valid values for this option can be"
        "'enable', 'disable', and 'auto'",
    )
    # Exclusive `torch.compile` API option.
    # Torch Dynamo passes two arguments to the compiler: `fx.graph` and `example_inputs`.
    # From torch==2.4, `example_inputs` is an actual tensor (not FakeTensor, non-symbolic).
    # We use `fx.graph` to get symbolic shapes because `example_inputs` doesn't contain such info.
    #
    # On the other hand, vllm passes symbolic `example_inputs` if dynamic shapes are expected and
    # non-symbolic `example_inputs` if static shapes are expected. `fx.graph` remains the same in both cases.
    # With `internal.torch_api_use_example_input_shapes` set to True, we use the shape of `example_inputs` to determine
    # the requested shapes.
    register_option(
        name='internal.torch_api_use_example_input_shapes',
        type_hint='bool',
        default_value=False,
        description='Applicable when using torch.compile only. Use `example_inputs` shapes instead of fx.graph shapes.',
    )

    # Load hidet config
    config_file_path = os.path.join(os.path.expanduser('~'), '.config', 'hidet', 'hidet.toml')
    if os.path.exists(config_file_path):
        _load_config(config_file_path)


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
        """
        Get the current option context.

        Returns
        -------
        ret: OptionContext
            The current option context.
        """
        return OptionContext.stack[-1]

    def load_from_file(self, config_path: str):
        import configparser

        config = configparser.ConfigParser()
        config.read(config_path)
        for section in config.sections():
            for option in config.options(section):
                value = config.get(section, option)
                entry_name = '{}.{}'.format(section, option)
                if entry_name not in OptionRegistry.registered_options:
                    raise KeyError(
                        'Option {} found in config file {} is not registered.'.format(entry_name, config_path)
                    )
                self.set_option(entry_name, value)

    def set_option(self, name: str, value: Any):
        """
        Set the value of an option in the self option context.

        Parameters
        ----------
        name: str
            The name of the option.

        value: Any
            The value of the option.
        """
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
        """
        Get the value of an option in the self option context.

        Parameters
        ----------
        name: str
            The name of the option.

        Returns
        -------
        ret: Any
            The value of the option.
        """
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
        The dumped options, as a dict.
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


def parallel_tune(max_parallel_jobs: int = -1, mem_gb_per_job: float = 1.5, max_candidates_per_job: int = 32):
    """
    Specify the maximum number of parallel compilation jobs to do,
    and the number of GiB preserved for each job.

    Parameters
    ----------
    max_parallel_jobs: int
        The maximum number of parallel jobs allowed, default -1
        (the number of available vcpu returned by `os.cpu_count()`).
    mem_gb_per_job: float
        The minimum amount of memory (in GiB) reserved for each tuning job, default 1.5GiB.
    """
    warnings.warn('`parallel_tune` option is deprecated.' 'Please use the `num_local_workers` option instead.')
    OptionContext.current().set_option('parallel_tune', (max_parallel_jobs, mem_gb_per_job, max_candidates_per_job))


def get_parallel_tune() -> Tuple[int, float, int]:
    """
    Get the option value of whether to build operators in parallel.

    Returns
    -------
    ret: Tuple[int, float]
        Get the maximum number of jobs and minumum amount of memory reserved for tuning.

    """
    warnings.warn('`parallel_tune` option is deprecated.' 'Please use the `num_local_workers` option instead.')
    return (get_num_local_workers(), 1.5, 32)


def fix_gpu_frequency_for_tuning(enabled: bool = False):
    """
    Whether to fix the GPU frequency during tuning to avoid frequency throttling.

    Parameters
    ----------
    enabled: bool
        Whether to fix the GPU frequency during tuning to avoid frequency throttling.
    """
    OptionContext.current().set_option('fix_gpu_frequency_for_tuning', enabled)


def is_fix_gpu_frequency_for_tuning() -> bool:
    """
    Get the option value of whether to fix the GPU frequency during tuning to avoid frequency throttling.

    Returns
    -------
    ret: bool
        Whether to fix the GPU frequency during tuning to avoid frequency throttling.
    """
    return OptionContext.current().get_option('fix_gpu_frequency_for_tuning')


def num_local_workers(num_workers: Optional[int] = None):
    """
    Set the number of local worker processes to use for parallel compilation/tuning.

    Parameters
    ----------
    num_workers: Optional[int]
        The number of local worker processes. If None, use os.cpu_count().
    """
    OptionContext.current().set_option('num_local_workers', num_workers if num_workers is not None else os.cpu_count())


def get_num_local_workers() -> int:
    """
    Get the number of local worker processes to use for parallel compilation/tuning.

    Returns
    -------
    ret: int
        The number of local worker processes.
    """
    return OptionContext.current().get_option('num_local_workers')


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
    """
    return OptionContext.current().get_option('save_lower_ir')


def debug_cache_tuning(enabled: bool = True):
    """
    Whether to cache the generated kernels during tuning.

    .. note::

        This option is only used for debugging purpose. It will generate a lot of files in the cache directory
        and take a lot of disk space.

    Parameters
    ----------
    enabled: bool
        Whether to debug cache tuning.
    """
    OptionContext.current().set_option('debug_cache_tuning', enabled)


def debug_enable_var_id(enable: bool = True):
    """
    Whether to enable var id in the IR.

    When this option is enabled, each variable (i.e., hidet.ir.Var) will have a unique id.
    Otherwise, each variable's ID will be 0.

    Parameters
    ----------
    enable: bool
        Whether to enable var id in the IR.
    """
    OptionContext.current().set_option('debug_enable_var_id', enable)


def debug_show_var_id(enable: bool = True):
    """
    Whether to show the var id in the IR.

    When this option is enabled, the IR will show the var id with the format `var@id`, like `x@1` and `d_1@1732`.
    Variable (i.e., hidet.ir.Var) a and b is the same var if and only if `a is b` evaluates to True in Python).

    Parameters
    ----------
    enable: bool
        Whether to show the var id in the IR.
    """
    if not OptionContext.current().get_option('debug_enable_var_id'):
        warnings.warn("Please use `hidet.option.debug_enable_var_id()` to enable the id first")
    OptionContext.current().set_option('debug_show_var_id', enable)


def debug_strict_broadcast_check(enable: bool = False):
    """
    Whether to enforce equality of shapes in symbolic broadcasts.

    If set to True, the symbolic equivalence checker is used to prove correctness of broadcasts,
    so broadcasting shapes [n] to [m] will raise ValueError. If set to False, broadcasting between
    shapes [n] and [m] will proceed assuming n == m.

    Parameters
    ----------
    enable: bool
        Whether to enforce equality of shapes in symbolic broadcasts.
    """
    OptionContext.current().set_option('debug_strict_broadcast_check', enable)


def runtime_check(enable: bool = True):
    """
    Whether to check shapes and dtypes of all input arguments to compiled Graphs or Tasks.

    Parameters
    ----------
    enable: bool
        Whether to check shapes and dtypes of all input arguments to compiled Graphs or Tasks.
    """
    OptionContext.current().set_option('runtime_check', enable)


def get_runtime_check() -> bool:
    """
    Get whether to check shapes and dtypes of all input arguments to compiled Graphs or Tasks.

    Returns
    -------
    ret: bool
        Get whether to check shapes and dtypes of all input arguments to compiled Graphs or Tasks.
    """
    return OptionContext.current().get_option('runtime_check')


def execution_mode(kind: str = 'compilation'):
    """
    Use 'symbolic', 'interpreter', or 'compilation' mode for run() function in Operator allowed.

    Parameters
    ----------
    kind: str
        Use 'symbolic', 'interpreter', or 'compilation' mode for run() function in Operator allowed.
    """
    OptionContext.current().set_option('execution_mode', kind)


def get_execution_mode() -> str:
    """
    Get which of 'symbolic', 'interpreter', 'compilation' mode to use for run() function in Operator allowed.

    Returns
    -------
    ret: str
        Get which of 'symbolic', 'interpreter', 'compilation' mode to use for run() function in Operator allowed.
    """
    return OptionContext.current().get_option('execution_mode')


def parallel_k(strategy: Union[str, int] = "default"):
    """
    Parallelization on k dimension of the matrix multiplication
    Candidates are: ``default``, ``disabled``, ``search``, 2, 3, 4...

    - ``default``:
        Default parallelization strategy. A heuristic strategy is used to decide whether to parallelize on k
        dimension and the size of split factor
    - ``disabled``:
        Disable parallelization on k dimension
    - ``search``:
        Search for the best parallelization strategy. Takes more time but usually achieves the best performance.

    Parameters
    ----------
    strategy: str
        The parallelization strategy.
    """
    OptionContext.current().set_option('parallel_k', strategy)


def get_parallel_k() -> Union[str, int]:
    """
    Get parallelization on k dimension of the matrix multiplication

    Returns
    -------
    ret: Union[str, int]
        Get parallelization strategy.
    """
    return OptionContext.current().get_option('parallel_k')


def hexcute_matmul(strategy: str = "enable"):
    """
    Whether to enable hexcute matmul kernels, such as the Hexcute matmul kernels.

    - ``enable``:
        Always enable the hexcute matmul kernels on all the GPU platforms.
    - ``disable``:
        Always disable the hexcute matmul kernels on all the GPU platforms.
    - ``auto``:
        Use a heuristic strategy to decide whether to enable the hexcute matmul kernels.
        The decision is based on the metrics of the current GPU platform, such as
        the memory bandwidth, the compute throughput, etc.

    Parameters
    ----------
    strategy: str
        The strategy to enable the hexcute matmul kernels.
    """
    OptionContext.current().set_option('hexcute_matmul', strategy)


def get_hexcute_matmul() -> str:
    """
    Get strategy to enable the hexcute matmul kernels.

    Returns
    -------
    ret: str
        Get strategy to enable the hexcute matmul kernels.
    """
    return OptionContext.current().get_option('hexcute_matmul')


def debug_show_verbose_flow_graph(enable: bool = True):
    """Whether to show verbose information (like task) when we convert flow graph in to human-readable text.

    Parameters
    ----------
    enable: bool
        Whether to show verbose information when we convert flow graph in to human-readable text.
    """
    OptionContext.current().set_option('debug_show_verbose_flow_graph', enable)


def is_option_exist(name: str) -> bool:
    """Checking is options exist/registered.

    Parameters
    ----------
    name: str
        Name of the option.

    Returns
    -------
    ret: bool
        True if option exists/registered, False otherwise.
    """
    return name in OptionRegistry.registered_options


def use_torch_stream(use: bool):
    """Set the flag for whether to use torch steam

    Parameters
    ----------
    use: bool
        whether to set
    """
    from hidet.ffi.runtime_api import runtime_api

    # set use torch stream flag according to the option
    runtime_api.use_torch_cuda_stream(use)


def is_use_torch_stream() -> bool:
    """Checking if currently is using torch stream


    Returns
    -------
    ret: bool
        True if using torch stream, False otherwise.
    """
    from hidet.ffi.runtime_api import runtime_api

    return runtime_api.get_use_torch_cuda_stream()


class cuda:
    """
    The CUDA related options.
    """

    @staticmethod
    def arch(arch: str = 'auto'):
        """
        Set the CUDA architecture to use when building CUDA kernels.

        Parameters
        ----------
        arch: Optional[str]
            The CUDA architecture, e.g., 'sm_35', 'sm_70', 'sm_80', etc. "auto" means
            using the architecture of the first CUDA GPU on the current machine. Default "auto".
        """
        OptionContext.current().set_option('cuda.arch', arch)

    @staticmethod
    def get_arch() -> str:
        """
        Get the CUDA architecture to use when building CUDA kernels.

        Returns
        -------
        ret: str
            The CUDA architecture, e.g., 'sm_35', 'sm_70', 'sm_80', etc.
        """
        arch: Optional[str] = OptionContext.current().get_option('cuda.arch')
        if arch == "auto":
            import hidet.cuda

            # get the architecture of the first CUDA GPU
            properties = hidet.cuda.properties(0)
            arch = 'sm_{}{}'.format(properties.major, properties.minor)
        return arch

    @staticmethod
    def get_arch_pair() -> Tuple[int, int]:
        """
        Get the CUDA architecture to use when building CUDA kernels, with major and minor version as a tuple.

        Returns
        -------
        ret: Tuple[int, int]
            The CUDA architecture, e.g., (3, 5), (7, 0), (8, 0), etc.
        """
        arch = cuda.get_arch()
        return int(arch[3]), int(arch[4])


class cpu:
    """
    The CPU related options.
    """

    @staticmethod
    def arch(arch: str = 'auto'):
        """
        Set the CPU architecture to use when building CPU kernels.

        Parameters
        ----------
        arch: Optional[str]
            The CPU architecture, e.g., 'x86-64', 'alderlake', etc. "auto" means
            using the architecture of the CPU on the current machine. Default "auto".
        """
        OptionContext.current().set_option('cpu.arch', arch)

    @staticmethod
    def get_arch() -> str:
        """
        Get the CPU architecture to use when building CPU kernels.

        Returns
        -------
        ret: str
            The CPU architecture, e.g., 'x86-64', 'alderlake', etc.
        """
        arch: Optional[str] = OptionContext.current().get_option('cpu.arch')
        if arch == "auto":
            cmd = ['gcc', '-march=native', '-Q', '--help=target']
            out = subprocess.check_output(cmd, text=True)
            begin = out.find('march=') + len('march=')
            end = out.find('\n', begin)
            arch = out[begin:end].strip()
        return arch


class compile_server:
    """
    Compilation server related options.
    """

    @staticmethod
    def addr(addr: str):
        """
        Set the address of the compile server.

        Parameters
        ----------
        addr: str
            The address of the compile server. Can be an IP address or a domain name.
        """
        OptionContext.current().set_option('compile_server.addr', addr)

    @staticmethod
    def port(port: int):
        """
        Set the port of the compile server.

        Parameters
        ----------
        port: int
            The port of the compile server.
        """
        OptionContext.current().set_option('compile_server.port', port)

    @staticmethod
    def enable(flag: bool = True):
        """
        Enable or disable the compile server.

        The compile server is disabled by default. We need to enable it before using it.

        Parameters
        ----------
        flag: bool
            Whether to enable the compile server.
        """
        OptionContext.current().set_option('compile_server.enabled', flag)

    @staticmethod
    def enabled() -> bool:
        """
        Get whether the compile server is enabled.

        Returns
        -------
        ret: bool
            Whether the compile server is enabled.
        """
        return OptionContext.current().get_option('compile_server.enabled')

    @staticmethod
    def username(username: str):
        """
        Set the username to access the compile server.

        Parameters
        ----------
        username: str
            The username to access the compile server.
        """
        OptionContext.current().set_option('compile_server.username', username)

    @staticmethod
    def password(password: str):
        """
        Set the password to access the compile server.

        Parameters
        ----------
        password: str
            The password to access the compile server.
        """
        OptionContext.current().set_option('compile_server.password', password)

    @staticmethod
    def repo(repo_url: str, version: str = 'main'):
        """
        Set the repository that the remote server will use.

        When we compile a tensor program with remote server, it will clone the given repository and checkout to the
        given version. Then, it will use the code in the repository to compile the tensor program. Thus, it is
        important to make sure the code in the repository is consistent with the code used to compile the tensor
        program.

        Parameters
        ----------
        repo_url: str
            The URL of the repository that the remote server will use. By default, it is the official repository
            of hidet https://github.com/hidet-org/hidet.
        version: str
            The version (e.g., branch, commit, or tag) that the remote server will use. By default, it is the main
            branch: 'main'.
        """
        OptionContext.current().set_option('compile_server.repo_url', repo_url)
        OptionContext.current().set_option('compile_server.repo_version', version)

    @staticmethod
    def num_workers(num_workers: int):
        """
        Set the number of worker processes of the compile server.

        Parameters
        ----------
        num_workers: int
            The number of worker processes of the compile server.
        """
        OptionContext.current().set_option('compile_server.num_workers', num_workers)

    @staticmethod
    def get_num_workers() -> int:
        """
        Get the number of worker processes of the compile server.

        Returns
        -------
        ret: int
            The number of worker processes of the compile server.
        """
        return OptionContext.current().get_option('compile_server.num_workers')


class internal:
    """
    Internal options.
    """

    @staticmethod
    def torch_api_use_example_input_shapes(enable: bool = False):
        """
        Applicable when using `torch.compile` only. Use `example_inputs` shapes instead of fx.graph shapes.

        Parameters
        ----------
        enable: bool
            Applicable when using torch.compile only. Use `example_inputs` shapes instead of fx.graph shapes.
        """
        OptionContext.current().set_option('internal.torch_api_use_example_input_shapes', enable)

    @staticmethod
    def is_torch_api_use_example_input_shapes():
        """
        Get whether to use `example_inputs` shapes instead of `fx.graph` shapes.

        Returns
        -------
        ret: bool
            Whether to use `example_inputs` shapes instead of `fx.graph` shapes.
        """
        return OptionContext.current().get_option('internal.torch_api_use_example_input_shapes')


# load the options from config file (e.g., ~/.config/hidet.config) if exists
_config_path = os.path.join(os.path.expanduser('~'), '.config', 'hidet.config')
if os.path.exists(_config_path):
    OptionContext.current().load_from_file(_config_path)
