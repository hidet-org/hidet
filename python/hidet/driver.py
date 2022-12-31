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
import subprocess
from typing import List, Optional, Sequence, Tuple
import os
import multiprocessing
import logging
from hashlib import sha256
import psutil
from tqdm import tqdm

from hidet import option
from hidet.transforms import lower, PassContext, SaveIRInstrument, ProfileInstrument
from hidet.backend import codegen, compile_source, load_task_func, load_lib_func
from hidet.backend.build import CompilationFailed
from hidet.utils.py import cyan, green, Timer
from hidet.ir.task import Task
from hidet.ir.func import IRModule, Function
from hidet.ir.type import FuncType
from hidet.runtime.module import compiled_task_cache, CompiledFunction

logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def build_task(task: Task, target_device='cuda', load=True) -> Optional[CompiledFunction]:
    """
    Build a task into a compiled function.

    Parameters
    ----------
    task: Task
        The task to be built.
    target_device: str
        The target device. Candidates are 'cuda' and 'cpu'.
    load: bool
        Whether to load the compiled function. If False, the compiled function will not be loaded, and None is returned.
        Otherwise, the compiled function is loaded and returned.
    Returns
    -------
    compiled_func:
        When load is True, the compiled function is returned. Otherwise, None is returned.
    """
    task_string: str = str(task)
    compiled_func: Optional[CompiledFunction] = None

    space_level = option.get_option('search_space')
    op_cache_dir = os.path.join(option.get_option('cache_dir'), './ops')
    use_cache = option.get_option('cache_operator')

    # check in-memory cache
    if compiled_task_cache.contains(target_device, space_level, task_string):
        if load:
            compiled_func = compiled_task_cache.get(target_device, space_level, task_string)
    else:
        # check on-disk cache
        config_str = f'{target_device}_space_{space_level}'
        task_hash = sha256(task_string.encode()).hexdigest()[:16]
        task_dir = os.path.join(op_cache_dir, config_str, task.name, task_hash)
        src_path = os.path.join(task_dir, 'source.cu')
        lib_path = os.path.join(task_dir, 'lib.so')

        # use previously generated library when available
        if use_cache and os.path.exists(lib_path):
            logger.debug(f"Load cached task binary {green(task.name)} from path: \n{cyan(lib_path)}")
            if load:
                compiled_func = load_task_func(lib_path, task)
                compiled_task_cache.add(target_device, space_level, task_string, compiled_func)
        else:
            logger.info(f"Compiling {target_device} task {green(task.signature())}...")
            # build from scratch
            os.makedirs(task_dir, exist_ok=True)
            # write task
            with open(os.path.join(task_dir, 'task.txt'), 'w') as f:
                f.write(task_string)
            # implement task
            ir_module = task.implement(target=target_device, workding_dir=task_dir)
            # lower ir module
            if option.get_option('save_lower_ir'):
                instruments = [
                    SaveIRInstrument(out_dir=os.path.join(task_dir, './ir')),
                    ProfileInstrument(log_file=os.path.join(task_dir, './lower_time.txt')),
                ]
            else:
                instruments = []
            with PassContext(instruments=instruments):
                ir_module = lower(ir_module)
            # code generation
            codegen(ir_module, src_out_path=src_path)
            # compile source code
            compile_source(src_path, out_lib_path=lib_path, keep_ptx=False)
            # load function
            if load:
                compiled_func = load_task_func(lib_path, task)
                compiled_task_cache.add(target_device, space_level, task_string, compiled_func)
    return compiled_func


def _build_task_job(args):
    try:
        task, target_device, dumped_options = args
        option.restore_options(dumped_options)
        build_task(task, target_device, load=False)
        return True
    except CompilationFailed as e:
        if option.get_option('parallel_build'):
            return False
        else:
            raise e


def build_task_batch(tasks: List[Task], target_device: str = 'cuda', raise_on_error: bool = True):
    dumped_options = option.dump_options()
    jobs = [(task, target_device, dumped_options) for task in tasks]
    if option.get_option('parallel_build') and len(jobs) > 1:
        with multiprocessing.Pool() as pool:
            status_list = list(pool.map(_build_task_job, jobs))
    else:
        status_list = list(map(_build_task_job, jobs))
    if not all(status_list) and raise_on_error:
        msg = ['Failed to build {} tasks:'.format(sum(1 for s in status_list if not s))]
        for task, status in zip(tasks, status_list):
            if not status:
                msg.append(f'  {task.signature()}')
        msg.append('Please turn off parallel build to see the error message:')
        msg.append('  hidet.option.parallel_build(False)')
        raise RuntimeError('\n'.join(msg))


def build_ir_module(
    ir_module: IRModule,
    func_name: str,
    output_dir='./outs/ir_module',
    save_ir: bool = True,
    profile_pass: bool = True,
    load: bool = True,
    use_hash_dir: bool = True,
):
    if use_hash_dir:
        hash_dir = sha256(str(ir_module).encode()).hexdigest()[:16]
        output_dir = os.path.join(output_dir, hash_dir)

    src_path = os.path.join(output_dir, 'source.cu')
    lib_path = os.path.join(output_dir, 'lib.so')

    # get function type
    func: Function = ir_module.lookup(func_name)
    if func.kind == 'packed_func':
        packed_func = ir_module.lookup(func.attrs['packed_func'])
        func_type = FuncType.from_func(packed_func)
    else:
        func_type = FuncType.from_func(func)

    # lower ir module
    instruments = []
    if save_ir:
        instruments.append(SaveIRInstrument(out_dir=os.path.join(output_dir, './ir')))
    if profile_pass:
        instruments.append(ProfileInstrument(log_file=os.path.join(output_dir, './lower_time.txt')))
    with PassContext(instruments=instruments):
        ir_module = lower(ir_module)

    # code generation
    codegen(ir_module, src_out_path=src_path)

    # compile source code
    compile_source(src_path, out_lib_path=lib_path, keep_ptx=False)

    if load:
        # load function
        return load_lib_func(lib_path, func_name, func_type=func_type)
    else:
        return lib_path, func_name, func_type


def _build_ir_module_job(args) -> Optional[Tuple[str, str, FuncType]]:
    ir_module, func_name, output_dir, dumped_options = args
    option.restore_options(dumped_options)
    try:
        return build_ir_module(
            ir_module, func_name, output_dir, save_ir=False, profile_pass=False, load=False, use_hash_dir=False
        )
    except subprocess.CalledProcessError:
        print('Failed launch subprocess to compile the lowered source code via nvcc.')
        return None
    except CompilationFailed:
        print('Failed to compile the lowered source code via nvcc.')
        return None


def build_ir_module_batch(
    ir_modules: Sequence[IRModule], func_name: str, output_dir: str, parallel=True, verbose=False
) -> List[Optional[CompiledFunction]]:
    """
    Build a batch of ir modules.

    Parameters
    ----------
    ir_modules: Sequence[IRModule]
        A sequence of ir modules to build.

    func_name: str
        The name of the function to load after building.

    output_dir: str
        The output directory to save the compiled library and source code (lib.so and source.cu).

    parallel: bool
        Whether build in parallel. Default True.

    verbose: bool
        Whether show the progress and summary. Default False.

    Returns
    -------
    funcs:
        The compiled functions, in the same order as build_instances.
        When the build for a build instance failed, None for that instance is returned.
    """
    with Timer() as timer:
        dumped_options = option.dump_options()
        jobs = [
            (ir_module, func_name, os.path.join(output_dir, str(idx)), dumped_options)
            for idx, ir_module in enumerate(ir_modules)
        ]
        build_results = []
        if parallel:
            # Set the affinity of current process. Some package such as numpy will change affinity of current process,
            # which might limit the parallelism of compilation.
            os.sched_setaffinity(0, range(os.cpu_count()))

            # the maximum number of processes is limited by the number of cores and memory
            mem_for_worker = 1.5 * 1024 * 1024 * 1024  # 1.5 GiB
            num_workers = min(max(int(psutil.virtual_memory().available // mem_for_worker), 1), psutil.cpu_count())

            with multiprocessing.Pool(processes=num_workers) as pool:
                for build_result in tqdm(
                    pool.imap(_build_ir_module_job, jobs),
                    desc='Compiling',
                    total=len(jobs),
                    disable=not verbose,
                    ncols=80,
                ):
                    build_results.append(build_result)
        else:
            # sequential build
            build_results = list(map(_build_ir_module_job, jobs))

        # load compiled functions
        funcs: List[Optional[CompiledFunction]] = []
        for build_result in build_results:
            if build_result is not None:
                lib_path, func_name, func_type = build_result
                funcs.append(load_lib_func(lib_path, func_name, func_type))
            else:
                funcs.append(None)
    if verbose:
        print(
            'Batch build {} modules within {:.3f} seconds, on average {:.1f} seconds per module.'.format(
                len(jobs), timer.elapsed_seconds(), timer.elapsed_seconds() / len(jobs)
            )
        )
    return funcs
