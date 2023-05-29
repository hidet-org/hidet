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
from typing import List, Optional, Sequence, Tuple, Dict
import subprocess
import os
import logging
import pickle
from hashlib import sha256
import psutil
from tqdm import tqdm

import hidet.cuda
from hidet import option
from hidet.transforms import lower, PassContext, SaveIRInstrument, ProfileInstrument
from hidet.backend import codegen, compile_source
from hidet.backend.build import CompilationFailed
from hidet.utils.py import cyan, green, Timer
from hidet.ir.task import Task
from hidet.ir.func import IRModule
from hidet.ir.type import FuncType
from hidet.runtime.module import compiled_task_cache, CompiledModule, load_compiled_module, compiled_module_exists
from hidet.runtime.device import Device
from hidet.utils.multiprocess import parallel_imap

logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def build_task(task: Task, target_device='cuda', load=True) -> Optional[CompiledModule]:
    """
    Build a task into a compiled function.

    Parameters
    ----------
    task: Task
        The task to be built.
    target_device: str or Device
        The target device. Candidates are 'cuda' and 'cpu'.
    load: bool
        Whether to load the compiled function. If False, the compiled function will not be loaded, and None is returned.
        Otherwise, the compiled function is loaded and returned.
    Returns
    -------
    ret: CompiledModule
        When load is True, the compiled function is returned. Otherwise, None is returned.
    """
    task_string: str = str(task)
    compiled_module: Optional[CompiledModule] = None

    if isinstance(target_device, Device):
        target_device = target_device.type

    space_level = option.get_option('search_space')
    op_cache_dir = os.path.join(option.get_option('cache_dir'), './ops')
    use_cache = option.get_option('cache_operator')

    # check in-memory cache
    if compiled_task_cache.contains(target_device, space_level, task_string):
        if load:
            compiled_module = compiled_task_cache.get(target_device, space_level, task_string)
    else:
        # check on-disk cache
        config_str = f'{target_device}_space_{space_level}'
        task_hash = sha256(task_string.encode()).hexdigest()[:16]
        task_dir = os.path.join(op_cache_dir, config_str, task.name, task_hash)
        lib_path = os.path.join(task_dir, 'lib.so')
        version_path = os.path.join(task_dir, 'version.txt')

        version_matched = False
        if os.path.exists(version_path):
            with open(version_path, 'r') as f:
                version = f.read()
                if version.strip() == hidet.__version__:
                    version_matched = True

        # use previously generated library when available
        if use_cache and version_matched and compiled_module_exists(task_dir):
            logger.debug(f"Load cached task binary {green(task.name)} from path: \n{cyan(lib_path)}")
            if load:
                compiled_module = load_compiled_module(task_dir)
                compiled_task_cache.add(target_device, space_level, task_string, compiled_module)
        else:
            logger.info(f"Compiling {target_device} task {green(task.signature())}...")

            # build from scratch
            os.makedirs(task_dir, exist_ok=True)

            # write task
            with open(os.path.join(task_dir, 'task.txt'), 'w') as f:
                f.write(task_string)

            # write version
            with open(version_path, 'w') as f:
                f.write(hidet.__version__)

            # implement task
            ir_module = task.implement(target=target_device, working_dir=task_dir)

            # compile ir module
            build_ir_module(
                ir_module,
                output_dir=task_dir,
                save_ir=option.get_option('save_lower_ir'),
                load=False,
                use_hash_dir=False,
            )
            if load:
                compiled_module = load_compiled_module(task_dir)
                compiled_task_cache.add(target_device, space_level, task_string, compiled_module)
    return compiled_module


def _lazy_initialize_cuda():
    # We intentionally query the cuda device information to put the properties of all devices to the lru_cache.
    #
    # Reasons:
    #   Hidet relies on the multiprocessing to parallelize the compilation. During the process, the forked process will
    #   query the properties of the device. If we do not cache the properties, the forked process will query the device
    #   via the cuda runtime API. However, the cuda runtime API does not work when the multiprocessing package is
    #   working in the fork mode. With the properties of all the GPUs cached, the forked process will not run any cuda
    #   runtime API and will not cause any problem.
    if getattr(_lazy_initialize_cuda, '_initialized', False):
        return
    _lazy_initialize_cuda._initialized = True  # pylint: disable=protected-access
    if hidet.cuda.available():
        for i in range(hidet.cuda.device_count()):
            hidet.cuda.properties(i)
            hidet.cuda.compute_capability(i)


def build_task_batch(task_device_pairs: List[Tuple[Task, Device]], raise_on_error: bool = True):
    dumped_options = option.dump_options()
    jobs = [(task, device, dumped_options) for task, device in task_device_pairs]

    def build_job(args):
        try:
            task, target_device, dumped_options = args
            option.restore_options(dumped_options)
            build_task(task, target_device, load=False)
            return True
        except Exception:  # pylint: disable=broad-except
            if option.get_option('parallel_build'):
                return False
            else:
                raise

    if option.get_option('parallel_build') and len(jobs) > 1:
        _lazy_initialize_cuda()
        status_list = list(parallel_imap(build_job, jobs))
    else:
        status_list = list(map(build_job, jobs))
    if not all(status_list) and option.get_option('parallel_build') and raise_on_error:
        msg = ['Failed to build {} tasks:'.format(sum(1 for s in status_list if not s))]
        for (task, device), status in zip(task_device_pairs, status_list):
            if not status:
                msg.append(f'  [{device.type}] {task.signature()}')
        msg.append('Please turn off parallel build to see the error message:')
        msg.append('  hidet.option.parallel_build(False)')
        raise RuntimeError('\n'.join(msg))


def build_ir_module(
    ir_module: IRModule,
    output_dir='./outs/ir_module',
    save_ir: bool = True,
    load: bool = True,
    use_hash_dir: bool = True,
) -> Optional[CompiledModule]:
    if use_hash_dir:
        hash_dir = sha256(str(ir_module).encode()).hexdigest()[:16]
        output_dir = os.path.join(output_dir, hash_dir)

    src_path = (
        os.path.join(output_dir, 'source.cu') if hidet.cuda.available() else os.path.join(output_dir, 'source.cc')
    )
    lib_path = os.path.join(output_dir, 'lib.so')

    # lower ir module
    instruments = []
    if save_ir:
        instruments.append(SaveIRInstrument(out_dir=os.path.join(output_dir, './ir')))
        instruments.append(ProfileInstrument(log_file=os.path.join(output_dir, './lower_time.txt')))
    with PassContext(instruments=instruments):
        ir_module = lower(ir_module)

    # code generation
    codegen(ir_module, src_out_path=src_path)

    # compile source code
    compile_source(src_path, out_lib_path=lib_path)

    # write the function types
    func_types: Dict[str, FuncType] = {
        func.name: FuncType.from_func(func) for func in ir_module.functions.values() if func.kind == 'public'
    }
    with open(os.path.join(output_dir, 'func_types.pickle'), 'wb') as f:
        pickle.dump(func_types, f)

    # load the compiled module, if needed
    if load:
        return load_compiled_module(output_dir)
    else:
        return None


def build_ir_module_batch(
    ir_modules: Sequence[IRModule], output_dir: str, parallel=True, verbose=False
) -> List[Optional[CompiledModule]]:
    """
    Build a batch of ir modules.

    Parameters
    ----------
    ir_modules: Sequence[IRModule]
        A sequence of ir modules to build.

    output_dir: str
        The output directory to save the compiled library and source code (lib.so and source.cu).

    parallel: bool
        Whether build in parallel. Default True.

    verbose: bool
        Whether show the progress and summary. Default False.

    Returns
    -------
    ret:
        The compiled modules, in the same order as ir_modules.
        When the build failed for an ir module, None for that ir module is returned.
    """

    def build_job(args) -> bool:
        ir_module, output_dir, dumped_options = args
        option.restore_options(dumped_options)
        try:
            build_ir_module(ir_module, output_dir, save_ir=False, load=False, use_hash_dir=False)
            return True
        except subprocess.CalledProcessError:
            print('Failed launch subprocess to compile the lowered source code via nvcc.')
            return False
        except CompilationFailed:
            print('Failed to compile the lowered source code via nvcc.')
            return False

    with Timer() as timer:
        dumped_options = option.dump_options()
        output_dirs: List[str] = [os.path.join(output_dir, str(idx)) for idx in range(len(ir_modules))]
        jobs = [(ir_module, output_dir, dumped_options) for ir_module, output_dir in zip(ir_modules, output_dirs)]
        build_results: List[bool] = []
        if parallel:
            cpu_count = os.cpu_count()
            max_jobs, mem_for_worker = option.get_parallel_tune()
            max_jobs = cpu_count if max_jobs == -1 else min(max_jobs, cpu_count)
            mem_for_worker *= 1024**3
            # Set the affinity of current process. Some package such as numpy will change affinity of current process,
            # which might limit the parallelism of compilation.
            os.sched_setaffinity(0, range(cpu_count))

            num_workers = min(max(int(psutil.virtual_memory().available // mem_for_worker), 1), max_jobs)

            _lazy_initialize_cuda()
            for build_result in tqdm(
                parallel_imap(build_job, jobs, num_workers),
                desc='Compiling',
                total=len(jobs),
                disable=not verbose,
                ncols=80,
            ):
                build_results.append(build_result)
        else:
            # sequential build
            for job in tqdm(jobs, desc='Compiling', disable=not verbose, ncols=80):
                build_results.append(build_job(job))
            build_results = list(map(build_job, jobs))

        # load compiled functions
        modules: List[Optional[CompiledModule]] = []
        for build_result, module_dir in zip(build_results, output_dirs):
            if build_result:
                # successfully built the module
                modules.append(load_compiled_module(module_dir))
            else:
                modules.append(None)
    if verbose:
        print(
            'Batch build {} modules within {:.3f} seconds, on average {:.1f} seconds per module.'.format(
                len(jobs), timer.elapsed_seconds(), timer.elapsed_seconds() / len(jobs)
            )
        )
    return modules
