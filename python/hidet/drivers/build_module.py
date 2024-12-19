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
from typing import Sequence, Dict, Union
import logging
import os
import pickle
import random
from tqdm import tqdm

import hidet.cuda
from hidet import option
from hidet.backend import codegen, compile_source
from hidet.drivers.utils import lazy_initialize_cuda
from hidet.ir.module import IRModule
from hidet.ir.type import FuncType
from hidet.ir.target import Target
from hidet.transforms import lower, PassContext, SaveIRInstrument, ProfileInstrument
from hidet.utils.multiprocess import parallel_imap, get_parallel_num_workers
from hidet.utils.stack_limit import set_stack_limit
from hidet.utils.folder_lock import FolderLock

logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def can_remote_build(ir_module: IRModule) -> bool:
    def can_remote_single_build(ir_module: IRModule) -> bool:
        return not (
            len(ir_module.object_files) > 0 or len(ir_module.linking_dirs) > 0 or len(ir_module.include_dirs) > 0
        )

    if isinstance(ir_module, IRModule):
        return can_remote_single_build(ir_module)
    else:
        return all(can_remote_single_build(m) for m in ir_module)


def create_instruments(output_dir: str, ir_module: IRModule):
    instruments = []
    if hidet.option.get_save_lower_ir():
        ir_candidate_dir = os.path.join(output_dir, 'ir', ir_module.namespace)
        instruments.extend(
            [
                SaveIRInstrument(out_dir=ir_candidate_dir),
                ProfileInstrument(log_file=os.path.join(ir_candidate_dir, 'lower_time.txt')),
            ]
        )
    return instruments


def configure_target(target):
    if target.name == 'cuda':
        if 'arch' in target.attrs:
            hidet.option.cuda.arch(target.attrs['arch'])
        if 'cpu_arch' in target.attrs:
            hidet.option.cpu.arch(target.attrs['cpu_arch'])
    elif target.name == 'cpu' and 'arch' in target.attrs:
        hidet.option.cpu.arch(target.attrs['arch'])


def build_ir_module(
    ir_module: Union[IRModule, Sequence[IRModule]],
    output_dir: str,
    target: str,
    output_kind: str = '.so',
    force: bool = False,
):
    """
    Build an IR module to a shared library or object file.

    This driver function performs the following steps to build an IR module:
    1. Lower and optimize the IR module with a sequence of pre-defined passes.
    2. Generate source code from the lowered IR module.
    3. Call the underlying compiler (e.g., gcc or nvcc) to compile the generated source code into a shared library
       (when `output_kind == '.so'`) or an object file (when `output_kind == '.o'`).

    To ensure safe parallel execution in a multiprocessing environment, a file-based lock (`.lock` file in the
    `output_dir`) is used. This guarantees that only one process can build the IR module for a given `output_dir`
    at any given time.

    Parameters
    ----------
    ir_module: Union[IRModule, Sequence[IRModule]]
        The IR module to be built. This can be a single IRModule or a sequence of IRModules.

    output_dir: str
        The directory to save the generated source code and the compiled library.

    target: str
        The target to build the IR module. Supported targets are `cpu` and `cuda`. Attributes
        (e.g., 'cuda --arch=sm_70') can also be specified.

    output_kind: str
        The output kind. Supported kinds are `'.so'` and `'.o'`.
        - `'.so'`: Compile the IR module to a shared library.
        - `'.o'`: Compile the IR module to an object file.

    force: bool
        Whether to force re-build the IR module. By default, the IR module will not be re-built if the library
        already exists in the specified output directory.

    Notes
    -----
    - **File Locking:** A `.lock` file is created in the `output_dir` to synchronize access. If another process
      tries to build the same IR module concurrently, it will wait until the lock is released.

    - **Parallel Safety:** The file lock ensures that only one process builds the IR module for a specific
      `output_dir`.
    """
    lib_name = get_library_name(output_kind)
    lib_path = os.path.join(output_dir, lib_name)

    # Acquire file lock for this output directory
    with FolderLock(output_dir):  # Locks on .lock file in the output directory
        if should_skip_build(lib_path, output_kind, output_dir, force):
            return

        if hidet.option.compile_server.enabled() and can_remote_build(ir_module):
            from hidet.apps.compile_server import remote_build

            remote_build(ir_module, output_dir, target=target, output_kind=output_kind)
            return

        target = Target.from_string(target) if isinstance(target, str) else target
        src_path = get_source_path(output_dir, target)

        # Set the recursion limit for lowering
        set_stack_limit()

        # Lower the IR module
        ir_module = lower_ir_module(ir_module, output_dir, target)

        # Generate source code
        codegen(ir_module, src_out_path=src_path, target=target)

        # Collect dependencies for compilation
        include_dir, linking_dir, linking_lib, object_file = collect_dependencies(ir_module)

        # Compile source code
        compile_source(
            src_path,
            output_library_file=lib_path,
            target=target,
            include_dirs=include_dir,
            linking_dirs=linking_dir,
            linking_libraries=linking_lib,
            object_files=object_file,
        )

        # Write function types for shared libraries
        if output_kind == '.so':
            write_function_types(ir_module, output_dir)


def build_ir_module_batch(
    ir_modules: Sequence[IRModule], output_dirs: Sequence[str], output_kind: str, target: str, force: bool = False
):
    """
    Build a batch of IR modules.

    Parameters
    ----------
    ir_modules: Sequence[IRModule]
        A sequence of IR modules to build.
    output_dirs: Sequence[str]
        Directories for compilation artifacts.
    output_kind: str
        The output kind of the compiled library. Can be `'.so'` or `'.o'`.
    target: str
        The target of the compilation. Can be 'cuda' or 'cpu'.
    force: bool
        Whether to force re-build the IR module. By default, the IR module will not be re-built if the library already
        exists in the specified output directory.
    """

    def build_job(args):
        ir_module, output_dir = args
        build_ir_module(ir_module, output_dir, output_kind=output_kind, target=target, force=force)

    def regroup_modules(modules, num_workers):
        """
        Regroup IR modules for parallel processing.
        """
        from hidet.utils import cdiv

        max_candidates_per_job = option.get_parallel_tune()[2]
        len_modules = len(modules)

        if len_modules <= num_workers:
            return modules

        num_new_jobs = cdiv(len_modules, num_workers * max_candidates_per_job) * num_workers
        job_per_worker = len_modules // num_new_jobs
        num_modules_for_1st_pass = job_per_worker * num_new_jobs

        grouped_modules = [modules[i : i + job_per_worker] for i in range(0, num_modules_for_1st_pass, job_per_worker)]
        remainder = modules[num_modules_for_1st_pass:]

        for i, module in enumerate(remainder):
            grouped_modules[i % len(grouped_modules)].append(module)

        assert sum(len(group) for group in grouped_modules) == len(modules)
        return grouped_modules

    def check_function_singular(module_list):
        """
        Ensure no duplicate function names exist after regrouping.
        """
        if not module_list or isinstance(module_list[0], IRModule):
            return True
        name_set = set()
        for modules in module_list:
            for module in modules:
                namespace = module.namespace
                for func_name in module.extern_functions.keys() | module.functions.keys():
                    func_str = f"{namespace}::{func_name}"
                    if func_str in name_set:
                        return False
                    name_set.add(func_str)
        return True

    # Determine number of workers
    max_num_worker, mem_for_worker, _ = option.get_parallel_tune()
    if hidet.option.compile_server.enabled():
        num_workers = min(len(ir_modules), 128)
    else:
        num_workers = get_parallel_num_workers(max_num_worker, mem_for_worker)

    # Shuffle modules for balanced workloads
    random.seed(42)
    random.shuffle(ir_modules)
    random.seed()

    if num_workers > 1 and len(ir_modules) > 1:
        lazy_initialize_cuda()
        ir_modules_list = regroup_modules(ir_modules, num_workers)
        assert check_function_singular(ir_modules_list), "Duplicate function names detected in regrouped modules."

        jobs = [(group, output_dir) for group, output_dir in zip(ir_modules_list, output_dirs[: len(ir_modules_list)])]

        for _ in tqdm(
            parallel_imap(build_job, jobs, num_workers, mem_for_worker), desc="Compiling", total=len(jobs), ncols=80
        ):
            pass

        return output_dirs[: len(ir_modules_list)]
    else:
        build_ir_module(ir_modules, output_dir=output_dirs[0], output_kind=output_kind, target=target, force=force)
        return [output_dirs[0]]


def get_library_name(output_kind):
    if output_kind == '.so':
        return 'lib.so'
    elif output_kind == '.o':
        return 'lib.o'
    else:
        raise ValueError(f"Invalid output kind: {output_kind}")


def should_skip_build(lib_path, output_kind, output_dir, force):
    '''lib_path always contains .lock file'''
    return (
        os.path.exists(lib_path)
        and os.path.getsize(lib_path) > 1
        and (output_kind != '.so' or os.path.exists(os.path.join(output_dir, 'func_types.pickle')))
        and not force
    )


def get_source_path(output_dir, target):
    if target.name == 'cuda':
        return os.path.join(output_dir, 'source.cu')
    elif target.name == 'cpu':
        return os.path.join(output_dir, 'source.cc')
    else:
        raise ValueError(f"Invalid target: {target}")


def lower_ir_module(ir_module, output_dir, target):
    configure_target(target)
    if isinstance(ir_module, Sequence):
        for i in range(len(ir_module)):
            instruments = create_instruments(output_dir, ir_module[i])
            with PassContext(instruments=instruments):
                ir_module[i] = lower(ir_module[i])
    else:
        instruments = create_instruments(output_dir, ir_module)
        with PassContext(instruments=instruments):
            ir_module = lower(ir_module)
    return ir_module


def collect_dependencies(ir_module):
    include_dir, linking_dir, linking_lib, object_file = [], [], [], []
    if isinstance(ir_module, Sequence):
        for im in ir_module:
            include_dir.extend(im.include_dirs)
            linking_dir.extend(im.linking_dirs)
            linking_lib.extend(im.linking_libs)
            object_file.extend(im.object_files)
    else:
        include_dir.extend(ir_module.include_dirs)
        linking_dir.extend(ir_module.linking_dirs)
        linking_lib.extend(ir_module.linking_libs)
        object_file.extend(ir_module.object_files)
    return include_dir, linking_dir, linking_lib, object_file


def write_function_types(ir_module, output_dir):
    """
    Write function types for public functions in the IR module.
    """
    func_types: Dict[str, FuncType] = {
        func.name: FuncType.from_func(func) for func in ir_module.functions.values() if func.kind == 'public'
    }
    with open(os.path.join(output_dir, 'func_types.pickle'), 'wb') as f:
        pickle.dump(func_types, f)
