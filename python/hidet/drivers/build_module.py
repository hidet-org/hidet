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


def build_ir_module(
    ir_module: Union[IRModule, Sequence[IRModule]], output_dir: str, target: str, output_kind: str = '.so', force=False
):
    """
    Build an IR module to a shared library or object file.

    This driver function performs the following steps to build an IR module:

    1. Lower and optimize the IR module with a sequence of pre-defined passes.
    2. Generate source code from the lowered IR module.
    3. Call the underlying compiler (e.g., gcc or nvcc) to compile the generated source code to a shared library (when
       `output_kind == '.so'`) or an object file (when `output_kind == '.o'`).

    Parameters
    ----------
    ir_module: Union[IRModule, Sequence[IRModule]]
        The IR module to be built. This can be a single IRModule or a sequence of IRModules.

    output_dir: str
        The directory to save the generated source code and the compiled library.

    target: str
        The target to build the IR module. Currently, we support two targets: `cpu` and `cuda`. The target can also
        specify attributes (e.g., 'cuda --arch=sm_70').

    output_kind: str
        The output kind. Currently, we support two kinds: `'.so'` and `'.o'`. The former means that the IR module will
        be compiled to a shared library, while the latter means that the IR module will be compiled to an object file.

    force: bool
        Whether to force re-build the IR module. By default, we will not re-build the IR module if the library has been
        built at the specified output directory.
    """
    if output_kind == '.so':
        lib_name = 'lib.so'
    elif output_kind == '.o':
        lib_name = 'lib.o'
    else:
        raise ValueError(f'Invalid output kind: {output_kind}')
    lib_path = os.path.join(output_dir, lib_name)

    if (
        os.path.exists(lib_path)
        and os.path.getsize(lib_path) > 0
        and (output_kind != '.so' or os.path.exists(os.path.join(output_dir, 'func_types.pickle')))
        and not force
    ):
        # the library has been built
        return

    if hidet.option.compile_server.enabled() and can_remote_build(ir_module):
        from hidet.apps.compile_server import remote_build

        remote_build(ir_module, output_dir, target=target, output_kind=output_kind)
        return

    if isinstance(target, str):
        target = Target.from_string(target)

    if target.name == 'cuda':
        src_path = os.path.join(output_dir, 'source.cu')
    elif target.name == 'cpu':
        src_path = os.path.join(output_dir, 'source.cc')
    else:
        raise ValueError(f'Invalid target: {target}')

    # set the recursion limit before every lowering, because some other packages might change this value to a lower
    # value that we need
    set_stack_limit()

    # lower ir module
    instruments = []
    if hidet.option.get_save_lower_ir():
        instruments.append(SaveIRInstrument(out_dir=os.path.join(output_dir, './ir')))
        instruments.append(ProfileInstrument(log_file=os.path.join(output_dir, './lower_time.txt')))
    with hidet.option.context():
        if target.name == 'cuda' and 'arch' in target.attrs:
            hidet.option.cuda.arch(target.attrs['arch'])
        if target.name == 'cuda' and 'cpu_arch' in target.attrs:
            hidet.option.cpu.arch(target.attrs['cpu_arch'])
        if target.name == 'cpu' and 'arch' in target.attrs:
            hidet.option.cpu.arch(target.attrs['arch'])
        with PassContext(instruments=instruments):
            if isinstance(ir_module, Sequence):
                for i in range(len(ir_module)):
                    ir_module[i] = lower(ir_module[i])
            else:
                ir_module = lower(ir_module)

    # code generation
    codegen(ir_module, src_out_path=src_path, target=target)

    include_dir = []
    linking_dir = []
    linking_lib = []
    object_file = []
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
    # compile source code
    compile_source(
        src_path,
        output_library_file=lib_path,
        target=target,
        include_dirs=include_dir,
        linking_dirs=linking_dir,
        linking_libraries=linking_lib,
        object_files=object_file,
    )

    # write the function types
    if output_kind == '.so':
        func_types: Dict[str, FuncType] = {
            func.name: FuncType.from_func(func) for func in ir_module.functions.values() if func.kind == 'public'
        }
        with open(os.path.join(output_dir, 'func_types.pickle'), 'wb') as f:
            pickle.dump(func_types, f)


def build_ir_module_batch(
    ir_modules: Sequence[IRModule], output_dirs: Sequence[str], output_kind: str, target: str, force: bool = False
):
    """
    Build a batch of ir modules.

    Parameters
    ----------
    ir_modules: Sequence[IRModule]
        A sequence of ir modules to build.

    output_dirs: Squence[str]
        Directories for compilation artifacts

    output_kind: str
        The output kind of the compiled library. Can be '.so' or '.o'.

    target: str
        The target of the compilation. Can be 'cuda' or 'cpu'.

    force: bool
        Whether to force re-build the IR module. By default, we will not re-build the IR module if the library has been
        built at the specified output directory.
    """

    def build_job(args):
        ir_module, output_dir = args
        build_ir_module(ir_module, output_dir, output_kind=output_kind, target=target, force=force)

    def regroup_modules(modules, per_worker_jobs, num_workers):
        if len(modules) >= num_workers:
            initial_list_len = per_worker_jobs * num_workers
            # first assign equal amount of the jobs to every worker
            initial_list = [modules[i : i + per_worker_jobs] for i in range(0, initial_list_len, per_worker_jobs)]
            # take the remaining jobs and assign them each to a worker, adding at most one job to each worker
            reminder_len = len(modules) - initial_list_len
            for i, j in zip(range(initial_list_len, len(modules)), range(reminder_len)):
                initial_list[j].append(modules[i])
            return initial_list
        else:
            return modules

    # check if regrouped IRModules have unique function names
    def check_function_singular(module_list: Union[Sequence[IRModule], Sequence[Sequence[IRModule]]]) -> bool:
        if len(module_list) == 0 or isinstance(module_list[0], IRModule):
            return True
        name_set = set()
        for modules in module_list:
            for module in modules:
                namespace_str = module.namespace
                function_name_list = list(module.extern_functions.keys()) + list(module.functions.keys())
                for func_name in function_name_list:
                    func_str = namespace_str + '::' + func_name
                    if func_str in name_set:
                        return False
                    else:
                        name_set.add(func_str)
        return True

    # calculate the number of workers
    max_num_worker, mem_for_worker = option.get_parallel_tune()
    if hidet.option.compile_server.enabled():
        num_workers = min(len(ir_modules), 128)
    else:
        num_workers = get_parallel_num_workers(max_num_worker, mem_for_worker)
    # shuffle the candidates to avoid grouping long-compilation time candidates together
    random.shuffle(ir_modules)
    if num_workers > 1 and len(ir_modules) > 1:
        lazy_initialize_cuda()
        per_worker_jobs = 1 if len(ir_modules) < num_workers else len(ir_modules) // num_workers
        ir_modules_list = regroup_modules(ir_modules, per_worker_jobs, num_workers)
        assert check_function_singular(
            ir_modules_list
        ), 'duplicate function names detected after regrouping candidates for batch compilation'
        jobs = [
            (ir_modules, output_dir)
            for ir_modules, output_dir in zip(ir_modules_list, output_dirs[: len(ir_modules_list)])
        ]

        for _ in tqdm(
            parallel_imap(build_job, jobs, num_workers, mem_for_worker), desc='Compiling', total=len(jobs), ncols=80
        ):
            pass
        return output_dirs[: len(ir_modules_list)]
    else:
        build_ir_module(ir_modules, output_dir=output_dirs[0], output_kind=output_kind, target=target, force=force)
        return [output_dirs[0]]
