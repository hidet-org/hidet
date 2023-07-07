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
from typing import Sequence, Dict
import logging
import os
import pickle

import psutil
from tqdm import tqdm

import hidet.cuda
from hidet import option
from hidet.backend import codegen, compile_source
from hidet.drivers.utils import lazy_initialize_cuda
from hidet.ir.module import IRModule
from hidet.ir.type import FuncType
from hidet.ir.target import Target
from hidet.transforms import lower, PassContext, SaveIRInstrument, ProfileInstrument
from hidet.utils.multiprocess import parallel_imap

logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def can_remote_build(ir_module: IRModule) -> bool:
    return not (len(ir_module.object_files) > 0 or len(ir_module.linking_dirs) > 0 or len(ir_module.include_dirs) > 0)


def build_ir_module(ir_module: IRModule, output_dir: str, *, target: str, output_kind: str = '.so', force=False):
    """
    Build an IR module to a shared library or object file.

    This driver function performs the following steps to build an IR module:

    1. Lower and optimize the IR module with a sequence of pre-defined passes.
    2. Generate source code from the lowered IR module.
    3. Call the underlying compiler (e.g., gcc or nvcc) to compile the generated source code to a shared library (when
       `output_kind == '.so'`) or an object file (when `output_kind == '.o'`).

    Parameters
    ----------
    ir_module: IRModule
        The IR module to be built.

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

    # lower ir module
    instruments = []
    if hidet.option.get_save_lower_ir():
        instruments.append(SaveIRInstrument(out_dir=os.path.join(output_dir, './ir')))
        instruments.append(ProfileInstrument(log_file=os.path.join(output_dir, './lower_time.txt')))
    with hidet.option.context():
        if target.name == 'cuda' and 'arch' in target.attrs:
            hidet.option.cuda.arch(target.attrs['arch'])
        with PassContext(instruments=instruments):
            ir_module = lower(ir_module)

    # code generation
    codegen(ir_module, src_out_path=src_path, target=target)

    # compile source code
    compile_source(
        src_path,
        output_library_file=lib_path,
        target=target,
        include_dirs=ir_module.include_dirs,
        linking_dirs=ir_module.linking_dirs,
        linking_libraries=ir_module.linking_libs,
        object_files=ir_module.object_files,
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

    output_dirs: Sequence[str]
        The output directory to save the compiled library and source code (lib.so and source.cu).

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

    jobs = [(ir_module, output_dir) for ir_module, output_dir in zip(ir_modules, output_dirs)]

    # calculate the number of workers
    cpu_count = os.cpu_count()
    if hidet.option.compile_server.enabled():
        num_workers = min(len(jobs), 128)
    else:
        max_jobs, mem_for_worker = option.get_parallel_tune()
        max_jobs = cpu_count if max_jobs == -1 else min(max_jobs, cpu_count)
        mem_for_worker *= 1024**3
        num_workers = min(max(int(psutil.virtual_memory().available // mem_for_worker), 1), max_jobs)

    if num_workers > 1 and len(jobs) > 1:
        # Set the affinity of current process. Some package such as numpy will change affinity of current process,
        # which might limit the parallelism of compilation.
        from contextlib import suppress

        with suppress(OSError):
            os.sched_setaffinity(0, range(cpu_count))

        lazy_initialize_cuda()

        for _ in tqdm(parallel_imap(build_job, jobs, num_workers), desc='Compiling', total=len(jobs), ncols=80):
            pass
    else:
        # sequential build
        for job in tqdm(jobs, desc='Compiling', ncols=80, disable=len(jobs) == 1):
            build_job(job)
