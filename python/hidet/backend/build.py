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
from typing import Optional
import functools
import warnings
import os
import shutil
import tempfile
import subprocess
from subprocess import PIPE

import hidet.cuda
from hidet.libinfo import get_include_dirs
from hidet.ir.type import FuncType
from hidet.runtime import CompiledFunction
from hidet.ffi import PackedFunc
from hidet.ffi.ffi import library_paths
from hidet.ffi.shared_lib import SharedLibrary


class CompilationFailed(Exception):
    def __init__(self, source_path: str, msg: str):
        super().__init__()
        self.source_path = source_path
        self.msg = msg

    def __str__(self):
        lines = ['failed to compile file://{}'.format(self.source_path), '{}'.format(self.msg)]
        return '\n'.join(lines)


@functools.lru_cache()
def nvcc_path() -> str:
    path: Optional[str] = shutil.which('nvcc')
    if path is not None:
        return path
    try_dirs = ['/usr/local/cuda/bin/', '/usr/bin']
    for try_dir in try_dirs:
        path = os.path.join(try_dir, 'nvcc')
        if os.path.exists(path):
            return path
    raise FileNotFoundError('Can not find nvcc compiler.')


def compile_source(src_path: str, out_lib_path: str, keep_ptx=False) -> None:
    """
    Compile the source code in 'src_path' file and output the library to 'out_lib_path'.

    Parameters
    ----------
    src_path: str
        The path to source code.
    out_lib_path: str
        The path to output library.
    keep_ptx: bool, default False
        Whether to keep the ptx code in the same directory of output library.
    """
    # pylint: disable=too-many-locals
    src_path = os.path.abspath(src_path)
    out_lib_path = os.path.abspath(out_lib_path)
    cc = hidet.cuda.compute_capability()

    # dir contains the runtime header file 'hidet/runtime.h'
    include_dirs = get_include_dirs()
    # dir contains the runtime library 'libhidet_runtime.so'
    library_dirs = [os.path.dirname(library_paths['hidet_runtime'])]

    cc_code = '{}{}'.format(cc[0], cc[1])
    # The following command compiles the cuda source code to a shared library
    # See https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html for more information about nvcc compilation.
    command = [
        # the path to nvcc compiler
        nvcc_path(),
        # the included directories.
        *['-I{}'.format(include_dir) for include_dir in include_dirs],
        # the library directories.
        *['-L{}'.format(library_dir) for library_dir in library_dirs],
        # keep option will keep the intermediate results during compilation, including PTX.
        '-keep' if keep_ptx else '',
        # the target PTX and SASS version.
        '-gencode',
        f'arch=compute_{cc_code},code=sm_{cc_code}',
        # allow ptxas (PTX assembler) to output information like register/smem usage.
        '--ptxas-options=-v',
        # compile into position independent code.
        '--compiler-options',
        "'-fPIC'",
        # embed the line information into the binary, allow Nsight Compute to get the source code for profiling.
        '-lineinfo',
        # link the hidet runtime, all APIs for communication between kernels and host system are in hidet runtime.
        '-lhidet_runtime',
        # shared cuda runtime library is used (.so), instead of static one (.a). used to reduce binary size.
        '--cudart',
        'shared',
        # supress some warnings
        # see https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#generic-tool-options-diag-suppress
        # supress warming no 177 like: "warning #177-D: variable "xxx" was declared but never referenced"
        '--diag-suppress 177',
        # supress warning no 179 like: "warning #179-D: right operand of "%" is zero"
        '--diag-suppress 179',
        # supress warning no 39 like: "warning #39-D: division by zero"
        '--diag-suppress 39',
        # generate shared library (lib.so).
        '--shared',
        # the source path.
        src_path,
        # the output library path.
        '-o',
        out_lib_path,
    ]

    try:
        # the directory to store the library "lib.so"
        out_lib_dir = os.path.dirname(out_lib_path)

        # write the compilation command to "compile.sh"
        with open(os.path.join(out_lib_dir, 'compile.sh'), 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write(" ".join(command))
            f.write("\n")

        # run the compilation command
        with tempfile.TemporaryDirectory() as working_dir:
            result = subprocess.run(" ".join(command).split(), stderr=PIPE, stdout=PIPE, cwd=working_dir, check=False)
            if result.returncode:
                message = "Command: " + " ".join(command) + "\n"
                if result.stdout:
                    message += result.stdout.decode().strip() + '\n'
                if result.stderr:
                    message += result.stderr.decode().strip()
                ptx_name = os.path.basename(src_path).replace('.cu', '.ptx')
                if keep_ptx and os.path.exists(os.path.join(working_dir, ptx_name)):
                    ptx_path = os.path.join(working_dir, ptx_name)
                    target_ptx_path = os.path.join(out_lib_dir, ptx_name)
                    shutil.move(ptx_path, target_ptx_path)
                raise CompilationFailed(src_path, message)
            if keep_ptx:
                ptx_name = os.path.basename(src_path).replace('.cu', '.ptx')
                ptx_path = os.path.join(working_dir, ptx_name)
                target_ptx_path = os.path.join(out_lib_dir, ptx_name)
                shutil.move(ptx_path, target_ptx_path)
                # os.rename(ptx_path, target_ptx_path)
            with open(os.path.join(out_lib_dir, 'nvcc_log.txt'), 'w') as f:
                output = '\n'.join([result.stdout.decode('utf-8').strip(), result.stderr.decode('utf-8').strip()])
                f.write(output)

                lines = output.split('\n')
                warning_lines = [line for line in lines if 'warning' in line]
                warning_lines = warning_lines[: len(warning_lines) // 2]  # nvcc would print the same warning twice
                if len(warning_lines) > 0:
                    warnings.warn('Compilation warnings:\n' + '\n'.join(warning_lines))
    except subprocess.CalledProcessError as e:
        print(' '.join(command))
        print(e.stderr.decode('utf-8'))
        raise e


def load_task_func(lib_path: str, task) -> CompiledFunction:
    """
    Load task's entry function from dynamic linked library.

    Parameters
    ----------
    lib_path: str
        The dynamic library path.
    task: hidet.graph.task.Task
        The task that corresponds to the dynamic library.

    Returns
    -------
    ret: CompiledFunction
        The loaded function that can be directly called in python.
    """
    try:
        lib = SharedLibrary(lib_path)
    except OSError as e:
        print("Removed the file '{}'".format(lib_path))
        os.remove(lib_path)
        raise e
    func_name = 'hidet_{}'.format(task.name)
    param_types = [param.ttype for param in task.parameters]
    packed_func = PackedFunc(param_types=param_types, c_func_pointer=lib[func_name])

    potential_src_path = os.path.join(os.path.dirname(lib_path), 'source.cu')
    if os.path.isfile(potential_src_path):
        src_path = potential_src_path
    else:
        src_path = None

    return CompiledFunction(name=task.name, packed_func=packed_func, lib_path=lib_path, src_path=src_path)


def load_lib_func(lib_path: str, func_name: str, func_type: FuncType) -> CompiledFunction:
    try:
        lib = SharedLibrary(lib_path)
    except OSError as e:
        print("Removed the file '{}'".format(lib_path))
        os.remove(lib_path)
        raise e
    func_name = 'hidet_{}'.format(func_name)
    packed_func = PackedFunc(param_types=list(func_type.param_types), c_func_pointer=lib[func_name])
    return CompiledFunction(name=func_name, packed_func=packed_func)
