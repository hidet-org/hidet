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
from typing import Optional, List, Dict
import functools
import warnings
import os
import shutil
import tempfile
import subprocess
from subprocess import PIPE

import hidet.cuda
from hidet.libinfo import get_include_dirs
from hidet.ffi.ffi import library_paths


class CompilationFailed(Exception):
    def __init__(self, source_path: str, msg: str):
        super().__init__(source_path, msg)
        self.source_path = source_path
        self.msg = msg

    def __str__(self):
        lines = ['failed to compile file://{}'.format(self.source_path), '{}'.format(self.msg)]
        return '\n'.join(lines)


class SourceCompiler:
    """
    The base class of source compiler.
    """

    def compile(self, src_path: str, out_lib_path: str, options: Optional[Dict[str, str]] = None) -> None:
        raise NotImplementedError()

    @staticmethod
    def run_compile_command(command: str, src_path, out_lib_path: str):
        try:
            # the directory to store the library "lib.so"
            out_lib_dir = os.path.dirname(out_lib_path)

            # write the compilation command to "compile.sh"
            with open(os.path.join(out_lib_dir, 'compile.sh'), 'w') as f:
                f.write("#!/bin/bash\n\n")
                f.write(command)
                f.write("\n")

            # run the compilation command
            with tempfile.TemporaryDirectory() as working_dir:
                result = subprocess.run(command.split(), stderr=PIPE, stdout=PIPE, cwd=working_dir, check=False)

                # if the compilation failed, raise an exception
                if result.returncode:
                    message = "Command: {}\n".format(command)
                    if result.stdout:
                        message += result.stdout.decode().strip() + '\n'
                    if result.stderr:
                        message += result.stderr.decode().strip()
                    raise CompilationFailed(src_path, message)

                # write the compilation log
                with open(os.path.join(out_lib_dir, 'compiler.log'), 'w') as f:
                    output = '\n'.join([result.stdout.decode('utf-8').strip(), result.stderr.decode('utf-8').strip()])
                    f.write(output.strip())

                    lines = output.split('\n')
                    warning_lines = [line for line in lines if 'warning' in line]
                    warning_lines = warning_lines[: len(warning_lines) // 2]  # nvcc would print the same warning twice
                    if len(warning_lines) > 0:
                        warnings.warn('Compilation warnings:\n' + '\n'.join(warning_lines))

        except subprocess.CalledProcessError as e:
            print(command)
            print(e.stderr.decode('utf-8'))
            raise e


class NVCC(SourceCompiler):
    def __init__(self):
        super().__init__()
        self.nvcc_path: str = self._resolve_nvcc_path()  # e.g., /usr/local/cuda/bin/nvcc
        self.include_dirs: List[str] = get_include_dirs()
        self.library_dirs: List[str] = [os.path.dirname(library_paths['hidet_runtime'])]

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _resolve_nvcc_path():
        path: Optional[str] = shutil.which('nvcc')
        if path is not None:
            return path
        try_dirs = ['/usr/local/cuda/bin/', '/usr/bin']
        for try_dir in try_dirs:
            path = os.path.join(try_dir, 'nvcc')
            if os.path.exists(path):
                return path
        raise FileNotFoundError('Can not find nvcc compiler.')

    def compile(self, src_path: str, out_lib_path: str, options: Optional[Dict[str, str]] = None) -> None:
        cc = hidet.cuda.compute_capability()
        cc_code = '{}{}'.format(cc[0], cc[1])

        # The following command compiles the cuda source code to a shared library
        # See https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
        # for more information about nvcc compilation.
        command = [
            # the path to nvcc compiler
            self.nvcc_path,
            # the included directories.
            *['-I{}'.format(include_dir) for include_dir in self.include_dirs],
            # the library directories.
            *['-L{}'.format(library_dir) for library_dir in self.library_dirs],
            # optimize host side code via -O3
            '-O3',
            # host compiler options: enable openmp, avx2, unroll loops and fast math
            '-Xcompiler -fopenmp,-fPIC,-m64,-mavx2,-march=native,-O3,-funroll-loops,-ffast-math',
            # the target PTX and SASS version.
            '-gencode arch=compute_{cc},code=sm_{cc}'.format(cc=cc_code),
            # allow ptxas (PTX assembler) to output information like register/smem usage.
            '--ptxas-options=-v',
            # compile into position independent code.
            # '--compiler-options -fPIC,-m64,-mavx2,-march=native, -O3',
            # embed the line information into the binary, allow Nsight Compute to get the source code for profiling.
            '-lineinfo',
            # ftz=true and prec-div=false for fast math
            '-ftz=true',
            '-prec-div=false',
            # link the hidet runtime, all APIs for communication between kernels and host system are in hidet runtime.
            '-lhidet_runtime',
            # shared cuda runtime library is used (.so), instead of static one (.a). used to reduce binary size.
            '--cudart shared',
            # allow constexpr function to be called from device code.
            # '--expt-relaxed-constexpr',
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

        self.run_compile_command(" ".join(command), src_path, out_lib_path)


class GCC(SourceCompiler):
    def __init__(self):
        super().__init__()
        self.gcc_path: str = self._resolve_gcc_path()
        self.include_dirs: List[str] = get_include_dirs()
        self.library_dirs: List[str] = [os.path.dirname(library_paths['hidet_runtime'])]

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _resolve_gcc_path():
        path: Optional[str] = shutil.which('g++')
        if path is not None:
            return path
        raise FileNotFoundError('Can not find g++ compiler.')

    def compile(self, src_path: str, out_lib_path: str, options: Optional[Dict[str, str]] = None) -> None:
        command = [
            # the path to nvcc compiler
            self.gcc_path,
            # the included directories.
            *['-I{}'.format(include_dir) for include_dir in self.include_dirs],
            # the library directories.
            *['-L{}'.format(library_dir) for library_dir in self.library_dirs],
            # apply -O3 optimization.
            '-O3',
            # support avx intrinsics
            '-mavx2',
            '-m64',
            '-march=native',
            # compile into position independent code.
            '-fPIC',
            # enable OpenMP.
            '-fopenmp',
            # link the hidet runtime, all APIs for communication between kernels and host system are in hidet runtime.
            '-lhidet_runtime',
            # generate shared library (lib.so).
            '-shared',
            # the source path.
            src_path,
            # the output library path.
            '-o',
            out_lib_path,
        ]

        self.run_compile_command(" ".join(command), src_path, out_lib_path)


def compile_source(src_path: str, out_lib_path: str) -> None:
    """
    Compile the source code in 'src_path' file and output the library to 'out_lib_path'.

    Parameters
    ----------
    src_path: str
        The path to source code.
    out_lib_path: str
        The path to output library.
    """
    src_path = os.path.abspath(src_path)
    out_lib_path = os.path.abspath(out_lib_path)

    if hidet.cuda.available():
        compiler = NVCC()
    else:
        compiler = GCC()

    compiler.compile(src_path, out_lib_path)
