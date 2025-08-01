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
from typing import Optional, List, Sequence, Union
import time
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
from hidet.ir.target import Target


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

    def compile(
        self,
        src_path: str,
        out_lib_path: str,
        target: Target,
        include_dirs: Sequence[str] = (),
        linking_dirs: Sequence[str] = (),
        linking_libs: Sequence[str] = (),
        object_files: Sequence[str] = (),
    ) -> None:
        raise NotImplementedError()

    def run_compile_command(self, command: str, src_path, out_lib_path: str):
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

                t1 = time.time()
                result = subprocess.run(command.split(), stderr=PIPE, stdout=PIPE, cwd=working_dir, check=False)
                t2 = time.time()

                # if the compilation failed, raise an exception
                if result.returncode:
                    message = "Command: {}\n".format(command)
                    if result.stdout:
                        message += result.stdout.decode().strip() + '\n'
                    if result.stderr:
                        message += result.stderr.decode().strip()
                    raise CompilationFailed(src_path, message)

                # write the compilation log
                log_name = self.__class__.__name__.lower() + '_output.txt'
                with open(os.path.join(out_lib_dir, log_name), 'w', encoding='utf-8') as f:
                    output = '\n'.join([result.stdout.decode('utf-8').strip(), result.stderr.decode('utf-8').strip()])
                    f.write(output.strip())
                    f.write('\n')
                    f.write('elapsed time: {:.3f} seconds'.format(t2 - t1))

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

    def compile(
        self,
        src_path: str,
        out_lib_path: str,
        target: Target,
        include_dirs: Sequence[str] = (),
        linking_dirs: Sequence[str] = (),
        linking_libs: Sequence[str] = (),
        object_files: Sequence[str] = (),
    ) -> None:
        if len(object_files) > 0 and out_lib_path.endswith('.o'):
            raise ValueError('Can not compile multiple objects into a single object file.')

        if 'arch' in target.attrs:
            arch = target.attrs['arch']
        else:
            arch = hidet.option.cuda.get_arch()

        # We use the `a` suffix for all `sm_90` and `sm_100` GPUs.
        need_suffix = {'sm_90', 'sm_100'}
        arch = arch + 'a' if arch in need_suffix else arch

        if 'cpu_arch' in target.attrs:
            cpu_arch = target.attrs['cpu_arch']
        else:
            cpu_arch = hidet.option.cpu.get_arch()

        # The following command compiles the cuda source code to a shared library
        # See https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
        # for more information about nvcc compilation.
        command = [
            # the path to nvcc compiler
            self.nvcc_path,
            # the included directories.
            *['-I{}'.format(include_dir) for include_dir in self.include_dirs + list(include_dirs)],
            # the library directories.
            *['-L{}'.format(library_dir) for library_dir in self.library_dirs + list(linking_dirs)],
            *['-l{}'.format(library) for library in [*linking_libs, 'cuda']],
            # optimize host side code via -O3
            '-O3',
            # host compiler options: enable openmp, avx2, unroll loops and fast math
            '-Xcompiler -fPIC,-m64,-march={cpu_arch},-O3,-funroll-loops,-ffast-math'.format(cpu_arch=cpu_arch),
            # use c++11 standard
            '-std=c++11',
            # the target PTX and SASS version.
            '-gencode arch=compute_{cc},code=sm_{cc}'.format(cc=arch[len('sm_') :]),
            # allow ptxas (PTX assembler) to output information like register/smem usage.
            '--ptxas-options=-v',
            # compile into position independent code.
            # '--compiler-options -fPIC,-m64,-mavx2,-march=native, -O3',
            # embed the line information into the binary, allow Nsight Compute to get the source code for profiling.
            '-lineinfo',
            # ftz=true and prec-div=false for fast math
            '-ftz={}'.format('true' if hidet.option.get_option('cuda.build.ftz') else 'false'),
            '-prec-div={}'.format('true' if hidet.option.get_option('cuda.build.prec_div') else 'false'),
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
            '--shared' if out_lib_path.endswith('.so') else '--compile',
            # the linking objects.
            ' '.join(object_files),
            # the source path.
            src_path,
            # the output library path.
            '-o',
            out_lib_path,
        ]

        self.run_compile_command(" ".join(command), src_path, out_lib_path)


class HIPCC(SourceCompiler):
    def __init__(self) -> None:
        super().__init__()
        self.hipcc_path: str = self._resolve_hipcc_path()  # e.g., /opt/rocm/bin/hipcc
        self.include_dirs: List[str] = get_include_dirs()
        self.library_dirs: List[str] = [os.path.dirname(library_paths['hidet_runtime'])]

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _resolve_hipcc_path():
        path: Optional[str] = shutil.which('hipcc')
        if path is not None:
            return path
        try_dirs = ['/opt/rocm/bin/', '/usr/bin']
        for try_dir in try_dirs:
            path = os.path.join(try_dir, 'hipcc')
            if os.path.exists(path):
                return path
        raise FileNotFoundError('Can not find hipcc compiler.')

    def compile(
        self,
        src_path: str,
        out_lib_path: str,
        target: Target,
        include_dirs: Sequence[str] = (),
        linking_dirs: Sequence[str] = (),
        linking_libs: Sequence[str] = (),
        object_files: Sequence[str] = (),
    ) -> None:
        if len(object_files) > 0 and out_lib_path.endswith('.o'):
            raise ValueError('Can not compile multiple objects into a single object file.')

        os.environ['HIP_PLATFORM'] = 'amd'

        if 'arch' in target.attrs:
            arch = target.attrs['arch']
        else:
            arch = hidet.option.hip.get_arch()

        # The following command compiles the hip source code to a shared library
        # See https://sep5.readthedocs.io/en/latest/Programming_Guides/HIP-GUIDE.html#hip-guide,
        # and command `hipcc --help` for more information about hipcc compilation.
        command = [
            # the path to hipcc compiler
            self.hipcc_path,
            # the included directories.
            *['-I{}'.format(include_dir) for include_dir in self.include_dirs + list(include_dirs)],
            # the library directories.
            *['-L{}'.format(library_dir) for library_dir in self.library_dirs + list(linking_dirs)],
            *['-l{}'.format(library) for library in linking_libs],
            # optimize host side code via -O3
            '-O3',
            # (in host compiler) compile into position independent code.
            '-Xcompiler -fPIC',
            # use c++11 standard
            '-std=c++11',
            # link the hidet runtime, all APIs for communication between kernels and host system are in hidet runtime.
            '-lhidet_runtime',
            '-Wno-unused-command-line-argument',
            '-funroll-loops',
            '-ffast-math',
            '--offload-arch={}'.format(arch),
            # allow constexpr function to be called from device code.
            # '--expt-relaxed-constexpr',
            # generate shared library (lib.so).
            '--shared' if out_lib_path.endswith('.so') else '-c',
            # the linking objects.
            ' '.join(object_files),
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

    def compile(
        self,
        src_path: str,
        out_lib_path: str,
        target: Target,
        include_dirs: Sequence[str] = (),
        linking_dirs: Sequence[str] = (),
        linking_libs: Sequence[str] = (),
        object_files: Sequence[str] = (),
    ) -> None:
        if len(object_files) > 0 and out_lib_path.endswith('.o'):
            raise ValueError('Can not compile multiple objects into a single object file.')

        if 'arch' in target.attrs:
            arch = target.attrs['arch']
        else:
            arch = hidet.option.cpu.get_arch()

        command = [
            # the path to nvcc compiler
            self.gcc_path,
            # the included directories.
            *['-I{}'.format(include_dir) for include_dir in self.include_dirs + list(include_dirs)],
            # the library directories.
            *['-L{}'.format(library_dir) for library_dir in self.library_dirs + list(linking_dirs)],
            *['-l{}'.format(library) for library in linking_libs],
            # apply -O3 optimization.
            '-O3',
            # use c++11 standard
            '-std=c++11',
            # support avx intrinsics
            '-mavx2',
            '-m64',
            '-ffast-math',
            '-march={arch}'.format(arch=arch),
            # compile into position independent code.
            '-fPIC',
            # enable OpenMP.
            '-fopenmp',
            # link the hidet runtime, all APIs for communication between kernels and host system are in hidet runtime.
            '-Wl,--no-as-needed -lhidet_runtime',
            # generate shared library (lib.so).
            '-shared' if out_lib_path.endswith('.so') else '--compile',
            # the linking objects.
            ' '.join(object_files),
            # the source path.
            src_path,
            # the output library path.
            '-o',
            out_lib_path,
        ]

        self.run_compile_command(" ".join(command), src_path, out_lib_path)


def compile_source(
    source_file: str,
    output_library_file: str,
    target: Union[str, Target],
    include_dirs: Sequence[str] = (),
    linking_dirs: Sequence[str] = (),
    linking_libraries: Sequence[str] = (),
    object_files: Sequence[str] = (),
) -> None:
    """
    Compile the source code in 'src_path' file and output the library to 'out_lib_path'.

    Parameters
    ----------
    source_file: str
        The path to source code.
    output_library_file: str
        The path to output library.
    target: str or Target
        The target platform. Currently only support 'cpu' and 'gpu'.
    include_dirs: Sequence[str]
        The include directories.
    linking_dirs: Sequence[str]
        The library directories.
    linking_libraries:
        The libraries to link to the output library.
    object_files: Sequence[str]
        The path to object files. If not None, the object files will be linked to the output library.
    """
    source_file = os.path.abspath(source_file)
    output_library_file = os.path.abspath(output_library_file)
    if object_files is not None:
        object_files = [os.path.abspath(object_file) for object_file in object_files]

    if isinstance(target, str):
        target = Target.from_string(target)

    if target.name == 'cuda':
        if not hidet.cuda.available():
            raise RuntimeError('CUDA is not available.')
        compiler = NVCC()
    elif target.name == 'hip':
        if not hidet.hip.available():
            raise RuntimeError('HIP is not available.')
        compiler = HIPCC()
    elif target.name == 'cpu':
        compiler = GCC()
    else:
        raise ValueError('Unknown target platform: {}'.format(target))

    object_files = object_files or []
    compiler.compile(
        source_file,
        output_library_file,
        target,
        include_dirs=include_dirs,
        linking_dirs=linking_dirs,
        linking_libs=linking_libraries,
        object_files=object_files,
    )
