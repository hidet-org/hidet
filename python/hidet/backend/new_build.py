import ctypes
import os
import subprocess
import tempfile
from subprocess import PIPE
from hidet.ir.task import Task
from hidet.runtime import CompiledFunction
from hidet.ffi import PackedFunc
from hidet.utils import cuda


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
    src_path = os.path.abspath(src_path)
    out_lib_path = os.path.abspath(out_lib_path)
    cc = cuda.query_compute_capability()
    cc_code = '{}{}'.format(cc[0], cc[1])
    command = ['nvcc',
               '-keep' if keep_ptx else '',
               '-gencode', f'arch=compute_{cc_code},code=sm_{cc_code}',
               '--ptxas-options=-v',
               '--compiler-options', "'-fPIC'",
               '-lineinfo',
               '-o', out_lib_path,
               '--shared', src_path]
    try:
        with tempfile.TemporaryDirectory() as working_dir:
            result = subprocess.run(" ".join(command).split(), stderr=PIPE, stdout=PIPE, cwd=working_dir)
            if result.returncode:
                message = ''
                if result.stdout:
                    message += result.stdout.decode() + '\n'
                if result.stderr:
                    message += result.stderr.decode()
                raise Exception('Failed to compile file "{}":\n\n{}'.format(src_path, message))
            out_lib_dir = os.path.dirname(out_lib_path)
            if keep_ptx:
                ptx_name = os.path.basename(src_path).replace('.cu', '.ptx')
                ptx_path = os.path.join(working_dir, ptx_name)
                target_ptx_path = os.path.join(out_lib_dir, ptx_name)
                os.rename(ptx_path, target_ptx_path)
            with open(os.path.join(out_lib_dir, 'nvcc_log.txt'), 'w') as f:
                f.write('Command: {}\n'.format(" ".join(result.args)))
                f.write(result.stdout.decode('utf-8'))
                f.write(result.stderr.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print(' '.join(command))
        print(e.stderr.decode('utf-8'))
        raise e


def load_task_func(lib_path: str, task: Task) -> CompiledFunction:
    """
    Load task's entry function from dynamic linked library.

    Parameters
    ----------
    lib_path: str
        The dynamic library path.
    task: Task
        The task that corresponds to the dynamic library.

    Returns
    -------
    ret: CompiledFunction
        The loaded function that can be directly called in python.
    """
    lib = ctypes.CDLL(lib_path)
    func_name = 'hidet_{}'.format(task.name)
    packed_func = PackedFunc(task.param_types(), c_func_pointer=lib[func_name])
    return CompiledFunction(name=task.name, packed_func=packed_func)


def load_ntask_func(lib_path: str, task) -> CompiledFunction:
    """
    Load task's entry function from dynamic linked library.

    Parameters
    ----------
    lib_path: str
        The dynamic library path.
    task: hidet.tos.task.Task
        The task that corresponds to the dynamic library.

    Returns
    -------
    ret: CompiledFunction
        The loaded function that can be directly called in python.
    """
    try:
        lib = ctypes.CDLL(lib_path)
    except OSError as e:
        print(e)
        print("Removed the file '{}'".format(lib_path))
        os.remove(lib_path)
        exit(0)
    func_name = 'hidet_{}'.format(task.name)
    param_types = [param.data_type for param in task.parameters]
    packed_func = PackedFunc(param_types=param_types, c_func_pointer=lib[func_name])
    return CompiledFunction(name=task.name, packed_func=packed_func)
