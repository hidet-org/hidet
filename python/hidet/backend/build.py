import ctypes
import os.path
import subprocess
import uuid
import io
from subprocess import PIPE

from hidet import utils
from hidet.backend import codegen
from hidet.ffi import PackedFunc
from hidet.ir.dialects.lowlevel import VoidType
from hidet.ir.func import IRModule
from hidet.runtime.module import CompiledModule, CompiledFunction
from hidet.transforms import lower
from hidet.utils import Timer, COLORS


def compile_src_code(src_path, keep=True, working_dir=None, keep_dir=None):
    if working_dir is None:
        working_dir = os.path.join(os.path.dirname(src_path), 'info')
    if keep_dir is None:
        keep_dir = os.path.dirname(src_path)
    src_path, working_dir, keep_dir = [os.path.abspath(path) for path in [src_path, working_dir, keep_dir]]
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(keep_dir, exist_ok=True)

    # use random lib name to avoid the dlopen caching loading the old library
    out_lib_path = os.path.join(working_dir, str(uuid.uuid4().hex)[-6:] + '.so')
    cc = utils.cuda.get_attribute('compute_capacity')
    cc_code = f'{cc[0]}{cc[1]}'
    command = ['nvcc',
               '-keep' if keep else '--verbose',
               '-gencode', f'arch=compute_{cc_code},code=sm_{cc_code}',
               '--ptxas-options=-v',
               '--compiler-options', "'-fPIC'",
               '-lineinfo',
               '-o',  out_lib_path,
               '--shared', src_path]
    try:
        result = subprocess.run(command, stderr=PIPE, stdout=PIPE, check=True, cwd=working_dir)
        # move source.ptx to be the same directory as source.cu
        if keep:
            # move the ptx code to the same directory as source code
            ptx_name = os.path.basename(src_path).replace('.cu', '.ptx')
            ptx_path = os.path.join(working_dir, ptx_name)
            dest_ptx_path = os.path.join(keep_dir, ptx_name)
            os.rename(ptx_path, dest_ptx_path)
            # output the compilation log to 'nvcc.stdout' file in the same directory as source code
            with open(os.path.join(keep_dir, 'nvcc_output.txt'), 'w') as f:
                f.write("Command: " + " ".join(result.args) + "\n")
                f.write(result.stdout.decode('utf-8'))
                f.write(result.stderr.decode('utf-8'))
        return out_lib_path
    except subprocess.CalledProcessError as e:
        print(' '.join(command))
        print(e.stderr.decode('utf-8'))
        raise e


def build(ir_module: IRModule, output_dir, keep=True, verbose=True) -> CompiledModule:
    # lower
    with Timer() as lower_timer:
        ir_module = lower(ir_module)

    # codegen
    os.makedirs(output_dir, exist_ok=True)
    src_code, func_name_map = codegen(ir_module)

    # write source code to disk
    src_path = os.path.join(output_dir, 'source.cu')
    with open(src_path, 'w') as f:
        f.write(src_code)

    # call target compiler to get dynamic library
    with Timer() as target_compile_timer:
        lib_path = compile_src_code(src_path, keep=keep)
    if verbose:
        info = [
            ('hidet lower time', lower_timer.elapsed_seconds()),
            ('nvcc compile time', target_compile_timer.elapsed_seconds())
        ]
        for name, time in info:
            print('{:>30} {}{:.3f}{} seconds'.format(name, COLORS.OKGREEN, time, COLORS.ENDC))
    # load dynamic library
    lib = ctypes.CDLL(lib_path)
    compiled_funcs = {}
    for func in ir_module.functions.values():
        # only load the packed function into python CompiledFunction
        if func.get_attr('packed_func') is not None:
            assert isinstance(func.ret_type, VoidType)
            target_func = ir_module.lookup(func.get_attr('packed_func'))
            target_func_param_types = [p.type for p in target_func.params]
            packed_func = PackedFunc(target_func_param_types, lib[func_name_map[func.name]])
            compiled_funcs[func.name] = CompiledFunction(func.name, func, packed_func)

    return CompiledModule(ir_module, compiled_funcs, src_code)
