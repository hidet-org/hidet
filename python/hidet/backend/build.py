import string
import random
import os.path
import shutil
import ctypes
import subprocess
from subprocess import PIPE
from hidet.ir.func import IRModule
from hidet.ir.task import Host
from hidet.ir.dialects.lowlevel import VoidType
from hidet.runtime.module import CompiledModule, CompiledFunction
from hidet.backend import codegen
from hidet.ffi import PackedFunc
from hidet.transforms import generate_packed_func_pass, flatten_tensor_pass, const_expr_simplifier_pass, eliminate_dead_device_function_pass

from hidet import utils
import uuid

import pycuda


def compile_src_code(src_path, working_dir=None):
    src_path = os.path.realpath(src_path)
    if working_dir is None:
        working_dir = os.path.join(os.path.dirname(src_path), 'info')
        os.makedirs(working_dir, exist_ok=True)

    out_lib_path = os.path.join(working_dir, str(uuid.uuid4().hex)[-6:] + 'so')
    # ''.join(random.choice(string.ascii_lowercase) for _ in range(8)) + '.so'
    cc = utils.cuda.get_attribute('compute_capacity')
    cc_code = f'{cc[0]}{cc[1]}'
    command = ['nvcc',
               '-keep',
               '-gencode', f'arch=compute_{cc_code},code=sm_{cc_code}',
               '--ptxas-options=-v',
               '--compiler-options', "'-fPIC'",
               '-o',  out_lib_path,
               '--shared', src_path]
    try:
        subprocess.run(command, stderr=PIPE, stdout=PIPE, check=True, cwd=working_dir)
        # move source.ptx to be the same directory as source.cu
        ptx_name = os.path.basename(src_path).replace('.cu', '.ptx')
        ptx_path = os.path.join(working_dir, ptx_name)
        dest_ptx_path = os.path.join(os.path.dirname(src_path), ptx_name)
        os.rename(ptx_path, dest_ptx_path)
        return out_lib_path
    except subprocess.CalledProcessError as e:
        print(' '.join(command))
        print(e.stderr.decode('utf-8'))
        raise e


def lower(ir_module: IRModule) -> IRModule:
    transforms = [
        eliminate_dead_device_function_pass(),
        generate_packed_func_pass(),
        flatten_tensor_pass(),
        const_expr_simplifier_pass(),
    ]

    for transform in transforms:
        ir_module = transform(ir_module)

    return ir_module


def build(ir_module: IRModule, output_dir) -> CompiledModule:
    # lower
    ir_module = lower(ir_module)

    # codegen
    os.makedirs(output_dir, exist_ok=True)
    src_code, func_name_map = codegen(ir_module)

    # write source code to disk
    src_path = os.path.join(output_dir, 'source.cu')
    with open(src_path, 'w') as f:
        f.write(src_code)

    # call target compiler to get dynamic library
    lib_path = compile_src_code(src_path)

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
