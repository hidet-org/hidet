import os.path
import ctypes
import subprocess
from subprocess import PIPE
from hidet.ir.func import IRModule
from hidet.ir.task import Host
from hidet.ir.dialects.lowlevel import VoidType
from hidet.runtime.module import CompiledModule, CompiledFunction
from hidet.backend import codegen
from hidet.ffi import PackedFunc
from hidet.transforms import generate_packed_func_pass, flatten_tensor_pass, const_expr_simplifier_pass


def compile_src_code(src_path, out_lib_path):
    # nvcc --ptxas-options=-v --compiler-options '-fPIC' -o mylib.so --shared mykernel.cu
    command = ['nvcc', '--ptxas-options=-v', "--compiler-options",  "'-fPIC'", '-o',  out_lib_path, '--shared', src_path]
    try:
        subprocess.run(command, stderr=PIPE, stdout=PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print(str(e.stderr))
        raise e


def lower(ir_module: IRModule) -> IRModule:
    transforms = [
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
    lib_path = os.path.join(output_dir, 'lib.so')
    with open(src_path, 'w') as f:
        f.write(src_code)

    # call target compiler to get dynamic library
    compile_src_code(src_path, lib_path)

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
