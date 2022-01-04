import os.path
import ctypes
import subprocess
from subprocess import PIPE
from hidet.ir.func import IRModule, Function
from hidet.ir.type import VoidType
from hidet.core.worker import Host
from hidet.runtime.module import CompiledModule, CompiledFunction
from hidet.backend.cuda import codegen
from hidet.ffi import PackedFunc


def compile_src_code(src_path, out_lib_path):
    # nvcc --ptxas-options=-v --compiler-options '-fPIC' -o mylib.so --shared mykernel.cu
    command = ['nvcc', '--ptxas-options=-v', "--compiler-options",  "'-fPIC'", '-o',  out_lib_path, '--shared', src_path]
    try:
        subprocess.run(command, stderr=PIPE, stdout=PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print(e)
        print(str(e.stderr))
        raise e


def build(ir_module: IRModule, output_dir) -> CompiledModule:
    src_code, func_name_map = codegen(ir_module)
    src_path = os.path.join(output_dir, 'source.cu')
    lib_path = os.path.join(output_dir, 'lib.so')
    with open(src_path, 'w') as f:
        f.write(src_code)
    compile_src_code(src_path, lib_path)
    lib = ctypes.CDLL(lib_path)

    compiled_funcs = {}
    for func in ir_module.functions.values():
        if isinstance(func.get_attr('worker'), Host):
            assert isinstance(func.ret_type, VoidType)
            grid_func = ir_module.lookup(func.name.replace('.host', '.grid'))
            grid_func_param_types = [p.type for p in grid_func.params]
            packed_func = PackedFunc(grid_func_param_types, lib[func_name_map[func.name]])
            compiled_funcs[func.name] = CompiledFunction(func.name, func, packed_func)
    return CompiledModule(ir_module, compiled_funcs, src_code)
