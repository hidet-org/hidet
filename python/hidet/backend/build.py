from typing import List, Tuple
import multiprocessing
import ctypes
import os.path
import subprocess
import uuid
from subprocess import PIPE

from hidet import utils
from hidet.backend import codegen
from hidet.ffi import PackedFunc
from hidet.ir.dialects.lowlevel import VoidType
from hidet.ir.func import IRModule
from hidet.runtime.module import CompiledModule, CompiledFunction
from hidet.transforms import lower, PassContext
from hidet.utils import Timer, COLORS


def compile_src_code(src_path, nvcc_keep=True, working_dir=None, keep_dir=None):
    if working_dir is None:
        working_dir = os.path.join(os.path.dirname(src_path), 'info')
    if keep_dir is None:
        keep_dir = os.path.dirname(src_path)
    src_path, working_dir, keep_dir = [os.path.abspath(path) for path in [src_path, working_dir, keep_dir]]
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(keep_dir, exist_ok=True)

    # use random lib name to avoid the dlopen caching loading the old library
    out_lib_path = os.path.join(working_dir, str(uuid.uuid4().hex)[-6:] + '.so')
    # cc = utils.cuda.get_attribute('compute_capacity')
    cc = utils.cuda.get_compute_capability()
    cc_code = f'{cc[0]}{cc[1]}'
    command = ['nvcc',
               '-keep' if nvcc_keep else '--verbose',
               '-gencode', f'arch=compute_{cc_code},code=sm_{cc_code}',
               '--ptxas-options=-v',
               '--compiler-options', "'-fPIC'",
               '-lineinfo',
               '-o', out_lib_path,
               '--shared', src_path]
    try:
        result = subprocess.run(command, stderr=PIPE, stdout=PIPE, check=True, cwd=working_dir)
        # move source.ptx to be the same directory as source.cu
        if nvcc_keep:
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


def build(ir_module: IRModule, output_dir, keep_ir=False, nvcc_keep=True, verbose=True) -> CompiledModule:
    lib_path, ir_module = lower_and_compile(ir_module, output_dir, keep_ir, nvcc_keep, verbose)
    return load_compiled_module(lib_path=lib_path, lowered_ir_module=ir_module)


class BuildInstance:
    def __init__(self, ir_module: IRModule, output_dir, keep_ir=False, nvcc_keep=True, verbose=True):
        self.ir_module = ir_module
        self.output_dir = output_dir
        self.keep_ir = keep_ir
        self.nvcc_keep = nvcc_keep
        self.verbose = verbose

    def get(self):
        return self.ir_module, self.output_dir, self.keep_ir, self.nvcc_keep, self.verbose


def lower_and_compile(ir_module: IRModule, output_dir, keep_ir: bool = False, nvcc_keep: bool = False, verbose: bool = False) -> Tuple[str, IRModule]:
    # lower
    with Timer() as lower_timer:
        with PassContext(save_lowering_results=keep_ir, save_dir=os.path.join(output_dir, 'ir')):
            ir_module = lower(ir_module)

    # codegen
    os.makedirs(output_dir, exist_ok=True)
    src_code = codegen(ir_module)

    # write source code to disk
    src_path = os.path.join(output_dir, 'source.cu')
    with open(src_path, 'w') as f:
        f.write(src_code)

    # call target compiler to get dynamic library
    with Timer() as target_compile_timer:
        lib_path = compile_src_code(src_path, nvcc_keep=nvcc_keep)
    if verbose:
        info = [
            ('hidet lower time', lower_timer.elapsed_seconds()),
            ('nvcc compile time', target_compile_timer.elapsed_seconds())
        ]
        for name, time in info:
            print('{:>30} {}{:.3f}{} seconds'.format(name, COLORS.OKGREEN, time, COLORS.ENDC))
    return lib_path, ir_module


def load_compiled_module(lib_path: str, lowered_ir_module: IRModule) -> CompiledModule:
    # load dynamic library
    lib = ctypes.CDLL(lib_path)
    compiled_funcs = {}
    for func in lowered_ir_module.functions.values():
        # only load the packed function into python CompiledFunction
        if func.get_attr('packed_func') is not None:
            assert isinstance(func.ret_type, VoidType)
            target_func = lowered_ir_module.lookup(func.get_attr('packed_func'))
            target_func_param_types = [p.type for p in target_func.params]
            packed_func = PackedFunc(target_func_param_types, lib[func.name])
            compiled_funcs[func.name] = CompiledFunction(func.name, func, packed_func)

    return CompiledModule(lowered_ir_module, compiled_funcs)


def batch_build(build_instances: List[BuildInstance], parallel=True, verbose=False) -> List[CompiledModule]:
    with Timer() as timer:
        if parallel:
            # Set the affinity of current process. Some package such as numpy will change affinity of current process,
            # which might limit the parallelism of compilation.
            os.sched_setaffinity(0, range(os.cpu_count()))
            with multiprocessing.Pool() as pool:
                # We doing the lower_and_compile in parallel instead of build because we can not transfer ctypes pointer
                pairs = list(pool.starmap(lower_and_compile, [instance.get() for instance in build_instances]))
            ret = [load_compiled_module(lib_path, ir_module) for lib_path, ir_module in pairs]
        else:
            ret = [build(*ins.get()) for ins in build_instances]
    if verbose:
        print('batch build {} modules within {:.3f} seconds'.format(len(build_instances), timer.elapsed_seconds()))
    return ret
