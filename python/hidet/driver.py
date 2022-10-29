from typing import List, Optional
import os
import multiprocessing
import logging
from hashlib import sha256
from hidet.transforms import lower, PassContext, SaveIRInstrument, ProfileInstrument
from hidet.backend import codegen, compile_source, load_task_func, load_lib_func
from hidet.utils import hidet_cache_dir
from hidet.utils.py import cyan, green
from hidet.ir.task import Task, TaskContext
from hidet.ir.func import IRModule
from hidet.ir.type import FuncType
from hidet.runtime.module import compiled_task_cache, CompiledFunction

logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

cache_disabled = False


def disable_cache(disable: bool = False):
    global cache_disabled
    cache_disabled = not disable


def build_task(
    task: Task,
    space_level: int = 0,
    target_device: str = 'cuda',
    warmup: int = 3,
    number: int = 10,
    repeat: int = 3,
    use_cache=True,
    cache_dir=None,
    load=True,
):
    # pylint: disable=too-many-arguments, too-many-locals
    task_string: str = str(task)
    compiled_func: Optional[CompiledFunction] = None

    # check in-memory cache
    if compiled_task_cache.contains(target_device, space_level, task_string):
        if load:
            compiled_func = compiled_task_cache.get(target_device, space_level, task_string)
    else:
        # check on-disk cache
        if cache_dir is None:
            cache_dir = os.path.join(hidet_cache_dir(), 'ops')
        config_str = f'{target_device}_space_{space_level}'
        task_hash = sha256(task_string.encode()).hexdigest()[:16]
        task_dir = os.path.join(cache_dir, config_str, task.name, task_hash)
        src_path = os.path.join(task_dir, 'source.cu')
        lib_path = os.path.join(task_dir, 'lib.so')

        # use previously generated library when available
        if not cache_disabled and use_cache and os.path.exists(lib_path):
            logger.debug(f"Load cached task binary {green(task.name)} from path: \n{cyan(lib_path)}")
            if load:
                compiled_func = load_task_func(lib_path, task)
                compiled_task_cache.add(target_device, space_level, task_string, compiled_func)
        else:
            logger.info(f"Compiling task {green(task.name)}...")
            # build from scratch
            os.makedirs(task_dir, exist_ok=True)
            # write task
            with open(os.path.join(task_dir, 'task.txt'), 'w') as f:
                f.write(task_string)
            # implement task
            with TaskContext(space_level, warmup, number, repeat, resolve_out_dir=task_dir):
                ir_module = task.implement(target=target_device)
            # lower ir module
            with PassContext(
                instruments=[
                    SaveIRInstrument(out_dir=os.path.join('./outs/ir', task.name, task_hash)),
                    ProfileInstrument(log_file=os.path.join('./outs/ir', task.name, task_hash, 'lower_time.txt')),
                ]
            ):
                ir_module = lower(ir_module)
            # code generation
            codegen(ir_module, src_out_path=src_path)
            # compile source code
            compile_source(src_path, out_lib_path=lib_path, keep_ptx=False)
            # load function
            if load:
                compiled_func = load_task_func(lib_path, task)
                compiled_task_cache.add(target_device, space_level, task_string, compiled_func)
    return compiled_func


def _build_task_job(args):
    task, space_level, target_device, warmup, number, repeat, use_cache, cache_dir, load = args
    build_task(task, space_level, target_device, warmup, number, repeat, use_cache, cache_dir, load)


def build_batch_task(
    tasks: List[Task],
    space_level: int,
    target_device: str = 'cuda',
    warmup: int = 3,
    number: int = 10,
    repeat: int = 3,
    parallel=True,
    use_cache=True,
    cache_dir=None,
):
    # pylint: disable=too-many-arguments
    jobs = [(task, space_level, target_device, warmup, number, repeat, use_cache, cache_dir, False) for task in tasks]
    if parallel and len(tasks) > 1:
        with multiprocessing.Pool() as pool:
            pool.map(_build_task_job, jobs)
    else:
        map(_build_task_job, jobs)


def build_ir_module(
    ir_module: IRModule,
    func_name: str,
    keep_ptx=False,
    working_dir='./outs',
    verbose=False,
    func_type: Optional[FuncType] = None,
):
    module_string = str(ir_module)
    module_hash = sha256(module_string.encode()).hexdigest()[:16]
    working_dir = os.path.join(working_dir, 'ir_module', module_hash)
    src_path = os.path.join(working_dir, 'source.cu')
    lib_path = os.path.join(working_dir, 'lib.so')

    if verbose:
        print(f'Compiling {src_path}')
    # lower ir module
    with PassContext(
        instruments=[
            SaveIRInstrument(out_dir=working_dir),
            ProfileInstrument(log_file=os.path.join(working_dir, 'lower_time.txt')),
        ]
    ):
        ir_module = lower(ir_module)
    # code generation
    codegen(ir_module, src_out_path=src_path)
    # compile source code
    compile_source(src_path, out_lib_path=lib_path, keep_ptx=keep_ptx)
    if func_type is None:
        func = ir_module.lookup(func_name)
        func_type = FuncType.from_func(func)
    return load_lib_func(lib_path, func_name, func_type=func_type)


if __name__ == '__main__':
    print(sha256('abc'.encode()).hexdigest())
    print(hex(hash('abc')))
    print(type(hex(1)))
