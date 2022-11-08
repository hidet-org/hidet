from typing import List, Optional
import os
import multiprocessing
import logging
from hashlib import sha256

from hidet import option
from hidet.transforms import lower, PassContext, SaveIRInstrument, ProfileInstrument
from hidet.backend import codegen, compile_source, load_task_func, load_lib_func
from hidet.utils.py import cyan, green
from hidet.ir.task import Task
from hidet.ir.func import IRModule
from hidet.ir.type import FuncType
from hidet.runtime.module import compiled_task_cache, CompiledFunction

logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def build_task(task: Task, target_device='cuda', load=True) -> Optional[CompiledFunction]:
    """
    Build a task into a compiled function.

    Parameters
    ----------
    task: Task
        The task to be built.
    target_device: str
        The target device. Candidates are 'cuda' and 'cpu'.
    load: bool
        Whether to load the compiled function. If False, the compiled function will not be loaded, and None is returned.
        Otherwise, the compiled function is loaded and returned.
    Returns
    -------
    compiled_func:
        When load is True, the compiled function is returned. Otherwise, None is returned.
    """
    task_string: str = str(task)
    compiled_func: Optional[CompiledFunction] = None

    space_level = option.get_option('search_space')
    op_cache_dir = os.path.join(option.get_option('cache_dir'), './ops')
    use_cache = option.get_option('cache_operator')

    # check in-memory cache
    if compiled_task_cache.contains(target_device, space_level, task_string):
        if load:
            compiled_func = compiled_task_cache.get(target_device, space_level, task_string)
    else:
        # check on-disk cache
        config_str = f'{target_device}_space_{space_level}'
        task_hash = sha256(task_string.encode()).hexdigest()[:16]
        task_dir = os.path.join(op_cache_dir, config_str, task.name, task_hash)
        src_path = os.path.join(task_dir, 'source.cu')
        lib_path = os.path.join(task_dir, 'lib.so')

        # use previously generated library when available
        if use_cache and os.path.exists(lib_path):
            logger.debug(f"Load cached task binary {green(task.name)} from path: \n{cyan(lib_path)}")
            if load:
                compiled_func = load_task_func(lib_path, task)
                compiled_task_cache.add(target_device, space_level, task_string, compiled_func)
        else:
            logger.info(f"Compiling task {green(task.signature())}...")
            # build from scratch
            os.makedirs(task_dir, exist_ok=True)
            # write task
            with open(os.path.join(task_dir, 'task.txt'), 'w') as f:
                f.write(task_string)
            # implement task
            ir_module = task.implement(target=target_device, workding_dir=task_dir)
            # lower ir module
            if option.get_option('save_lower_ir'):
                instruments = [
                    SaveIRInstrument(out_dir=os.path.join(task_dir, './ir')),
                    ProfileInstrument(log_file=os.path.join(task_dir, './lower_time.txt')),
                ]
            else:
                instruments = []
            with PassContext(instruments=instruments):
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
    task, target_device, dumped_options = args
    option.restore_options(dumped_options)
    build_task(task, target_device, load=False)


def build_batch_task(tasks: List[Task], target_device: str = 'cuda'):
    dumped_options = option.dump_options()
    jobs = [(task, target_device, dumped_options) for task in tasks]
    if option.get_option('parallel_build') and len(jobs) > 1:
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
