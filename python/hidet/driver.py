from typing import List
import os
import multiprocessing
import logging
from hashlib import sha256
from hidet.transforms import lower, PassContext, SaveIRInstrument, ProfileInstrument
from hidet.backend import codegen, compile_source, load_task_func, load_lib_func
from hidet.utils import COLORS, hidet_cache_dir
from hidet.utils.py import cyan, green
from hidet.ir.task import Task, TaskContext
from hidet.ir.func import IRModule
from hidet.ir.type import FuncType

logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

cache_disabled = False


def disable_cache(disable: bool = False):
    global cache_disabled
    cache_disabled = not disable


def build_task(task: Task, space_level, use_cache=True, cache_dir=None, load=True):
    # resolve task dir
    if cache_dir is None:
        cache_dir = os.path.join(hidet_cache_dir(), 'ops')
    config_str = 'space_{}'.format(space_level)
    task_string = str(task)
    task_hash = sha256(task_string.encode()).hexdigest()[:16]
    task_dir = os.path.join(cache_dir, config_str, task.name, task_hash)
    src_path = os.path.join(task_dir, 'source.cu')
    lib_path = os.path.join(task_dir, 'lib.so')

    # use previously generated library when available
    if not cache_disabled and use_cache and os.path.exists(lib_path):
        logger.debug("Load cached task binary {} from path: \n{}".format(green(task.name), cyan(lib_path)))
        if not load:
            return None
        return load_task_func(lib_path, task)

    logger.info("Compiling task {}{}{}...".format(COLORS.OKGREEN, task.name, COLORS.ENDC))
    # print(task)
    # exit(0)

    # build from scratch
    os.makedirs(task_dir, exist_ok=True)
    # write task
    with open(os.path.join(task_dir, 'task.txt'), 'w') as f:
        f.write(task_string)
    # implement task
    with TaskContext(space_level=space_level, resolve_out_dir=task_dir):
        ir_module = task.implement(target='cuda')
    # lower ir module
    with PassContext(instruments=[
                         SaveIRInstrument(out_dir=os.path.join('./outs/ir', task.name, task_hash)),
                         ProfileInstrument(log_file=os.path.join('./outs/ir', task.name, task_hash, 'lower_time.txt'))
                     ]):
        ir_module = lower(ir_module)
    # code generation
    codegen(ir_module, src_out_path=src_path)
    # compile source code
    compile_source(src_path, out_lib_path=lib_path, keep_ptx=False)
    # load function
    if not load:
        return None
    return load_task_func(lib_path, task)


def _build_task_job(args):
    task, space_level, use_cache, cache_dir, load = args
    build_task(task, space_level, use_cache, cache_dir, load)


def build_batch_task(tasks: List[Task], space_level: int, parallel=True, use_cache=True, cache_dir=None):
    if parallel and len(tasks) > 1:
        with multiprocessing.Pool() as pool:
            pool.map(_build_task_job, [(task, space_level, use_cache, cache_dir, False) for task in tasks])
    else:
        map(_build_task_job, [(task, space_level, use_cache, cache_dir, False) for task in tasks])


def build_ir_module(ir_module: IRModule, func_name: str, keep_ptx=False, working_dir='./outs'):
    module_string = str(ir_module)
    module_hash = sha256(module_string.encode()).hexdigest()[:16]
    working_dir = os.path.join(working_dir, 'ir_module', module_hash)
    src_path = os.path.join(working_dir, 'source.cu')
    lib_path = os.path.join(working_dir, 'lib.so')

    # lower ir module
    with PassContext(instruments=[
                         SaveIRInstrument(out_dir=working_dir),
                         ProfileInstrument(log_file=os.path.join(working_dir, 'lower_time.txt'))
                     ]):
        ir_module = lower(ir_module)
    # code generation
    codegen(ir_module, src_out_path=src_path)
    # compile source code
    compile_source(src_path, out_lib_path=lib_path, keep_ptx=keep_ptx)
    func = ir_module.lookup(func_name + '_grid')
    return load_lib_func(lib_path, func_name, func_type=FuncType.from_func(func))


if __name__ == '__main__':
    print(sha256('abc'.encode()).hexdigest())
    print(hex(hash('abc')))
    print(type(hex(1)))
