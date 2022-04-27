import os
import logging
from hashlib import sha256
from hidet.transforms import lower, pass_context
from hidet.backend import codegen, compile_source, load_task_func
from hidet.utils import COLORS, hidet_cache_dir
from hidet.ir.task import Task

logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def build_task(task: Task, space_level, opt_level, use_cache=True, cache_dir=None):
    # resolve task dir
    if cache_dir is None:
        cache_dir = os.path.join(hidet_cache_dir(), 'nops')
    config_str = 'space_{}_opt_{}'.format(space_level, opt_level)
    task_string = str(task)
    task_hash = sha256(task_string.encode()).hexdigest()[:16]
    task_dir = os.path.join(cache_dir, config_str, task.name, task_hash)
    src_path = os.path.join(task_dir, 'source.cu')
    lib_path = os.path.join(task_dir, 'lib.so')

    # use previously generated library when available
    if use_cache and os.path.exists(lib_path):
        return load_task_func(lib_path, task)

    logger.info("Compiling task {}{}{}...".format(COLORS.OKGREEN, task.name, COLORS.ENDC))

    # build from scratch
    os.makedirs(task_dir, exist_ok=True)
    # with impl_context(allowed=ImplementerContext.current().allowed, space_level=space_level):
    # write task
    with open(os.path.join(task_dir, 'task.txt'), 'w') as f:
        f.write(task_string)
    # implement task
    ir_module = task.implement(target='cuda', space_level=space_level)
    # lower ir module
    # todo: turn off keep_ir after debug
    with pass_context(opt_level=opt_level, keep_ir=True, keep_ir_dir='./outs/ir'):
        ir_module = lower(ir_module)
    # code generation
    codegen(ir_module, src_out_path=src_path)
    # compile source code
    compile_source(src_path, out_lib_path=lib_path, keep_ptx=False)
    # load function
    return load_task_func(lib_path, task)


if __name__ == '__main__':
    print(sha256('abc'.encode()).hexdigest())
    print(hex(hash('abc')))
    print(type(hex(1)))
