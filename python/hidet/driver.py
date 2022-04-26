import os
import logging
import hidet
from hashlib import sha256
from hidet.implement import impl_context, implement, ImplementerContext
from hidet.transforms import lower, pass_context
from hidet.backend import codegen, compile_source, load_task_func, load_ntask_func
from hidet.utils import COLORS, hidet_cache_dir
from hidet.runtime import CompiledFunction

logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def build_task(task, space_level, opt_level, use_cache=True, cache_dir=None) -> CompiledFunction:
    from hidet.tos.task import Task as NTask
    if isinstance(task, NTask):
        return build_ntask(task, space_level, opt_level, use_cache, cache_dir)
    # resolve task dir
    if cache_dir is None:
        cache_dir = os.path.join(hidet_cache_dir(), 'ops')
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
    # if task.name == 'matmul':
    #     allowed = [hidet.implement.cuda.CudaGridStaticMatmulImplementer]
    # else:
    #     allowed = None  # allow any matched implementer
    with impl_context(allowed=ImplementerContext.current().allowed, space_level=space_level):
        # write task
        with open(os.path.join(task_dir, 'task.txt'), 'w') as f:
            f.write(task_string)
        # implement task
        ir_module = implement(task)
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


def build_ntask(task: 'hidet.tos.task.Task', space_level, opt_level, use_cache=True, cache_dir=None):
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
        return load_ntask_func(lib_path, task)

    logger.info("Compiling task {}{}{}...".format(COLORS.OKGREEN, task.name, COLORS.ENDC))

    # build from scratch
    os.makedirs(task_dir, exist_ok=True)
    with impl_context(allowed=ImplementerContext.current().allowed, space_level=space_level):
        # write task
        with open(os.path.join(task_dir, 'task.txt'), 'w') as f:
            f.write(task_string)
        # implement task
        ir_module = task.implement(target='cuda')
        # ir_module = implement(task)
        # lower ir module
        # todo: turn off keep_ir after debug
        with pass_context(opt_level=opt_level, keep_ir=True, keep_ir_dir='./outs/ir'):
            ir_module = lower(ir_module)
        # code generation
        codegen(ir_module, src_out_path=src_path)
        # compile source code
        compile_source(src_path, out_lib_path=lib_path, keep_ptx=False)
        # load function
        return load_ntask_func(lib_path, task)


if __name__ == '__main__':
    print(sha256('abc'.encode()).hexdigest())
    print(hex(hash('abc')))
    print(type(hex(1)))
