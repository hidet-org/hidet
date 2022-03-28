import os
from hashlib import sha256
from hidet.implement import impl_context, implement
from hidet.transforms import lower, pass_context
from hidet.backend import codegen, compile_source, load_task_func
from hidet.utils.git_utils import repo_root
from hidet.runtime import CompiledFunction


def build_task(task, space_level, opt_level, use_cache=True, cache_dir=None) -> CompiledFunction:
    # resolve task dir
    if cache_dir is None:
        cache_dir = repo_root()
    config_str = 'space_{}_opt_{}'.format(space_level, opt_level)
    task_string = str(task)
    # task_hash = format(abs(hash(task_string)), 'x')
    task_hash = sha256(task_string.encode()).hexdigest()[:16]
    task_dir = os.path.join(cache_dir, '.opcache', config_str, task.name, task_hash)
    src_path = os.path.join(task_dir, 'source.cu')
    lib_path = os.path.join(task_dir, 'lib.so')

    # use previously generated library when available
    if use_cache and os.path.exists(lib_path):
        return load_task_func(lib_path, task)

    # build from scratch
    os.makedirs(task_dir, exist_ok=True)
    with impl_context(space_level=space_level):
        # write task
        with open(os.path.join(task_dir, 'task.txt'), 'w') as f:
            f.write(task_string)
        # implement task
        ir_module = implement(task)
        # lower ir module
        with pass_context(opt_level=opt_level):
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
