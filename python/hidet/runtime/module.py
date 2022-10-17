from typing import Dict, Tuple, Optional
from collections import namedtuple
from hidet.ir.task import Task
from hidet.ir.func import Function, IRModule


class CompiledModule:
    def __init__(self, ir_module, funcs):
        self.ir_module: IRModule = ir_module
        self.funcs: Dict[str, CompiledFunction] = funcs

    def __getitem__(self, item: str):
        return self.funcs[item]


class CompiledFunction:
    def __init__(self, name, packed_func):
        from hidet.ffi import PackedFunc
        self.name: str = name
        self.packed_func: PackedFunc = packed_func

    def __call__(self, *args):
        self.packed_func(*args)

    def profile(self, *args, warmup=1, number=1, repeat=10):
        return self.packed_func.profile(*args, warmup=warmup, number=number, repeat=repeat)


CompiledTaskKey = namedtuple('CompiledTaskKey', ['device', 'space', 'task_str'])


class CompiledTaskCache:
    def __init__(self):
        self.cached: Dict[Tuple[str, int, str], CompiledFunction] = {}

    def contains(self, device: str, space: int, task_str: str) -> bool:
        key = CompiledTaskKey(device, space, task_str)
        return key in self.cached

    def get(self, device: str, space: int, task_str: str) -> Optional[CompiledFunction]:
        key = CompiledTaskKey(device, space, task_str)
        return self.cached.get(key) if key in self.cached else None

    def add(self, device: str, space: int, task_str: str, func: CompiledFunction):
        key = CompiledTaskKey(device, space, task_str)
        self.cached[key] = func


compiled_task_cache = CompiledTaskCache()

