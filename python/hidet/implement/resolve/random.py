import random
from hidet.ir.func import IRModule, FunctionGroup


def random_resolve(ir_module: IRModule, seed=None) -> IRModule:
    if seed:
        random.seed(seed)
    for name, func in ir_module.functions.items():
        if isinstance(func, FunctionGroup):
            ir_module.functions[name] = random.choice(func.group)
    return ir_module
