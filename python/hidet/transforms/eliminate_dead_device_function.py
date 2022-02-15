from hidet.ir.functors import simplify
from hidet.ir.expr import Expr
from hidet.ir.func import Function, IRModule
from hidet.ir.task import ThreadBlock, Warp, Thread
from hidet.ir.utils.call_graph import CallGraph
from hidet.transforms.base import Pass


class EliminateDeadDeviceFunction(Pass):
    def __call__(self, ir_module: IRModule) -> IRModule:
        while True:
            funcs = ir_module.functions
            call_graph = CallGraph(ir_module)

            updated = False
            for name in list(funcs.keys()):
                func = funcs[name]
                if isinstance(func.get_attr('worker'), (ThreadBlock, Warp, Thread)):
                    if len(call_graph.func2node[func].callers) == 0:
                        # unused device function
                        updated = True
                        del funcs[name]
            if not updated:
                return ir_module


def eliminate_dead_device_function_pass():
    return EliminateDeadDeviceFunction()
