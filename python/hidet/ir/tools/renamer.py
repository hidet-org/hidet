from typing import Dict, List
from hidet.ir.expr import Var
from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.tools.rewriter import rewrite
from hidet.ir.tools.util_functors import collect


def rename_funcs(ir_module: IRModule, rmap: Dict[str, str]) -> IRModule:
    """
    Rename functions in an IRModule.

    Parameters
    ----------
    ir_module: IRModule
        The IRModule.
    rmap: Dict[str, str]
        The renaming map.

    Returns
    -------
    ret: IRModule
        The renamed IRModule.
    """
    used_vars: List[Var] = collect(ir_module, node_types=Var)
    func_vars: List[Var] = [v for v in used_vars if v.type.is_func_type()]

    # rename the variables
    name2var: Dict[str, Var] = {}
    remap = {}
    for func_var in func_vars:
        if func_var.name in rmap:
            if func_var.name not in name2var:
                name2var[func_var.name] = Var(hint=None, type=func_var.type, name=rmap[func_var.name])
            remap[func_var] = name2var[func_var.name]

    ir_module: IRModule = rewrite(ir_module, remap)

    # rename functions
    new_functions: Dict[str, Function] = {}
    for name, func in ir_module.functions.items():
        if name in rmap:
            new_functions[rmap[name]] = Function(
                name=rmap[name],
                params=func.params,
                body=func.body,
                ret_type=func.ret_type,
                kind=func.kind,
                attrs=func.attrs,
            )
        else:
            new_functions[name] = func

    # rename global vars
    global_vars: Dict[str, Var] = {}
    for name, var in ir_module.global_vars.items():
        if name in rmap:
            assert var.name == rmap[name]
            global_vars[rmap[name]] = var
        else:
            global_vars[name] = var

    ir_module.reset_funcs(functions=new_functions, global_vars=global_vars)
    return ir_module
