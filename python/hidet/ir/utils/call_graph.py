from typing import Union, List
from collections import defaultdict
from hidet.ir.expr import Call
from hidet.ir.func import IRModule, Function
from hidet.ir.functors import collect
from hidet.ir.primitives import is_primitive_function


class CallGraphNode:
    def __init__(self, func):
        self.func = func
        self.callers = []
        self.callees = []

    def add_caller(self, caller):
        if caller not in self.callers:
            self.callers.append(caller)

    def add_callee(self, callee):
        if callee not in self.callees:
            self.callees.append(callee)


class CallGraph:
    def __init__(self, ir_module: IRModule):
        self.nodes: List[CallGraphNode] = []
        self.func2node = {}

        self.order: List[CallGraphNode] = []  # topological order, from caller to callee
        self.reversed_order: List[CallGraphNode] = []

        for func in ir_module.functions.values():
            node = CallGraphNode(func)
            self.func2node[func] = node
            self._add_node(node)

        for func in ir_module.functions.values():
            caller = func
            for call in collect(func.body, Call):
                if is_primitive_function(call.func_var.hint):
                    continue
                callee = ir_module.lookup(call.func_var.hint)
                self._add_edge(caller, callee)

        self._init_order()

    def _add_node(self, node):
        if node not in self.nodes:
            self.nodes.append(node)

    def _add_edge(self, caller: Union[Function, CallGraphNode], callee: Union[Function, CallGraphNode]):
        if isinstance(caller, Function):
            caller = self.func2node[caller]
        if isinstance(callee, Function):
            callee = self.func2node[callee]
        caller.add_callee(callee)
        callee.add_caller(caller)

    def _init_order(self):
        in_degree = defaultdict(int)
        # do not support recursive calling now
        for node in self.nodes:
            for callee in node.callees:
                in_degree[callee] += 1
        qu = []
        for node in self.nodes:
            if in_degree[node] == 0:
                qu.append(node)
        self.order = []
        while len(qu) > 0:
            u = qu.pop()
            self.order.append(u)
            for callee in u.callees:
                in_degree[callee] -= 1
                if in_degree[callee] == 0:
                    qu.append(callee)

        self.reversed_order = list(reversed(self.order))

