from typing import Type, List
from .base import GraphPass
from .resolve_variant_rules import ResolveRule, Conv2dResolveRule, MatmulResolveRule
from hidet.graph.ir import FlowGraph, GraphRewriter, Tensor, Operator
from hidet.utils import strict_zip, same_list


class ResolveVariantRewriter(GraphRewriter):
    def __init__(self, rule: ResolveRule):
        super().__init__()
        self.rule = rule

    def visit_Operator(self, op: Operator):
        op_cls = self.rule.op_cls()
        if not isinstance(op, op_cls):
            return GraphRewriter.visit_Operator(self, op)
        inputs = [self(x) for x in op.inputs]
        if same_list(inputs, op.inputs):
            resolve_op = op
        else:
            updated_outputs = op.reforward(inputs)
            resolve_op = updated_outputs[0].op
        outs = self.rule.resolve(resolve_op)

        if outs is None:
            # keep the original operator
            # we still need to update memo in case inputs changed
            for original, updated in strict_zip(op.outputs, resolve_op.outputs):
                assert original not in self.memo
                self.memo[original] = updated
        else:
            # update output of resolved operator
            for original, updated in strict_zip(op.outputs, outs):
                assert original not in self.memo
                self.memo[original] = updated


class ResolveVariantPass(GraphPass):
    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        rule_seq: List[ResolveRule] = [
            Conv2dResolveRule(),
            MatmulResolveRule()
        ]
        for rule in rule_seq:
            resolver = ResolveVariantRewriter(rule)
            while True:
                updated_graph = resolver(graph)
                if updated_graph is graph:
                    break
                else:
                    graph = updated_graph
        return graph


def resolve_variant_pass() -> GraphPass:
    return ResolveVariantPass()
