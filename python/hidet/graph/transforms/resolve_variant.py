from typing import Type, List, Optional, Union
from hidet.graph.ir import FlowGraph, GraphRewriter, Tensor, Operator
from hidet.utils import strict_zip, same_list, repeat_until_converge
from .base import GraphPass, PassContext


class ResolveRule:
    def op_cls(self) -> Type[Operator]:
        raise NotImplementedError()

    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        """
        Parameters
        ----------
        op: Operator
            The operator to be resolved.

        Returns
        -------
        ret: Optional[List[Tensor]]
            None - indicates the operator has not been resolved, keep the original operator.
            List[Tensor] - the output of resolved operators.
        """
        raise NotImplementedError()

    def get_config(self, name, default=None):
        return PassContext.current().configs.get(name, default)


registered_resolve_rules: List[ResolveRule] = []


def register_resolve_rule(rule_or_cls: Union[Type[ResolveRule], ResolveRule]):
    if isinstance(rule_or_cls, ResolveRule):
        rule = rule_or_cls
    elif issubclass(rule_or_cls, ResolveRule):
        rule = rule_or_cls()
    else:
        raise ValueError("Expect a ResolveRule instance or subclass, got {}".format(type(rule_or_cls)))
    registered_resolve_rules.append(rule)
    return rule_or_cls


def get_registered_resolve_rules() -> List[ResolveRule]:
    return registered_resolve_rules


class ResolveVariantRewriter(GraphRewriter):
    def __init__(self, rule: ResolveRule):
        super().__init__()
        self.rule = rule

    def visit_Operator(self, op: Operator):
        op_cls = self.rule.op_cls()
        if not isinstance(op, op_cls):
            GraphRewriter.visit_Operator(self, op)
            return
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
        def apply_rules(graph: FlowGraph) -> FlowGraph:
            for rule in get_registered_resolve_rules():
                rewriter = ResolveVariantRewriter(rule)
                graph = rewriter(graph)
            return graph

        return repeat_until_converge(apply_rules, graph, limit=None)
        # for rule in rule_seq:
        #     resolver = ResolveVariantRewriter(rule)
        #     while True:
        #         updated_graph = resolver(graph)
        #         if updated_graph is graph:
        #             break
        #         graph = updated_graph
        # return graph


def resolve_variant_pass() -> GraphPass:
    return ResolveVariantPass()
