from typing import Type, List, Optional, Dict
from hidet.graph.ir import FlowGraph, GraphRewriter, Tensor, Operator
from hidet.utils import strict_zip, same_list, repeat_until_converge
from .base import GraphPass, PassContext


class ResolveRule:
    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        """
        Parameters
        ----------
        op: Operator
            The operator to be resolved.

        Returns
        -------
        ret:
            None - indicates the operator has not been resolved, keep the original operator.
            List[Tensor] - the output of resolved operators.
        """
        raise NotImplementedError()

    def get_config(self, name, default=None):
        return PassContext.current().configs.get(name, default)


class ResolveRuleChain:
    def __init__(self, op_cls: Type[Operator], rules: List[ResolveRule]):
        self.op_cls: Type[Operator] = op_cls
        self.rules: List[ResolveRule] = list(rules)

    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        # apply rules in reverse order, so that the latest rule has the highest priority
        for rule in reversed(self.rules):
            if self.op_cls == type(op):
                return rule.resolve(op)
        return None


registered_resolve_rules: Dict[Type[Operator], ResolveRuleChain] = {}


def register_resolve_rule(op_cls: Type[Operator]):
    if not issubclass(op_cls, Operator):
        raise ValueError("Expect a subclass of Operator, got {}".format(type(op_cls)))

    def wrapper(rule_cls: Type[ResolveRule]):
        if not issubclass(rule_cls, ResolveRule):
            raise ValueError("Expect a subclass of ResolveRule, got {}".format(type(rule_cls)))

        if op_cls not in registered_resolve_rules:
            registered_resolve_rules[op_cls] = ResolveRuleChain(op_cls, [])
        chain = registered_resolve_rules[op_cls]
        chain.rules.append(rule_cls())
        return rule_cls

    return wrapper


def get_resolve_chain(op_cls: Type[Operator]) -> Optional[ResolveRuleChain]:
    if op_cls not in registered_resolve_rules:
        return None
    return registered_resolve_rules[op_cls]


class ResolveVariantRewriter(GraphRewriter):
    def __init__(self, op_cls: Type[Operator], rule_chain: ResolveRuleChain):
        super().__init__()
        self.op_cls: Type[Operator] = op_cls
        self.rule_chain: ResolveRuleChain = rule_chain

    def visit_Operator(self, op: Operator):
        if not isinstance(op, self.op_cls):
            GraphRewriter.visit_Operator(self, op)
            return
        inputs = [self(x) for x in op.inputs]
        if same_list(inputs, op.inputs):
            resolve_op = op
        else:
            updated_outputs = op.reforward(inputs)
            resolve_op = updated_outputs[0].op
        outs = self.rule_chain.resolve(resolve_op)

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
    def process_graph(self, input_graph: FlowGraph) -> FlowGraph:
        def apply_rules(graph: FlowGraph) -> FlowGraph:
            for op_cls, rule_chain in registered_resolve_rules.items():
                rewriter = ResolveVariantRewriter(op_cls, rule_chain)
                graph = rewriter(graph)
            return graph

        return repeat_until_converge(apply_rules, input_graph, limit=None)


def resolve_variant_pass() -> GraphPass:
    return ResolveVariantPass()
