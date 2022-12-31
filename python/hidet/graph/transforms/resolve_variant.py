# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Type, List, Optional, Dict
from hidet.graph.ir import FlowGraph, GraphRewriter, Tensor, Operator
from hidet.utils import strict_zip, same_list, repeat_until_converge
from .base import GraphPass, PassContext


class ResolveRule:
    """
    A resolve rule defines how to resolve an operator to other operators.
    """

    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        """
        When define a resolve rule, the user should subclass this class and override this method.

        Parameters
        ----------
        op: Operator
            The operator to be resolved.

        Returns
        -------
        ret: Optional[List[Tensor]]
            This function should return a list of tensors if the operator can be resolved, otherwise return None.
            In the first case, the returned tensors will be used to replace the outputs of the original operator, thus
            the number of tensors should be the same as the number of outputs of the original operator.
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
            outs = rule.resolve(op)
            if outs is not None:
                return outs
        return None


registered_resolve_rules: Dict[Type[Operator], ResolveRuleChain] = {}


def register_resolve_rule(op_cls: Type[Operator]):
    """
    Register a resolve rule for an operator class.

    Parameters
    ----------
    op_cls: Type[Operator]
        The operator class to be registered.

    Returns
    -------
    ret: Callable[[Type[ResolveRule]], Type[ResolveRule]]
        The decorator function.

    Notes
    -----

    In the following example, we define a resolve rule for operator ``PowOp`` to resolve ``pow(x, 2.0)``
    to ``square(x)``.

    .. code-block:: python

        from hidet.ir import Tensor
        from hidet import ops
        from hidet.graph.ops.definitions import PowOp
        from hidet.graph.transforms import ResolveRule, register_resolve_rule

        @register_resolve_rule(PowOp)
        class AddResolveRule(ResolveRule):
            def resolve(self, op: PowOp) -> Optional[List[Tensor]]:
                a: Tensor = op.inputs[0]
                b: Tensor = op.inputs[1]
                if not b.is_symbolic() and len(b.shape) == 0 and b.scalar() == 2:
                    return [ops.square(a)]
                return None

    """
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
            if not isinstance(outs, (list, tuple)):
                raise ValueError(
                    "The resolve rule of operator '{}' should return a list of tensors, but got {}".format(
                        op.name, type(outs)
                    )
                )
            if len(outs) != len(op.outputs):
                raise ValueError(
                    "The resolve rule of operator '{}' should return {} tensors, but got {} ones".format(
                        op.name, len(op.outputs), len(outs)
                    )
                )
            for i, (original, updated) in enumerate(strict_zip(op.outputs, outs)):
                assert original not in self.memo
                if (original.dtype, tuple(original.shape)) != (updated.dtype, tuple(updated.shape)):
                    raise ValueError(
                        (
                            "The resolve rule of operator '{}' should return tensors with the same dtype and "
                            "shape as the original ones. The {}-th tensor expect {}{} but got {}{}"
                        ).format(op.name, i, original.dtype, list(original.shape), updated.dtype, list(updated.shape))
                    )
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
