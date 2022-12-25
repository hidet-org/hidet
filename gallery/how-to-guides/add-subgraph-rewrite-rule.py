"""
Add Sub-Graph Rewrite Rule
==========================

This tutorial shows how to add a sub-graph rewrite rule in the graph optimization pipeline. Sub-graph rewriting is an
important technique in graph optimization. It is used to replace a sub-graph with another sub-graph, which is usually
more efficient than the original one. For example, we can replace a sub-graph with two matrix multiplications sharing
the same input and one addition with a concatenation and a single matrix multiplication:

.. figure:: /_static/img/subgraph-rewrite-example.svg
    :align: center
    :scale: 70%

    The sub-graph rewrite rule that fuses two matrix multiplications.

.. seealso::
    :class: margin

    TASO :cite:`taso` systematically studies the sub-graph rewrite optimization for deep learning workloads.

After the rewrite, the graph becomes more efficient as we only need to run a single kernel and the `fused` matrix
multiplication usually exposes more parallelism to utilize the underlying hardware. We can also fuse multiple
convolutions into a single one, or do other sub-graph rewrites.

Sub-graph rewrite in Hidet
--------------------------

In Hidet, we use a *sub-graph rewrite rule* to describe the rewrite. A sub-graph rewrite rule contains two parts:

- **Sub-graph pattern**: a sub-graph pattern that we use to match the sub-graph in the graph. The pattern is a directed
  acyclic graph (DAG) where each node is an operator pattern and each edge is a tensor pattern. We only specify the
  operator type for each node and whether the (input) tensors are symbolic or concrete.

- **Target sub-graph constructor**: when we find a sub-graph that matches the pattern, we use the constructor to
  construct the target sub-graph that replaces the matched sub-graph. When constructing the target sub-graph, we can
  use the matched tensors and nodes to further determine whether the rewrite is applicable. If applicable, the
  constructor returns the target sub-graph, otherwise it returns ``None``.

In the above example, the sub-graph pattern contains three input tensors, where x1 is a symbolic tensor and w1, w2 are
two concrete tensors (i.e., we know the concrete values of w1 and w2). There are three operators in the pattern, where
the first two are matrix multiplications and the last one is an addition. The output tensor of the addition is the
output tensor of the pattern. When we find a sub-graph that matches the pattern, we use the constructor to construct
the target sub-graph and replace the matched sub-graph with the target sub-graph.

.. note::

  **Difference between sub-graph rewrite and operator resolving**. Although
  :ref:`operator resolving <add-operator-resolve-rule>` can be conceptually considered as a special case of
  sub-graph rewrite, we use a different mechanism to implement them and their execution order is different. The operator
  resolving can be performed efficiently thus we can add arbitrary number of operator resolve rules. But the sub-graph
  rewrite is usually more expensive. Second, we run the sub-graph rewrite pass before the operator resolving pass, so
  that we can use the generic operators in the sub-graph patterns without worrying about the operator resolving.


Add a sub-graph rewrite rule
----------------------------

Let's implement the sub-graph rewrite rule shown in the above example. Before we start, we first create a new model
that contains the sub-graph we want to rewrite:

"""
from typing import Optional, List

import hidet
from hidet import Tensor, FlowGraph, Operator
from hidet import ops
from hidet.graph.transforms.graph_patterns import MatchDict


def example_model(x: Tensor, w0: Tensor, w1: Tensor, w2: Tensor):
    x = ops.matmul(x, w0)
    y1 = ops.matmul(x, w1)
    y2 = ops.matmul(x, w2)
    y = ops.relu(ops.concat([y1, y2], axis=1))
    return y


x = hidet.symbol([3, 3])
w0, w1, w2 = hidet.randn([3, 3]), hidet.randn([3, 3]), hidet.randn([3, 3])
y = example_model(x, w0, w1, w2)
graph: FlowGraph = hidet.trace_from(y, inputs=[x])
print(graph)

# %%
# Then, we define and register the sub-graph rewrite rule.
#
from hidet.graph.ops.definitions import MatmulOp, ConcatOp
from hidet.graph.transforms import TensorPattern, SubgraphRewriteRule
from hidet.graph.transforms import op_pattern, register_rewrite_rule
from hidet.utils import same_list


# register the rewrite rule, only registered rewrite rules will be applied
@register_rewrite_rule
class FuseTwoMatmulRewriteRule(SubgraphRewriteRule):
    def __init__(self):
        super().__init__(name="new: [matmul(x, c1), matmul(x,c2)] => matmul(x, [c1, c2])")
        self.x = TensorPattern()  # x can match either a symbolic or concrete tensor
        self.c1 = TensorPattern(is_const=True)  # c1 can only match a concrete tensor
        self.c2 = TensorPattern(is_const=True)  # c2 can only match a concrete tensor
        self.y1 = op_pattern(MatmulOp, [self.x, self.c1])  # pattern: y1 = matmul(x, c1)
        self.y2 = op_pattern(MatmulOp, [self.x, self.c2])  # pattern: y2 = matmul(x, c2)
        self.y = op_pattern(ConcatOp, [self.y1, self.y2])  # pattern: y = concat(y1, y2)

    def source(self) -> List[TensorPattern]:
        # Return the output tensors of the source sub-graph pattern.
        return [self.y]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        # The target sub-graph constructor
        # The matched dictionary has type Dict[TensorPattern, Tensor]
        # that maps the patterns to the matched tensors.
        x, c1, c2, y = [matched[t] for t in [self.x, self.c1, self.c2, self.y]]
        concat: Operator = y.op

        # We can use the matched tensors to determine whether the rewrite is applicable.
        # For example, in this case, we check whether the two weight matrices share the
        # same shape except the last dimension.
        if (
            2 <= len(c1.shape) == len(c2.shape)
            and same_list(c1.shape[:-1], c2.shape[:-1])
            and concat.attrs["axis"] == len(y.shape) - 1
        ):
            # If applicable, we construct the target sub-graph and return the output tensors.
            c = ops.concat([c1, c2], axis=-1)
            y = ops.matmul(x, c)
            return [y]
        else:
            # If not, we return None to indicate that the rewrite is not applicable.
            return None


# %%
# We can check that the rewrite rule has been registered:
from hidet.graph.transforms import registered_rewrite_rules

print('Registered rewrite rules:')
for rule in registered_rewrite_rules:
    assert isinstance(rule, SubgraphRewriteRule)
    print(rule.name)

# %%
# Apply the rewrite rule
# ----------------------
# Besides the predefined rewrite rules, we can see that the rewrite rule we just registered is also included at the
# last line. In this tutorial, to prevent the default rewrite rules from being applied, we first clear the registered
# rewrite rules and then register the rewrite rule we just defined:
registered_rewrite_rules.clear()
register_rewrite_rule(
    FuseTwoMatmulRewriteRule()
)  # a second way to register the rewrite rule

# %%
# The rewrite process is done in a graph optimization pass called `subgraph_rewrite_pass`.
from hidet.graph.transforms import subgraph_rewrite_pass

rewrite_pass = subgraph_rewrite_pass()
rewritten_graph: FlowGraph = rewrite_pass(graph)
print(rewritten_graph)

# %%
# We can see that the rewritten graph contains 2 matmul operators instead of 3. There is no concat operator in the
# rewritten graph, because the inputs of concat operator are constant tensors and thus have been folded.
#
# We do not need to call the rewrite pass explicitly. It will be called automatically when we call
# :func:`hidet.graph.optimize`, together with other graph optimization passes.
graph_opt: FlowGraph = hidet.graph.optimize(graph)
print(graph_opt)

# %%
# Summary
# -------
# In this tutorial, we have learned how to define and register a sub-graph rewrite rule. It is an important
# component of the graph optimization framework. Hidet uses it to implement some horizontal fusion rules.

# %%
# References
# ----------
# .. bibliography::
