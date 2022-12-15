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

    TASO :cite:`taso` for more details about optimizations about graph-rewrites.

After the rewrite, the graph becomes more efficient as we only need to run a single kernel and the **fused** matrix
multiplication usually exposes more parallelism to utilize the underlying hardware. We can also fuse multiple
convolutions into a single one.

Sub-graph rewrite in Hidet
--------------------------

In Hidet, we use a *sub-graph rewrite rule* to describe the rewrite. A sub-graph rewrite rule contains two parts:

- **Sub-graph pattern**: a sub-graph pattern to match.
- **Target sub-graph constructor**: a sub-graph to replace the matched sub-graph.




.. bibliography::

"""