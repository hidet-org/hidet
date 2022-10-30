"""
Define Computation Task
=======================

.. _define-computation-task:

Each operator takes a list of input tensors and produces a list of output tensors:

.. code-block:: python

    inputs: List[Tensor]
    outputs: List[Tensor] = operator(inputs)

.. note::
  :class: margin

  Our pioneers `Halide <https://halide-lang.org/>`_ and `Apache TVM <https://tvm.apache.org/>`_ also employ a similar
  DSL to define the mathematical definition of an operator.

The precise mathematical definition of each operator in Hidet is defined through a domain-specific-language (DSL).
In this article, we will show how to define the mathematical definition of a new operator in Hidet using this DSL,
which is defined in the :py:mod:`hidet.ir.compute` module.


Compute Primitives
------------------
This module provides compute primitives to define the mathematical computation of an operator:

.. py:function:: tensor_input(name: str, dtype: str, shape: List[int])
    :noindex:

    The :py:func:`~hidet.ir.compute.tensor_input` primitive defines a tensor input by specifying the name hint, scalar
    data type, and shape of the tensor.

    .. code-block:: python
      :caption: Examples

      a = tensor_input('a', dtype='float32', shape=[10, 10])
      b = tensor_input('b', dtype='float32', shape=[])
      b = tensor_input('data', dtype='float16', shape=[1, 3, 224, 224])


.. py:function:: compute(name: str, shape: List[int], fcompute: Callable[[Var,...], Expr])
    :noindex:

    The :py:func:`~hidet.ir.compute.compute` primitive defines a tensor by specifying

    - the name of the tensor, just a hint for what the tensor represents,
    - the shape of the tensor, and
    - a function that maps an index to the expression that computes the value of the tensor at that index.

    The computation of each element of the tensor is *independent* with each other and can be computed in parallel.

    .. code-block:: python
        :caption: Semantics

        # compute primitive
        out = compute(
            name='hint_name',
            shape=[n1, n2, ..., nk],
            fcompute=lambda i1, i2, ..., ik: f(i1, i2, ..., ik)
        )

        # semantics
        for i1 in range(n1):
          for i2 in range(n2):
            ...
              for ik in range(nk):
                out[i1, i2, ..., ik] = f(i1, i2, ..., ik)

    .. note::
        :class: margin

        In the last example, we used an :py:func:`~hidet.ir.expr.if_then_else` expression to define a conditional
        expression.

    .. code-block:: python
      :caption: Examples

      # define an input tensor
      a = tensor_input('a', dtype='float32', shape=[10, 10])

      # example 1: slice the first column of a
      b = compute('slice', shape=[10], fcompute=lambda i: a[i, 0])

      # example 2: reverse the rows of matrix a
      c = compute('reverse', shape=[10, 10], fcompute=lambda i, j: a[9 - i, j])

      # example 3: add 1 to the diagonal elements of a
      from hidet.ir.expr import if_then_else
      d = compute(
        name='diag_add',
        shape=[10, 10],
        fcompute=lambda i, j: if_then_else(i == j, then_expr=a[i, j] + 1.0, else_expr=a[i, j])
      )


.. py:function:: reduce(shape: List[int], fcompute: Callable[[Var, ...], Expr], reduce_type='sum')
    :noindex:

    The :py:func:`~hidet.ir.compute.reduce` primitive conducts a reduction operation on a domain with the given shape.
    It returns a scalar value and can be used in :py:func:`~hidet.ir.compute.compute` primitive.

    .. code-block:: python
        :caption: Semantics

        # reduce primitive
        out = reduce(
            name='hint_name',
            shape=[n1, n2, ..., nk],
            fcompute=lambda i1, i2, ..., ik: f(i1, i2, ..., ik)
            reduce_type='sum' | 'max' | 'min' | 'avg'
        )

        # semantics
        values = []
        for i1 in range(n1):
          for i2 in range(n2):
            ...
              for ik in range(nk):
                values.append(f(i1, i2, ..., ik))
        out = reduce_type(values)

    .. code-block:: python
      :caption: Examples

      # define an input tensor
      a = tensor_input('a', dtype='float32', shape=[10, 10])

      # example 1: sum all elements of a
      c = reduce(shape=[10, 10], fcompute=lambda i, j: a[i, j], reduce_type='sum')

      # example 2: sum the first column of a
      d = reduce(shape=[10], fcompute=lambda i: a[i, 0], reduce_type='sum')

      # example 3: matrix multiplication
      b = tensor_input('b', dtype='float32', shape=[10, 10])
      e = compute(
          name='e',
          shape=[10, 10],
          fcompute=lambda i, j: reduce(
              shape=[10],
              fcompute=lambda k: a[i, k] * b[k, j],
              reduce_type='sum'
          )
      )



.. py:function:: arg_reduce(extent: int, fcompute: Callable[[Var], Expr], reduce_type='max')
    :noindex:

    Similar to :py:func:`~hidet.ir.compute.reduce`, the :py:func:`~hidet.ir.compute.arg_reduce` primitive conducts a
    reduction operation on a domain with the given extent. The difference is that it returns the index of the element
    that corresponds to the reduction result, instead of the result itself.

    .. code-block:: python
        :caption: Semantics

        # arg_reduce primitive
        out = arg_reduce(extent, fcompute=lambda i: f(i), reduce_type='max' | 'min')

        # semantics
        values = []
        for i in range(extent):
          values.append(f(i))
        out = index of the max/min value in values

    .. code-block:: python
        :caption: Examples

        # define an input tensor
        a = tensor_input('a', dtype='float32', shape=[10, 10])

        # example: find the index of the max element in each row of a
        b = compute('b', [10], lambda i: arg_reduce(10, lambda j: a[i, j], reduce_type='max'))


Computation Task
----------------
The computation of each operator can be described as a directed acyclic graph (DAG). The DAG is composed of tensor
nodes. Both :py:func:`~hidet.ir.compute.tensor_input` and :py:func:`~hidet.ir.compute.compute` primitives create tensor
nodes. The edges of the DAG are the dependencies between the tensor nodes.

.. graphviz::
    :caption: An example of computation DAG. In this example, there are 5 tensor nodes, where node A and B are inputs
              and node D and E are outputs. The computation of node C depends on the computation of node A and B.

    digraph {
        // rankdir=LR;
        splines=curved;
        node [
            shape=box, style="rounded",
            height=0.4, width=0.6
        ];
        graph [style="rounded, dashed"]
        subgraph cluster_0 {
            graph [style="rounded, dashed", margin="12"];
            node [group=0];
            label="Inputs";
            a [label="A"];
            b [label="B"];
        }
        subgraph cluster_1 {
            graph [style="rounded, dashed", labelloc="b", margin="15"];
            node [group=1];
            labeljust="b";
            d [label="D"];
            e [label="E"];
            label="Outputs";
        }
        c [label="C"];
        a -> c -> d
        a -> e
        b -> e
    }

Such a DAG is defined by a :py:class:`~hidet.ir.task.Task` object.

.. py:class:: Task(name: str, inputs: List[TensorNode], outputs: List[TensorNode])
    :noindex:


Examples
--------

The computation of each operator in Hidet is defined
"""
from typing import List
import hidet
from hidet.ir.compute import tensor_input, reduce, compute, arg_reduce, TensorNode
from hidet.ir.task import Task


def print_nodes(nodes: List[TensorNode]):
    """Print computation nodes in a human-readable format."""
    from hidet.ir.functors.printer import IRPrinter
    printer = IRPrinter()
    text = str(printer.print_tensor_nodes(nodes))
    print(text.strip())
    print()


def compile_compute(inputs: List[TensorNode], outputs: List[TensorNode], task_name: str):
    """Compile computation nodes into a callable function."""
    task = Task(name=task_name, inputs=inputs, outputs=outputs)
    func: hidet.runtime.CompiledFunction = hidet.driver.build_task(task, target_device='cuda')
    return func


def add_example():
    a: TensorNode = tensor_input(name='a', dtype='float32', shape=[10])
    b: TensorNode = tensor_input(name='b', dtype='float32', shape=[10])
    c: TensorNode = compute(name='c', shape=[10], fcompute=lambda i: a[i] + b[i])
    print_nodes([c])


add_example()


def reduce_sum_example():
    a = tensor_input('a', dtype='float32', shape=[10, 20])
    b = compute(
        'b',
        shape=[10],
        fcompute=lambda i: reduce(shape=[20], fcompute=lambda j: a[i, j], reduce_type='sum')
    )
    print([b])


reduce_sum_example()


def arg_max_example():
    a = tensor_input('a', dtype='float32', shape=[5, 3])
    b = compute(
        'b',
        shape=[5],
        fcompute=lambda i: arg_reduce(extent=3, fcompute=lambda j: a[i, j], reduce_type='max')
    )
    print_nodes([b])


arg_max_example()


def matmul_example():
    a = tensor_input('a', dtype='float32', shape=[10, 10])
    b = tensor_input('b', dtype='float32', shape=[10, 10])
    c = compute(
        'c',
        shape=[10, 10],
        fcompute=lambda i, j: reduce(
            shape=[10],
            fcompute=lambda k: a[i, k] * b[k, j],
            reduce_type='sum'
        )
    )
    print_nodes([c])


matmul_example()


def softmax_example():
    from hidet.ir.primitives import exp
    a = tensor_input('a', dtype='float32', shape=[10])
    max_val = reduce(shape=[10], fcompute=lambda i: a[i], reduce_type='max')
    b = compute('b', shape=[10], fcompute=lambda i: a[i] - max_val)
    exp_a = compute('exp', shape=[10], fcompute=lambda i: exp(b[i]))
    exp_sum = reduce(shape=[10], fcompute=lambda i: exp_a[i], reduce_type='sum')
    softmax = compute('softmax', shape=[10], fcompute=lambda i: exp_a[i] / exp_sum)
    print_nodes([softmax])


softmax_example()
