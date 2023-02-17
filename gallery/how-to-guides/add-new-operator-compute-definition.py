"""
Define Operator Computation
===========================

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
In this tutorial, we will show how to define the mathematical definition of a new operator in Hidet using this DSL,
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


Define a Computation Task
-------------------------
The computation of each operator can be described as a directed acyclic graph (DAG). The DAG is composed of tensor
nodes. Both :py:func:`~hidet.ir.compute.tensor_input` and :py:func:`~hidet.ir.compute.compute` primitives create tensor
nodes. The edges of the DAG are the dependencies between the tensor nodes. Such a DAG is stored in a 
:py:class:`~hidet.ir.task.Task` object. 

.. py:class:: Task(name: str, inputs: List[TensorNode], outputs: List[TensorNode])
    :noindex:

Each task has a name, a list of inputs, and a list of outputs, correspongding to the inputs and outputs of the operator.
The following example shows how to create a task.
"""


def demo_task():
    from hidet.ir.compute import tensor_input, compute
    from hidet.ir.task import Task

    # define the computation DAG through the compute primitives
    a = tensor_input('a', dtype='float32', shape=[10])
    b = tensor_input('b', dtype='float32', shape=[10])
    c = compute('c', [10], lambda i: a[i] + i)
    d = compute('d', [10], lambda i: c[9 - i])
    e = compute('e', [10], lambda i: a[i] + b[i])

    # create a task object
    task = Task(name='task', inputs=[a, b], outputs=[d, e])
    print(task)


demo_task()

# %%
# Its computation DAG can be visualized as follows.
#
# .. graphviz::
#   :caption: An example of computation DAG. In this example, there are 5 tensor nodes, where node A and B are inputs
#             and node D and E are outputs. The computation of node C depends on the computation of node A and B.
#
#   digraph {
#       // rankdir=LR;
#       splines=curved;
#       node [
#           shape=box, style="rounded",
#           height=0.4, width=0.6
#       ];
#       graph [style="rounded, dashed"]
#           subgraph cluster_0 {
#               graph [style="rounded, dashed", margin="12"];
#               node [group=0];
#               label="Inputs";
#               a [label="A"];
#               b [label="B"];
#           }
#           subgraph cluster_1 {
#               graph [style="rounded, dashed", labelloc="b", margin="15"];
#               node [group=1];
#               labeljust="b";
#               d [label="D"];
#               e [label="E"];
#               label="Outputs";
#           }
#           c [label="C"];
#           a -> c -> d
#           a -> e
#           b -> e
#   }

# %%
# Build and Run a Task
# --------------------
# We provide a driver function :py:func:`hidet.driver.build_task` to build a task into callable function. The
# :py:func:`~hidet.driver.build_task` function does the following steps to lower the task into a callable function:
#
# .. note::
#   :class: margin
#
#   A scheduler is a function that takes a task as input and returns an scheduled tensor program defined in an IRModule.
#
# 1. Dispatch the task to a **scheduler** according to the target device and task.
# 2. The scheduler lowers the task into a tensor program, defined with :py:class:`~hidet.ir.func.IRModule`.
# 3. Lower and optimize the IRModule.
# 4. Code generation that translates the IRModule into the target source code (e.g., **source.cu**).
# 5. Call compiler (e.g., **nvcc**) to compile the source code into a dynamic library (i.e., **lib.so**).
# 6. Load the dynamic library and wrap it to :py:class:`~hidet.runtime.CompiledFunction` that can be directly called.
#
# We can define the following function to build and run a task.

from typing import List
import hidet
from hidet.ir.task import Task


def run_task(task: Task, inputs: List[hidet.Tensor], outputs: List[hidet.Tensor]):
    """Run given task and print inputs and outputs"""
    from hidet.runtime import CompiledFunction

    # build the task
    func: CompiledFunction = hidet.driver.build_task(task, target_device='cpu')
    params = inputs + outputs

    # run the compiled task
    func(*params)

    print('Task:', task.name)
    print('Inputs:')
    for tensor in inputs:
        print(tensor)
    print('Output:')
    for tensor in outputs:
        print(tensor)
    print()


# %%
# The following code shows how to 1) define the computation, 2) define the task, and 3) build and run the task.
#
# .. note::
#  :class: margin
#
#  Please pay attention to the difference between :class:`~hidet.graph.Tensor` and
#  :class:`~hidet.ir.compute.TensorNode`. The former is a tensor object that can be used to store data and trace the
#  high-level computation graph of a deep learning model. The latter is a tensor node in the domain-specific language
#  that is used to describe the computation of a single operator.

from hidet.ir.compute import tensor_input, reduce, compute, arg_reduce, TensorNode

# sphinx_gallery_start_ignore
# Hidet use numpy for tensor printing, this line reduce the number of printed digits
import numpy as np

np.set_printoptions(precision=2, suppress=True)
# sphinx_gallery_end_ignore


def add_example():
    a: TensorNode = tensor_input(name='a', dtype='float32', shape=[5])
    b: TensorNode = tensor_input(name='b', dtype='float32', shape=[5])
    c: TensorNode = compute(name='c', shape=[5], fcompute=lambda i: a[i] + b[i])
    task = Task(name='add', inputs=[a, b], outputs=[c])
    run_task(task, [hidet.randn([5]), hidet.randn([5])], [hidet.empty([5])])


add_example()


# %%
# More Examples
# -------------
#
# .. tip::
#   :class: margin
#
#   All the hidet operators are defined in :py:mod:`hidet.graph.ops` submodule. And all of existing operators
#   are defined through the compute primitives described in this tutorial. Feel free to check the source code to learn
#   more about how to define the computation of different operators.
#
# At last, we show more examples of using the compute primitives to define operator computation.
#
# ReduceSum
# ^^^^^^^^^


def reduce_sum_example():
    a = tensor_input('a', dtype='float32', shape=[4, 3])
    b = compute(
        'b',
        shape=[4],
        fcompute=lambda i: reduce(
            shape=[3], fcompute=lambda j: a[i, j], reduce_type='sum'
        ),
    )
    task = Task('reduce_sum', inputs=[a], outputs=[b])
    run_task(task, [hidet.randn([4, 3])], [hidet.empty([4])])


reduce_sum_example()


# %%
# ArgMax
# ^^^^^^


def arg_max_example():
    a = tensor_input('a', dtype='float32', shape=[4, 3])
    b = compute(
        'b',
        shape=[4],
        fcompute=lambda i: arg_reduce(
            extent=3, fcompute=lambda j: a[i, j], reduce_type='max'
        ),
    )
    task = Task('arg_max', inputs=[a], outputs=[b])
    run_task(task, [hidet.randn([4, 3])], [hidet.empty([4], dtype='int64')])


arg_max_example()


# %%
# MatMul
# ^^^^^^
def matmul_example():
    a = tensor_input('a', dtype='float32', shape=[3, 3])
    b = tensor_input('b', dtype='float32', shape=[3, 3])
    c = compute(
        'c',
        shape=[3, 3],
        fcompute=lambda i, j: reduce(
            shape=[3], fcompute=lambda k: a[i, k] * b[k, j], reduce_type='sum'
        ),
    )
    task = Task('matmul', inputs=[a, b], outputs=[c])
    run_task(task, [hidet.randn([3, 3]), hidet.randn([3, 3])], [hidet.empty([3, 3])])


matmul_example()


# %%
# Softmax
# ^^^^^^^
def softmax_example():
    from hidet.ir.primitives import exp

    a = tensor_input('a', dtype='float32', shape=[3])
    max_val = reduce(shape=[3], fcompute=lambda i: a[i], reduce_type='max')
    b = compute('b', shape=[3], fcompute=lambda i: a[i] - max_val)
    exp_a = compute('exp', shape=[3], fcompute=lambda i: exp(b[i]))
    exp_sum = reduce(shape=[3], fcompute=lambda i: exp_a[i], reduce_type='sum')
    softmax = compute('softmax', shape=[3], fcompute=lambda i: exp_a[i] / exp_sum)

    task = Task('softmax', inputs=[a], outputs=[softmax])
    run_task(task, [hidet.randn([3])], [hidet.empty([3])])


softmax_example()

# %%
# Summary
# -------
# In this tutorial, we introduced the compute primitives that are used to define the computation of operators in Hidet.
# After that, we showed how to wrap the computation DAG into a task and build and run the task. In the next step, we
# will show you how to use these compute primitives to define new operators in Hidet.
#
