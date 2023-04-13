"""
Using Rule-based Scheduling
===========================

In the previous tutorial, we have learned how to define the computation using compute primitives and wrap it into a
:py:class:`~hidet.ir.task.Task`. In this tutorial, we will learn how to add an operator (i.e.,
:py:class:`~hidet.graph.Operator`) with given computation definition, and use hidet's provided rule-based scheduler to
automatically schedule the computation into a tensor program.

Three steps to define a new operator
------------------------------------

There are three steps to define a new operator in Hidet.

1. Define the computation task class by inheriting :py:class:`~hidet.ir.task.Task`.
2. Define the operator class by inheriting :py:class:`~hidet.graph.Operator`.
3. Define a function to create the operator instance.

Batch Matrix Multiplication Example
-----------------------------------

We will take the batch matrix multiplication as an example to illustrate the three steps.

1. Define the computation task class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We define the computation task class **BatchMatmulTask** by inheriting :py:class:`~hidet.ir.task.Task` class. The
**BatchMatmulTask** class's constructor function takes two arguments, **a** and **b** that are the input tensor nodes
of the batch matrix multiplication.
"""

# sphinx_gallery_start_ignore
# Hidet use numpy for tensor printing, this line reduce the number of printed digits
import numpy as np

np.set_printoptions(precision=2, suppress=True)
# sphinx_gallery_end_ignore
from hidet.ir.compute import TensorNode, compute, reduce
from hidet.ir.task import Task


class BatchMatmulTask(Task):
    def __init__(self, a: TensorNode, b: TensorNode):
        # get the input sizes
        batch_size, m_size, k_size = a.const_shape()
        batch_size, k_size, n_size = b.const_shape()

        # define the computation
        c = compute(
            name='c',
            shape=[batch_size, m_size, n_size],
            fcompute=lambda p, i, j: reduce(
                shape=[k_size],
                fcompute=lambda k: a[p, i, k] * b[p, k, j],
                reduce_type='sum',
            ),
        )

        # call the parent class constructor to initialize the task
        super().__init__(
            name='batch_matmul',  # the name of the task
            inputs=[a, b],  # the input tensor nodes
            outputs=[c],  # the output tensor nodes
        )


# %%
# 2. Define the operator class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Our next step is to define the operator class **BatchMatmulOp** by inheriting :py:class:`~hidet.graph.Operator` class.
from hidet.graph import Operator, Tensor
from hidet.graph.ops.definitions.utils import input_like


class BatchMatmulOp(Operator):
    def __init__(self, a: Tensor, b: Tensor):
        # call the parent class constructor to initialize the operator
        super().__init__(
            inputs=[a, b],  # the input tensors
            attributes={},
            task=BatchMatmulTask(  # the task of the operator
                # create tensor nodes (TensorNode) with the same shape and dtype as the tensors (Tensor)
                input_like(a, 'a'),
                input_like(b, 'b'),
            ),
        )


# %%
# 3. Define a function to create the operator instance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We define a function **batch_matmul** to create the operator instance **BatchMatmulOp** and return the output tensor.


def batch_matmul(a: Tensor, b: Tensor) -> Tensor:
    # get_output(0) returns the first output tensor of the operator
    return BatchMatmulOp(a, b).get_output(0)


# %%
# Use the defined operator
# ~~~~~~~~~~~~~~~~~~~~~~~~
# The new operator has no difference with the hidet provided operators, as we define hidet operators in the same way.
# For example, when we optimize the flow graph, this new operator can also fuse surrounding operators.
import hidet


def demo_usage():
    a = hidet.randn([2, 2, 3])
    b = hidet.randn([2, 3, 2])
    c = batch_matmul(a, b)
    print(a)
    print(b)
    print(c)


demo_usage()

# %%
# Two Scheduling Machanisms
# -------------------------
# We only define the computation of the operator, and leave the scheduling to the rule-based scheduler provided by
# hidet. We call this method of scheduling as **rule-based scheduling**. Most hidet operators are using the same
# rule-based scheduler as we used in this example. Our experience shows that the rule-based
# scheduler can achieve good performance for operators that do not have large amount of reduction. However, for
# operators like matrix multiplication, convolution, etc., the rule-based scheduler may not be able to achieve the
# best performance as it does not use shared memory to cache the data loading. Thus, hidet also provides another
# scheduling mechanism, the **template-based scheduling**.
#

# %%
# Summary
# -------
# In this tutorial, we have learned how to define a new operator with given computation definition, and use hidet's
# provided rule-based scheduler to automatically schedule the computation into a tensor program. In the next tutorial,
# we will learn how to use the template-based scheduling to achieve better performance.
