"""
Define Computation
==================

Each operator takes a list of input tensors and produces a list of output tensors:

.. code-block:: python

    inputs: List[Tensor]
    outputs: List[Tensor] = operator(inputs)

The precise mathematical definition of each operator in Hidet is defined through a domain-specific-language (DSL).

In this article, we will show how to define the mathematical definition of a new operator in Hidet using this DSL.
The DSL is defined in the :py:mod:`hidet.ir.compute` module.
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
