Type System
===========

In hidet script, we have a type system that contains scalar types, tensor type, as well as pointer types.

Scalar types
------------

Hidet supports the following scalar types:
- integer types: ``i8``, ``i16``, ``i32``, ``i64`` (``int8``, ``int16``, ``int32``, ``int64``)
- floating point types: ``f16``, ``f32``, ``f64``, ``bf16``, ``tf32`` (``float16``, ``float32``, ``float64``, ``bfloat16``, ``tfloat32``)
- boolean type: ``bool``
- complex types: ``c64``, ``c128`` (``complex64``, ``complex128``)

Some types have both short names and long names. For example, ``i8`` and ``int8`` are the same type.

There are also vectorized scalar types:
- vectorized integer types: ``i8x4`` (``int8x4``)
- vectorized float types: ``f16x2``, ``f32x4`` (``float16x2``, ``float32x4``)

Tensor type
-----------

Hidet is designed to simplify the tensor program writing. Therefore, we have a powerful tensor type that
represents a tensor with a specific element data type, shape, and memory layout. More specifically, a
tensor type has the following attributes:
- ``dtype``: the data type of the tensor elements, can be any scalar type.
- ``shape``: a list of expressions that represents the shape of the tensor.
- ``layout``: the memory layout of the tensor.

The following code snippet shows how to define a tensor type:

.. code-block::

    import hidet
    from hidet.lang import attrs, printf
    from hidet.lang.types import tensor, f32

    with hidet.script_module() as script_module:
      @hidet.script
      def kernel():
        attrs.func_kind = 'cpu_kernel'

        # by default, the layout is a row-major layout
        a = tensor(dtype=f32, shape=[1024, 1024])

        a[0, 0] = 0.0

        printf("a[%d, %d] = %.1f\n", 0, 0, a[0, 0])

    module = script_module.build()
    module()


Tensor shape
~~~~~~~~~~~~

The shape of the tensor must be determined at the compile time. Therefore, the shape of the tensor can only
be defined with constant expressions. If we want to access a tensor with shape determined at runtime with
variable expressions, we can use *tensor pointer* (will be discussed later).


Tensor layout
~~~~~~~~~~~~~

The layout of a tensor defines how to map the coordinates of a tensor element to the linear position of the
element in the memory space. Generally speaking, a layout maps a :math:`n`-dimensional coordinate
:math:`(c_0, c_1, \dots, c_{n-1})` to a linear index:

.. math::

    index = layout(c_0, c_1, ..., c_{n-1})


The most commonly used layout is the row-major layout. In row-major layout, the linear index is calculated as:


.. math::

  index = c_0 \times s_1 \times s_2 \times \dots \times s_{n-1} + c_1 \times s_2 \times \dots \times s_{n-1} + \dots + c_{n-2} \times s_{n-1} + c_{n-1}

where :math:`s_i` is the size of the :math:`i`-th dimension of the tensor: :math:`shape=(s_0, s_1, \dots, s_{n-1})`.


Similar to the row-major layout, we can also define a column-major layout as follows:

.. math::

  index = c_{n-1} \times s_{n-2} \times \dots \times s_1 \times s_0 + c_{n-2} \times \dots \times s_1 \times s_0 + \dots + c_1 \times s_0 + c_0

The row-major layout is the default layout if we do not specify the layout of a tensor. We can also specify
the layout of a tensor with the ``layout`` argument of the ``tensor`` type. For example, we can define a tensor with
column-major layout as follows:

.. code-block::

    from hidet.lang.layout import column_major
    from hidet.lang.types import tensor, f32
    # ...
    a = tensor(dtype=f32, shape=[1024, 1024], layout=column_major(1024, 1024))
    # or ignore shape if the layout is specified
    b = tensor(dtype=f32, layout=column_major(1024, 1024))


Both row-major layout and column-major layout are special cases of the strided layout.
In hidet, we can define a strided layout like


.. code-block::

    from hidet.lang.layout import strided_layout
    from hidet.lang.types import tensor, f32

    # equivalent to row-major layout
    a = tensor(dtype=f32, layout=strided_layout(shape=[1024, 1024], ranks=[0, 1]))
    # equivalent to column-major layout
    b = tensor(dtype=f32, layout=strided_layout(shape=[1024, 1024], ranks=[1, 0]))
    # the ranks define the order of the dimensions from the one that changes the slowest to the one that changes the fastest
    c = tensor(dtype=f32, layout=strided_layout(shape=[2, 2, 2], ranks=[0, 2, 1]))
    # c[coordinate] -> index
    # c[0, 0, 0] -> 0
    # c[0, 1, 0] -> 1
    # c[0, 0, 1] -> 2
    # c[0, 1, 1] -> 3
    # c[1, 0, 0] -> 4
    # c[1, 1, 0] -> 5
    # c[1, 0, 1] -> 6
    # c[1, 1, 1] -> 7

Given two layouts $f$ and $g$, we can define a new layout $h$ as the composition of $f$ and $g$ with $f$ as the outer
layout and $g$ as the inner layout:

.. math::

    h(\textbf{c}) = f(\textbf{c}/\textbf{s}_{g}) * n_g + g(\textbf{c} \mod \textbf{s}_{g})

where :math:`\textbf{c}` is the coordinate of the tensor element, :math:`\textbf{s}_{g}` is the shape of the inner
layout :math:`g`, and :math:`n_g` is the number of elements in the inner layout :math:`g`. The division and modulo
operations are performed element-wise. The composed layout $h$ has the same number of dimensions as the outer and inner
layouts, and the shape of the composed layout is the elementwise product of the shapes of the outer and inner layouts.

In hidet script, we can use the *multiply* operator ``*`` to compose two layouts. For example, we can define a
composed layout as follows:

.. code-block::

    from hidet.lang.layout import row_major, column_major

    c = row_major(2, 1) * row_major(2, 2)
    # c shape=[4, 2]
    # c[0, 0] -> 0
    # c[0, 1] -> 1
    # c[1, 0] -> 2
    # c[1, 1] -> 3
    # c[2, 0] -> 4
    # c[2, 1] -> 5
    # c[3, 0] -> 6
    # c[3, 1] -> 7

    d = row_major(2, 1) * column_major(2, 2)
    # d shape=[4, 2]
    # d[0, 0] -> 0
    # d[1, 0] -> 1
    # d[0, 1] -> 2
    # d[1, 1] -> 3
    # d[2, 0] -> 4
    # d[3, 0] -> 5
    # d[2, 1] -> 6
    # d[3, 1] -> 7

We can apply the composition operation multiple times to compose multiple layouts. For example,

.. code-block::

    from hidet.lang.layout import row_major, column_major

    e = row_major(2, 1) * row_major(2, 2) * column_major(2, 2)   # e shape=[8, 4]

The composition operation is associative, i.e., :math:`(f * g) * h = f * (g * h)`, but not commutative, i.e.,
it is highly likely :math:`f * g \neq g * f`.


Pointer types
~~~~~~~~~~~~~

In hidet, we can define a pointer type with the same semantics as the pointer type in C/C++.

To construct a pointer type, we use the ``~`` operator to apply to a scalar type or pointer type:

- ``~i32``: a pointer to ``i32`` type
- ``~(~f16)``: a pointer to a pointer to ``f16`` type


Void type
~~~~~~~~~

The ``void`` type can be used as the return type of a function, or used to define a ``void`` pointer type
(i.e., ``~void``).
