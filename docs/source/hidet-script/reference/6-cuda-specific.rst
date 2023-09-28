CUDA Specifics
==============

.. todo:

    make is more comprehensive and detailed


Function attributes
-------------------

The ``cuda_kernel`` function kind has the following attributes:
- ``attrs.cuda.block_dim`` (required): the block dimensions
- ``attrs.cuda.grid_dim`` (required): the grid dimensions
- ``attrs.cuda.dynamic_smem_bytes`` (optional): the dynamic shared memory size to use
- ``attrs.cuda.min_blocks`` (optional): the minimum number of blocks this kernel will be launched.

Memory scope
------------

To define a tensor that resides in the shared memory, we can specify the ``scope`` argument of
the ``hidet.lang.types.tensor`` constructor:

.. code-block::

    from hidet.lang.types import tensor, f32, DeclareScope

    # define a tensor in the shared memory
    a = tensor(dtype=f32, shape=[10, 10], scope='shared')   # use the string to specify the scope
    b = tensor(dtype=f32, shape=[10, 10], scope=DeclareScope.Shared)  # use the enum to specify the scope

    # similarly, we can define a tensor that resides in the register file
    # please note that each thread will have a f32[10, 10] tensor
    c = tensor(dtype=f32, shape=[10, 10], scope='register')
    d = tensor(dtype=f32, shape=[10, 10], scope=DeclareScope.Register)

Primitive functions
-------------------

Hidet provides some primitive functions that can be used in the cuda kernel functions. The primitive functions
are defined in the ``hidet.lang.cuda`` module. The following table lists the commonly used primitive functions:

.. todo::

  make a full list in the reference section.

- ``threadIdx``, ``blockIdx``, ``blockDim``, ``gridDim``: the thread index, block index, block dimension and grid dimension.
- ``syncthreads()``: synchronize all threads in the same block.
- ``ldmatrix(...)``: load a matrix from shared memory to the register file.
- ``mma_sync(...)``: perform matrix-matrix multiplication using the tensor cores.
- ``atomic_add(...)``: perform atomic add operation (other atomic functions like ``atomic_max`` are also included).
- ``shfl_sync(...)``: warp shuffle operation.
- ``dynamic_shared_memory(...)``: access the dynamic allocated shared memory

Please refer to the ``hidet.lang.cuda`` module for the complete list of supported primitive functions
