from typing import Mapping

from hidet.implement.common import init_task, transfer_task
from hidet.implement.implementer import Implementer, register_impl
from hidet.ir.builders import FunctionBuilder, TaskBuilder, StmtBuilder
from hidet.ir.dialects.compute import TensorCompute, ReduceCompute
from hidet.ir.dialects.compute import compute, TensorInput, reduce_sum
from hidet.ir.dialects.lowlevel import TensorPointerType
from hidet.ir.dialects.pattern import TaskPattern, any_const_int
from hidet.ir.expr import Var, var
from hidet.ir.func import IRModule
from hidet.ir.layout import TaskLayout, row_major_layout, full_layout, DataLayout
from hidet.ir.node import Node
from hidet.ir.primitives import block_idx, syncthreads
from hidet.ir.task import Task
from hidet.ir.task import ThreadBlock, Grid
from hidet.ir.type import TensorType
from hidet.ir.type import scalar_type, Scope
from hidet.utils.py import prod


@register_impl('cuda_block_static_matmul_no_pipe_implementer')
class CudaBlockStaticMatmulNoPipeImplementer(Implementer):
    def __init__(self):
        # const definition
        self.block_size = any_const_int()
        self.task_m = any_const_int()
        self.task_n = any_const_int()
        self.task_k = any_const_int()

        # inputs
        self.A = TensorInput('A', dtype=scalar_type('float32'), shape=None)
        self.B = TensorInput('B', dtype=scalar_type('float32'), shape=None)

        # compute
        self.axis_i = var('i')
        self.axis_j = var('j')
        self.axis_k = var('k')
        self.value = self.A[self.axis_i, self.axis_k] * self.B[self.axis_k, self.axis_j]
        self.reduce = ReduceCompute(value=self.value,
                                    shape=[self.task_k],
                                    axis=self.axis_k,
                                    reduce_type=None)
        self.computation = TensorCompute(name='C',
                                         shape=[self.task_m, self.task_n],
                                         axes=[self.axis_i, self.axis_j],
                                         value=self.reduce)

        # inputs and output types
        self.A_layout = DataLayout()
        self.B_layout = DataLayout()
        self.C_layout = DataLayout()
        self.A_type = TensorType(Scope('global'), scalar_type('float32'), shape=None, layout=self.A_layout)
        self.B_type = TensorType(Scope('global'), scalar_type('float32'), shape=None, layout=self.B_layout)
        self.C_type = TensorType(Scope('global'), scalar_type('float32'), shape=None, layout=self.C_layout)

        # pattern
        self.pattern = TaskPattern(
            compute_pattern=self.computation,
            required_params=[self.A, self.B, self.computation],
            required_param_types=[self.A_type, self.B_type, self.C_type],
            allow_tensor_extra_params=False,
            worker=ThreadBlock(block_dim=self.block_size)
        )

    def priority(self) -> int:
        return 1

    def task_pattern(self) -> TaskPattern:
        return self.pattern

    def implement(self, task: Task, match: Mapping[Node, Node]) -> IRModule:
        ir_module = IRModule()

        # Task-related constants. The workload is a matrix multiplication C[M, N] = A[M, K] * B[K, N].
        task_m, task_n, task_k = [int(match[v]) for v in [self.task_m, self.task_n, self.task_k]]

        # Kernel configs
        #                          32
        #           +-------------------------------+ +-------------------------------+
        #         B |                               | |                               | 1
        #           +-------------------------------+ +-------------------------------+
        #      A
        #    +---+  +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
        #    |   |  |t0 |t1 |t2 |t3 |t4 |t5 |t6 |t7 | |t0 |   |   |   |   |   |   |   |
        #    |   |  +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
        #    |   |  |t8 |t9 |t10|t11|t12|t13|t14|t15| |   |   |   |   |   |   |   |   |
        # 16 |   |  +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
        #    |   |  |t16|t17|t18|t19|t20|t21|t22|t23| |   |   |   |   |   |   |   |   |                      128
        #    |   |  +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+              +-----------------+
        #    |   |  |t24|t25|t26|t27|t28|t29|t30|t31| |   |   |   |   |   |   |   |   |            B |                 | 8
        #    +---+  +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+              +-----------------+
        #                                                                                       A
        #    +---+  +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+       +---+  +--------+--------+
        #    |   |  |t0 |   |   |   |   |   |   |   | |t0 |   |   |   |   |   |   |   |       |   |  | warp 0 | warp 1 |
        #    |   |  +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+       |   |  +--------+--------+
        #    |   |  |   |   |   |   |   |   |   |   | |   |   |   |   |   |   |   |   |       |   |  | warp 2 | warp 3 |
        #    |   |  +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+   128 |   |  +--------+--------+
        #    |   |  |   |   |   |   |   |   |   |   | |   |   |   |   |   |   |   |   |       |   |  | warp 4 | warp 5 |
        #    |   |  +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+       |   |  +--------+--------+
        #    |   |  |   |   |   |   |   |   |   |   | |   |   |   |   |   |   |   |   |       |   |  | warp 6 | warp 7 |
        #    +---+  +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+       +---+  +--------+--------+
        #      1                                     C                                                        C
        #                                       warp layout                                             block layout
        #
        #   Each warp has 32 threads. Each thread will work on 2 * 2 * 4 * 4 = 64 elements of C matrix.
        #   Inner part is 4 * 4 = 16, and there are 2 * 2 = 4 such parts.
        #   Each thread block contains 4 * 2 = 8 warps. Each warp tile has shape (32, 64), and the block tile has shape (128, 128).
        warp_inner = [4, 4]
        warp_middle = [4, 8]
        warp_outer = [2, 2]
        block_warps = [4, 2]
        block_k, warp_k = 8, 1

        # Define task layouts.
        # A task layout is a mapping from the worker (e.g., a thread or a warp) to the unit tasks that it works on.
        # For example, a row major task layout with 8 workers on task grid (4, 2) has mapping w -> [(w / 2, w % 2)].
        # A special task layout called full layout has a single worker. If the task grid has shape (2, 2), we would
        # have mapping w -> [(0, 0), (0, 1), (1, 0), (1, 1)].
        # We can construct above task layouts by calling row_major_layout(4, 2) and full_layout(2, 2).
        warp_inner_layout: TaskLayout = full_layout(*warp_inner)
        warp_middle_layout: TaskLayout = row_major_layout(*warp_middle)
        warp_outer_layout: TaskLayout = full_layout(*warp_outer)
        block_warps_layout: TaskLayout = row_major_layout(*block_warps)
        # Task layout can be composed to get a larger task layout. For example, the task layout for each warp shown
        # in above figure can be derived from composing: warp_layout = full_layout(2, 2) * row_major_layout(4, 8) * full_layout(4, 4)
        # In the next line, we get the task layout of thread block by composing. In this layout, each thread is a worker.
        block_layout: TaskLayout = block_warps_layout * warp_outer_layout * warp_middle_layout * warp_inner_layout
        block_size = block_layout.num_workers
        block_shape = block_layout.task_shape
        # We can decompose this kernel into different sub-tasks, such as transferring A's data from global memory to shared memory (a_g2s),
        # and transferring A's data from shared to registers (a_s2r). The task layout for these sub-tasks are derived from
        # task composition and transformation (e.g., projection).
        a_g2s_layout: TaskLayout = row_major_layout(block_size // block_k, block_k) * full_layout(block_shape[0] // (block_size // block_k), 1)
        b_g2s_layout: TaskLayout = full_layout(1, block_shape[1] // (block_size // block_k)) * row_major_layout(block_k, block_size // block_k)
        a_s2r_layout: TaskLayout = (block_warps_layout * full_layout(warp_outer[0], 1) * warp_middle_layout * full_layout(warp_inner[0], 1)).projection({1: 0})
        b_s2r_layout: TaskLayout = (block_warps_layout * full_layout(1, warp_outer[1]) * warp_middle_layout * full_layout(1, warp_inner[1])).projection({0: 0})

        # We check some requirement assumed by following implementation.
        self.check(block_size % block_k == 0)  # required by subtask transferring data from global memory to shared memory
        self.check(task_m % block_shape[0] == 0)  # this implementation has no bound checking, thus only support matmul that perfectly matches the kernel
        self.check(task_n % block_shape[1] == 0)  # that means the matmul must have shape M % 128 = 0, N % 128 = 0, K % 8 = 0.
        self.check(task_k % (block_k * warp_k) == 0)
        self.check(block_size == int(match[self.block_size]))

        # Define data layouts.
        # Each tensor has its data layout on memory because the storage is linear but the tensor has multiple dimensions. We need a mapping
        # that maps the high dimension index into scalar index. The most commonly used data layouts are row major layout and column major
        # layout. Both of them are strides layout, that each dimension has a stride and the scalar index is a dot product of the strides and
        # tensor index. A special index in hidet is local index that maps all tensor indices into 0. This is used to describe the data layout
        # of local storage. For example, we have a tensor with shape [32]. The i-th element is stored in the i-th thread. Thread i accesses
        # i-th element of the tensor by index 0. Using local layout directly is not useful, but we also support the data layout compositions.
        # There are two kinds of composition: product and concat. Product of data layout A[M, N] and B[P, Q] is a layout C[M * P, N * Q] with
        # mapping (i, j) -> A[i / P, j / Q] * B_size + B[i % P, j % Q]. We overloaded the __mul__ operator for data layout used to do product
        # composition.
        row_major = DataLayout.row_major
        column_major = DataLayout.column_major
        local = DataLayout.local
        # We can split a dimension by a factor. The following line split a row major layout [M, K] into [M / B, B, K] where 0 is the dimension
        # to split and B (i.e., block_shape[0]) is the split factor. Please refer function DataLayout.split(...) for details.
        gmem_a_layout = match[self.A_layout].split(dim2factor={0: block_shape[0]})
        gmem_b_layout = match[self.B_layout].split(dim2factor={1: block_shape[1]})
        gmem_c_layout = match[self.C_layout].split(dim2factor={0: block_shape[0], 1: block_shape[1]})
        smem_a_layout = column_major([block_shape[0], block_k * warp_k])
        smem_b_layout = row_major([block_k * warp_k, block_shape[1]])
        regs_a_layout = local((block_warps[0], 1)) * row_major((warp_outer[0], 1)) * local((warp_middle[0], 1)) * row_major((warp_inner[0], 1))
        regs_b_layout = local((1, block_warps[1])) * row_major((1, warp_outer[1])) * local((1, warp_middle[1])) * row_major((1, warp_inner[1]))
        regs_c_layout = local(block_warps) * row_major(warp_outer) * local(warp_middle) * row_major(warp_inner)

        # Declare inputs and outputs and their types shared by all subtasks
        dtype = 'float32'
        # We can slice out some dimensions of a data layout. In the next line, we slice out the first dimension of data layout with
        # shape [M / B, B, K] and get a new data layout with shape [B, K]. The mapping of new data layout is (i, j) -> original[0, i, j].
        gmem_a_type = TensorType(scope='global', dtype=dtype, layout=gmem_a_layout.slice_out([0]))
        gmem_b_type = TensorType(scope='global', dtype=dtype, layout=gmem_b_layout.slice_out([1]))
        gmem_c_type = TensorType(scope='global', dtype=dtype, layout=gmem_c_layout.slice_out([0, 2]))
        smem_a_type = TensorType(scope='shared', dtype=dtype, layout=smem_a_layout)
        smem_b_type = TensorType(scope='shared', dtype=dtype, layout=smem_b_layout)
        regs_a_type = TensorType(scope='register', dtype=dtype, layout=regs_a_layout)
        regs_b_type = TensorType(scope='register', dtype=dtype, layout=regs_b_layout)
        regs_c_type = TensorType(scope='register', dtype=dtype, layout=regs_c_layout)

        # Define subtasks.
        # We can decompose different components of the kernel into subtasks. Because transfer task is very common, we can directly create a
        # transfer task by calling transfer_task function. It returns a callable object. We call it to construct the Call node that calls the
        # function that implements the subtask.
        c_init = init_task(f'{task.name}_init', dst_type=regs_c_type, init_value=0.0, worker=ThreadBlock(task_layout=block_layout), parent_module=ir_module)
        a_g2s = transfer_task(f'{task.name}_a_g2s', src_type=gmem_a_type, dst_type=smem_a_type, worker=ThreadBlock(task_layout=a_g2s_layout), parent_module=ir_module)
        b_g2s = transfer_task(f'{task.name}_b_g2s', src_type=gmem_b_type, dst_type=smem_b_type, worker=ThreadBlock(task_layout=b_g2s_layout), parent_module=ir_module)
        a_s2r = transfer_task(f'{task.name}_a_s2r', src_type=smem_a_type, dst_type=regs_a_type, worker=ThreadBlock(task_layout=a_s2r_layout), parent_module=ir_module)
        b_s2r = transfer_task(f'{task.name}_b_s2r', src_type=smem_b_type, dst_type=regs_b_type, worker=ThreadBlock(task_layout=b_s2r_layout), parent_module=ir_module)
        c_r2g = transfer_task(f'{task.name}_c_r2g', src_type=regs_c_type, dst_type=gmem_c_type, worker=ThreadBlock(task_layout=block_layout), parent_module=ir_module)
        # If there is not existing short-cut way to create a sub task, we can also use TaskBuilder to create it. In the following lines, we create
        # a subtask that conducts a matrix-multiplication accumulation task: C = C + matmul(A, B). The returned mma is callable like above subtasks.
        with TaskBuilder('mma', ThreadBlock(task_layout=block_layout), ir_module) as mma:
            # We declare the inputs of the subtasks.
            a = TensorInput('regs_A', dtype)
            b = TensorInput('regs_B', dtype)
            # Here k is the reduction axis.
            k = var('k')
            # Define the computation. This is similar with what Halide and TVM/TE provide. Here accumulate='sum' means
            # we want the computation results to be added into the output tensor instead of assignment.
            c = compute('regs_C', shape=block_shape, fcompute=lambda i, j: reduce_sum(a[i, k] * b[k, j], axis=k, shape=[warp_k]), accumulate='sum')
            mma.set_computation(c)
            # Add the parameter types
            mma.append_param(a, regs_a_type)
            mma.append_param(b, regs_b_type)
            mma.append_param(c, regs_c_type)

        # Define function.
        # We use function builder to define the kernel. It allows us to write low-level code such as if statement and for loop.
        # We need to specify the grid dimension and thread block dimension in the function attributes so that backend know how
        # to launch the kernel.
        with FunctionBuilder(name=task.name, attrs={'worker': ThreadBlock(block_dim=block_size)}) as fb:
            """
            The pseudo code of matmul we want to implement.

            for block_tile_idx in range(task_k / block_k)
                gmem -> smem
                block_sync
                for frag_tile_idx in range(block_k / warp_k)
                    smem -> regs
                    regs -> regs (compute)
                block_sync
            """

            # Declare parameters. The matmul kernel takes three parameters A, B, and C.
            gmem_a = Var('A', TensorPointerType('global', dtype=dtype, layout=gmem_a_layout))
            gmem_b = Var('B', TensorPointerType('global', dtype=dtype, layout=gmem_b_layout))
            gmem_c = Var('C', TensorPointerType('global', dtype=dtype, layout=gmem_c_layout))
            fb.extend_params([gmem_a, gmem_b, gmem_c])

            # Declare A, B shared memory. Shared memory are used to store the fragments of A and B, reducing the number of accesses to
            # global memory. There are local variables so we need to add them to the local variable list of current function.
            smem_a = Var('smem_A', smem_a_type)
            smem_b = Var('smem_B', smem_b_type)
            fb.extend_local_vars([smem_a, smem_b])

            # Declare A, B, C registers. Register is faster than shared memory.
            regs_a = Var('regs_A', regs_a_type)
            regs_b = Var('regs_B', regs_b_type)
            regs_c = Var('regs_C', regs_c_type)
            fb.extend_local_vars([regs_a, regs_b, regs_c])

            # Define the function body. We use a statement builder to construct our function body.
            sb = StmtBuilder()
            # Init regs c by calling the sub-task's function.
            sb += c_init(regs_c)
            # Use a for loop to enumerate the fragments of A and B to do the mma.
            with sb.for_loop('block_k_tile', task_k // (block_k * warp_k)) as block_tile_idx:
                # Transfer fragments of A and B from global memory to shared memory. We use ~gmem_a[i, j, k] to represents the
                # address of element gmem_a[i, j, k] (i.e., &gmem_a[i][j][k] in C/C++).
                sb += a_g2s(~gmem_a[0, 0, block_tile_idx * (block_k * warp_k)], smem_a)
                sb += b_g2s(~gmem_b[block_tile_idx * (block_k * warp_k), 0, 0], smem_b)
                # Synchronize all the threads in the thread block, which makes the data in shared memory loaded by threads visible
                # to each other.
                sb += syncthreads()
                # The k dimension of fragment of A and B is 8 (block_k). We do mma 8 times.
                with sb.for_loop('warp_k_tile', block_k) as warp_tile_idx:
                    # smem -> regs
                    sb += a_s2r(~smem_a[0, warp_tile_idx], regs_a)
                    sb += b_s2r(~smem_b[warp_tile_idx, 0], regs_b)
                    # compute
                    sb += mma(regs_a, regs_b, regs_c)
                    # sync
                sb += syncthreads()
            # regs -> gmem
            sb += c_r2g(regs_c, gmem_c)
            # Set the function body when we finish the body construction.
            fb.set_body(sb.finish())

        func = fb.get()
        ir_module.add(func.name, func)
        return ir_module
