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
from typing import List, Union
from hidet.ir.module import IRModule
from hidet.ir.library import tune
from hidet.utils.py import cdiv
from hidet.graph.tensor import from_torch
from .utils import Task, Operator, Tensor, TensorNode, compute, input_like


class TransposeTask2D(Task):
    def __init__(self, input: TensorNode):
        self.input_shape = input.shape
        self.input_dtype = input.type.dtype
        self.output_shape = [self.input_shape[1], self.input_shape[0]]

        output = compute(name='output', shape=self.output_shape, fcompute=lambda i, j: input[j, i])

        super().__init__(name='transpose2d', inputs=[input], outputs=[output], attributes={})

    def allow_prologue(self) -> bool:
        return False

    def allow_epilogue(self) -> bool:
        return False

    def implement_cuda(self, working_dir: str) -> Union[IRModule, List[IRModule]]:
        return tune.extract_ir_modules(self.cuda_schedule_threads_coarsening_transpose)

    @staticmethod
    def get_coo_for_1d_arr(tx, shape):
        if shape[0] == 1:
            return 0, tx
        else:
            return tx, 0

    @staticmethod
    def get_size(input_shape, coarsen_factor_row, coarsen_factor_col, tile_size_baseline):
        num_elements_per_thread_row, num_elements_per_thread_col = coarsen_factor_row, coarsen_factor_col
        block_size_row = min(cdiv(input_shape[0], num_elements_per_thread_row), tile_size_baseline)
        block_size_col = min(cdiv(input_shape[1], num_elements_per_thread_col), tile_size_baseline)
        num_elements_per_block_row = num_elements_per_thread_row * block_size_row
        num_elements_per_block_col = num_elements_per_thread_col * block_size_col
        return (
            num_elements_per_thread_row,
            num_elements_per_thread_col,
            block_size_row,
            block_size_col,
            num_elements_per_block_row,
            num_elements_per_block_col,
        )

    @staticmethod
    def get_launch_params(
        input_shape, coarsen_factor_row, coarsen_factor_col, tile_size_baseline, smem_size, input_dtype
    ):
        smem_max_elements = smem_size / input_dtype.nbytes
        (
            num_elements_per_thread_row,
            num_elements_per_thread_col,
            block_size_row,
            block_size_col,
            num_elements_per_block_row,
            num_elements_per_block_col,
        ) = TransposeTask2D.get_size(input_shape, coarsen_factor_row, coarsen_factor_col, tile_size_baseline)
        if num_elements_per_block_row * (num_elements_per_thread_col + 1) < smem_max_elements:
            return (
                num_elements_per_thread_row,
                num_elements_per_thread_col,
                block_size_row,
                block_size_col,
                num_elements_per_block_row,
                num_elements_per_block_col,
            )
        else:
            num_elements_per_block = num_elements_per_block_row * (num_elements_per_thread_col + 1)
            while True:
                ratio = smem_max_elements / num_elements_per_block
                coarsen_factor_row, coarsen_factor_col = int(coarsen_factor_row * ratio), int(
                    coarsen_factor_col * ratio
                )
                (
                    num_elements_per_thread_row,
                    num_elements_per_thread_col,
                    block_size_row,
                    block_size_col,
                    num_elements_per_block_row,
                    num_elements_per_block_col,
                ) = TransposeTask2D.get_size(input_shape, coarsen_factor_row, coarsen_factor_col, tile_size_baseline)
                num_elements_per_block = num_elements_per_block_row * (num_elements_per_thread_col + 1)
                if num_elements_per_block < smem_max_elements:
                    break
            return (
                num_elements_per_thread_row,
                num_elements_per_thread_col,
                block_size_row,
                block_size_col,
                num_elements_per_block_row,
                num_elements_per_block_col,
            )

    @tune.space(1, coarsen_factor_row=[1], coarsen_factor_col=[1], tile_size_baseline=[32])
    def cuda_schedule_threads_coarsening_transpose(
        self, coarsen_factor_row=1, coarsen_factor_col=1, tile_size_baseline=32
    ) -> IRModule:
        # pylint: disable=unused-variable
        import hidet
        from hidet.lang.cuda import blockIdx, threadIdx, blockDim, gridDim, syncthreads
        from hidet.lang.cuda import shared_tensor
        from hidet.lang import attrs, grid
        from hidet.cuda.device import properties

        input, output = self.inputs[0], self.outputs[0]
        (
            num_elements_per_thread_row,
            num_elements_per_thread_col,
            block_size_row,
            block_size_col,
            num_elements_per_block_row,
            num_elements_per_block_col,
        ) = TransposeTask2D.get_launch_params(
            self.input_shape,
            coarsen_factor_row,
            coarsen_factor_col,
            tile_size_baseline,
            properties().sharedMemPerBlock,
            self.input_dtype,
        )
        shared_mem_size_row, shared_mem_size_col = num_elements_per_block_row, num_elements_per_block_col
        smem_padding = shared_mem_size_col % 16 == 0  # may improve by a bit-level condition!
        block_size = (block_size_row, block_size_col)
        grid_size = (
            cdiv(self.input_shape[0], num_elements_per_block_row),
            cdiv(self.input_shape[1], num_elements_per_block_col),
        )
        num_elements = self.input_shape[0] * self.input_shape[1]

        with hidet.script_module() as module:
            if self.input_shape[0] == 1 or self.input_shape[1] == 1:

                @hidet.script
                def transpose_kernel(
                    input: self.input_dtype[self.input_shape], output: self.input_dtype[self.output_shape]
                ):
                    attrs.cuda.block_dim = 1024
                    attrs.cuda.grid_dim = cdiv(num_elements, 1024)
                    tx = threadIdx.x + blockIdx.x * blockDim.x
                    coo_x, coo_y = TransposeTask2D.get_coo_for_1d_arr(tx, self.input_shape)
                    while tx < num_elements:
                        output[coo_y, coo_x] = input[coo_x, coo_y]
                        tx += gridDim.x

            else:

                @hidet.script
                def transpose_kernel(
                    input: self.input_dtype[self.input_shape], output: self.input_dtype[self.output_shape]
                ):
                    attrs.cuda.grid_dim = grid_size
                    attrs.cuda.block_dim = block_size
                    if smem_padding:
                        tile = shared_tensor(self.input_dtype, shape=[shared_mem_size_row, shared_mem_size_col + 1])
                    else:
                        tile = shared_tensor(self.input_dtype, shape=[shared_mem_size_row, shared_mem_size_col])
                    for kx, ky in grid(coarsen_factor_row, coarsen_factor_col):
                        tx = threadIdx.x + block_size_row * kx
                        ty = threadIdx.y + block_size_col * ky
                        x_index = blockIdx.x * num_elements_per_block_row + tx
                        y_index = blockIdx.y * num_elements_per_block_col + ty

                        if x_index < self.input_shape[0] and y_index < self.input_shape[1]:
                            tile[tx, ty] = input[x_index, y_index]

                    syncthreads()
                    for kx, ky in grid(coarsen_factor_row, coarsen_factor_col):
                        tx = threadIdx.x + block_size_row * kx
                        ty = threadIdx.y + block_size_col * ky

                        x_index = blockIdx.x * num_elements_per_block_row + tx
                        y_index = blockIdx.y * num_elements_per_block_col + ty
                        if y_index < self.output_shape[0] and x_index < self.output_shape[1]:
                            output[y_index, x_index] = tile[tx, ty]

        ir_module = module.ir_module()
        return ir_module


class TransposeOp2D(Operator):
    def __init__(self, input: Tensor):
        super().__init__(inputs=[input], attributes={}, task=TransposeTask2D(input_like(input, 'input')))

    def run_torch(self):
        x_torch = self.inputs[0].torch()
        return [from_torch(x_torch.T)]
