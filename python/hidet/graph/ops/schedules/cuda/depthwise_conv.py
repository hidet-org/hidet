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
from typing import List, Tuple, Union, TypeVar
import os

import hidet.cuda
from hidet import option
from hidet.ir.func import IRModule
from hidet.graph.ops.definitions.conv2d.conv2d import Conv2dTask
from hidet.graph.ops.schedules.common import Schedule, NotSupportedError
from hidet.graph.ops.schedules.resolve import resolve_ir_modules
from hidet.utils import prod
from hidet.transforms.tools import fuse_and_pack


T = TypeVar('T', bound=Tuple)


def tuple_divide(lhs: T, rhs: T, ceil=False) -> T:
    assert len(lhs) == len(rhs)
    if not ceil:
        assert all(a % b == 0 for a, b in zip(lhs, rhs))
    return tuple((a + b - 1) // b for a, b in zip(lhs, rhs))


class DepthwiseConv2dSchedule(Schedule):
    def __init__(
        self,
        task: Conv2dTask,
        task_shape: Tuple[int, int, int, int],
        block_shape: Tuple[int, int, int, int],
        repeat_shape: Tuple[int, int, int, int],
    ):
        self.task_shape: Tuple[int, int, int, int] = task_shape
        self.block_shape: Tuple[int, int, int, int] = block_shape
        self.repeat_shape: Tuple[int, int, int, int] = repeat_shape
        self.thread_shape: Tuple[int, int, int, int] = (1, 1, 1, 1)
        self.block_count: Tuple[int, int, int, int] = tuple_divide(task_shape, block_shape, ceil=True)
        self.repeat_count: Tuple[int, int, int, int] = tuple_divide(block_shape, repeat_shape)
        self.thread_count: Tuple[int, int, int, int] = tuple_divide(repeat_shape, self.thread_shape)
        self.blocks = prod(self.block_count)
        self.threads = prod(self.thread_count)
        strides = task.stride
        kernels = task.inputs[1].const_shape()[2:]
        self.smem_nbytes = (
            block_shape[0]
            * block_shape[1]
            * ((block_shape[2] - 1) * strides[0] + kernels[0])
            * ((block_shape[3] - 1) * strides[1] + kernels[1])
            + block_shape[1] * kernels[0] * kernels[1]
        ) * 4
        if self.smem_nbytes > hidet.cuda.properties().sharedMemPerBlock:
            raise NotSupportedError(self)
        if self.threads > 1024 or self.threads < 32:
            raise NotSupportedError(self)

    @staticmethod
    def schedules_for(task: Conv2dTask):
        # pylint: disable=too-many-nested-blocks
        task_shape: Tuple[int, int, int, int] = tuple(task.outputs[0].const_shape())
        space_level = option.get_option('search_space')
        if space_level == 0:
            sch = DepthwiseConv2dSchedule(
                task=task, task_shape=task_shape, block_shape=(1, 36, 7, 7), repeat_shape=(1, 12, 7, 7)
            )
            return [sch]
        elif space_level == 1:
            raise NotImplementedError()
        elif space_level == 2:
            schedules = []
            for block_c in [1, 2, 4, 6, 12, 24, 36, 48]:
                for block_h in [1, 4, 7, 14, 28]:
                    for repeat_c in [1, 2, 4, 6, 12]:
                        for repeat_h in [1, 4, 7]:
                            if block_c % repeat_c != 0:
                                continue
                            if block_h % repeat_h != 0:
                                continue
                            block_shape = (1, block_c, block_h, block_h)
                            repeat_shape = (1, repeat_c, repeat_h, repeat_h)
                            try:
                                schedules.append(DepthwiseConv2dSchedule(task, task_shape, block_shape, repeat_shape))
                            except NotSupportedError:
                                pass
            return schedules
        else:
            raise ValueError()

    def keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        return [
            ('task', '{}x{}x{}x{}'.format(*self.task_shape)),
            ('block', '{}x{}x{}x{}'.format(*self.block_shape)),
            ('repeat', '{}x{}x{}x{}'.format(*self.repeat_shape)),
            ('thread', '{}x{}x{}x{}'.format(*self.thread_shape)),
        ]

    def derived_keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        return [
            ('block_count', '{}x{}x{}x{}'.format(*self.block_count)),
            ('repeat_count', '{}x{}x{}x{}'.format(*self.repeat_count)),
            ('thread_count', '{}x{}x{}x{}'.format(*self.thread_count)),
            ('blocks', '{}'.format(self.blocks)),
            ('threads', '{}'.format(self.threads)),
        ]


def schedule_depthwise_conv2d(task: Conv2dTask, workding_dir: str) -> IRModule:
    data, weight, output = task.inputs[0], task.inputs[1], task.outputs[0]
    stride_height, stride_width = task.stride
    batch_size, channels, in_height, in_width = data.const_shape()
    _, _, height, width = output.const_shape()
    _, _, kernel_height, kernel_width = weight.const_shape()
    # print(batch_size, channels, height, width, stride_height, kernel_height, in_height, in_width)
    schedules = DepthwiseConv2dSchedule.schedules_for(task)
    ir_modules = [
        schedule_depthwise_conv2d_kernel(
            task,
            sch,
            batch_size,
            channels,
            height,
            width,
            kernel_height,
            kernel_width,
            in_height,
            in_width,
            stride_width,
            stride_height,
        )
        for sch in schedules
    ]
    return resolve_ir_modules(
        ir_modules=ir_modules,
        schedules=schedules,
        func_name=task.name,
        target_device='cuda',
        output_dir=os.path.join(workding_dir, './resolve'),
        parallel=True,
        verbose=True,
    )


def schedule_depthwise_conv2d_kernel(
    task: Conv2dTask,
    sch: DepthwiseConv2dSchedule,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    kernel_height: int,
    kernel_width: int,
    in_height: int,
    in_width: int,
    stride_width: int,
    stride_height: int,
) -> IRModule:
    from hidet.lang import f32, tensor, attr, grid
    from hidet.lang.mapping import spatial, repeat
    from hidet.lang.cuda import threadIdx, blockIdx, syncthreads

    # print(sch)

    block_n, block_c, block_h, block_w = sch.block_shape
    count_n, count_c, count_h, count_w = sch.repeat_count
    group_n, group_c, group_h, group_w = sch.repeat_shape
    tcn, tcc, tch, tcw = sch.thread_count
    bcn, bcc, bch, bcw = sch.block_count
    func_name = task.name + '_grid'

    with hidet.lang.script_module(task) as script_module:

        smem_x_num_elements = (
            block_n
            * block_c
            * ((block_h - 1) * stride_height + kernel_height)
            * ((block_w - 1) * stride_width + kernel_width)
        )
        smem_x_repeats = (smem_x_num_elements + sch.threads - 1) // sch.threads
        smem_w_num_elements = block_c * kernel_height * kernel_width
        smem_w_repeats = (smem_w_num_elements + sch.threads - 1) // sch.threads

        @hidet.lang.script
        def conv2d_grid(
            gmem_x: f32[batch_size, channels, in_height, in_width],
            gmem_w: f32[channels, 1, kernel_height, kernel_width],
            gmem_y: f32[batch_size, channels, height, width],
        ):
            attr.func_name = func_name
            attr.cuda_grid_dim = sch.blocks
            attr.cuda_block_dim = sch.threads

            smem_x = tensor(
                'shared',
                'float32',
                [
                    block_n,
                    block_c,
                    (block_h - 1) * stride_height + kernel_height,
                    (block_w - 1) * stride_width + kernel_width,
                ],
            )
            smem_w = tensor('shared', 'float32', [block_c, kernel_height, kernel_width])
            regs_y = tensor('register', 'float32', [count_n, count_c, count_h, count_w])

            offset_n, offset_c, offset_h, offset_w = spatial(bcn, bcc, bch, bcw).single_task_of(blockIdx.x)
            for worker in repeat(smem_x_repeats).spatial(sch.threads).on(threadIdx.x):
                if worker < smem_x_num_elements:
                    n, c, h, w = spatial(
                        block_n,
                        block_c,
                        (block_h - 1) * stride_height + kernel_height,
                        (block_w - 1) * stride_width + kernel_width,
                    ).single_task_of(worker)
                    nn, cc, hh, ww = (
                        offset_n * block_n + n,
                        offset_c * block_c + c,
                        offset_h * block_h * stride_height + h,
                        offset_w * block_w * stride_width + w,
                    )
                    smem_x[n, c, h, w] = (
                        gmem_x[nn, cc, hh, ww]
                        if nn < batch_size and cc < channels and hh < in_height and ww < in_width
                        else 0.0
                    )

            offset_n, offset_c, offset_h, offset_w = spatial(bcn, bcc, bch, bcw).single_task_of(blockIdx.x)
            for worker in repeat(smem_w_repeats).spatial(sch.threads).on(threadIdx.x):
                if worker < smem_w_num_elements:
                    c, r, s = spatial(block_c, kernel_height, kernel_width).single_task_of(worker)
                    cc = offset_c * block_c + c
                    smem_w[c, r, s] = gmem_w[cc, 0, r, s] if cc < channels else 0.0

            syncthreads()

            offset_n, offset_c, offset_h, offset_w = spatial(bcn, bcc, bch, bcw).single_task_of(blockIdx.x)
            for rn, rc, rh, rw in grid(count_n, count_c, count_h, count_w):
                for tn, tc, th, tw in spatial(tcn, tcc, tch, tcw).on(threadIdx.x):
                    regs_y[rn, rc, rh, rw] = 0.0
                    nn, cc, hh, ww = rn * group_n + tn, rc * group_c + tc, rh * group_h + th, rw * group_w + tw
                    gn, gc, gh, gw = (
                        offset_n * block_n + nn,
                        offset_c * block_c + cc,
                        offset_h * block_h + hh,
                        offset_w * block_w + ww,
                    )
                    for r, s in grid(kernel_height, kernel_width):
                        regs_y[rn, rc, rh, rw] = (
                            regs_y[rn, rc, rh, rw]
                            + smem_x[nn, cc, hh * stride_height + r, ww * stride_width + s] * smem_w[cc, r, s]
                        )
                    if gn < batch_size and gc < channels and gh < height and gw < width:
                        gmem_y[gn, gc, gh, gw] = regs_y[rn, rc, rh, rw]

    return fuse_and_pack(script_module.ir_module(), conv2d_grid, task)
