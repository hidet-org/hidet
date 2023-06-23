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
from typing import List, Tuple, Sequence, Union

from hidet.graph.tensor import Tensor
from hidet.ir import dtypes, logical_and
from hidet.ir.dtypes import float16
from hidet.ir.expr import if_then_else, is_constant, Int
from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.compute import TensorNode
from hidet.ir.task import Task
from hidet.ir.compute import compute, reduce
from hidet.graph.ops.matmul import matmul
from hidet.graph.ops.utils import Operator, input_like
from hidet.graph.ops.utils import normalize_kernel, normalize_stride, tune
from hidet.utils.py import is_power_of_two, cdiv
from .utils import infer_conv2d_shape


class Conv2dGemmImageTransformTask(Task):
    def __init__(self, x: TensorNode, kernel: List[int], stride: List[int], dilations: List[int], groups: int):
        n, c, h, w = x.shape
        kx, ky = kernel
        sx, sy = stride
        dilx, dily = dilations
        p, q = (h - dilx * (kx - 1) - 1) // sx + 1, (w - dily * (ky - 1) - 1) // sy + 1
        self._assert(
            c % groups == 0,
            msg='Conv2d expect in_channels % groups == 0, but got in_channels {} and groups {}'.format(c, groups),
        )
        gc = c // groups  # group channels
        gemm_x = compute(
            name='gemm_x',
            shape=[groups, n * p * q, gc * kx * ky],
            fcompute=lambda g, i, k: x[
                i // (p * q), g * gc + k // (kx * ky), i // q % p * sx + k // ky % kx * dilx, i % q * sy + k % ky * dily
            ],
        )
        super().__init__(name='conv2d_gemm_image_transform', inputs=[x], outputs=[gemm_x])


class Conv2dGemmImageTransformOp(Operator):
    def __init__(self, x: Tensor, kernel, stride, dilations, groups):
        kernel = normalize_kernel(kernel)
        stride = normalize_stride(stride)
        super().__init__(
            inputs=[x],
            attributes={'kernel': kernel, 'stride': stride, 'groups': groups, 'dilations': dilations},
            task=Conv2dGemmImageTransformTask(input_like(x, 'x'), kernel, stride, dilations, groups),
        )


def conv2d_gemm_image_transform(
    x: Tensor, kernel: Sequence[int], stride: Sequence[int], dilations: Sequence[int], groups: int = 1
) -> Tensor:
    return Conv2dGemmImageTransformOp(x, kernel, stride, dilations, groups).get_output(0)


def conv2d_gemm_filter_transform(w: Tensor, groups: int = 1) -> Tensor:
    # weight shape: [oc, c, kx, ky]
    # output shape: [groups, c * kx * ky, ogc] where ogc = oc // groups
    oc, c, kx, ky = w.shape
    # TODO: current assertion mechanism does not cover this use case (only on the task-level)
    if is_constant(oc, groups) and oc % groups != 0:
        raise ValueError('invalid conv2d groups {} for out channels {}'.format(groups, oc))
    ogc = oc // groups
    w = w.reshape([groups, ogc, c, kx, ky])  # [groups, ogc, c, kx, ky]
    w = w.rearrange([[0], [2, 3, 4], [1]])  # [groups, c * kx * ky, ogc]
    return w


def conv2d_gemm_inverse_transform(gemm_y: Tensor, out_height, out_width) -> Tensor:
    # gemm_y shape: [groups, n * p * q, ogc]
    # output shape: [n, oc, p, q] where oc = groups * ogc
    p, q = out_height, out_width
    groups, npq, ogc = gemm_y.shape
    # TODO: current assertion mechanism does not cover this use case (only on the task-level)
    if is_constant(npq, p, q) and npq % (p * q) != 0:
        raise ValueError('invalid conv2d output shape {} for height {} and width {}'.format(npq, p, q))
    n = npq // (p * q)
    y = gemm_y.reshape([groups, n, p, q, ogc])
    y = y.rearrange([[1], [0, 4], [2], [3]])
    return y


def conv2d_gemm(data: Tensor, weight: Tensor, stride, dilations: List[int], groups: int = 1) -> Tensor:
    gemm_x = conv2d_gemm_image_transform(
        data, kernel=weight.shape[2:], stride=stride, dilations=dilations, groups=groups
    )
    gemm_w = conv2d_gemm_filter_transform(weight, groups=groups)
    gemm_y = matmul(gemm_x, gemm_w, require_prologue=True)

    y_shape = infer_conv2d_shape(data.shape, weight.shape, stride, groups, dilations)
    y = conv2d_gemm_inverse_transform(gemm_y, out_height=y_shape[2], out_width=y_shape[3])
    return y



class Conv2dGemmFp16PretransformTask(Task):
    def __init__(self, img: TensorNode, padding: Union[int, Tuple[int, int]], pad_value: float, channel_mul_8: bool = False):
        self._assert(len(img.shape) == 4, "expected images to have 4 dimensions")
        if isinstance(padding, tuple):
            self._assert(len(padding) == 2, "padding must be a tuple of 2 ints")
            pad_h = padding[0]
            pad_w = padding[1]
        else:
            pad_h = padding
            pad_w = padding
        self._assert(pad_h >= 0, f"padding size must be greater or equal to 0, got {padding}")
        self._assert(pad_w >= 0, f"padding size must be greater or equal to 0, got {padding}")

        n, c, h, w = img.shape
        if channel_mul_8:
            new_channel = cdiv(c, 8) * 8
        else:
            new_channel = c
        self.channel_mul_8 = channel_mul_8
        self.pad_value = pad_value
        self.pad_h = pad_h
        self.pad_w = pad_w

        y = compute(
            name="y",
            shape=[n, h + pad_h * 2, w + pad_w * 2, new_channel],
            fcompute=lambda ni, hi, wi, ci: if_then_else(
                ci < c,
                if_then_else(
                    logical_and(pad_h <= hi, hi < h + pad_h, pad_w <= wi, wi < w + pad_w),
                    img[ni, ci, hi - pad_h, wi - pad_w],
                    float16(pad_value)
                ),
                float16(0.0)
            )
        )
        super().__init__(
            name='Conv2dGemmFp16PretransformTask',
            inputs=[img],
            outputs=[y],
            attributes={
                'pad_value': pad_value,
                'pad_h': pad_h,
                'pad_w': pad_w,
                'channel_mul_8': channel_mul_8
            }
        )
    
    def allow_prologue(self) -> bool:
        return True

    def allow_epilogue(self) -> bool:
        return True

    def implement_cuda(self, working_dir: str) -> List[IRModule]:
        return tune.extract_ir_modules(self.schedule)
    
    @tune.space(2, block_n = [8, 16, 32, 64, 128], block_m = [8, 16, 32, 64, 128])    
    @tune.space(1, block_n = [16, 32], block_m = [16, 32])
    def schedule(self, block_n = 32, block_m = 32):
        import hidet
        from hidet.lang import attrs, view, u32, tensor_pointer, grid
        from hidet.lang.cuda import shared_tensor, syncthreads, cp_async, cp_async_wait_all, threadIdx, blockIdx

        N, C, H, W = self.inputs[0].shape
        _, HN, WN, CN = self.outputs[0].shape

        tune.check(block_n * block_m <= 1024)
        
        tiles_hw = cdiv(HN * WN, block_n)
        tiles_c = cdiv(CN, block_m)

        with hidet.script_module() as module:
            @hidet.script
            def conv2dfp16_pretransform_kernel(img: float16[N, C, H, W], y: float16[N, HN, WN, CN]):
                attrs.cuda.block_dim = block_n, block_m, 1
                attrs.cuda.grid_dim = tiles_hw, tiles_c, N

                smem = shared_tensor(float16, [block_n, block_m])
                offset_hw = blockIdx.x * block_n + threadIdx.x
                offset_c = blockIdx.y * block_m + threadIdx.y

                offset_hn = offset_hw // WN
                offset_wn = offset_hw % WN

                val = float16(0.0)
                if offset_c < C:
                    if self.pad_h <= offset_hn < H + self.pad_h and self.pad_w <= offset_wn < W + self.pad_w:
                        val = img[blockIdx.z, offset_c, offset_hn - self.pad_h, offset_wn - self.pad_w]
                    else:
                        val = self.pad_value
                else:
                    val = float16(0.0)
                

                smem[threadIdx.x, threadIdx.y] = val
                syncthreads()
                tid = threadIdx.x + threadIdx.y * block_n
                row_idx = tid % block_m
                col_idx = tid // block_m

                offset_c = row_idx + blockIdx.y * block_m
                offset_hw = col_idx + blockIdx.x * block_n
                offset_hn = offset_hw // WN
                offset_wn = offset_hw % WN

                if offset_hn < HN and offset_c < CN:
                    y[blockIdx.z, offset_hn, offset_wn, offset_c] = smem[col_idx, row_idx]
        ir_module = module.ir_module()
        assert isinstance(conv2dfp16_pretransform_kernel, Function)

        return ir_module  
                

class Conv2dGemmFp16PretransformV2Task(Task):
    def __init__(self, img: TensorNode, padding: Union[int, Tuple[int, int]], pad_value: float, channel_mul_8: bool = False):
        self._assert(len(img.shape) == 4, "expected images to have 4 dimensions")
        if isinstance(padding, tuple):
            self._assert(len(padding) == 2, "padding must be a tuple of 2 ints")
            pad_h = padding[0]
            pad_w = padding[1]
        else:
            pad_h = padding
            pad_w = padding
        self._assert(pad_h >= 0, f"padding size must be greater or equal to 0, got {padding}")
        self._assert(pad_w >= 0, f"padding size must be greater or equal to 0, got {padding}")

        n, c, h, w = img.shape
        if channel_mul_8:
            new_channel = cdiv(c, 8) * 8
        else:
            new_channel = c
        self.channel_mul_8 = channel_mul_8
        self.pad_value = pad_value
        self.pad_h = pad_h
        self.pad_w = pad_w

        y = compute(
            name="y",
            shape=[n, h + pad_h * 2, w + pad_w * 2, new_channel],
            fcompute=lambda ni, hi, wi, ci: if_then_else(
                ci < c,
                if_then_else(
                    logical_and(pad_h <= hi, hi < h + pad_h, pad_w <= wi, wi < w + pad_w),
                    img[ni, ci, hi - pad_h, wi - pad_w],
                    float16(pad_value)
                ),
                float16(0.0)
            )
        )
        super().__init__(
            name='Conv2dGemmFp16PretransformTask',
            inputs=[img],
            outputs=[y],
            attributes={
                'pad_value': pad_value,
                'pad_h': pad_h,
                'pad_w': pad_w,
                'channel_mul_8': channel_mul_8
            }
        )
    
    def allow_prologue(self) -> bool:
        return True

    def allow_epilogue(self) -> bool:
        return True

    def implement_cuda(self, working_dir: str) -> List[IRModule]:
        return tune.extract_ir_modules(self.schedule)
    
    def load_dtype(self, load_width):
        from hidet.lang import attrs, view, u16, u32, u64, tensor_pointer, grid
        if load_width == 1:
            load_type = u16
        elif load_width == 2:
            load_type = u32
        elif load_width == 4:
            load_type = u64
        else:
            load_type = None
        return load_type
        
    @tune.space(2, block_n = [4, 8, 16, 32, 64, 128], block_m = [8, 16, 32, 64, 128], load_width = [1, 2, 4])    
    @tune.space(1, block_n = [8, 16, 32], block_m = [16, 32], load_width = [1, 2])
    def schedule(self, block_n = 8, block_m = 32, load_width = 1, write_width = 1): # load_width is multiples of f16
        import hidet
        from hidet.lang import attrs, view, u16, u32, u64, tensor_pointer, grid
        from hidet.lang.cuda import shared_tensor, register_tensor, syncthreads, cp_async, cp_async_wait_all, threadIdx, blockIdx, gridDim

        N, C, H, W = self.inputs[0].shape
        _, HN, WN, CN = self.outputs[0].shape

        tune.check(block_n * block_m <= 1024)
        
        last_dim = H * W
        tune.check(last_dim % load_width == 0)
        tune.check(block_n % load_width == 0)
        load_dtype = self.load_dtype(load_width)
        tune.check(load_dtype is not None)

        write_width = load_width
        write_dtype = self.load_dtype(write_width)
        tune.check(write_dtype is not None)
        tune.check(CN % write_width == 0 and block_m % write_width == 0)

        tune.check(block_m % write_width == 0)
        tiles_hw = cdiv(H * W, block_n * load_width)
        tiles_c = cdiv(C, block_m)

        with hidet.script_module() as module:
            @hidet.script
            def conv2dfp16_pretransform_kernelv2(img: float16[N, C, H * W], y: float16[N, HN * WN, CN]):
                attrs.cuda.block_dim = block_n, block_m, 1
                attrs.cuda.grid_dim = tiles_hw, tiles_c, N

                bid = blockIdx.x + blockIdx.y * tiles_hw + blockIdx.z * tiles_hw * tiles_c
                global_stride = tiles_hw * tiles_c * N * block_n * block_m * write_width
                tid = (threadIdx.x + threadIdx.y * block_n + bid * block_n * block_m) * write_width

                write_ptr = tensor_pointer(write_dtype, [N * CN * HN * WN // write_width])
                write_ptr = y

                # pack larger dtype with pad value
                # is is valid because last dimension CN is a multiple of write_width
                pad_regs = register_tensor(write_dtype, [1])
                pad_ptr = tensor_pointer(float16, [write_width])
                pad_ptr = pad_regs
                for i in range(write_width):
                    pad_ptr[i] = float16(self.pad_value)
                
                # fill padding first
                while (tid < N * CN * HN * WN // write_width):
                    c_idx = (tid // (HN * WN)) % CN
                    if c_idx < C:
                        write_ptr[tid] = pad_regs
                    else:
                        write_ptr[tid] = 0
                    tid += global_stride
                
                smem = shared_tensor(float16, [block_n * load_width, block_m])

                offset_hw = blockIdx.x * block_n + threadIdx.x
                offset_c = blockIdx.y * block_m + threadIdx.y
                load_ptr = tensor_pointer(load_dtype, [N, C, H * W // load_width])
                load_ptr = img

                regs = register_tensor(float16, [load_width])
                reg_ptr = tensor_pointer(load_dtype, [1])
                reg_ptr = regs

                if offset_hw < H * W // load_width:
                    reg_ptr[0] = load_ptr[blockIdx.z, offset_c, offset_hw]
                for i in range(load_width):
                    smem[threadIdx.x * load_width + i, threadIdx.y] = regs[i]
                syncthreads()

                tid = threadIdx.x + threadIdx.y * block_n
                row_idx = tid % (block_m // write_width)
                col_idx = tid // (block_m // write_width)
                offset_hw = blockIdx.x * block_n * load_width + col_idx
                offset_c = (blockIdx.y * block_m + row_idx) * write_width

                initial_pad_offset = CN * (self.pad_h * HN * WN + 2 * self.pad_h + self.pad_w) + self.pad_w * CN
                im_block = offset_hw // W
                w_pad = im_block * 2 * self.pad_w * CN

                offset_hw = initial_pad_offset + w_pad + offset_hw % W
                if offset_hw < HN * WN and offset_c < CN:
                    batch_offset = blockIdx.z * HN * WN * CN
                    global_offset = batch_offset + offset_hw * CN + offset_c
                    write_ptr[global_offset // write_width] = smem[col_idx, row_idx]


        ir_module = module.ir_module()
        assert isinstance(conv2dfp16_pretransform_kernelv2, Function)

        return ir_module  

class Conv2dGemmFp16PretransformV3Task(Task):
    def __init__(self, img: TensorNode, padding: Union[int, Tuple[int, int]], pad_value: float, channel_mul_8: bool = False):
        self._assert(len(img.shape) == 4, "expected images to have 4 dimensions")
        if isinstance(padding, tuple):
            self._assert(len(padding) == 2, "padding must be a tuple of 2 ints")
            pad_h = padding[0]
            pad_w = padding[1]
        else:
            pad_h = padding
            pad_w = padding
        self._assert(pad_h >= 0, f"padding size must be greater or equal to 0, got {padding}")
        self._assert(pad_w >= 0, f"padding size must be greater or equal to 0, got {padding}")

        n, c, h, w = img.shape
        if channel_mul_8:
            new_channel = cdiv(c, 8) * 8
        else:
            new_channel = c
        self.channel_mul_8 = channel_mul_8
        self.pad_value = pad_value
        self.pad_h = pad_h
        self.pad_w = pad_w

        y = compute(
            name="y",
            shape=[n, h + pad_h * 2, w + pad_w * 2, new_channel],
            fcompute=lambda ni, hi, wi, ci: if_then_else(
                ci < c,
                if_then_else(
                    logical_and(pad_h <= hi, hi < h + pad_h, pad_w <= wi, wi < w + pad_w),
                    img[ni, ci, hi - pad_h, wi - pad_w],
                    float16(pad_value)
                ),
                float16(0.0)
            )
        )
        super().__init__(
            name='Conv2dGemmFp16PretransformV3Task',
            inputs=[img],
            outputs=[y],
            attributes={
                'pad_value': pad_value,
                'pad_h': pad_h,
                'pad_w': pad_w,
                'channel_mul_8': channel_mul_8
            }
        )
    
    def allow_prologue(self) -> bool:
        return True

    def allow_epilogue(self) -> bool:
        return True

    def implement_cuda(self, working_dir: str) -> List[IRModule]:
        return tune.extract_ir_modules(self.schedule)
    
    def load_dtype(self, load_width):
        from hidet.lang import attrs, view, u16, u32, u64, tensor_pointer, grid
        if load_width == 1:
            load_type = u16
        elif load_width == 2:
            load_type = u32
        elif load_width == 4:
            load_type = u64
        else:
            load_type = None
        return load_type
        
    @tune.space(2, block_n = [4, 8, 16, 32, 64, 128], block_m = [8, 16, 32, 64, 128], load_width = [1, 2, 4])    
    @tune.space(1, block_n = [8, 16, 32], block_m = [16, 32], load_width = [1, 2])
    def schedule(self, block_n = 8, block_m = 32, load_width = 1): # load_width is multiples of f16
        import hidet
        from hidet.lang import attrs, view, u16, u32, u64, tensor_pointer, grid
        from hidet.lang.cuda import shared_tensor, register_tensor, syncthreads, cp_async, cp_async_wait_all, threadIdx, blockIdx, gridDim

        N, C, H, W = self.inputs[0].shape
        _, HN, WN, CN = self.outputs[0].shape

        tune.check(block_n * block_m <= 1024)
        
        last_dim = H * W
        tune.check(last_dim % load_width == 0)
        tune.check(block_n % load_width == 0)
        load_dtype = self.load_dtype(load_width)
        tune.check(load_dtype is not None)

        write_width = load_width
        write_dtype = self.load_dtype(write_width)
        tune.check(write_dtype is not None)
        tune.check(CN % write_width == 0 and block_m % write_width == 0)
        tune.check(C % write_width == 0)

        tune.check(block_m % write_width == 0)
        tiles_hw = cdiv(H * W, block_n * load_width)
        tiles_c = cdiv(C, block_m)

        with hidet.script_module() as module:
            @hidet.script
            def write_pad(y: write_dtype[N, HN * WN * CN // write_width]):
                # pack larger dtype with pad value
                # is is valid because last dimension CN is a multiple of write_width
                pad_regs = register_tensor(write_dtype, [1])
                pad_ptr = tensor_pointer(float16, [write_width])
                pad_ptr = pad_regs
                for i in range(write_width):
                    pad_ptr[i] = float16(self.pad_value)
                
                stride = tiles_hw * tiles_c * block_n * block_m
                bid = blockIdx.x + blockIdx.y * tiles_hw
                tid = threadIdx.x + threadIdx.y * block_n + bid * block_n * block_m

                while (tid < HN * WN * CN // write_width):
                    row_id = tid // (CN // write_width)
                    col_id = tid % (CN // write_width)
                    middle_square = False
                    if row_id >= WN * self.pad_h and row_id < HN * WN - WN * self.pad_h:
                        r_id = (row_id - WN * self.pad_h) % WN
                        if r_id >= self.pad_w and r_id < self.pad_w + W:
                            middle_square = True
                    if not middle_square:
                        if col_id < C:
                            y[blockIdx.z, tid] = pad_regs[0]
                        else:
                            y[blockIdx.z, tid] = 0
                    tid += stride
            

            @hidet.script
            def conv2dfp16_pretransform_kernelv2(img: float16[N, C, H * W], y: float16[N, HN * WN, CN]):
                attrs.cuda.block_dim = block_n, block_m, 1
                attrs.cuda.grid_dim = tiles_hw, tiles_c, N

                write_ptr = tensor_pointer(write_dtype, [N * CN * HN * WN // write_width])
                write_ptr = y
                write_pad(write_ptr)

                smem = shared_tensor(float16, [block_n * load_width, block_m])

                offset_hw = (blockIdx.x * block_n + threadIdx.x) * load_width
                offset_c = blockIdx.y * block_m + threadIdx.y
                load_ptr = tensor_pointer(load_dtype, [N, C, H * W // load_width])
                load_ptr = img

                regs = register_tensor(float16, [load_width])
                reg_ptr = tensor_pointer(load_dtype, [1])
                reg_ptr = regs

                if offset_hw < H * W and offset_c < C:
                    reg_ptr[0] = load_ptr[blockIdx.z, offset_c, offset_hw // load_width]
                for i in range(load_width):
                    smem[threadIdx.x * load_width + i, threadIdx.y] = regs[i]
                syncthreads()

                tid = threadIdx.x + threadIdx.y * block_n
                row_idx = tid % (block_m // write_width)
                col_idx = tid // (block_m // write_width)
                offset_hw = blockIdx.x * block_n * load_width + col_idx
                offset_c = blockIdx.y * block_m + row_idx * write_width

                initial_pad_offset = self.pad_h * WN * CN + self.pad_w * CN
                im_block = offset_hw // W
                w_pad = im_block * 2 * self.pad_w * CN

                smem_ptr = tensor_pointer(write_dtype, [block_n * write_width, block_m // write_width])
                smem_ptr = smem

                if offset_hw < H * W and offset_c < C:
                    batch_offset = blockIdx.z * HN * WN * CN
                    global_offset = batch_offset + offset_hw * CN + offset_c + initial_pad_offset + w_pad
                    write_ptr[global_offset // write_width] = smem_ptr[col_idx, row_idx]

        ir_module = module.ir_module()
        assert isinstance(conv2dfp16_pretransform_kernelv2, Function)

        return ir_module  

class Conv2dGemmFp16PretransformV3OP(Operator):
    def __init__(self, img: Tensor, padding: Union[int, Tuple[int, int]], pad_value=0.0, channel_mul_8: bool = False):
        super().__init__(
            inputs=[img],
            attributes={
                'pad_value': pad_value,
                'padding': padding,
                'channel_mul_8': channel_mul_8
            },
            task=Conv2dGemmFp16PretransformV3Task(
                input_like(img, 'img'),
                padding,
                pad_value,
                channel_mul_8
            )
        )

def pre_transform_imgv3(img: Tensor, padding: Union[int, Tuple[int, int]], pad_value=0.0, make_multiple_8=False):
    return Conv2dGemmFp16PretransformV3OP(img, padding, pad_value, make_multiple_8).get_output(0)

class Conv2dGemmFp16Task(Task):
    def __init__(
        self,
        img: TensorNode,
        weight: TensorNode,
        orig_weight_shape: List[int],
        stride: List[int],
        dilations: List[int],
        groups: int = 1,
        parallel_k_parts: int = 1,
        disable_cp_async: bool = False,
    ):
        # Channel last
        # This kernel expects the weight to be transformed in the following way:
        # weight.shape [OC, WC, KY, KX] -> [KY * KX * WC, OC]
        self._assert(len(img.shape) == 4, f"expect img shape to be in NHWC format, got {img.shape}")
        self._assert(
            len(weight.shape) == 2,
            f"expected weight to be transformed from [OC, WC, KY, kX] to [KY * KX * WC, OC], got {weight.shape}",
        )
        self._assert(img.type.dtype == float16 and weight.type.dtype == float16, 'Both inputs must be float16 tensors')

        self.groups = groups
        self.dilations = dilations
        self.stride = stride
        self.img_shape = img.shape
        self.orig_weight_shape = orig_weight_shape
        self.disable_cp_async = disable_cp_async

        DILY, DILX = dilations
        STRY, STRX = stride
        # orig_weight_shape == [OC, WC, KY, KX]
        N, H, W, C = img.shape
        OC, WC, KY, KX = orig_weight_shape

        self._assert(C % groups == 0, f"expected input channels to be divisible by groups, got {C}")
        self._assert(OC % groups == 0, f"expected output channels to be divisible by groups, got {OC}")
        self._assert(
            groups * WC == C,
            f"expected groups * WC == C, got groups: {groups}, WC: {WC}, C: {C}; make sure the image is channels last!",
        )
        self._assert(
            DILX > 0 and DILY > 0 and STRX > 0 and STRY > 0,
            f"dilations and strides must be larger than 0, got strides={(STRY, STRX)}, dilations={(DILY, DILX)}",
        )
        self._assert(parallel_k_parts > 0, "expected parallel_k_parts to be greater than 0")
        self._assert(H >= KY and W >= KX, "expected image dimensions to be greater than filter dimensions")

        OUT_H = (H - DILY * (KY - 1) - 1) // STRY + 1
        OUT_W = (W - DILX * (KX - 1) - 1) // STRX + 1

        self.out_shape = [parallel_k_parts, N, OUT_H, OUT_W, OC]

        k_size = WC * KY * KX
        k_part_extent = cdiv(k_size, parallel_k_parts)

        # k is tiled from [ky, kx, wc]
        #   this compute definition is not ever going to be used, since we always
        #   implement cuda on fp16
        def f_compute(k, ni, hi, wi, oci):
            wci = k % WC
            ky = (k // (WC * KX)) % KY
            kx = (k // WC) % KX
            out_group_size = OC // groups
            return (
                img[ni, hi * STRY + ky * DILY, wi * STRX + kx * DILX, (oci // out_group_size) * WC + wci]
                * weight[k, oci]
            )

        c = compute(
            name='c',
            shape=self.out_shape,
            fcompute=lambda kpi, ni, hi, wi, oci: reduce(
                shape=[k_part_extent],
                fcompute=lambda k: if_then_else(
                    kpi * k_part_extent + k < k_size, f_compute(kpi * k_part_extent + k, ni, hi, wi, oci), float16(0.0)
                ),
                reduce_type='sum',
            ),
        )

        if not disable_cp_async:
            name = 'conv_gemm_fp16_pk'
        else:
            name = 'conv_gemm_fp16_pk_disable_cp_async'
        super().__init__(
            name=name,
            inputs=[img, weight],
            outputs=[c],
            attributes={
                'stride': stride,
                'dilations': dilations,
                'orig_weight_shape': orig_weight_shape,
                'groups': groups,
                'parallel_k_parts': parallel_k_parts,
                'disable_cp_async': disable_cp_async,
            },
        )

    def allow_prologue(self) -> bool:
        return self.disable_cp_async

    def allow_epilogue(self) -> bool:
        return True

    def implement_cuda(self, working_dir: str) -> List[IRModule]:
        return tune.extract_ir_modules(self.schedule)

    @tune.space(
        2,
        block_m=[32, 64, 128, 256],
        block_n=[32, 64, 128, 256],
        block_k=[8, 16, 32, 64, 128],
        warp_m=[16, 32, 48, 64],
        warp_n=[16, 32, 48, 64],
        warp_k=[8, 16, 32, 64],
        mma=['m16n8k16'],
    )
    @tune.space(1, block_m=[128], block_n=[128], block_k=[16], warp_m=[64], warp_n=[64], warp_k=[16], mma=['m16n8k16'])
    def schedule(
        self, block_m=64, block_n=128, block_k=16, warp_m=32, warp_n=64, warp_k=16, mma: str = 'm16n8k16'
    ) -> IRModule:
        # pylint: disable=unused-variable
        import hidet
        from hidet.ir.type import tensor_type
        from hidet.lang import attrs, view, u32, tensor_pointer, grid
        from hidet.lang.layout import row_layout
        from hidet.lang.mapping import spatial, auto_map
        from hidet.lang.cuda import blockIdx, threadIdx, syncthreads, dynamic_shared_memory
        from hidet.lang.cuda import MmaConfig, mma_sync, cp_async, cp_async_wait_all, ldmatrix
        from hidet.lang.cuda import register_tensor

        DILY, DILX = self.dilations
        STRY, STRX = self.stride
        N, H, W, C = self.img_shape
        OC, WC, KY, KX = self.orig_weight_shape
        GROUPS = self.groups

        GROUP_C = C // GROUPS
        GROUP_OC = OC // GROUPS
        # actual shape = [KY * KX * WC, OC]

        K_PARTS, _, OUT_H, OUT_W, _ = self.out_shape

        # the problem is that the block_k is not contiguous across the channel dimension, depending on certain
        # configuration of parameters
        TILES_K = cdiv(GROUP_C, block_k) * KX * KY
        K_TILES_PER_BLOCK = cdiv(TILES_K, K_PARTS)  # number of tiles assigned to each block

        # schedule parameters
        mma_configs = {'m16n8k8': MmaConfig.m16n8k8_f16_f16(), 'm16n8k16': MmaConfig.m16n8k16_f16_f16()}
        tune.check(mma in mma_configs)
        mma_config = mma_configs[mma]

        # number of elements each warp handles at once
        mma_m, mma_n, mma_k = mma_config.m, mma_config.n, mma_config.k  # 16, 8, 16
        # number of warps in each dimension
        warp_count_m, warp_count_n, warp_count_k = block_m // warp_m, block_n // warp_n, block_k // warp_k
        # number of repeats that each warp has to do
        mma_count_m, mma_count_n, mma_count_k = warp_m // mma_m, warp_n // mma_n, warp_k // mma_k
        threads = warp_count_m * warp_count_n * warp_count_k * 32

        grid_dim: Tuple[Int, Int, Int] = cdiv(OUT_H * OUT_W, block_m), cdiv(GROUP_OC, block_n), N * K_PARTS * GROUPS
        dynamic_smem_bytes = max(2 * (block_m + block_n) * block_k * 2, block_m * block_n * 2)

        ### checks
        tune.check(block_m % warp_m == block_n % warp_n == block_k % warp_k == 0, 'warp dims divide block dims')
        tune.check(warp_m % mma_m == warp_n % mma_n == warp_k % mma_k == 0, 'mma dims divide warp dims')
        tune.check(threads <= 1024, 'threads in a block <= 1024')
        maximum_smem_bytes = 49152
        tune.check(dynamic_smem_bytes <= maximum_smem_bytes, 'dynamic shared memory <= 49152')

        tune.check(block_n % 64 == 0, 'block_n must be multiple of 64, required by async gmem -> smem loading')
        tune.check(block_k % 8 == 0)
        tune.check(is_power_of_two(block_k // 8))

        smem_img_type = tensor_type(
            'float16',
            shape=[block_m, block_k],
            layout=row_layout(block_m, block_k // 8).swizzle(1) * row_layout(1, 8)
            # layout=row_layout(block_m, block_k)
        )
        smem_weight_type = tensor_type(
            'float16',
            shape=[block_k, block_n],
            layout=row_layout(block_k // 8, block_n // 64) * row_layout(8, 8).swizzle(1) * row_layout(1, 8),
            # layout=row_layout(block_k, block_n)
        )
        load_smem_a_map = auto_map(block_m, block_k // 8, workers=threads, on_fail=lambda msg: tune.check(False, msg))
        load_smem_b_map = auto_map(block_k, block_n // 8, workers=threads, on_fail=lambda msg: tune.check(False, msg))
        store_smem_c_map = auto_map(block_m, block_n, workers=threads, on_fail=lambda msg: tune.check(False, msg))

        with hidet.script_module() as module:

            @hidet.script
            def load_regs_a(mi: int, k1: int, smem_a: smem_img_type, regs_a: float16[mma_config.a_elements]):
                # mi - mma_count_m
                # k1 - mma_count_k
                # block - [warp_count_m, warp_count_n, warp_count_k]
                # each warp handles: [warp_m, warp_k] == [mma_count_m * mma_m, mma_count_k * mma_k]
                # smem_a - [block_m, block_k]
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                wk = warp_id % warp_count_k
                wi = warp_id // (warp_count_k * warp_count_n)
                p = lane_id % 16
                q = lane_id // 16
                row_addr = ~smem_a[wi * warp_m + mi * mma_m + p, wk * warp_k + k1 * mma_k + q * 8]
                b32_regs = view(regs_a, u32[4])
                ldmatrix(
                    regs=[b32_regs[0], b32_regs[1], b32_regs[2], b32_regs[3]],
                    smem_addr=row_addr,
                    shared_space_addr=False,
                    trans=False,
                )

            @hidet.script
            def load_regs_b(mj: int, k1: int, smem_b: smem_weight_type, regs_b: float16[mma_config.b_elements]):
                # mj - mma_count_n
                # k1 - mma_count_k
                # each warp handles: [warp_k, warp_n] == [mma_count_k * mma_k, mma_count_n * mma_n]
                # smem_b - [block_k, block_n]
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                wj = (warp_id // warp_count_k) % warp_count_n
                wk = warp_id % warp_count_k

                p = lane_id % 16
                # have not used q as we only use the address of the first 16 threads to load 2 of 8x8 f16 matrix.
                row_addr = ~smem_b[wk * warp_k + k1 * mma_k + p, wj * warp_n + mj * mma_n]
                regs = view(regs_b, u32[2])
                ldmatrix(regs=[regs[0], regs[1]], smem_addr=row_addr, trans=True)

            @hidet.script
            def warp_mma(
                regs_a: float16[mma_config.a_elements],
                regs_b: float16[mma_config.b_elements],
                regs_c: float16[mma_config.c_elements],
            ):
                mma_sync(mma_config, regs_a, regs_b, regs_c)

            @hidet.script
            def load_smem_img(k0: int, img: float16[N, H, W, C], smem_img: smem_img_type):
                offset_m = blockIdx.x * block_m  # this is the output pixel index

                # the current global tile index, where each tile is of size [block_k]
                k_tile_idx = (blockIdx.z // (N * GROUPS)) * K_TILES_PER_BLOCK + k0

                batch_idx = (blockIdx.z // GROUPS) % N

                group_idx = blockIdx.z % GROUPS
                num_tiles_per_channel = cdiv(GROUP_C, block_k)
                channel_idx = k_tile_idx // num_tiles_per_channel
                channel_group_offset = (k_tile_idx % num_tiles_per_channel) * block_k
                filter_y = channel_idx // KX
                filter_x = channel_idx % KX

                for i, k_seg in load_smem_a_map.on(threadIdx.x):
                    k = k_seg * 8

                    # tiling the output image spatial dimension [OUT_H, OUT_W]
                    img_spatial = i + offset_m
                    oh_idx = img_spatial // OUT_W
                    ow_idx = img_spatial % OUT_W

                    # these are the input pixel coordinates
                    ih_idx = oh_idx * STRY + filter_y * DILY
                    iw_idx = ow_idx * STRX + filter_x * DILX

                    channel_offset = channel_group_offset + k + group_idx * GROUP_C

                    src_size = 0
                    if iw_idx < W and ih_idx < H and channel_group_offset + k < GROUP_C:
                        src_size = min(8, GROUP_C - (channel_group_offset + k))

                    # a bit strange, the two branches should be the same, but gives different results
                    #   but only when GROUP_C % 8 != 0
                    if GROUP_C % 8 == 0 and not self.disable_cp_async:
                        cp_async(
                            ~smem_img[i, k],
                            ~img[batch_idx, ih_idx, iw_idx, channel_offset],
                            cp_size=16,
                            src_size=src_size * 2,
                            cache_level='global',
                        )
                    else:
                        for ki in range(src_size):
                            smem_img[i, k + ki] = img[batch_idx, ih_idx, iw_idx, channel_offset + ki]
                        for ki in range(8 - src_size):
                            smem_img[i, k + ki + src_size] = 0

            @hidet.script
            def load_smem_weight(k0: int, weight: float16[KX * KY * WC, OC], smem_weight: smem_weight_type):
                group_idx = blockIdx.z % GROUPS
                offset_n_group = blockIdx.y * block_n

                k_tile_idx = (blockIdx.z // (N * GROUPS)) * K_TILES_PER_BLOCK + k0
                offset_k = 0

                num_tiles_per_channel = cdiv(GROUP_C, block_k)
                channel_idx = k_tile_idx // num_tiles_per_channel
                channel_offset = k_tile_idx % num_tiles_per_channel
                filter_y = channel_idx // KX
                filter_x = channel_idx % KX
                offset_k = filter_y * KX * WC + filter_x * WC + channel_offset * block_k

                for k, j_seg in load_smem_b_map.on(threadIdx.x):
                    j = j_seg * 8
                    # we don't need to mask channel wise, since we have already done so for the img
                    #   so the extra bits are not relevant when multipled by zeros
                    offset_n = offset_n_group + group_idx * GROUP_OC
                    src_size = (
                        0
                        if (offset_n_group + j >= GROUP_OC or offset_k + k >= KY * KX * WC)
                        else min(8, GROUP_OC - (offset_n_group + j))
                    )

                    # also quite strange; the two branches should be the same, but gives different
                    #   results when GROUP_OC % 8 != 0
                    if GROUP_OC % 8 == 0 and not self.disable_cp_async:
                        cp_async(
                            ~smem_weight[k, j],
                            ~weight[offset_k + k, offset_n + j],
                            cp_size=16,
                            src_size=src_size * 2,
                            cache_level='global',
                        )
                    else:
                        for ji in range(src_size):
                            smem_weight[k, j + ji] = weight[offset_k + k, offset_n + j + ji]
                        for ji in range(8 - src_size):
                            smem_weight[k, j + ji + src_size] = 0

            @hidet.script
            def conv2d_gemm_f16_kernel(
                img: float16[N, H, W, C], weight: float16[KX * KY * WC, OC], res: float16[K_PARTS, N, OUT_H, OUT_W, OC]
            ):
                # matrix multiplication, using mma instruction
                attrs.cuda.grid_dim = grid_dim
                attrs.cuda.block_dim = threads
                # the second 2 means '2 bytes per float16'
                attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes
                # smem_storage = dyn_smem_storage
                smem_img = tensor_pointer(
                    'float16', shape=[2, block_m, block_k], layout=row_layout(2) + smem_img_type.layout
                )
                smem_weight = tensor_pointer(
                    'float16', shape=[2, block_k, block_n], layout=row_layout(2) + smem_weight_type.layout
                )
                smem_img = dynamic_shared_memory(byte_offset=0, dtype=float16)
                smem_weight = dynamic_shared_memory(byte_offset=2 * block_m * block_k * 2, dtype=float16)
                regs_a = register_tensor(float16, [2, mma_count_m, mma_config.a_elements])
                regs_b = register_tensor(float16, [2, mma_count_n, mma_config.b_elements])
                regs_c = register_tensor(float16, [mma_count_m, mma_count_n, mma_config.c_elements])

                for i, j, p in grid(mma_count_m, mma_count_n, mma_config.c_elements):
                    regs_c[i, j, p] = 0.0

                load_smem_img(0, img, ~smem_img[0, 0, 0])
                load_smem_weight(0, weight, ~smem_weight[0, 0, 0])
                if GROUP_OC % 8 == 0 and not self.disable_cp_async:
                    cp_async_wait_all()

                syncthreads()
                for k0 in range(K_TILES_PER_BLOCK):
                    load_smem_img(k0 + 1, img, ~smem_img[(k0 + 1) % 2, 0, 0])
                    load_smem_weight(k0 + 1, weight, ~smem_weight[(k0 + 1) % 2, 0, 0])

                    for mi in range(mma_count_m):
                        load_regs_a(mi, 0, ~smem_img[k0 % 2, 0, 0], ~regs_a[0, mi, 0])
                    for mj in range(mma_count_n):
                        load_regs_b(mj, 0, ~smem_weight[k0 % 2, 0, 0], ~regs_b[0, mj, 0])
                    for mk in range(mma_count_k):
                        if mk + 1 < mma_count_k:
                            for mi in range(mma_count_m):
                                load_regs_a(mi, mk + 1, ~smem_img[k0 % 2, 0, 0], ~regs_a[(mk + 1) % 2, mi, 0])
                            for mj in range(mma_count_n):
                                load_regs_b(mj, mk + 1, ~smem_weight[k0 % 2, 0, 0], ~regs_b[(mk + 1) % 2, mj, 0])
                        for mi, mj in grid(mma_count_m, mma_count_n):
                            warp_mma(~regs_a[mk % 2, mi, 0], ~regs_b[mk % 2, mj, 0], ~regs_c[mi, mj, 0])
                    if GROUP_OC % 8 == 0 and not self.disable_cp_async:
                        cp_async_wait_all()
                    syncthreads()

                # store back
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                offset_m, offset_n = blockIdx.x * block_m, blockIdx.y * block_n
                k_part_idx = blockIdx.z // (N * GROUPS)
                batch_idx = (blockIdx.z // GROUPS) % N
                group_idx = blockIdx.z % GROUPS
                group_offset = group_idx * GROUP_OC

                if warp_count_k == 1:
                    wi = warp_id // (warp_count_n * warp_count_k)
                    wj = (warp_id // warp_count_k) % warp_count_n
                    wk = warp_id % warp_count_k

                    for mi in range(mma_count_m):
                        for mj in range(mma_count_n):
                            p = 0
                            for i, j in mma_config.c_store_map.on(lane_id):
                                res_spatial = wi * warp_m + mi * mma_m + i + offset_m
                                channel_group_idx = wj * warp_n + mj * mma_n + j + offset_n

                                channel_idx = channel_group_idx + group_offset
                                res_x = res_spatial % OUT_W
                                res_y = res_spatial // OUT_W
                                in_bound = (res_spatial < OUT_H * OUT_W) and (channel_group_idx < GROUP_OC)
                                if in_bound:
                                    res[k_part_idx, batch_idx, res_y, res_x, channel_idx] = regs_c[mi, mj, p]
                                p += 1
                else:
                    smem_c = tensor_pointer('float16', shape=[block_m, block_n])
                    smem_c = dynamic_shared_memory(byte_offset=0, dtype=float16)

                    for k_round in range(warp_count_k):
                        for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                            if wk == k_round:
                                for mi, mj in grid(mma_count_m, mma_count_n):
                                    p = 0
                                    for i, j in mma_config.c_store_map.on(lane_id):
                                        delta_m = wi * warp_m + mi * mma_m + i
                                        delta_n = wj * warp_n + mj * mma_n + j
                                        in_bound = (offset_m + delta_m < OUT_H * OUT_W) and (offset_n + delta_n < OC)
                                        if in_bound:
                                            if k_round == 0:
                                                smem_c[delta_m, delta_n] = regs_c[mi, mj, p]
                                            else:
                                                smem_c[delta_m, delta_n] += regs_c[mi, mj, p]
                                        p += 1
                        if warp_count_k > 1:
                            syncthreads()
                    for i, j in store_smem_c_map.on(threadIdx.x):
                        res_spatial = i + offset_m
                        channel_group_idx = j + offset_n
                        channel_idx = channel_group_idx + group_offset

                        res_x = res_spatial % OUT_W
                        res_y = res_spatial // OUT_W
                        if res_spatial < OUT_H * OUT_W and channel_group_idx < GROUP_OC:
                            res[k_part_idx, batch_idx, res_y, res_x, channel_idx] = smem_c[i, j]

        ir_module = module.ir_module()
        assert isinstance(conv2d_gemm_f16_kernel, Function)

        return ir_module

class Conv2dGemmFp16PretransformOP(Operator):
    def __init__(self, img: Tensor, padding: Union[int, Tuple[int, int]], pad_value=0.0, channel_mul_8: bool = False):
        super().__init__(
            inputs=[img],
            attributes={
                'pad_value': pad_value,
                'padding': padding,
                'channel_mul_8': channel_mul_8
            },
            task=Conv2dGemmFp16PretransformTask(
                input_like(img, 'img'),
                padding,
                pad_value,
                channel_mul_8
            )
        )

class Conv2dGemmFp16Op(Operator):
    def __init__(
        self,
        img: Tensor,
        weight: Tensor,
        orig_weight_shape: List[int],
        stride: List[int],
        dilations: List[int],
        groups: int,
        parallel_k_parts=1,
        disable_cp_async=False,
    ):
        if not (isinstance(parallel_k_parts, int) and not isinstance(parallel_k_parts, bool)):
            raise ValueError('parallel_k_parts must be an integer, got {}'.format(parallel_k_parts))

        super().__init__(
            inputs=[img, weight],
            attributes={
                'stride': stride,
                'dilations': dilations,
                'orig_weight_shape': orig_weight_shape,
                'groups': groups,
                'parallel_k_parts': parallel_k_parts,
                'disable_cp_async': disable_cp_async,
            },
            task=Conv2dGemmFp16Task(
                input_like(img, 'img'),
                input_like(weight, 'weight'),
                orig_weight_shape,
                stride,
                dilations,
                groups=groups,
                parallel_k_parts=parallel_k_parts,
                disable_cp_async=disable_cp_async,
            ),
        )


def pre_transform_img(img: Tensor, padding: Union[int, Tuple[int, int]], pad_value=0.0, make_multiple_8=False):
    import hidet
    n, c, w, h = img.shape
    assert pad_value == 0.0
    img = hidet.ops.conv_pad(img, padding)
    img = hidet.ops.transpose(img, [0, 2, 3, 1])
    if make_multiple_8:
        pad_channel = cdiv(c, 8) * 8 - c
        img = hidet.ops.pad(img, [0, pad_channel])
    return img


def pre_transform_imgv2(img: Tensor, padding: Union[int, Tuple[int, int]], pad_value=0.0, make_multiple_8=False):
    return Conv2dGemmFp16PretransformOP(img, padding, pad_value, make_multiple_8).get_output(0)


# pylint: disable=dangerous-default-value
def parallel_part_heuristic(
    input_shape, weight_shape, stride: List[int] = [1, 1], dilation: List[int] = [1, 1], groups: int = 1
):
    N, H, W, _ = input_shape
    OC, WC, KY, KX = weight_shape
    DILY, DILX = dilation
    STRY, STRX = stride
    OUT_H = (H - DILY * (KY - 1) - 1) // STRY + 1
    OUT_W = (W - DILX * (KX - 1) - 1) // STRX + 1
    m_size = OUT_H * OUT_W
    n_size = OC // groups
    k_size = WC * KX * KY
    estimate_blocks = N * cdiv(m_size, 64) * cdiv(n_size, 64) * groups
    estimate_concurrent_blocks = 80 * 5
    max_k_parts = cdiv(k_size, 64)
    k_parts = min(cdiv(estimate_concurrent_blocks, estimate_blocks), max_k_parts)
    return k_parts


def conv2d_gemm_fp16_channel_last(
    img: Tensor,
    weight: Tensor,
    stride: List[int],
    dilations: List[int],
    groups: int,
    parallel_k_parts=1,
    disable_cp_async=False,
) -> Tensor:
    import hidet

    if len(img.shape) != 4 or len(weight.shape) != 4:
        raise ValueError('a and b must have 4 dimensions, got shape {} and {}'.format(img.shape, weight.shape))
    if img.dtype != dtypes.float16 or weight.dtype != dtypes.float16:
        raise ValueError('ConvGemmF16Op only support float16, got {} and {}'.format(img.dtype, weight.dtype))
    oc, wc, ky, kx = weight.shape
    weight = hidet.ops.transpose(weight, [2, 3, 1, 0]).reshape([ky * kx * wc, oc])
    return (
        Conv2dGemmFp16Op(
            img,
            weight,
            orig_weight_shape=[oc, wc, ky, kx],
            stride=stride,
            dilations=dilations,
            groups=groups,
            parallel_k_parts=parallel_k_parts,
            disable_cp_async=disable_cp_async,
        )
        .get_output(0)
        .sum(0)
    )


def conv2d_pointwise_fp16(img: Tensor, weight: Tensor, groups: int):
    import hidet

    n, c, h, w = img.shape
    oc, wc, ky, kx = weight.shape
    if ky == 1 and kx == 1:
        assert c % groups == 0 and wc * groups == c, "invalid group / channel size"
        img = hidet.ops.reshape(img, [n, groups, c // groups, h * w])
        weight = hidet.ops.reshape(weight, [1, groups, oc // groups, wc])
        out = hidet.ops.matmul(weight, img)
        return hidet.ops.reshape(out, [n, oc, h, w])
    else:
        raise ValueError(f"expected kernel sizes to be (1, 1), got {(ky, kx)}")


def conv2d_gemm_fp16(
    img: Tensor,
    weight: Tensor,
    padding: List[int],
    stride: List[int],
    dilations: List[int],
    groups: int,
    parallel_k_parts=1,
    disable_cp_async=False,
) -> Tensor:
    import hidet

    n, c, h, w = img.shape
    oc, wc, ky, kx = weight.shape
    sy, sx = stride
    dy, dx = dilations
    if ky == 1 and kx == 1 and sy == 1 and sx == 1 and dy == 1 and dx == 1:
        assert c % groups == 0 and wc * groups == c, "invalid group / channel size"
        img = hidet.ops.reshape(img, [n, groups, c // groups, h * w])
        weight = hidet.ops.reshape(weight, [1, groups, oc // groups, wc])
        out = hidet.ops.matmul(weight, img)
        return hidet.ops.reshape(out, [n, oc, h, w])

    img = pre_transform_imgv3(img, tuple(padding), pad_value=0, make_multiple_8=True)
    if groups == 1 and c % 8 != 0:
        pad_channel = cdiv(c, 8) * 8 - c
        weight = hidet.ops.pad(weight, [0, 0, 0, 0, 0, pad_channel, 0, 0])

    res = conv2d_gemm_fp16_channel_last(img, weight, stride, dilations, groups, parallel_k_parts, disable_cp_async)
    return hidet.ops.transpose(res, [0, 3, 1, 2])
