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
from typing import Any, Dict, List, Optional, Sequence
from hidet.graph.tensor import Tensor

from hidet.ir.expr import is_constant
from hidet.graph.ops.matmul import matmul, batch_matmul
from hidet.graph.ops.utils import Task, Operator, Tensor, compute, input_like, TensorNode
from hidet.graph.ops.utils import normalize_kernel, normalize_stride, normalize_dilations, reduce
from hidet.ir.task import Task
from .utils import infer_conv2d_shape


class Conv2dGemmImageTransformTask(Task):
    def __init__(self, x: TensorNode, kernel: List[int], stride: List[int], dilations: List[int], groups: int):
        n, c, h, w = x.shape
        kx, ky = kernel
        sx, sy = stride
        dilx, dily = dilations
        p, q = (h - dilx * (kx - 1) - 1) // sx + 1, (w - dily * (ky - 1) - 1) // sy + 1
        if is_constant(c) and c % groups != 0:
            msg = 'Conv2d expect in_channels % groups == 0, but got in_channels {} and groups {}'.format(c, groups)
            raise ValueError(msg)
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


class Conv2dGemmPostTransformTask(Task):
    def __init__(self, a: TensorNode, stride: List[int], dilations: List[int]):
        n, ky, kx, oc, h, w = a.shape
        stride_y, stride_x = stride
        dilation_y, dilation_x = dilations
        out_h = (h - dilation_y * (ky - 1) - 1) // stride_y + 1
        out_w = (w - dilation_x * (kx - 1) - 1) // stride_x + 1
        out = compute(
            name='post_transform',
            shape=[n, oc, out_h, out_w], 
            fcompute=lambda ni, oci, hi, wi: reduce(
                shape=[ky, kx],
                reduce_type='sum',
                fcompute=lambda kyi, kxi: a[ni, kyi, kxi, oci, hi * stride_y + kyi * dilation_y, wi * stride_x + kxi * dilation_x]
            )
        )
        super().__init__(name="conv2d_gemm_post_transform", inputs=[a], outputs=[out])


class Conv2dGemmPostTransformOp(Operator):
    def __init__(self, a: Tensor, stride: List[int], dilations: List[int]):
        stride = normalize_stride(stride)
        dilations = normalize_dilations(dilations)
        super().__init__(
            inputs=[a],
            attributes={'stride': stride, 'dilations': dilations},
            task=Conv2dGemmPostTransformTask(input_like(a, 'a'), stride, dilations),
        )


def conv2d_gemm_image_transform(
    x: Tensor, kernel: Sequence[int], stride: Sequence[int], dilations: Sequence[int], groups: int = 1
) -> Tensor:
    return Conv2dGemmImageTransformOp(x, kernel, stride, dilations, groups).get_output(0)


def conv2d_gemm_post_transform(
    a: Tensor, stride: Sequence[int], dilations: Sequence[int]
) -> Tensor:
    """
    Equivalent to the following algorithm:

    def post_transform(a, stride, dilations):
        n, ky, kx, oc, h, w = a.shape
        stride_y = stride[0]
        stride_x = stride[1]
        dilation_y = dilations[0]
        dilation_x = dilations[1]
        out_h = (h - dilation_y * (ky - 1) - 1) // stride_y + 1
        out_w = (w - dilation_x * (kx - 1) - 1) // stride_x + 1
        y = torch.zeros([n, oc, out_h, out_w], device=x.device, dtype=x.dtype)

        for i in range(out_h):
            for j in range(out_w):
                for kyi in range(ky):
                    for kxi in range(kx):
                        iy = i * stride_y + kyi * dilation_y
                        ix = j * stride_x + kxi * dilation_x
                        y[:, :, i, j] += a[:, kyi, kxi, :, iy, ix]
        return y
    """
    return Conv2dGemmPostTransformOp(a, stride, dilations).get_output(0)


def conv2d_gemm_filter_transform(w: Tensor, groups: int = 1) -> Tensor:
    # weight shape: [oc, c, kx, ky]
    # output shape: [groups, c * kx * ky, ogc] where ogc = oc // groups
    oc, c, kx, ky = w.shape
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
    gemm_y = matmul(gemm_x, gemm_w)

    y_shape = infer_conv2d_shape(data.shape, weight.shape, stride, groups, dilations)
    y = conv2d_gemm_inverse_transform(gemm_y, out_height=y_shape[2], out_width=y_shape[3])
    return y


def conv2d_gemm_1(data: Tensor, gemm_w: Tensor, weight_shape, stride, dilations: List[int], groups: int = 1) -> Tensor:
    gemm_x = conv2d_gemm_image_transform(
        data, kernel=weight_shape[2:], stride=stride, dilations=dilations, groups=groups
    )
    
    gemm_y = matmul(gemm_x, gemm_w)

    y_shape = infer_conv2d_shape(data.shape, weight_shape, stride, groups, dilations)
    y = conv2d_gemm_inverse_transform(gemm_y, out_height=y_shape[2], out_width=y_shape[3])
    return y


def conv2d_gemm_fp16(data: Tensor, weight: Tensor, stride, dilations: List[int], groups: int = 1, mma='simt') -> Tensor:
    gemm_x = conv2d_gemm_image_transform(
        data, kernel=weight.shape[2:], stride=stride, dilations=dilations, groups=groups
    )
    gemm_w = conv2d_gemm_filter_transform(weight, groups=groups)
    gemm_y = batch_matmul(gemm_x, gemm_w, mma=mma)

    y_shape = infer_conv2d_shape(data.shape, weight.shape, stride, groups, dilations)
    y = conv2d_gemm_inverse_transform(gemm_y, out_height=y_shape[2], out_width=y_shape[3])
    return y


def conv2d_gemm_2(data: Tensor, weight: Tensor, stride, dilations: List[int], groups: int = 1) -> Tensor:
    """
    Equivalent to the following algorithm:

    def conv2d(x: torch.Tensor, weight: torch.Tensor, stride=[1, 1], dilation=[1, 1], groups=1):
        n, c, h, w = x.shape
        oc, wc, ky, kx = weight.shape
        assert c % groups == 0 and oc % groups == 0
        assert wc * groups == c

        stride_y = stride[0]
        stride_x = stride[1]
        dilation_y = dilation[0]
        dilation_x = dilation[1]
        x = x.reshape([n, 1, groups, wc, w * h])
        weight = weight.permute(2, 3, 0, 1).reshape([ky * kx, groups, oc // groups, wc])

        a = weight @ x   
        a = a.reshape([n, ky, kx, oc, h, w])
        out_h = (h - dilation_y * (ky - 1) - 1) // stride_y + 1
        out_w = (w - dilation_x * (kx - 1) - 1) // stride_x + 1
        y = torch.zeros([n, oc, out_h, out_w], device=x.device, dtype=x.dtype)

        for i in range(out_h):
            for j in range(out_w):
                for kyi in range(ky):
                    for kxi in range(kx):
                        iy = i * stride_y + kyi * dilation_y
                        ix = j * stride_x + kxi * dilation_x
                        y[:, :, i, j] += a[:, kyi, kxi, :, iy, ix]
        return y
    
    """
    from hidet.graph.ops import transpose
    stride = normalize_stride(stride)
    dilations = normalize_dilations(dilations)
    
    n, c, h, w = data.shape
    oc, wc, ky, kx = weight.shape
    data = data.reshape([n, 1, groups, wc, w * h])
    weight = transpose(weight, [2, 3, 0, 1]).reshape([ky * kx, groups, oc // groups, wc])

    # a [n, ky * kx, groups, oc // groups, w * h]
    a = matmul(weight, data).reshape([n, ky, kx, oc, h, w])
    y = conv2d_gemm_post_transform(a, stride, dilations)
    return y


def conv2d_gemm_2_1(data: Tensor, weight: Tensor, weight_shape: List[int], stride, dilations: List[int], groups: int = 1) -> Tensor:
    stride = normalize_stride(stride)
    dilations = normalize_dilations(dilations)
    
    n, c, h, w = data.shape
    oc, wc, ky, kx = weight_shape
    data = data.reshape([n, 1, groups, wc, w * h])

    a = matmul(weight, data).reshape([n, ky, kx, oc, h, w])
    y = conv2d_gemm_post_transform(a, stride, dilations)
    return y


from typing import List, Tuple
from hidet.ir import dtypes
from hidet.ir.dtypes import float16
from hidet.ir.expr import if_then_else, Int, Expr
from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.compute import TensorNode
from hidet.ir.task import Task
from hidet.ir.compute import compute, reduce
from hidet.graph.ops.utils import input_like, broadcast_shape, can_mutually_broadcast
from hidet.graph.ops.utils import tune
from hidet.graph.operator import Operator, Tensor
from hidet.utils.py import is_power_of_two, cdiv, prod
from hidet.graph.ops.utils import broadcast_indices


class FusedConvGemmF16Task(Task):
    def __init__(self, x: TensorNode, weight: TensorNode, ky: int, kx: int, stride: List[int], dilation: List[int], groups=1):
        # x.shape == [n, c, h, w]
        # w.shape == [ky * kx, groups, oc // groups, wc]
        if not x.type.dtype == float16 or not weight.type.dtype == float16:
            raise ValueError('Both inputs must be float16 tensors')

        if not (isinstance(x.shape[-1], Expr) or isinstance(weight.shape[-1], Expr)) and x.shape[-1] != weight.shape[-2]:
            raise ValueError(
                'Matrix multiplication expect tensor A and B with shape [..., M, K] and [..., K, N]'
                ', got {} and {}'.format(x.shape, weight.shape)
            )

        if not can_mutually_broadcast(x.shape[:-2], weight.shape[:-2]):
            raise ValueError(
                'Matrix multiplication expect tensor A and B with compatible broadcast shape, '
                'got {} and {}'.format(x.shape, weight.shape)
            )

        n, c, h, w = x.shape
        _, groups, gp, wc = weight.shape
        self.groups = groups
        self.oc = groups * gp
        self.stride_y = stride[0]
        self.stride_x = stride[1]
        self.dilation_y = dilation[0]
        self.dilation_x = dilation[1]
        self.out_h = (h - self.dilation_y * (ky - 1) - 1) // self.stride_y + 1
        self.out_w = (w - self.dilation_x * (kx - 1) - 1) // self.stride_x + 1
        self.ky = ky
        self.kx = kx
        
        out_group_size = self.oc // groups
        
        c_shape = [n, self.oc, self.out_h, self.out_w]

        c = compute(
            name='out',
            shape=c_shape,
            fcompute=lambda ni, oci, pi, qi: reduce(
                shape=[wc, ky, kx],
                fcompute=lambda wci, kxi, kyi: (
                    x[ni, (oci // out_group_size) * wc + wci, pi * self.stride_y + kxi * self.dilation_y, qi * self.stride_x + kyi * self.dilation_x]
                    * weight[oci, wci, kxi, kyi]
                ),
                reduce_type='sum',
            ),
        )

        super().__init__(
            name='matmul_f16_pk', inputs=[x, weight], outputs=[c]
        )

    def allow_prologue(self) -> bool:
        return False

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
        from hidet.lang import attrs, col_spatial, view, u32, tensor_pointer, grid
        from hidet.lang.layout import row_layout
        from hidet.lang.mapping import spatial, auto_map
        from hidet.lang.cuda import blockIdx, threadIdx, syncthreads, dynamic_shared_memory
        from hidet.lang.cuda import MmaConfig, mma_sync, cp_async, cp_async_wait_all, ldmatrix
        from hidet.lang.cuda import register_tensor, atomic_add

        # input shapes
        node_a, node_b = self.inputs[0].ttype.shape, self.inputs[1].ttype.shape
        N, C, HW = node_a[0], node_a[1], node_a[2] * node_a[3]
        KYKX, GROUPS, GP, WC = node_b
        KY, KX = self.ky, self.kx

        OC, OUTH, OUTW = self.oc, self.out_h, self.out_w

        dilation_y, dilation_x = self.dilation_y, self.dilation_x
        stride_y, stride_x = self.stride_y, self.stride_x

        # a_shape = [1, KYKX, GROUPS, GP, WC]
        # b_shape = [N, 1, GROUPS, WC, HW]

        m_size, n_size, k_size = GP, HW, WC

        # schedule parameters
        mma_configs = {'m16n8k8': MmaConfig.m16n8k8_f16_f16(), 'm16n8k16': MmaConfig.m16n8k16_f16_f16()}
        tune.check(mma in mma_configs)
        mma_config = mma_configs[mma]

        mma_m, mma_n, mma_k = mma_config.m, mma_config.n, mma_config.k  # 16, 8, 16
        warp_count_m, warp_count_n, warp_count_k = block_m // warp_m, block_n // warp_n, block_k // warp_k
        mma_count_m, mma_count_n, mma_count_k = warp_m // mma_m, warp_n // mma_n, warp_k // mma_k
        threads = warp_count_m * warp_count_n * warp_count_k * 32
        
        grid_dim: Tuple[Int, Int, Int] = cdiv(m_size, block_m), cdiv(n_size, block_n), N * KYKX * GROUPS
        dynamic_smem_bytes = max(2 * (block_m + block_n) * block_k * 2, block_m * block_n * 2)

        tune.check(block_m % warp_m == block_n % warp_n == block_k % warp_k == 0, 'warp dims divide block dims')
        tune.check(warp_m % mma_m == warp_n % mma_n == warp_k % mma_k == 0, 'mma dims divide warp dims')
        tune.check(threads <= 1024, 'threads in a block <= 1024')
        maximum_smem_bytes = 49152
        tune.check(dynamic_smem_bytes <= maximum_smem_bytes, 'dynamic shared memory <= 49152')

        tune.check(block_n % 64 == 0, 'block_n must be multiple of 64, required by async gmem -> smem loading')
        tune.check(block_k % 8 == 0)
        tune.check(is_power_of_two(block_k // 8))
        smem_a_type = tensor_type(
            'float16', shape=[block_m, block_k], layout=row_layout(block_m, block_k // 8).swizzle(1) * row_layout(1, 8)
        )
        smem_b_type = tensor_type(
            'float16',
            shape=[block_k, block_n],
            layout=row_layout(block_k // 8, block_n // 64) * row_layout(8, 8).swizzle(1) * row_layout(1, 8),
        )
        load_smem_a_map = auto_map(block_m, block_k // 8, workers=threads, on_fail=lambda msg: tune.check(False, msg))
        load_smem_b_map = auto_map(block_k, block_n // 8, workers=threads, on_fail=lambda msg: tune.check(False, msg))
        store_smem_c_map = auto_map(block_m, block_n, workers=threads, on_fail=lambda msg: tune.check(False, msg))

        with hidet.script_module() as module:

            @hidet.script
            def load_regs_a(mi: int, k1: int, smem_a: smem_a_type, regs_a: float16[mma_config.a_elements]):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    p, q = col_spatial(16, 2).map(lane_id)
                    row_addr = ~smem_a[wi * warp_m + mi * mma_m + p, wk * warp_k + k1 * mma_k + q * 8]
                    b32_regs = view(regs_a, u32[4])
                    ldmatrix(
                        regs=[b32_regs[0], b32_regs[1], b32_regs[2], b32_regs[3]],
                        smem_addr=row_addr,
                        shared_space_addr=False,
                        trans=False,
                    )

            @hidet.script
            def load_regs_b(mj: int, k1: int, smem_b: smem_b_type, regs_b: float16[mma_config.b_elements]):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    p, q = col_spatial(16, 2).map(lane_id)
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
            def load_smem_a(k0: int, a: float16[KYKX, GROUPS, GP, WC], smem_a: smem_a_type):
                gp_idx = blockIdx.z % GROUPS
                kykx_idx = (blockIdx.z // GROUPS) % KYKX

                offset_m = blockIdx.x * block_m
                offset_k = k0 * block_k

                gmem_a = a[kykx_idx, gp_idx, offset_m:, offset_k:]
                for i, k_seg in load_smem_a_map.on(threadIdx.x):
                    k = k_seg * 8
                    src_size = (
                        0
                        if (offset_m + i >= m_size or offset_k + k >= k_size)
                        else min(k_size - (offset_k + k), 8)
                    )
                    cp_async(~smem_a[i, k], ~gmem_a[i, k], cp_size=16, src_size=src_size * 2, cache_level='global')
                    cp_async_wait_all()

            @hidet.script
            def load_smem_b(k0: int, b: float16[N, GROUPS, WC, HW], smem_b: smem_b_type):
                gp_idx = blockIdx.z % GROUPS
                n_idx = (blockIdx.z // GROUPS) // KYKX

                offset_n = blockIdx.y * block_n
                offset_k = k0 * block_k

                gmem_b = b[n_idx, gp_idx, offset_k:, offset_n:]
                for k, j_seg in load_smem_b_map.on(threadIdx.x):
                    j = j_seg * 8
                    src_size = (
                        0 if (offset_k + k >= k_size or offset_n + j >= n_size) else min(n_size - (offset_n + j), 8)
                    )
                    cp_async(~smem_b[k, j], ~gmem_b[k, j], cp_size=16, src_size=src_size * 2, cache_level='global')

            @hidet.script
            def matmul_f16_kernel(
                a: float16[KYKX, GROUPS, GP, WC],
                b: float16[N, GROUPS, WC, HW],
                c: float16[N, OC, OUTH, OUTW],
            ):
                # matrix multiplication, using mma instruction
                attrs.cuda.grid_dim = grid_dim
                attrs.cuda.block_dim = threads
                # the second 2 means '2 bytes per float16'
                attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes
                # smem_storage = dyn_smem_storage
                smem_a = tensor_pointer(
                    'float16', shape=[2, block_m, block_k], layout=row_layout(2) + smem_a_type.layout
                )
                smem_b = tensor_pointer(
                    'float16', shape=[2, block_k, block_n], layout=row_layout(2) + smem_b_type.layout
                )
                smem_a = dynamic_shared_memory(byte_offset=0, dtype=float16)
                smem_b = dynamic_shared_memory(byte_offset=2 * block_m * block_k * 2, dtype=float16)
                regs_a = register_tensor(float16, [2, mma_count_m, mma_config.a_elements])
                regs_b = register_tensor(float16, [2, mma_count_n, mma_config.b_elements])
                regs_c = register_tensor(float16, [mma_count_m, mma_count_n, mma_config.c_elements])

                for i, j, p in grid(mma_count_m, mma_count_n, mma_config.c_elements):
                    regs_c[i, j, p] = 0.0

                load_smem_a(0, a, ~smem_a[0, 0, 0])
                load_smem_b(0, b, ~smem_b[0, 0, 0])
                cp_async_wait_all()

                syncthreads()
                for k0 in range((k_size + block_k - 1) // block_k):
                    load_smem_a(k0 + 1, a, ~smem_a[(k0 + 1) % 2, 0, 0])
                    load_smem_b(k0 + 1, b, ~smem_b[(k0 + 1) % 2, 0, 0])
                    for mi in range(mma_count_m):
                        load_regs_a(mi, 0, ~smem_a[k0 % 2, 0, 0], ~regs_a[0, mi, 0])
                    for mj in range(mma_count_n):
                        load_regs_b(mj, 0, ~smem_b[k0 % 2, 0, 0], ~regs_b[0, mj, 0])
                    for mk in range(mma_count_k):
                        if mk + 1 < mma_count_k:
                            for mi in range(mma_count_m):
                                load_regs_a(mi, mk + 1, ~smem_a[k0 % 2, 0, 0], ~regs_a[(mk + 1) % 2, mi, 0])
                            for mj in range(mma_count_n):
                                load_regs_b(mj, mk + 1, ~smem_b[k0 % 2, 0, 0], ~regs_b[(mk + 1) % 2, mj, 0])
                        for mi, mj in grid(mma_count_m, mma_count_n):
                            warp_mma(~regs_a[mk % 2, mi, 0], ~regs_b[mk % 2, mj, 0], ~regs_c[mi, mj, 0])
                    cp_async_wait_all()
                    syncthreads()

                # store back
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                offset_m, offset_n = blockIdx.x * block_m, blockIdx.y * block_n

                gp_idx = blockIdx.z % GROUPS
                kykx_idx = (blockIdx.z // GROUPS) % KYKX
                ky_idx = kykx_idx // KX
                kx_idx = kykx_idx % KX
                n_idx = (blockIdx.z // GROUPS) // KYKX

                # [1, ky * kx, groups, gp, wc] * [n, 1, groups, wc, w * h] = [n, ky * kx, groups, gp, w * h]
                # gp * groups == oc
                
                if warp_count_k == 1:
                    for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                        for mi, mj in grid(mma_count_m, mma_count_n):
                            p = 0
                            for i, j in mma_config.c_store_map.on(lane_id):
                                delta_m = wi * warp_m + mi * mma_m + i + offset_m
                                delta_n = wj * warp_n + mj * mma_n + j + offset_n
                                
                                iy = delta_n // OUTW
                                ix = delta_n % OUTW
                                
                                im_y = iy - ky_idx * dilation_y
                                im_x = ix - kx_idx * dilation_x
                                in_bound = (delta_m < m_size) and (delta_n < n_size)
                                is_stride_multiple = im_y % stride_y == 0 and im_x % stride_x == 0
                                if in_bound and is_stride_multiple:
                                    im_y = im_y // stride_y
                                    im_x = im_x // stride_x
                                    col_idx = gp_idx * GP + delta_n

                                    atomic_add(~c[n_idx, col_idx, im_y, im_x], regs_c[mi, mj, p])
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
                                        in_bound = (offset_m + delta_m < m_size) and (offset_n + delta_n < n_size)
                                        if in_bound:
                                            if k_round == 0:
                                                smem_c[delta_m, delta_n] = regs_c[mi, mj, p]
                                            else:
                                                smem_c[delta_m, delta_n] += regs_c[mi, mj, p]
                                        p += 1
                        if warp_count_k > 1:
                            syncthreads()
                    for i, j in store_smem_c_map.on(threadIdx.x):
                        delta_m = i + offset_m
                        delta_n = j + offset_n
                        iy = delta_n // OUTW
                        ix = delta_n % OUTW
                        
                        im_y = iy - ky_idx * dilation_y
                        im_x = ix - kx_idx * dilation_x
                        in_bound = (delta_m < m_size) and (delta_n < n_size)
                        is_stride_multiple = im_y % stride_y == 0 and im_x % stride_x == 0

                        if in_bound and is_stride_multiple:
                            im_y = im_y // stride_y
                            im_x = im_x // stride_x
                            col_idx = gp_idx * GP + delta_n
                            atomic_add(~c[n_idx, col_idx, im_y, im_x], smem_c[i, j])

        ir_module = module.ir_module()
        assert isinstance(matmul_f16_kernel, Function)

        return ir_module


class ConvMatmulF16Op(Operator):
    def __init__(self, a: Tensor, b: Tensor, parallel_k_parts=1):
        if not (isinstance(parallel_k_parts, int) and not isinstance(parallel_k_parts, bool)):
            raise ValueError('parallel_k_parts must be an integer, got {}'.format(parallel_k_parts))
        super().__init__(
            inputs=[a, b],
            attributes={'parallel_k_parts': parallel_k_parts},
            task=FusedConvGemmF16Task(input_like(a, 'a'), input_like(b, 'b'), parallel_k_parts),
        )


def matmul_f16(a: Tensor, b: Tensor, parallel_k_parts=1) -> Tensor:
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError('a and b must have at least 2 dimensions, got shape {} and {}'.format(a.shape, b.shape))
    # TODO: impliment dynamic run-time shape assertion
    if not (isinstance(a.shape[-1], Expr) or isinstance(b.shape[-1], Expr)) and (
        a.shape[-1] % 8 != 0 or b.shape[-1] % 8 != 0
    ):
        raise ValueError('Expect the last dimension of the input tensors to be a multiple of 8')
    if a.dtype != dtypes.float16 or b.dtype != dtypes.float16:
        raise ValueError('BatchMatmulF16Op only support float16, got {} and {}'.format(a.dtype, b.dtype))
    return ConvMatmulF16Op(a, b, parallel_k_parts).get_output(0)
