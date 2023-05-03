import numpy.testing
from hidet.ir import Constant
from hidet.ir.stmt import DeclareScope

import hidet


def matmul_kernel5():
    from hidet.transforms.generate_packed_func import add_packed_func
    from hidet.lang import attr
    from hidet.lang import float32, int32
    from hidet.lang import as_tensor_pointer, tensor
    from hidet.lang.mapping import repeat, spatial, auto_map
    from hidet.lang.layout import row_layout, local_layout, col_layout

    from hidet.lang.avx import avx_f32x4_broadcast, avx_f32x4_fmadd, avx_f32x4_load, avx_f32x4_store
    from hidet.lang.avx import avx_f32x8_store, avx_f32x8_broadcast, avx_f32x8_fmadd, avx_f32x8_load
    from hidet.lang.avx import avx_free, avx_malloc, x86_memset

    MC = 1024
    NC = 256
    KC = 256

    MR = 8
    NR = 8

    MC = MC
    NC = NC
    KC = KC
    MR = MR
    NR = NR

    aip_outer_rows = MC // MR
    bip_outer_cols = NC // NR

    aip_outer_rows = aip_outer_rows
    bip_outer_cols = bip_outer_cols

    with hidet.lang.script_module() as script_module:
        @hidet.lang.script
        def micro_kernel_8x8(a_ptr: ~float32,
                             b_ptr: ~float32,
                             c_ptr: ~float32,
                             pb: int32,
                             m_size: int32,
                             n_size: int32):
            a = as_tensor_pointer(a_ptr, dtype=float32,
                                  layout=row_layout(aip_outer_rows, 1) * col_layout(MR, KC))
            b = as_tensor_pointer(b_ptr, dtype=float32,
                                  layout=row_layout(1, bip_outer_cols) * row_layout(KC, NR))
            c = as_tensor_pointer(c_ptr, dtype=float32, shape=[m_size, n_size])

            c0 = avx_f32x8_load(~c[0, 0])
            c1 = avx_f32x8_load(~c[1, 0])
            c2 = avx_f32x8_load(~c[2, 0])
            c3 = avx_f32x8_load(~c[3, 0])
            c4 = avx_f32x8_load(~c[4, 0])
            c5 = avx_f32x8_load(~c[5, 0])
            c6 = avx_f32x8_load(~c[6, 0])
            c7 = avx_f32x8_load(~c[7, 0])

            for pp in range(pb):
                bb_0to7 = avx_f32x8_load(~b[pp, 0])

                aa = avx_f32x8_broadcast(~a[0, pp])
                c0 = avx_f32x8_fmadd(aa, bb_0to7, c0)
                aa = avx_f32x8_broadcast(~a[1, pp])
                c1 = avx_f32x8_fmadd(aa, bb_0to7, c1)
                aa = avx_f32x8_broadcast(~a[2, pp])
                c2 = avx_f32x8_fmadd(aa, bb_0to7, c2)
                aa = avx_f32x8_broadcast(~a[3, pp])
                c3 = avx_f32x8_fmadd(aa, bb_0to7, c3)
                aa = avx_f32x8_broadcast(~a[4, pp])
                c4 = avx_f32x8_fmadd(aa, bb_0to7, c4)
                aa = avx_f32x8_broadcast(~a[5, pp])
                c5 = avx_f32x8_fmadd(aa, bb_0to7, c5)
                aa = avx_f32x8_broadcast(~a[6, pp])
                c6 = avx_f32x8_fmadd(aa, bb_0to7, c6)
                aa = avx_f32x8_broadcast(~a[7, pp])
                c7 = avx_f32x8_fmadd(aa, bb_0to7, c7)
            avx_f32x8_store(~c[0, 0], c0)
            avx_f32x8_store(~c[1, 0], c1)
            avx_f32x8_store(~c[2, 0], c2)
            avx_f32x8_store(~c[3, 0], c3)
            avx_f32x8_store(~c[4, 0], c4)
            avx_f32x8_store(~c[5, 0], c5)
            avx_f32x8_store(~c[6, 0], c6)
            avx_f32x8_store(~c[7, 0], c7)

        @hidet.lang.script
        def macro_kernel(
                a_ptr: ~float32,
                b_ptr: ~float32,
                c_ptr: ~float32,
                ib: int32,
                jb: int32,
                pb: int32,
                m_size: int32,
                n_size: int32
        ):
            a = as_tensor_pointer(a_ptr, dtype=float32,
                                  layout=row_layout(aip_outer_rows, 1) * col_layout(MR, KC))
            b = as_tensor_pointer(b_ptr, dtype=float32,
                                  layout=row_layout(1, bip_outer_cols) * row_layout(KC, NR))
            c = as_tensor_pointer(c_ptr, dtype=float32, shape=[m_size, n_size])

            mpanels = (ib + MR - 1) // MR
            npanels = (jb + NR - 1) // NR
            _mr = ib % MR
            _nr = jb % NR
            # Loop 2
            for mpanel in range(mpanels):
                mr = MR if mpanel != mpanels - 1 or _mr == 0 else _mr
                ii = mpanel * MR
                # Loop 1
                for npanel in range(npanels):
                    nr = NR if npanel != npanels - 1 or _nr == 0 else _nr
                    jj = npanel * NR
                    # micro-kernel
                    if mr == MR and nr == NR:
                        micro_kernel_8x8(~a[ii, 0], ~b[0, jj], ~c[ii, jj],
                                         pb, m_size, n_size)
                    else:
                        temp_c = tensor(
                            scope=DeclareScope.Default,
                            dtype=float32,
                            layout=row_layout(MR, NR)
                        )
                        for tempi in range(MR):
                            for tempj in range(NR):
                                temp_c[tempi, tempj] = 0.0
                        micro_kernel_8x8(~a[ii, 0], ~b[0, jj], temp_c,
                                         pb, MR, NR)
                        for remain_row in range(mr):
                            for remain_col in range(nr):
                                c[ii + remain_row, jj+remain_col] += temp_c[remain_row, remain_col]

        @hidet.lang.script
        def matmul_kernel(
                a_ptr: ~float32,
                b_ptr: ~float32,
                c_ptr: ~float32,
                m_size: int32,
                n_size: int32,
                k_size: int32
        ):
            a = as_tensor_pointer(a_ptr, float32, [m_size, k_size])
            b = as_tensor_pointer(b_ptr, float32, [k_size, n_size])
            c = as_tensor_pointer(c_ptr, float32, [m_size, n_size])

            _mc = m_size % MC
            _nc = n_size % NC
            _kc = k_size % KC

            aip_packed = tensor(
                scope=DeclareScope.Default,
                dtype=float32,
                layout=row_layout(aip_outer_rows, 1) * col_layout(MR, KC)
            )

            bpj_packed = tensor(
                scope=DeclareScope.Default,
                dtype=float32,
                layout=row_layout(1, bip_outer_cols) * row_layout(KC, NR)
            )

            i = 0
            while i < m_size:
                ib = min(MC, m_size - i)
                p = 0
                while p < k_size:
                    # pack A into contiguous memory
                    pb = min(KC, k_size - p)
                    mp = ib // MR
                    mr = ib % MR
                    for micropanel_idx in range(mp):
                        panel_row_start = micropanel_idx * MR
                        for micropanel_col in range(pb):
                            for micropanel_row in range(MR):
                                aip_packed[panel_row_start + micropanel_row, micropanel_col] = \
                                    a[i + micropanel_row + panel_row_start, p + micropanel_col]
                    # # TODO: pack the remaining if the shape is not 'nice'
                    if mr > 0:
                        remain_start_row = mp * MR
                        # assert remain_start_row + mr == m_size
                        for remain_col in range(pb):
                            for remain_row in range(mr):
                                aip_packed[remain_start_row + remain_row, remain_col] = \
                                    a[i + remain_start_row + remain_row, p + remain_col]
                            # zero-fill the rest
                            remain_row = mr
                            while remain_row < MR:
                                aip_packed[remain_start_row + remain_row, remain_col] = 0.0
                                remain_row += 1
                    # End of the packing of A...
                    # Start loop 3
                    j = 0
                    while j < n_size:
                        jb = min(NC, n_size - j)
                        # TODO: pack B into contiguous memory
                        np = jb // NR
                        nr = jb % NR
                        for micropanel_idx in range(np):
                            panel_col_start = micropanel_idx * NR
                            for micropanel_row in range(pb):
                                for micropanel_col in range(NR):
                                    bpj_packed[micropanel_row, micropanel_col + panel_col_start] = \
                                        b[p + micropanel_row, j + micropanel_col + panel_col_start]
                        if nr > 0:
                            remain_col_start = np * NR
                            # assert remain_col_start + nr == n_size
                            for remain_row in range(pb):
                                for remain_col in range(nr):
                                    bpj_packed[remain_row, remain_col + remain_col_start] = \
                                        b[p + remain_row, j + remain_col + remain_col_start]
                                remain_col = nr
                                while remain_col < NR:
                                    bpj_packed[remain_row, remain_col_start + remain_col] = 0.0
                                    remain_col += 1
                        # End of packing B into contiguous memory
                        # Start of the macro-kernel
                        macro_kernel(aip_packed, bpj_packed, ~c[i, j], ib, jb, pb, m_size, n_size)

                        j += NC
                    p += KC
                i += MC
    #################################################
    assert isinstance(matmul_kernel, hidet.ir.Function)
    matmul_kernel.kind = 'host_kernel'

    ir_module = script_module.ir_module()
    add_packed_func(ir_module, matmul_kernel, pack_func_name='matmul6')
    compiled_function = hidet.driver.build_ir_module(ir_module)
    return compiled_function


def ff():
    func = matmul_kernel5()

    for m, n, k in [(1, 74, 1), (64, 64, 64), (110, 111, 111), (101, 101, 37), (111, 367, 369), (224, 562, 325),
                    (256, 256, 256), (333, 444, 555), (512, 512, 512), (1024, 1024, 1024), (1024, 512, 768),
                    (480, 480, 480), (720, 720, 720), (720, 960, 1440), (1111, 1111, 1111), (1111, 1314, 533)]:
        a = hidet.randn([m, k], dtype='float32').cpu()
        b = hidet.randn([k, n], dtype='float32').cpu()
        c = hidet.zeros([m, n]).cpu()
        func(a, b, c, m, n, k)
        numpy.testing.assert_allclose(
            actual=c.cpu().numpy(),
            desired=a.cpu().numpy() @ b.cpu().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

        hidet_latency = hidet.utils.benchmark_func(
            lambda: func(a, b, c, m, n, k), repeat=10
        )

        np_latency = hidet.utils.benchmark_func(
            lambda: a.cpu().numpy() @ b.cpu().numpy(), repeat=10
        )

        print(f'{m} x {k} x {n}: hidet takes {hidet_latency:.2f} ms')
        print(f'{m} x {k} x {n}: numpy takes {np_latency: .2f} ms')


ff()

#### -O3
# 1 x 1 x 74: hidet takes 0.02 ms
# 1 x 1 x 74: numpy takes  0.02 ms
# 64 x 64 x 64: hidet takes 0.03 ms
# 64 x 64 x 64: numpy takes  0.04 ms
# 110 x 111 x 111: hidet takes 0.09 ms
# 110 x 111 x 111: numpy takes  0.13 ms
# 101 x 37 x 101: hidet takes 0.05 ms
# 101 x 37 x 101: numpy takes  0.10 ms
# 111 x 369 x 367: hidet takes 0.65 ms
# 111 x 369 x 367: numpy takes  0.21 ms
# 224 x 325 x 562: hidet takes 1.60 ms
# 224 x 325 x 562: numpy takes  0.28 ms
# 256 x 256 x 256: hidet takes 0.68 ms
# 256 x 256 x 256: numpy takes  0.14 ms
# 333 x 555 x 444: hidet takes 2.78 ms
# 333 x 555 x 444: numpy takes  1.00 ms
# 512 x 512 x 512: hidet takes 3.06 ms
# 512 x 512 x 512: numpy takes  0.64 ms
# 1024 x 1024 x 1024: hidet takes 23.58 ms
# 1024 x 1024 x 1024: numpy takes  2.70 ms
# 1024 x 768 x 512: hidet takes 8.96 ms
# 1024 x 768 x 512: numpy takes  1.19 ms
# 480 x 480 x 480: hidet takes 2.61 ms
# 480 x 480 x 480: numpy takes  0.76 ms
# 720 x 720 x 720: hidet takes 8.30 ms
# 720 x 720 x 720: numpy takes  1.46 ms
# 720 x 1440 x 960: hidet takes 21.75 ms
# 720 x 1440 x 960: numpy takes  2.99 ms
# 1111 x 1111 x 1111: hidet takes 30.28 ms
# 1111 x 1111 x 1111: numpy takes  3.54 ms
# 1111 x 533 x 1314: hidet takes 17.33 ms
# 1111 x 533 x 1314: numpy takes  1.99 ms
#
# Process finished with exit code 0

