import numpy.testing
from hidet.ir import Constant
from hidet.ir.stmt import DeclareScope

import hidet


def matmul_kernel5():
    from hidet.transforms.generate_packed_func import add_packed_func
    from hidet.lang import attr
    from hidet.lang import float32, int32
    from hidet.lang import as_tensor_pointer, tensor, grid
    from hidet.lang.mapping import repeat, spatial, auto_map
    from hidet.lang.layout import row_layout, local_layout, col_layout

    from hidet.lang.avx import avx_f32x4_broadcast, avx_f32x4_fmadd, avx_f32x4_load, avx_f32x4_store
    from hidet.lang.avx import avx_f32x8_store, avx_f32x8_broadcast, avx_f32x8_fmadd, avx_f32x8_load
    from hidet.lang.avx import avx_free, avx_malloc, x86_memset

    # MC = 2400
    # NC = 768
    # KC = 1024
    MC = 4800
    KC = 768
    NC = 384

    MR = 6
    NR = 16

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
        def micro_kernel_6x16(a_ptr: ~float32,
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
            c08 = avx_f32x8_load(~c[0, 8])
            c1 = avx_f32x8_load(~c[1, 0])
            c18 = avx_f32x8_load(~c[1, 8])
            c2 = avx_f32x8_load(~c[2, 0])
            c28 = avx_f32x8_load(~c[2, 8])
            c3 = avx_f32x8_load(~c[3, 0])
            c38 = avx_f32x8_load(~c[3, 8])
            c4 = avx_f32x8_load(~c[4, 0])
            c48 = avx_f32x8_load(~c[4, 8])
            c5 = avx_f32x8_load(~c[5, 0])
            c58 = avx_f32x8_load(~c[5, 8])

            for pp in range(pb):
                bb_0to7 = avx_f32x8_load(~b[pp, 0])
                bb_8to15 = avx_f32x8_load(~b[pp, 8])

                aa = avx_f32x8_broadcast(~a[0, pp])
                c0 = avx_f32x8_fmadd(aa, bb_0to7, c0)
                c08 = avx_f32x8_fmadd(aa, bb_8to15, c08)
                aa = avx_f32x8_broadcast(~a[1, pp])
                c1 = avx_f32x8_fmadd(aa, bb_0to7, c1)
                c18 = avx_f32x8_fmadd(aa, bb_8to15, c18)
                aa = avx_f32x8_broadcast(~a[2, pp])
                c2 = avx_f32x8_fmadd(aa, bb_0to7, c2)
                c28 = avx_f32x8_fmadd(aa, bb_8to15, c28)
                aa = avx_f32x8_broadcast(~a[3, pp])
                c3 = avx_f32x8_fmadd(aa, bb_0to7, c3)
                c38 = avx_f32x8_fmadd(aa, bb_8to15, c38)
                aa = avx_f32x8_broadcast(~a[4, pp])
                c4 = avx_f32x8_fmadd(aa, bb_0to7, c4)
                c48 = avx_f32x8_fmadd(aa, bb_8to15, c48)
                aa = avx_f32x8_broadcast(~a[5, pp])
                c5 = avx_f32x8_fmadd(aa, bb_0to7, c5)
                c58 = avx_f32x8_fmadd(aa, bb_8to15, c58)

            avx_f32x8_store(~c[0, 0], c0)
            avx_f32x8_store(~c[0, 8], c08)
            avx_f32x8_store(~c[1, 0], c1)
            avx_f32x8_store(~c[1, 8], c18)
            avx_f32x8_store(~c[2, 0], c2)
            avx_f32x8_store(~c[2, 8], c28)
            avx_f32x8_store(~c[3, 0], c3)
            avx_f32x8_store(~c[3, 8], c38)
            avx_f32x8_store(~c[4, 0], c4)
            avx_f32x8_store(~c[4, 8], c48)
            avx_f32x8_store(~c[5, 0], c5)
            avx_f32x8_store(~c[5, 8], c58)

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
            for mpanel in grid(mpanels, attrs='p32'):
                mr = MR if mpanel != mpanels - 1 or _mr == 0 else _mr
                ii = mpanel * MR
                # Loop 1
                for npanel in range(npanels):
                    nr = NR if npanel != npanels - 1 or _nr == 0 else _nr
                    jj = npanel * NR
                    # micro-kernel
                    if mr == MR and nr == NR:
                        micro_kernel_6x16(~a[ii, 0], ~b[0, jj], ~c[ii, jj],
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
                        micro_kernel_6x16(~a[ii, 0], ~b[0, jj], temp_c,
                                          pb, MR, NR)
                        for remain_row, remain_col in grid(mr, nr):
                            c[ii + remain_row, jj + remain_col] += temp_c[remain_row, remain_col]

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

            mbs = (m_size + MC - 1) // MC
            nbs = (n_size + NC - 1) // NC
            kbs = (k_size + KC - 1) // KC

            for mb in grid(mbs):
                i = mb * MC
                ib = min(MC, m_size - i)
                p = 0
                # while p < k_size:
                for kb in range(kbs):
                    p = kb * KC
                    # pack A into contiguous memory
                    pb = min(KC, k_size - p)
                    mp = ib // MR
                    mr = ib % MR
                    for micropanel_idx in grid(mp):
                        panel_row_start = micropanel_idx * MR
                        for micropanel_col in range(pb):
                            for micropanel_row in range(MR):
                                aip_packed[panel_row_start + micropanel_row, micropanel_col] = \
                                    a[i + micropanel_row + panel_row_start, p + micropanel_col]
                    # # TODO: pack the remaining if the shape is not 'nice'
                    if mr > 0:
                        remain_start_row = mp * MR
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
                    # j = 0
                    # while j < n_size:
                    for nb in range(nbs):
                        j = nb * NC
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
    #################################################
    assert isinstance(matmul_kernel, hidet.ir.Function)
    matmul_kernel.kind = 'host_kernel'

    ir_module = script_module.ir_module()
    # add_packed_func(ir_module, matmul_kernel, pack_func_name='matmul6')
    compiled_function = hidet.driver.build_ir_module(ir_module)
    return compiled_function


def ff():
    func = matmul_kernel5()

    # for m, n, k in [(64, 64, 64), (110, 111, 111), (101, 101, 37), (111, 367, 369), (224, 562, 325),
    #                 (256, 256, 256), (333, 444, 555), (512, 512, 512), (1024, 1024, 1024), (1111, 1111, 1111), (1111, 1314, 533),
    #                 (1440, 1440, 1440), (1920, 1920, 1920), (2023, 2023, 2023), (5247, 4202, 3175)]:
    for m, n, k in [(4096, 4096, 4096)]:
        a = hidet.randn([m, k], dtype='float32').cpu()
        b = hidet.randn([k, n], dtype='float32').cpu()
        c = hidet.zeros([m, n]).cpu()
        func(a, b, c, m, n, k)
        # numpy.testing.assert_allclose(
        #     actual=c.cpu().numpy(),
        #     desired=a.cpu().numpy() @ b.cpu().numpy(),
        #     atol=1e-4,
        #     rtol=1e-4
        # )

        hidet_latency = hidet.utils.benchmark_func(
            lambda: func(a, b, c, m, n, k), repeat=30
        )

        np_latency = hidet.utils.benchmark_func(
            lambda: a.cpu().numpy() @ b.cpu().numpy(), repeat=30
        )

        print(f'{m} x {k} x {n}: hidet takes {hidet_latency:.2f} ms')
        print(f'{m} x {k} x {n}: numpy takes {np_latency: .2f} ms')


ff()

#### -O3
# 1 x 1 x 74: hidet takes 0.03 ms
# 1 x 1 x 74: numpy takes  0.03 ms
# 64 x 64 x 64: hidet takes 0.04 ms
# 64 x 64 x 64: numpy takes  0.03 ms
# 110 x 111 x 111: hidet takes 0.08 ms
# 110 x 111 x 111: numpy takes  0.17 ms
# 101 x 37 x 101: hidet takes 0.04 ms
# 101 x 37 x 101: numpy takes  0.11 ms
# 111 x 369 x 367: hidet takes 0.46 ms
# 111 x 369 x 367: numpy takes  0.31 ms
# 224 x 325 x 562: hidet takes 0.71 ms
# 224 x 325 x 562: numpy takes  0.48 ms
# 256 x 256 x 256: hidet takes 0.26 ms
# 256 x 256 x 256: numpy takes  0.21 ms
# 333 x 555 x 444: hidet takes 1.15 ms
# 333 x 555 x 444: numpy takes  0.76 ms
# 512 x 512 x 512: hidet takes 1.26 ms
# 512 x 512 x 512: numpy takes  0.70 ms
# 1024 x 1024 x 1024: hidet takes 6.84 ms
# 1024 x 1024 x 1024: numpy takes  3.52 ms
# 1024 x 768 x 512: hidet takes 2.92 ms
# 1024 x 768 x 512: numpy takes  1.66 ms
# 480 x 480 x 480: hidet takes 0.99 ms
# 480 x 480 x 480: numpy takes  0.98 ms
# 720 x 720 x 720: hidet takes 2.71 ms
# 720 x 720 x 720: numpy takes  2.34 ms
# 720 x 1440 x 960: hidet takes 7.01 ms
# 720 x 1440 x 960: numpy takes  3.56 ms
# 1111 x 1111 x 1111: hidet takes 8.62 ms
# 1111 x 1111 x 1111: numpy takes  3.52 ms
# 1111 x 533 x 1314: hidet takes 4.96 ms
# 1111 x 533 x 1314: numpy takes  1.97 ms

# NC = 384, KC = 512:
# 1 x 1 x 74: hidet takes 0.04 ms
# 1 x 1 x 74: numpy takes  0.03 ms
# 64 x 64 x 64: hidet takes 0.06 ms
# 64 x 64 x 64: numpy takes  0.04 ms
# 110 x 111 x 111: hidet takes 0.09 ms
# 110 x 111 x 111: numpy takes  0.18 ms
# 101 x 37 x 101: hidet takes 0.06 ms
# 101 x 37 x 101: numpy takes  0.11 ms
# 111 x 369 x 367: hidet takes 0.32 ms
# 111 x 369 x 367: numpy takes  0.52 ms
# 224 x 325 x 562: hidet takes 0.65 ms
# 224 x 325 x 562: numpy takes  0.56 ms
# 256 x 256 x 256: hidet takes 0.31 ms
# 256 x 256 x 256: numpy takes  0.20 ms
# 333 x 555 x 444: hidet takes 1.01 ms
# 333 x 555 x 444: numpy takes  1.07 ms
# 512 x 512 x 512: hidet takes 1.03 ms
# 512 x 512 x 512: numpy takes  0.64 ms
# 1024 x 1024 x 1024: hidet takes 6.34 ms
# 1024 x 1024 x 1024: numpy takes  3.43 ms
# 1024 x 768 x 512: hidet takes 2.44 ms
# 1024 x 768 x 512: numpy takes  1.68 ms
# 480 x 480 x 480: hidet takes 1.16 ms
# 480 x 480 x 480: numpy takes  0.96 ms
# 720 x 720 x 720: hidet takes 2.70 ms
# 720 x 720 x 720: numpy takes  2.28 ms
# 720 x 1440 x 960: hidet takes 6.64 ms
# 720 x 1440 x 960: numpy takes  2.97 ms
# 1111 x 1111 x 1111: hidet takes 7.82 ms
# 1111 x 1111 x 1111: numpy takes  4.04 ms
# 1111 x 533 x 1314: hidet takes 4.35 ms
# 1111 x 533 x 1314: numpy takes  2.87 ms
#
# NC = 384, KC = 1024, p24:
# 1 x 1 x 74: hidet takes 0.06 ms
# 1 x 1 x 74: numpy takes  0.03 ms
# 64 x 64 x 64: hidet takes 0.07 ms
# 64 x 64 x 64: numpy takes  0.04 ms
# 110 x 111 x 111: hidet takes 0.09 ms
# 110 x 111 x 111: numpy takes  0.11 ms
# 101 x 37 x 101: hidet takes 0.07 ms
# 101 x 37 x 101: numpy takes  0.11 ms
# 111 x 369 x 367: hidet takes 0.43 ms
# 111 x 369 x 367: numpy takes  0.30 ms
# 224 x 325 x 562: hidet takes 0.66 ms
# 224 x 325 x 562: numpy takes  0.41 ms
# 256 x 256 x 256: hidet takes 0.33 ms
# 256 x 256 x 256: numpy takes  0.15 ms
# 333 x 555 x 444: hidet takes 1.01 ms
# 333 x 555 x 444: numpy takes  0.81 ms
# 512 x 512 x 512: hidet takes 1.31 ms
# 512 x 512 x 512: numpy takes  0.60 ms
# 1024 x 1024 x 1024: hidet takes 6.24 ms
# 1024 x 1024 x 1024: numpy takes  3.40 ms
# 1024 x 768 x 512: hidet takes 2.94 ms
# 1024 x 768 x 512: numpy takes  1.64 ms
# 480 x 480 x 480: hidet takes 0.90 ms
# 480 x 480 x 480: numpy takes  1.01 ms
# 720 x 720 x 720: hidet takes 2.17 ms
# 720 x 720 x 720: numpy takes  1.92 ms
# 720 x 1440 x 960: hidet takes 5.61 ms
# 720 x 1440 x 960: numpy takes  3.12 ms
# 1111 x 1111 x 1111: hidet takes 7.77 ms
# 1111 x 1111 x 1111: numpy takes  3.59 ms
# 1111 x 533 x 1314: hidet takes 4.38 ms
# 1111 x 533 x 1314: numpy takes  2.89 ms
#
# NC = 384, KC = 1024, p32:
# 1 x 1 x 74: hidet takes 0.06 ms
# 1 x 1 x 74: numpy takes  0.03 ms
# 64 x 64 x 64: hidet takes 0.19 ms
# 64 x 64 x 64: numpy takes  0.04 ms
# 110 x 111 x 111: hidet takes 0.29 ms
# 110 x 111 x 111: numpy takes  0.14 ms
# 101 x 37 x 101: hidet takes 0.08 ms
# 101 x 37 x 101: numpy takes  0.10 ms
# 111 x 369 x 367: hidet takes 0.44 ms
# 111 x 369 x 367: numpy takes  0.41 ms
# 224 x 325 x 562: hidet takes 0.65 ms
# 224 x 325 x 562: numpy takes  0.44 ms
# 256 x 256 x 256: hidet takes 0.33 ms
# 256 x 256 x 256: numpy takes  0.18 ms
# 333 x 555 x 444: hidet takes 1.00 ms
# 333 x 555 x 444: numpy takes  1.09 ms
# 512 x 512 x 512: hidet takes 1.25 ms
# 512 x 512 x 512: numpy takes  0.71 ms
# 1024 x 1024 x 1024: hidet takes 5.76 ms
# 1024 x 1024 x 1024: numpy takes  3.51 ms
# 1024 x 768 x 512: hidet takes 2.72 ms
# 1024 x 768 x 512: numpy takes  1.52 ms
# 480 x 480 x 480: hidet takes 1.14 ms
# 480 x 480 x 480: numpy takes  0.71 ms
# 720 x 720 x 720: hidet takes 2.49 ms
# 720 x 720 x 720: numpy takes  2.26 ms
# 720 x 1440 x 960: hidet takes 6.10 ms
# 720 x 1440 x 960: numpy takes  4.87 ms
# 1111 x 1111 x 1111: hidet takes 7.02 ms
# 1111 x 1111 x 1111: numpy takes  5.04 ms
# 1111 x 533 x 1314: hidet takes 3.94 ms
# 1111 x 533 x 1314: numpy takes  3.16 ms
