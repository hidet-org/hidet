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

    with hidet.lang.script_module() as script_module:
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

            msize_b = (m_size + MC - 1) // MC
            nsize_b = (n_size + NC - 1) // NC
            ksize_b = (k_size + KC - 1) // KC
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
                                        a[i+micropanel_row+panel_row_start, p+micropanel_col]
                    # TODO: pack the remaining if the shape is not 'nice'
                    if mr > 0:
                        remain_start_row = mp * MR
                        for remain_col in range(pb):
                            for remain_row in range(mr):
                                aip_packed[remain_start_row + remain_row, remain_col] = \
                                        a[i+remain_start_row+remain_row, p+remain_col]
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
                                            b[p+micropanel_row, j+micropanel_col+panel_col_start]
                        if nr > 0:
                            remain_col_start = np * NR
                            for remain_row in range(pb):
                                for remain_col in range(nr):
                                    bpj_packed[remain_row, remain_col+remain_col_start] = \
                                        b[i+remain_row, p+remain_col+remain_col_start]
                                remain_col = nr
                                while remain_col < NR:
                                    bpj_packed[remain_row, remain_col_start+remain_col] = 0.0
                                    remain_col += 1
                        # End of packing B into contiguous memory
                        # Start of the macro-kernel
                        mpanels = (ib + MR - 1) // MR
                        npanels = (jb + NR - 1) // NR
                        _mr = ib % MR
                        _nr = jb % NR
                        # Loop 2
                        for mpanel in range(mpanels):
                            mr = MR if mpanel != mpanels - 1 or _mr == 0 else _mr
                            ii = mpanel * MR
                            midx = i + ii
                            # Loop 1
                            for npanel in range(npanels):
                                nr = NR if npanel != npanels - 1 or _nr == 0 else _nr
                                jj = npanel * NR
                                nidx = j + jj
                                # micro-kernel
                                if mr == MR and nr == NR:
                                    c0_0to7 = avx_f32x8_load(~c[midx, nidx])
                                    c1_0to7 = avx_f32x8_load(~c[midx+1, nidx])
                                    c2_0to7 = avx_f32x8_load(~c[midx + 2, nidx])
                                    c3_0to7 = avx_f32x8_load(~c[midx + 3, nidx])
                                    c4_0to7 = avx_f32x8_load(~c[midx + 4, nidx])
                                    c5_0to7 = avx_f32x8_load(~c[midx + 5, nidx])
                                    c6_0to7 = avx_f32x8_load(~c[midx + 6, nidx])
                                    c7_0to7 = avx_f32x8_load(~c[midx + 7, nidx])
                                    for pp in range(pb):
                                        bb_0to7 = avx_f32x8_load(~bpj_packed[pp, jj])

                                        aa = avx_f32x8_broadcast(~aip_packed[ii, pp])
                                        c0_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c0_0to7)
                                        aa = avx_f32x8_broadcast(~aip_packed[ii + 1, pp])
                                        c1_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c1_0to7)
                                        aa = avx_f32x8_broadcast(~aip_packed[ii + 2, pp])
                                        c2_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c2_0to7)
                                        aa = avx_f32x8_broadcast(~aip_packed[ii + 3, pp])
                                        c3_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c3_0to7)
                                        aa = avx_f32x8_broadcast(~aip_packed[ii + 4, pp])
                                        c4_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c4_0to7)
                                        aa = avx_f32x8_broadcast(~aip_packed[ii + 5, pp])
                                        c5_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c5_0to7)
                                        aa = avx_f32x8_broadcast(~aip_packed[ii + 6, pp])
                                        c6_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c6_0to7)
                                        aa = avx_f32x8_broadcast(~aip_packed[ii + 7, pp])
                                        c7_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c7_0to7)
                                    avx_f32x8_store(~c[midx, nidx], c0_0to7)
                                    avx_f32x8_store(~c[midx + 1, nidx], c1_0to7)
                                    avx_f32x8_store(~c[midx + 2, nidx], c2_0to7)
                                    avx_f32x8_store(~c[midx + 3, nidx], c3_0to7)
                                    avx_f32x8_store(~c[midx + 4, nidx], c4_0to7)
                                    avx_f32x8_store(~c[midx + 5, nidx], c5_0to7)
                                    avx_f32x8_store(~c[midx + 6, nidx], c6_0to7)
                                    avx_f32x8_store(~c[midx + 7, nidx], c7_0to7)
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

    # for m, n, k in [(64, 64, 64), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (1024, 512, 768),
    #                 (480, 480, 480), (720, 720, 720), (720, 960, 1440)]:
    for m, n, k in [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (768, 768, 768), (768, 512, 1024)]:
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
            lambda: func(a, b, c, m, n, k), repeat=2
        )

        np_latency = hidet.utils.benchmark_func(
            lambda: a.cpu().numpy() @ b.cpu().numpy()
        )

        print(f'{m} x {k} x {n}: hidet takes {hidet_latency:.2f} ms')
        print(f'{m} x {k} x {n}: numpy takes {np_latency: .2f} ms')


ff()

#### -O3
# 256 x 256 x 256: hidet takes 0.58 ms
# 256 x 256 x 256: numpy takes  0.14 ms
# 512 x 512 x 512: hidet takes 4.42 ms
# 512 x 512 x 512: numpy takes  0.51 ms
# 1024 x 1024 x 1024: hidet takes 24.68 ms
# 1024 x 1024 x 1024: numpy takes  2.46 ms
# 768 x 768 x 768: hidet takes 12.01 ms
# 768 x 768 x 768: numpy takes  1.19 ms
# 768 x 1024 x 512: hidet takes 11.28 ms
# 768 x 1024 x 512: numpy takes  1.20 ms
