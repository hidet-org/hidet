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

            MC = 256
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
                # Loop 4
                p = 0
                while p < k_size:
                    pb = min(KC, k_size - p)
                    # # TODO: pack the column panel of A
                    # # panelA_start_row = i
                    panelA_row_offset = 0
                    while panelA_row_offset < ib:
                        # panelA_row = panelA_start_row + panelA_row_offset
                        for micropanelA_col in range(pb):
                            for micropanelA_row in range(MR):
                                aip_packed[panelA_row_offset + micropanelA_row, micropanelA_col] = a[i+micropanelA_row + panelA_row_offset, p+micropanelA_col]

                        panelA_row_offset += MR
                    ## End of packing A
                    # loop 3
                    j = 0
                    while j < n_size:
                        jb = min(NC, n_size - j)
                        # TODO: back the block of B into contiguous memory
                        blockB_col_offset = 0
                        while blockB_col_offset < jb:
                            for blockB_row in range(pb):
                                for blockB_column in range(NR):
                                    bpj_packed[blockB_row, blockB_column+blockB_col_offset] = b[p+blockB_row, j+blockB_column+blockB_col_offset]
                            blockB_col_offset += NR

                        # Loop 2
                        ii = 0
                        while ii < ib:
                            iidx = i + ii
                            # Loop 1
                            jj = 0
                            while jj < jb:
                                jidx = j + jj
                                # micro-kernel
                                c0_0to7 = avx_f32x8_load(~c[iidx, jidx])
                                c1_0to7 = avx_f32x8_load(~c[iidx+1, jidx])
                                c2_0to7 = avx_f32x8_load(~c[iidx+2, jidx])
                                c3_0to7 = avx_f32x8_load(~c[iidx+3, jidx])
                                c4_0to7 = avx_f32x8_load(~c[iidx+4, jidx])
                                c5_0to7 = avx_f32x8_load(~c[iidx+5, jidx])
                                c6_0to7 = avx_f32x8_load(~c[iidx+6, jidx])
                                c7_0to7 = avx_f32x8_load(~c[iidx+7, jidx])

                                for pp in range(pb):
                                    bb_0to7 = avx_f32x8_load(~bpj_packed[pp, jj])

                                    aa = avx_f32x8_broadcast(~aip_packed[ii, pp])
                                    c0_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c0_0to7)
                                    aa = avx_f32x8_broadcast(~aip_packed[ii+1, pp])
                                    c1_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c1_0to7)
                                    aa = avx_f32x8_broadcast(~aip_packed[ii+2, pp])
                                    c2_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c2_0to7)
                                    aa = avx_f32x8_broadcast(~aip_packed[ii+3, pp])
                                    c3_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c3_0to7)
                                    aa = avx_f32x8_broadcast(~aip_packed[ii+4, pp])
                                    c4_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c4_0to7)
                                    aa = avx_f32x8_broadcast(~aip_packed[ii+5, pp])
                                    c5_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c5_0to7)
                                    aa = avx_f32x8_broadcast(~aip_packed[ii+6, pp])
                                    c6_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c6_0to7)
                                    aa = avx_f32x8_broadcast(~aip_packed[ii+7, pp])
                                    c7_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c7_0to7)
                                avx_f32x8_store(~c[iidx, jidx], c0_0to7)
                                avx_f32x8_store(~c[iidx + 1, jidx], c1_0to7)
                                avx_f32x8_store(~c[iidx + 2, jidx], c2_0to7)
                                avx_f32x8_store(~c[iidx + 3, jidx], c3_0to7)
                                avx_f32x8_store(~c[iidx + 4, jidx], c4_0to7)
                                avx_f32x8_store(~c[iidx + 5, jidx], c5_0to7)
                                avx_f32x8_store(~c[iidx + 6, jidx], c6_0to7)
                                avx_f32x8_store(~c[iidx + 7, jidx], c7_0to7)
                                jj += NR
                            ii += MR
                        j += NC
                    p += KC
                i += MC

#################################################3
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
# 256 x 256 x 256: hidet takes 0.59 ms
# 256 x 256 x 256: numpy takes  0.14 ms
# 512 x 512 x 512: hidet takes 4.68 ms
# 512 x 512 x 512: numpy takes  0.48 ms
# 1024 x 1024 x 1024: hidet takes 26.53 ms
# 1024 x 1024 x 1024: numpy takes  3.36 ms
# 768 x 768 x 768: hidet takes 12.56 ms
# 768 x 768 x 768: numpy takes  1.02 ms
# 768 x 1024 x 512: hidet takes 11.78 ms
# 768 x 1024 x 512: numpy takes  1.55 ms