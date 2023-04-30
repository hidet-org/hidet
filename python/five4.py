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

            MC: int32 = 256
            NC: int32 = 256
            KC: int32 = 256

            MR: int32 = 8
            NR: int32 = 8

            i = 0
            while i < m_size:
                ib = min(MC, m_size - i)
                # Loop 4
                p = 0
                while p < k_size:
                    pb = min(KC, k_size - p)
                    # loop 3
                    j = 0
                    while j < n_size:
                        jb = min(NC, n_size - j)
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
                                    pi = p + pp
                                    bb_0to7 = avx_f32x8_load(~b[pi, jidx])
                                    aa = avx_f32x8_broadcast(~a[iidx, pi])
                                    c0_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c0_0to7)
                                    aa = avx_f32x8_broadcast(~a[iidx+1, pi])
                                    c1_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c1_0to7)
                                    aa = avx_f32x8_broadcast(~a[iidx+2, pi])
                                    c2_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c2_0to7)
                                    aa = avx_f32x8_broadcast(~a[iidx+3, pi])
                                    c3_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c3_0to7)
                                    aa = avx_f32x8_broadcast(~a[iidx+4, pi])
                                    c4_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c4_0to7)
                                    aa = avx_f32x8_broadcast(~a[iidx+5, pi])
                                    c5_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c5_0to7)
                                    aa = avx_f32x8_broadcast(~a[iidx+6, pi])
                                    c6_0to7 = avx_f32x8_fmadd(aa, bb_0to7, c6_0to7)
                                    aa = avx_f32x8_broadcast(~a[iidx+7, pi])
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

    for m, n, k in [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (1024, 512, 768),
                    (480, 480, 480), (720, 720, 720), (720, 960, 1440)]:
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
# 256 x 256 x 256: hidet takes 0.62 ms
# 256 x 256 x 256: numpy takes  0.23 ms
# 512 x 512 x 512: hidet takes 5.27 ms
# 512 x 512 x 512: numpy takes  0.78 ms
# 1024 x 1024 x 1024: hidet takes 38.82 ms
# 1024 x 1024 x 1024: numpy takes  2.32 ms
# 1024 x 768 x 512: hidet takes 13.60 ms
# 1024 x 768 x 512: numpy takes  1.13 ms
# 480 x 480 x 480: hidet takes 4.22 ms
# 480 x 480 x 480: numpy takes  0.56 ms
# 720 x 720 x 720: hidet takes 11.49 ms
# 720 x 720 x 720: numpy takes  1.42 ms
# 720 x 1440 x 960: hidet takes 25.72 ms
# 720 x 1440 x 960: numpy takes  4.75 ms
