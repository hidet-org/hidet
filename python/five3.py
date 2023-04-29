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

            MR: int32 = 4
            NR: int32 = 8

            j = 0
            while j < n_size:
                jb = min(NC, n_size - j)
                # Loop 4
                p = 0
                while p < k_size:
                    pb = min(KC, k_size - p)
                    # Loop 3
                    i = 0
                    while i < m_size:
                        ib = min(MC, m_size - i)
                        # Loop 2
                        jj = 0
                        while jj < jb:
                            jidx = j + jj
                            # Loop 1
                            ii = 0
                            while ii < ib:

                                iidx = i + ii
                                # micro-kernel
                                c0_0to7 = avx_f32x8_load(~c[iidx, jidx])

                                c1_0to7 = avx_f32x8_load(~c[iidx+1, jidx])

                                c2_0to7 = avx_f32x8_load(~c[iidx+2, jidx])

                                c3_0to7 = avx_f32x8_load(~c[iidx+3, jidx])

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

                                avx_f32x8_store(~c[iidx, jidx], c0_0to7)
                                avx_f32x8_store(~c[iidx+1, jidx], c1_0to7)
                                avx_f32x8_store(~c[iidx+2, jidx], c2_0to7)
                                avx_f32x8_store(~c[iidx+3, jidx], c3_0to7)

                                ii += MR
                            jj += NR
                        i += MC

                    p += KC

                j += NC

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
# 256 x 256 x 256: hidet takes 1.05 ms
# 256 x 256 x 256: numpy takes  0.18 ms
# 512 x 512 x 512: hidet takes 9.14 ms
# 512 x 512 x 512: numpy takes  0.69 ms
# 1024 x 1024 x 1024: hidet takes 75.51 ms
# 1024 x 1024 x 1024: numpy takes  3.63 ms
# 1024 x 768 x 512: hidet takes 21.92 ms
# 1024 x 768 x 512: numpy takes  1.10 ms
# 480 x 480 x 480: hidet takes 7.23 ms
# 480 x 480 x 480: numpy takes  0.58 ms
# 720 x 720 x 720: hidet takes 17.23 ms
# 720 x 720 x 720: numpy takes  1.40 ms
# 720 x 1440 x 960: hidet takes 44.92 ms
# 720 x 1440 x 960: numpy takes  2.86 ms

