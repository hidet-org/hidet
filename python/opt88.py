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

            mblk: int32 = 256
            kblk: int32 = 256
            p = 0
            while p < k_size:
                pb = min(k_size - p, kblk)
                i = 0
                while i < m_size:
                    ib = min(m_size - i, mblk)
                    jj = 0
                    while jj < n_size:
                        ii = 0
                        while ii < ib:

                            iidx = i+ii

                            c0_0123 = avx_f32x4_load(~c[iidx, jj])
                            c1_0123 = avx_f32x4_load(~c[iidx+1, jj])
                            c2_0123 = avx_f32x4_load(~c[iidx+2, jj])
                            c3_0123 = avx_f32x4_load(~c[iidx+3, jj])

                            for pp in range(pb):
                                pi = p + pp

                                bb_0123 = avx_f32x4_load(~b[pi, jj]) 

                                aidx = i + ii
                                aa = avx_f32x4_broadcast(~a[aidx, pi])

                                c0_0123 = avx_f32x4_fmadd(aa, bb_0123, c0_0123)

                                aa = avx_f32x4_broadcast(~a[aidx+1, pi])
                                c1_0123 = avx_f32x4_fmadd(aa, bb_0123, c1_0123)

                                aa = avx_f32x4_broadcast(~a[aidx+2, pi])
                                c2_0123 = avx_f32x4_fmadd(aa, bb_0123, c2_0123)

                                aa = avx_f32x4_broadcast(~a[aidx+3, pi])
                                c3_0123 = avx_f32x4_fmadd(aa, bb_0123, c3_0123)

                            idx = i + ii

                            avx_f32x4_store(~c[idx, jj], c0_0123)

                            avx_f32x4_store(~c[idx+1, jj], c1_0123)

                            avx_f32x4_store(~c[idx+2, jj], c2_0123)

                            avx_f32x4_store(~c[idx+3, jj], c3_0123)

                            ii += 4
                        jj += 4
                    i += mblk
                p += kblk



#################################################3
    assert isinstance(matmul_kernel, hidet.ir.Function)
    matmul_kernel.kind = 'host_kernel'

    ir_module = script_module.ir_module()
    add_packed_func(ir_module, matmul_kernel, pack_func_name='matmul6')
    compiled_function = hidet.driver.build_ir_module(ir_module)
    return compiled_function

def ff():
    func = matmul_kernel5()

    for m, n, k in [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (1024, 512, 768), (333, 444, 555),
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
            lambda: func(a, b, c, m, n, k), repeat=10
        )

        np_latency = hidet.utils.benchmark_func(
            lambda: a.cpu().numpy() @ b.cpu().numpy()
        )

        print(f'{m} x {k} x {n}: hidet takes {hidet_latency:.2f} ms')
        print(f'{m} x {k} x {n}: numpy takes {np_latency: .2f} ms')



ff()

#     256 x 256 x 256: hidet takes 1.73 ms
#     256 x 256 x 256: numpy takes  0.16 ms
#     512 x 512 x 512: hidet takes 11.73 ms
#     512 x 512 x 512: numpy takes  0.57 ms
#     1024 x 1024 x 1024: hidet takes 181.92 ms
#     1024 x 1024 x 1024: numpy takes  3.38 ms
#     1024 x 768 x 512: hidet takes 35.10 ms
#     1024 x 768 x 512: numpy takes  1.44 ms
#     333 x 555 x 444: hidet takes 5.92 ms
#     333 x 555 x 444: numpy takes  0.73 ms
#     480 x 480 x 480: hidet takes 7.98 ms
#     480 x 480 x 480: numpy takes  0.60 ms
#     720 x 720 x 720: hidet takes 27.06 ms
#     720 x 720 x 720: numpy takes  1.40 ms
#     720 x 1440 x 960: hidet takes 73.55 ms
#     720 x 1440 x 960: numpy takes  2.83 ms