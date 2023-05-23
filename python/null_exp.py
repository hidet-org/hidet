from hidet.ir.expr import cast

import hidet
from hidet.ir.layout import row_layout
from hidet.ir.stmt import DeclareScope
from hidet.ir.type import void_p

from hidet.lang.avx import aligned_alloc


def matmul_kernel5():
    from hidet.transforms.generate_packed_func import add_packed_func
    from hidet.lang import float32, int32
    from hidet.lang import as_tensor_pointer, tensor

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

            # aaa = tensor(scope=DeclareScope.Default, dtype=float32,
            #              layout=row_layout(10, 10))

            aaa = aligned_alloc(64, 2000)
            ap = as_tensor_pointer(aaa, float32, shape=[2000, 10]
            )

            nullptr = as_tensor_pointer(int32(0), float32, layout=row_layout(1, 1))


            # if a_ptr == 0:
            #     return
            # if b_ptr == nullptr:
            #     return
            # if c_ptr == nullptr2:
            #     return

            for i in range(m_size):
                for j in range(n_size):
                    for k in range(k_size):
                        c[i, j] += a[i, k] * b[k, j]
            for k in range(2000):
                for kk in range(10):
                    ap[k, kk] = k+kk


# ################################################3
    assert isinstance(matmul_kernel, hidet.ir.Function)
    matmul_kernel.kind = 'host_kernel'

    ir_module = script_module.ir_module()
    add_packed_func(ir_module, matmul_kernel, pack_func_name='matmul6')
    compiled_function = hidet.driver.build_ir_module(ir_module)
    return compiled_function


def ff():
    func = matmul_kernel5()

    for m, n, k in [(11, 11, 11)]:
        a = hidet.randn([m, k], dtype='float32').cpu()
        b = hidet.randn([k, n], dtype='float32').cpu()
        c = hidet.zeros([m, n]).cpu()
        func(a, b, c, m, n, k)

        hidet_latency = hidet.utils.benchmark_func(
            lambda: func(a, b, c, m, n, k), repeat=2
        )

        np_latency = hidet.utils.benchmark_func(
            lambda: a.cpu().numpy() @ b.cpu().numpy()
        )

        print(f'{m} x {k} x {n}: hidet takes {hidet_latency:.2f} ms')
        print(f'{m} x {k} x {n}: numpy takes {np_latency: .2f} ms')


ff()
