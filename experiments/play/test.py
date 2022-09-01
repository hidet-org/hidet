import hidet
import numpy as np


def demo_wmma():
    sub_type = 'bfloat16'
    mma = 'wmma_bf16_f32'
    a = hidet.randn([1, 128, 128], dtype=sub_type)
    b = hidet.randn([1, 128, 128], dtype=sub_type)
    c0 = hidet.ops.matmul(a.cast('float32'), b.cast('float32')).cast(sub_type)
    c2 = hidet.ops.matmul(a, b, algo='direct', mma=mma)
    np.set_printoptions(suppress=True, linewidth=180)
    print('a')
    print(a)
    print('b')
    print(b)
    print('c0')
    print(c0)
    print('c2')
    print(c2)
    print('c2 - c0')
    print(c2 - c0)
    np_c2 = c2.numpy()
    np_c0 = c0.numpy()
    print((np_c2 - np_c0).max())
    print(np_c2.max())
    print(np.any(np.isnan(np_c2)))
    np.testing.assert_allclose(actual=c2.numpy(), desired=c0.numpy(), atol=1e-1, rtol=1e-1)


def demo_bf16():
    a = hidet.randn([1, 16, 16], dtype='bfloat16')
    b = hidet.randn([1, 16, 16], dtype='bfloat16')
    c0 = hidet.ops.matmul(a.cast('float32'), b.cast('float32')).cast('bfloat16')
    c1 = hidet.ops.matmul(a, b, mma='wmma_bf16_f32')
    print(a)
    print(b)
    print(c0 - c1)


def demo_tf32():
    a = hidet.randn([1, 128, 128], dtype='float32')
    b = hidet.randn([1, 128, 128], dtype='float32')
    c0 = hidet.ops.matmul(a, b, mma='simt')
    c1 = hidet.ops.matmul(a, b, mma='wmma_tf32_f32')
    print(a)
    print(b)
    print(c0 - c1)


def demo_reduce():
    a = hidet.array(np.arange(9, dtype=np.float32).reshape([3, 3]))
    print(a)
    print(hidet.ops.reduce_mean(a, dims=[0], keep_dim=False))
    print(a.numpy().mean(axis=0, keepdims=False))


def demo_add():
    a = hidet.array(np.array([1.0], dtype=np.float16))
    b = hidet.array(np.array([2.0], dtype=np.float16))
    c = a + b
    print(a)
    print(b)
    print(c)


def demo_map():
    from hidet.ir.layout.task_layout import spatial_map
    a = spatial_map([3, 4])
    print(spatial_map([3, 1]) * spatial_map([1, 4]))
    print(spatial_map([1, 4]) * spatial_map([3, 1]))
    print(a)


def demo_matmul():
    hidet.space_level(0)
    m, n, k = 5120, 4096, 4096
    a = hidet.symbol([1, m, k])
    b = hidet.symbol([1, k, n])
    # c = hidet.ops.matmul(a, b, mma='wmma_f16_f32')
    c = hidet.ops.matmul(a, b, mma='simt')
    print(c.op.latency())


def demo_max():
    a = hidet.array([1, 2, 3])
    b = hidet.array([3, 2, 1])
    c = hidet.ops.max(a, b)
    print(c)


def demo_func():
    def f(b, *, c):
        print(b, c)
    f(1, c=2)


def demo_layout():
    from hidet.ir.layout import row_layout, col_layout, local_layout
    # print(row_layout(4, 4))
    print('A = row_layout(4, 4)')
    print(row_layout(4, 4))
    print('B = row_layout(4, 4).swizzle(dim=1)')
    print('B[i, j ^ i] = A[i, j]')
    print(row_layout(4, 4).swizzle(1))
    # print(col_layout(3, 4))
    # print(local_layout(3, 4))
    # print(row_layout(3, 4).swizzle(plan={1: 0}))
    # print((row_layout(3, 4) * row_layout(3, 4)))
    # print((row_layout(3, 4) * row_layout(3, 4)).swizzle(1))


if __name__ == '__main__':
    # demo_wmma()
    # demo_bf16()
    # demo_tf32()
    # demo_tf32()
    # demo_reduce()
    # demo_add()
    # demo_map()
    # demo_matmul()
    # demo_max()
    # demo_func()
    # demo_layout()
    demo_matmul_v2()
