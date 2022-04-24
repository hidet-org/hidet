import numpy as np
import hidet
from hidet.implement import impl_context
from hidet.implement.cuda import CudaStaticComputeImplementer, CudaGridStaticBatchedMatmulImplementer
from hidet.tos.operators import conv2d, conv2d_winograd, conv2d_default


def demo_check_conv2d_winograd():
    x = hidet.randn([2, 3, 12, 12])
    w = hidet.randn([6, 3, 3, 3])

    with impl_context(allowed=[CudaStaticComputeImplementer, CudaGridStaticBatchedMatmulImplementer]):
        y_win = conv2d_winograd(x, w, 0)
    y_gemm = conv2d_default(x, w, 0, 1)
    np.testing.assert_allclose(y_win.numpy(), y_gemm.numpy(), 1e-5, 1e-5)


def demo_conv2d_winograd():
    with impl_context(allowed=[CudaStaticComputeImplementer, CudaGridStaticBatchedMatmulImplementer]):
        x = hidet.ones([1, 2, 4, 4])
        # x = hidet.array(np.array(
        #     [[0, 0, 0, 0, 0, 0],
        #      [0, 1, 1, 1, 1, 0],
        #      [0, 1, 1, 1, 1, 0],
        #      [0, 1, 1, 1, 1, 0],
        #      [0, 1, 1, 1, 1, 0],
        #      [0, 0, 0, 0, 0, 0]]
        # ).astype(np.float32)).unsqueeze([0, 1])
        w = hidet.ones([2, 2, 3, 3])
        y = conv2d_winograd(x, w, padding=0)
        print(y)

        # x = hidet.randn([1, 6, 32, 32])
        # w = hidet.randn([12, 6, 3, 3])
        # y = conv2d_winograd(x, w, padding=1)
        # yy = conv2d_default(x, w, padding=1, stride=1)
        # print(y - yy)


def main():
    # demo_conv2d_winograd()
    demo_check_conv2d_winograd()


if __name__ == '__main__':
    main()
