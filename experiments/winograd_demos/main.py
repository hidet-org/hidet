import hidet
from hidet.tos.operators import conv2d, conv2d_winograd


def demo_conv2d_winograd():
    x = hidet.randn([1, 6, 32, 32])
    w = hidet.randn([6, 12, 3, 3])
    y = conv2d_winograd(x, w, padding=[1, 1])


def main():
    demo_conv2d_winograd()


if __name__ == '__main__':
    main()
