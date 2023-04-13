import os


def main():
    import hidet

    ret = os.fork()
    if ret == 0:
        # child process
        p = 'child'
        a = hidet.randn([3, 4], device='cuda')
    else:
        # parent process
        p = 'parent'
        a = hidet.randn([3, 4], device='cuda')
    print('I am the {} process, tensor: \n{}'.format(p, a))


main()
