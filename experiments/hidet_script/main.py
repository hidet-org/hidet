import hidet
from hidet.transforms.tools  import add_packed_func

hidet.option.cache_dir('./outs')


def main():
    from hidet.lang import attr
    from hidet.lang import float32
    from hidet.lang import grid

    with hidet.script_module() as script_module:

        @hidet.script
        def kernel(a: float32[2, 3], b: float32[2, 3]):
            attr.func_kind = "cuda_kernel"
            attr.cuda_block_dim = 1
            attr.cuda_grid_dim = 1
            for indices in grid(2, 3):
                b[indices] = a[[indices[0]] + [0]] + 1.0

    ir_module = script_module.ir_module()

    add_packed_func(ir_module, kernel, 'packed')

    func = hidet.driver.build_ir_module(
        ir_module,
        func_name='packed',
    )
    aa = hidet.ones([2, 3])
    bb = hidet.empty([2, 3])
    func(aa, bb)
    print(aa)
    print(bb)


if __name__ == '__main__':
    main()
