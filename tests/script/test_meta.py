import hidet


def test_args():
    from hidet.lang import attrs, meta, printf, int32

    with hidet.script_module() as script_module:

        @hidet.script
        def launch(args: meta.types([int, bool, float, int32]), second_args: int, thrid_args: meta.types([int32])):
            attrs.func_kind = 'public'

            printf("%d\n", args[0])
            printf("%d\n", args[1])
            printf("%f\n", args[2])
            printf("%d\n", args[3])
            printf("%d\n", second_args)
            printf("%d\n", thrid_args[0])

    module = script_module.build()
    module(1, True, 0.1, 2, 3, 4)


def test_meta_range():
    from hidet.lang import attrs, meta, printf

    with hidet.script_module() as script_module:

        @hidet.script
        def launch():
            attrs.func_kind = 'public'

            for i in meta.range(10):
                for j in range(i):
                    printf("%d ", j)
                printf("\n")

    module = script_module.build()
    module()
