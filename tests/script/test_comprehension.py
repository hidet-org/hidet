import pytest
import hidet


def test_list_comprehension():
    from hidet.lang import attrs, printf

    with hidet.script_module() as script_module:

        @hidet.script
        def func():
            attrs.func_kind = 'host_kernel'
            bs = 1.0 + 1
            shape = [bs, 3, 224, 224]
            a = [1, 2, 3]
            b = [i + 1 for i in range(3) if i != 2]
            c = [s / (i + 1) for i, s in enumerate(shape)]
            printf("%d %d %d\n", a[0], a[1], a[2])
            printf("%d\n", b[0])
            printf("%f\n", c[0])
            printf("%d\n", len(b))

    func = script_module.build()
    func()


def test_dict_comprehension():
    from hidet.lang import attrs, printf

    with hidet.script_module() as script_module:

        @hidet.script
        def func():
            attrs.func_kind = 'host_kernel'
            bs = 1.0 + 1
            shape = [bs, 3, 224, 224]
            a = {k: v for k, v in enumerate(shape)}
            b = {i: (j + 1) / 3 for i, j in enumerate(range(3)) if j != 2}
            printf("%d %d %d\n", a[0], a[1], a[2])
            printf("%f\n", b[1])
            printf("%d\n", len(b))

    func = script_module.build()
    func()


if __name__ == '__main__':
    pytest.main([__file__])
