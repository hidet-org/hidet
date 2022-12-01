import pytest
import hidet
from hidet.ir.func import IRModule, Function
from hidet.transforms.tools import add_packed_func

from hidet.lang import attr, printf, grid, spatial


def check(kernel: Function, output: str, capfd):
    with capfd.disabled():
        ir_module = IRModule()
        ir_module.add(kernel.name, kernel)
        add_packed_func(ir_module, kernel, 'func')
        func = hidet.driver.build_ir_module(ir_module, func_name='func')
    func()
    captured = capfd.readouterr()
    assert captured.out == output


def test_for_range(capfd):
    @hidet.script
    def kernel():
        attr.func_kind = 'host_kernel'
        for i in range(10):
            printf('%d ', i)

    check(kernel, '0 1 2 3 4 5 6 7 8 9 ', capfd)


def test_for_grid(capfd):
    @hidet.script
    def kernel():
        attr.func_kind = 'host_kernel'
        for i, j in grid(2, 3):
            printf(r'%d %d\n', i, j)

    expected = ''
    for i in range(2):
        for j in range(3):
            expected += '{} {}\n'.format(i, j)

    check(kernel, expected, capfd)


def test_for_task_mapping(capfd):
    @hidet.script
    def kernel():
        attr.func_kind = 'host_kernel'
        for w in range(6):
            for i, j in spatial(2, 3).on(w):
                printf(r'%d %d\n', i, j)

    expected = ''
    for w in range(6):
        i, j = w // 3, w % 3
        expected += '{} {}\n'.format(i, j)
    check(kernel, expected, capfd)


def test_tuple_as_index(capfd):
    @hidet.script
    def kernel():
        attr.func_kind = 'host_kernel'
        for axes in grid(2, 3):
            printf(r'%d %d\n', axes[0], axes[1])

    expected = ''
    for i in range(2):
        for j in range(3):
            expected += '{} {}\n'.format(i, j)

    check(kernel, expected, capfd)

