# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List
import numpy as np
import numpy.testing
import pytest
import hidet
from hidet import int32
from hidet.ir.func import IRModule, Function
from hidet.lang import attrs, printf, grid, spatial


def run(kernel: Function, shape: List[int]) -> hidet.Tensor:
    ir_module = IRModule()
    ir_module.add(kernel.name, kernel)
    func = hidet.driver.build_ir_module(ir_module)

    a = hidet.empty(shape, dtype=hidet.int32).cpu()
    func(a)
    return a


def test_for_range():
    @hidet.script
    def kernel(a: int32[10]):
        attrs.func_kind = 'cpu_kernel'
        p = 0
        for i in range(10):
            a[i] = p
            p = p + 1

    output = run(kernel, [10])
    assert np.all(output.numpy() == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))


def test_for_grid():
    @hidet.script
    def kernel(a: int32[2, 3]):
        attrs.func_kind = 'cpu_kernel'
        p = 0
        for i, j in grid(2, 3):
            a[i, j] = p
            p += 1

    expected = np.array([[0, 1, 2], [3, 4, 5]])
    assert np.all(run(kernel, [2, 3]).numpy() == expected)


def test_for_task_mapping():
    @hidet.script
    def kernel(a: int32[2, 3]):
        attrs.func_kind = 'cpu_kernel'
        p = 0
        for w in range(6):
            for i, j in spatial(2, 3).on(w):
                a[i, j] = p
            p += 1

    expected = np.array([[0, 1, 2], [3, 4, 5]])
    assert np.all(run(kernel, [2, 3]).numpy() == expected)


def test_tuple_as_index():
    @hidet.script
    def kernel(a: int32[2, 3]):
        attrs.func_kind = 'cpu_kernel'
        p = 0
        for axes in grid(2, 3):
            # the following three ways of indexing are equivalent
            a[axes] = p
            a[axes[0], axes[1]] = p
            a[axes[0]][axes[1]] = p
            p += 1

    expected = np.array([[0, 1, 2], [3, 4, 5]])
    assert np.all(run(kernel, [2, 3]).numpy() == expected)


@pytest.mark.parametrize('shape,axis', [([2, 3], 0), ([2, 3], 1), ([2, 3, 4], 1)])
def test_softmax(shape: List[int], axis: int):
    from hidet.lang import f32
    from hidet.lang import attrs
    from hidet.lang import tensor
    from hidet.lang import grid
    import math

    with hidet.script_module() as script_module:

        @hidet.script
        def kernel(x: f32[shape], y: f32[shape]):
            attrs.func_kind = 'cpu_kernel'
            spatial_shape = shape[:axis] + shape[axis + 1 :]
            reduce_extent = shape[axis]

            max_value = tensor('default', f32, shape=spatial_shape)  # max(x, axis)
            exp_value = tensor('default', f32, shape=shape)  # exp(x - max)
            sum_value = tensor('default', f32, shape=spatial_shape)  # sum(exp(x - max), axis)

            # max value
            for indices in grid(spatial_shape):
                max_value[indices] = -1e10
                for k in range(reduce_extent):
                    max_value[indices] = max(max_value[indices], x[indices[:axis] + (k,) + indices[axis:]])

            # exp(x - max)
            for indices in grid(shape):
                exp_value[indices] = math.exp(x[indices] - max_value[indices[:axis] + indices[axis + 1 :]])

            # sum(exp(x - max))
            for indices in grid(spatial_shape):
                sum_value[indices] = 0.0
                for k in range(reduce_extent):
                    sum_value[indices] += exp_value[indices[:axis] + (k,) + indices[axis:]]

            # exp(x - max) / sum(exp(x - max))
            for indices in grid(shape):
                y[indices] = exp_value[indices] / sum_value[indices[:axis] + indices[axis + 1 :]]

    func = script_module.build()
    x = hidet.randn(shape)
    y1 = hidet.ops.softmax(x, axis)
    y2 = hidet.empty(shape)
    func(x, y2)
    numpy.testing.assert_allclose(y1.numpy(), y2.numpy(), rtol=1e-5, atol=1e-5)


def test_bind_tuple():
    from hidet.lang import attrs
    from hidet.lang.mapping import spatial, repeat
    from hidet.lang import printf, grid

    with hidet.script_module() as script_module:

        @hidet.script
        def launch():
            attrs.func_kind = 'public'

            for w in grid(3, attrs='p'):
                for indices in repeat(2).spatial(3).on(w, bind_tuple=True):
                    printf("%d %d\n", w, indices[0])

            for w in grid(3, attrs='p'):
                for i in repeat(2).spatial(3).on(w):
                    printf("%d %d\n", w, i)

            for indices in grid(3, bind_tuple=True):
                printf("%d\n", indices[0])

            for indices in grid([3]):
                printf("%d\n", indices[0])

            for i in grid(3):
                printf("%d\n", i)

    cm = script_module.build()
    cm()
