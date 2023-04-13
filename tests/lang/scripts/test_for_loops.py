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
import pytest
import hidet
from hidet import int32
from hidet.ir.func import IRModule, Function
from hidet.lang import attr, printf, grid, spatial


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
        attr.func_kind = 'host_kernel'
        p = 0
        for i in range(10):
            a[i] = p
            p = p + 1

    output = run(kernel, [10])
    assert np.all(output.numpy() == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))


def test_for_grid():
    @hidet.script
    def kernel(a: int32[2, 3]):
        attr.func_kind = 'host_kernel'
        p = 0
        for i, j in grid(2, 3):
            a[i, j] = p
            p += 1

    expected = np.array([[0, 1, 2], [3, 4, 5]])
    assert np.all(run(kernel, [2, 3]).numpy() == expected)


def test_for_task_mapping():
    @hidet.script
    def kernel(a: int32[2, 3]):
        attr.func_kind = 'host_kernel'
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
        attr.func_kind = 'host_kernel'
        p = 0
        for axes in grid(2, 3):
            # the following three ways of indexing are equivalent
            a[axes] = p
            a[axes[0], axes[1]] = p
            a[axes[0]][axes[1]] = p
            p += 1

    expected = np.array([[0, 1, 2], [3, 4, 5]])
    assert np.all(run(kernel, [2, 3]).numpy() == expected)
