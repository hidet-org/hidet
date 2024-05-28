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
import pytest

import hidet
from hidet.ir.cute import *
from hidet.ir.cute.layout import filter, make_layout


coalesce_tests = [
    TensorLayout(1, 0),
    TensorLayout(1, 1),
    TensorLayout((2, 4)),
    TensorLayout((2, 4, 6)),
    TensorLayout((2, 1, 6), (1, 6, 2)),
    TensorLayout((2, 1, 6), (1, 7, 2)),
    TensorLayout((2, 1, 6), (1, 7, 8)),
    TensorLayout((2, 4, 6)),
    TensorLayout((2, 1, 3), (2, 4, 4)),
    TensorLayout((2, 1, 3), (2, 0, 4)),
    TensorLayout(((2, 2), (2, 2)), ((1, 4), (8, 32))),
]


@pytest.mark.parametrize("a", coalesce_tests)
def test_coalesce(a: TensorLayout):
    coalesce_layout = coalesce(a)

    print(f"{a} => {coalesce_layout}")

    assert coalesce_layout.depth() <= 1
    assert coalesce_layout.size() == a.size()

    for i in range(a.size()):
        assert coalesce_layout(i) == a(i)


composition_tests = [
    (TensorLayout(1, 0), TensorLayout(1, 0)),
    (TensorLayout(1, 0), TensorLayout(1, 1)),
    (TensorLayout(1, 1), TensorLayout(1, 0)),
    (TensorLayout(1, 1), TensorLayout(1, 1)),
    (TensorLayout((4)), TensorLayout((4))),
    (TensorLayout((4), (2)), TensorLayout((4))),
    (TensorLayout((4), (0)), TensorLayout((4))),
    (TensorLayout((4)), TensorLayout((4), (0))),
    (TensorLayout((4)), TensorLayout((1), (0))),
    (TensorLayout((4)), TensorLayout((2))),
    (TensorLayout((4), (2)), TensorLayout((2))),
    (TensorLayout((4)), TensorLayout((2), (2))),
    (TensorLayout((4), (2)), TensorLayout((2), (2))),
    (TensorLayout((4, 3)), TensorLayout((12))),
    (TensorLayout((12)), TensorLayout((4, 3))),
    (TensorLayout((12), (2)), TensorLayout((4, 3))),
    (TensorLayout((12)), TensorLayout((4, 3), (3, 1))),
    (TensorLayout((12), (2)), TensorLayout((4, 3), (3, 1))),
    (TensorLayout((12)), TensorLayout((2, 3), (2, 4))),
    (TensorLayout((4, 3)), TensorLayout((4, 3))),
    (TensorLayout((4, 3)), TensorLayout((6), (2))),
    (TensorLayout((4, 3)), TensorLayout((6, 2), (2, 1))),
    (TensorLayout((4, 3), (3, 1)), TensorLayout((4, 3))),
    (TensorLayout((4, 3), (3, 1)), TensorLayout((12))),
    (TensorLayout((4, 3), (3, 1)), TensorLayout((6), (2))),
    (TensorLayout((4, 3), (3, 1)), TensorLayout((6, 2), (2, 1))),
    (TensorLayout((8, 8)), TensorLayout(((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32)))),
    (TensorLayout((8, 8), (8, 1)), TensorLayout(((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32)))),
    (TensorLayout((4, 2), (1, 16)), TensorLayout((4, 2), (2, 1))),
    (TensorLayout((2, 2), (2, 1)), TensorLayout((2, 2), (2, 1))),
    (TensorLayout((4, 8, 2)), TensorLayout((2, 2, 2), (2, 8, 1))),
    (TensorLayout((4, 8, 2), (2, 8, 1)), TensorLayout((2, 2, 2), (1, 8, 2))),
    (TensorLayout((4, 8, 2), (2, 8, 1)), TensorLayout((4, 2, 2), (2, 8, 1))),
    (TensorLayout((4, 3), (3, 1)), TensorLayout((6, 2), (2, 1))),
    (TensorLayout((1), (0)), TensorLayout((4))),
    (TensorLayout((1), (1)), TensorLayout((4))),
    (TensorLayout((4)), TensorLayout((4), (2))),
    (TensorLayout((4, 3), (3, 1)), TensorLayout((8))),
    (TensorLayout((4, 3, 1), (3, 1, 0)), TensorLayout((24))),
    (TensorLayout(3, 1), TensorLayout(4, 1)),
    # remove this testcase because the divisibility check fails.
    # CUTLASS removes the divisibility check because they should support
    # dynamic shapes.
    # We keep the divisibility check to reject incorrect composition results
    # (TensorLayout((48, 24, 5), (1, 128, 3072)), TensorLayout(32, 1)),
    # FIXME: the testcase
    # (TensorLayout((4, 3), (3, 1)), TensorLayout((24))),
]


@pytest.mark.parametrize("a,b", composition_tests)
def test_composition(a: TensorLayout, b: TensorLayout):
    layoutR = composition(a, b)
    print(f"{a} o {b} => {layoutR}")
    assert compatible(b.shape, layoutR.shape)
    for i in range(layoutR.size()):
        assert layoutR(i) == a(b(i))

    # FIXME
    # a = TensorLayout((4, 3), (3, 1))
    # b = TensorLayout((24))
    # test(a, b)
    # divisibility


complement_tests = [
    (TensorLayout(1, 0), None),
    (TensorLayout(1, 0), 2),
    (TensorLayout(1, 1), None),
    (TensorLayout(1, 1), 2),
    (TensorLayout(1, 2), 1),
    (TensorLayout(1, 2), 2),
    (TensorLayout(1, 2), 8),
    (TensorLayout(4, 0), 1),
    (TensorLayout(4, 0), 2),
    (TensorLayout(4, 0), 8),
    (TensorLayout(4, 1), 1),
    (TensorLayout(4, 1), 2),
    (TensorLayout(4, 1), 8),
    (TensorLayout(4, 2), 1),
    (TensorLayout(4, 2), None),
    (TensorLayout(4, 2), 16),
    (TensorLayout(4, 4), 1),
    (TensorLayout(4, 4), None),
    (TensorLayout(4, 4), 17),
    (TensorLayout((2, 4)), None),
    (TensorLayout((2, 3)), None),
    (TensorLayout((2, 4), (1, 4)), None),
    (TensorLayout((2, 4, 8), (8, 1, 64)), None),
    (TensorLayout((2, 4, 8), (8, 1, 0)), None),
    (TensorLayout((2, 4, 8), (8, 1, 0)), 460),
    (TensorLayout(((2, 2), (2, 2)), ((1, 4), (8, 32))), None),
    (TensorLayout(((2, 2), (2, 2)), ((1, 32), (8, 4))), None),
    (TensorLayout((4, 6), (1, 6)), None),
    (TensorLayout((4, 2), (1, 10)), None),
    (TensorLayout((4, 2), (1, 16)), None),
    (TensorLayout(12), 1),
    (TensorLayout(12), None),
    (TensorLayout(12), 53),
    (TensorLayout(12), 128),
    (TensorLayout(12, 1), 1),
    (TensorLayout(12, 1), None),
    (TensorLayout(12, 1), 53),
    (TensorLayout(12, 1), 128),
    (TensorLayout(12, 2), 1),
    (TensorLayout(12, 2), None),
    (TensorLayout(12, 2), 53),
    (TensorLayout(12, 2), 128),
    (TensorLayout((3, 6), (1, 3)), None),
    (TensorLayout((3, 6), (1, 9)), None),
    (TensorLayout((3, 6), (1, 10)), None),
]


@pytest.mark.parametrize("a,cosize_hi", complement_tests)
def test_complement(a: TensorLayout, cosize_hi: int):
    if cosize_hi is None:
        cosize_hi = a.cosize()
    result = complement(a, cosize_hi)
    print(f"complement({a}, {cosize_hi}) => {result}")
    assert result.size() >= cosize_hi // filter(a).size()
    assert result.cosize() <= ceil_div(cosize_hi, a.cosize()) * a.cosize()
    for i in range(1, result.size()):
        assert result(i - 1) < result(i)
        for j in range(a.size()):
            assert result(i) != a(j)
    assert result.size() <= result.cosize()
    assert result.cosize() >= cosize_hi // filter(a).size()
    assert complement(make_layout(a, result)).size() == 1


left_inverse_tests = [
    TensorLayout((1), (0)),
    TensorLayout((1, 1), (0, 0)),
    TensorLayout((4), (1)),
    TensorLayout((4), (2)),
    TensorLayout((4), (2)),
    TensorLayout((8, 4)),
    TensorLayout((8, 4), (4, 1)),
    TensorLayout((2, 4, 6)),
    TensorLayout((2, 4, 6), (4, 1, 8)),
    TensorLayout((4, 2), (1, 16)),
]


@pytest.mark.parametrize("a", left_inverse_tests)
def test_left_inverse(a: TensorLayout):
    inv_layout = left_inverse(a)
    print(f"{a}^-1 => {inv_layout}")
    for i in range(a.size()):
        assert inv_layout(a(i)) == i
    print(f"Composition: {coalesce(composition(inv_layout, a))}")


right_inverse_tests = [
    TensorLayout(1, 0),
    TensorLayout(1, 1),
    TensorLayout(4, 0),
    TensorLayout(((1, 1)), ((0, 0))),
    TensorLayout(((3, 7)), ((0, 0))),
    TensorLayout((1), (1)),
    TensorLayout((4), (1)),
    TensorLayout((4), (2)),
    TensorLayout((2, 4), (0, 2)),
    TensorLayout((8, 4)),
    TensorLayout((8, 4), (4, 1)),
    TensorLayout((2, 4, 6)),
    TensorLayout((2, 4, 6), (4, 1, 8)),
    TensorLayout((2, 4, 4, 6), (4, 1, 0, 8)),
    TensorLayout((4, 2), (1, 16)),
    TensorLayout((4, 2), (1, 5)),
    TensorLayout((4, 2), (1, 4)),
]


@pytest.mark.parametrize("a", right_inverse_tests)
def test_right_inverse(a: TensorLayout):
    inv_layout = right_inverse(a)
    print(f"{a}^-1 => {inv_layout}")
    for i in range(inv_layout.size()):
        assert a(inv_layout(i)) == i
    print(f"Composition: {coalesce(composition(a, inv_layout))}")
