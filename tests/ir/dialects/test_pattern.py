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
from hidet.ir.expr import var, Add
from hidet.ir.dialects.pattern import match, PlaceholderExpr


def check_pairs(pairs):
    for pattern, target, expect in pairs:
        actual, msg = match(pattern, target)
        if expect is None:
            assert actual is None
        else:
            for p in expect:
                assert p in actual
                assert expect[p] is actual[p]


def test_normal_expr():
    a, b = PlaceholderExpr(), PlaceholderExpr()
    c, d = var('c'), var('d')

    pairs = [
        (Add(a, b), Add(c, d), {a: c, b: d}),
        (Add(a, a), Add(c, c), {a: c}),
        (Add(a, a), Add(c, d), None),
        (Add(a, b), Add(c, c), {a: c, b: c}),
    ]
    check_pairs(pairs)


def test_any_pattern():
    a, c, d = var('a'), var('c'), var('d')
    b = PlaceholderExpr()
    s = Add(a, b)
    any_expr = PlaceholderExpr()

    pairs = [(any_expr, s, {any_expr: s}), (Add(any_expr, b), Add(c, d), {any_expr: c, b: d})]
    check_pairs(pairs)


if __name__ == '__main__':
    pytest.main(__file__)
