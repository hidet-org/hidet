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
from collections import namedtuple

import hidet
from hidet.ir.expr import if_then_else
from hidet.transforms.rule_based_simplifier import RuleBasedSimplifier
from hidet.utils import repeat_until_converge


def test_rule_based_simplify():
    n = hidet.symbol_var("n")
    m = hidet.symbol_var("m")
    testcase = namedtuple("testcase", ["expr", "expected"])
    cases = [
        testcase(expr=(n + 1) - (1 + n), expected=0),
        testcase(expr=if_then_else(n > 0, 1, 1), expected=1),
        testcase(expr=(n + m - m) - (m + n - m), expected=0),
        testcase(expr=(n + m) - (m + n), expected=0),
        testcase(expr=n / n, expected=1),
    ]

    simp = RuleBasedSimplifier()
    for expr, expected in cases:
        assert repeat_until_converge(simp, expr) == expected
