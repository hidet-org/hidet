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
from typing import List, Optional
from hidet.utils.py import initialize
from .base import SubgraphRewriteRule, TensorPattern, MatchDict, register_rewrite_rule


class ArithmeticSubgraphRewriteRule(SubgraphRewriteRule):
    def __init__(self, name, fsrc, fdst):
        super().__init__(name)
        x, y = TensorPattern.tensors(2, is_symbolic=True)  # can not be const
        a, b = TensorPattern.tensors(2, is_const=True)  # can not be symbolic
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.src = fsrc(x, y, a, b)
        self.fdst = fdst

    def source(self) -> List[TensorPattern]:
        return [self.src]

    def target(self, matched: MatchDict) -> Optional[List[TensorPattern]]:
        x, y, a, b = [matched[v] if v in matched else None for v in [self.x, self.y, self.a, self.b]]
        return [self.fdst(x, y, a, b)]
        # constructor = GraphConstructor(matched)
        # return [constructor.visit(self.tgt)]


@initialize()
def arithmetic_patterns():
    # # tensors can be used as pattern inputs
    # x, y, z = TensorPattern.tensors(3, is_symbolic=True)  # can not be const
    # a, b, c = TensorPattern.tensors(3, is_const=True)  # can not be symbolic
    #
    # (source, target) pattern pairs
    pairs = [
        ['a + x => x + a', lambda x, y, a, b: a + x, lambda x, y, a, b: x + a],
        ['x - a => x + (-a)', lambda x, y, a, b: x - a, lambda x, y, a, b: x + (-a)],
        ['(x + a) + b => x + (a + b)', lambda x, y, a, b: (x + a) + b, lambda x, y, a, b: x + (a + b)],
        ['(x + a) * b => x * b + a * b', lambda x, y, a, b: (x + a) * b, lambda x, y, a, b: x * b + a * b],
        [
            '(x + a) + (y + b) => (x + y) + (a + b)',
            lambda x, y, a, b: (x + a) + (y + b),
            lambda x, y, a, b: (x + y) + (a + b),
        ],
    ]
    rules = [ArithmeticSubgraphRewriteRule(name, src, tgt) for name, src, tgt in pairs]
    for rule in rules:
        register_rewrite_rule(rule)
