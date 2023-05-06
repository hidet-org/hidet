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
from __future__ import annotations
from typing import Any, Tuple, Optional, Dict, ContextManager
from contextlib import ExitStack
from hidet.ir.node import Node
from hidet.ir.type import TypeNode
from hidet.ir.expr import Expr, Constant, Add, Sub, Multiply, Div, Mod, FloorDiv, LessThan, Equal, LessEqual
from hidet.ir.expr import BitwiseXor
from hidet.ir.expr import IfThenElse, LogicalAnd, LogicalOr, BinaryExpr


class PlaceholderExpr(Expr):
    def __init__(self, required_type: Optional[TypeNode] = None, require_const=False, require_non_const=False):
        super().__init__()
        self.required_type: Optional[TypeNode] = required_type
        self.require_const: bool = require_const
        self.require_non_const: bool = require_non_const


class NotMatchedError(Exception):
    def __init__(self, pattern, target, message=""):
        super().__init__(message)
        self.pattern = pattern
        self.target = target


class MatchContext:
    def __init__(self, matcher: PatternMatcher, pattern: Expr, target: Expr):
        self.matcher: PatternMatcher = matcher
        self.matched: Dict[Expr, Optional[Any]] = matcher.matched
        self.pattern: Expr = pattern
        self.target: Expr = target

    def __enter__(self):
        if self.pattern in self.matched:
            if self.matched[self.pattern] is not self.target:
                # we think the constant with the same value as the same object
                lhs, rhs = self.matched[self.pattern], self.target
                if isinstance(lhs, Constant) and isinstance(rhs, Constant) and lhs == rhs:
                    return
                raise NotMatchedError(self.pattern, self.target, 'Can not match a pattern to two different targets')
            else:
                return
        try:
            self.matched[self.pattern] = self.target
            self.matcher.match_dispatch(self.pattern, self.target)
        except NotMatchedError as e:
            # error from current <pattern, target>
            del self.matched[self.pattern]
            raise e

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type == NotMatchedError:
            # error from inner <pattern, target>
            # delete the matched target for pattern
            # do not return True, propagate the exception:
            #   1. it can be caught by pattern like CommutativeBinary to try other target,
            #   2. or, it will be caught by the of PatternMatcher.match , indicating failure of matching.
            if self.pattern is not None:
                del self.matched[self.pattern]


class PatternMatcher:
    """
    invariant: every time when we enter match(...)
        0 the self.matched[...] stored the matched patterns and targets, and ongoing matching must follow them
        1 if successful, all sub-expressions of pattern have been set in self.matched[...]
        2 if failed, it acted like we have not called this function (we treat `self.matched[v] = None`
          and v not in `self.matched` as the same state)
    """

    def __init__(self):
        from hidet.ir.tools import TypeInfer

        self.matched: Dict[Expr, Expr] = {}
        self.type_infer = TypeInfer()

    def __call__(self, pattern, target):
        return self.match(pattern, target)

    def match(self, pattern, target):
        self.matched.clear()
        try:
            with self.try_match(pattern, target):
                pass
            return self.matched, "Matched"
        except NotMatchedError as e:
            return None, str(e)

    def try_match(self, pattern: Expr, target: Expr) -> ContextManager:
        return MatchContext(self, pattern, target)

    def match_dispatch(self, pattern: Expr, target: Expr):
        if isinstance(pattern, (Add, Multiply, BitwiseXor, LogicalAnd, LogicalOr)):
            self.match_CommutativeBinary(pattern, target)
        elif isinstance(pattern, (Sub, Div, Mod, BitwiseXor, FloorDiv, LessThan, Equal, LessEqual)):
            self.match_Binary(pattern, target)
        elif isinstance(pattern, PlaceholderExpr):
            self.match_PlaceholderExpr(pattern, target)
        elif isinstance(pattern, Constant):
            self.match_Constant(pattern, target)
        elif isinstance(pattern, IfThenElse):
            self.match_IfThenElse(pattern, target)
        else:
            raise NotImplementedError(f'Pattern {type(pattern)} is not implemented')

    def match_CommutativeBinary(self, pattern: BinaryExpr, target: Expr):
        if not isinstance(target, type(pattern)):
            raise NotMatchedError(pattern, target, "Commutative binary op has not matched")
        try:
            with ExitStack() as stack:
                stack.enter_context(self.try_match(pattern.a, target.a))
                stack.enter_context(self.try_match(pattern.b, target.b))
        except NotMatchedError:
            pass
        else:
            return
        try:
            with ExitStack() as stack:
                stack.enter_context(self.try_match(pattern.a, target.b))
                stack.enter_context(self.try_match(pattern.b, target.a))
        except NotMatchedError:
            pass
        else:
            return

        raise NotMatchedError(pattern, target, "Commutative binary op has not matched")

    def match_Binary(self, pattern: BinaryExpr, target: Expr):
        if not isinstance(target, type(pattern)):
            raise NotMatchedError(pattern, target, "Binary op has not matched")
        with ExitStack() as stack:
            stack.enter_context(self.try_match(pattern.a, target.a))
            stack.enter_context(self.try_match(pattern.b, target.b))

    def match_IfThenElse(self, pattern: IfThenElse, target: Expr):
        if not isinstance(target, IfThenElse):
            raise NotMatchedError(pattern, target, "IfThenElse has not matched")
        with ExitStack() as stack:
            stack.enter_context(self.try_match(pattern.cond, target.cond))
            stack.enter_context(self.try_match(pattern.then_expr, target.then_expr))
            stack.enter_context(self.try_match(pattern.else_expr, target.else_expr))

    @staticmethod
    def match_Constant(pattern: Constant, target: Expr):
        if not isinstance(target, Constant):
            raise NotMatchedError(pattern, target, "Constant has not matched")
        if pattern.type != target.type:
            raise NotMatchedError(pattern, target, "Constant type has not matched")
        if pattern.value != target.value:
            raise NotMatchedError(pattern, target, "Constant value has not matched")

    def match_PlaceholderExpr(self, pattern: PlaceholderExpr, target: Expr):
        if pattern.required_type is not None:
            target_type = self.type_infer(target)
            if type(target_type) is not type(pattern.required_type):
                raise NotMatchedError(pattern, target, "PlaceholderExpr type class has not matched")
            if pattern.required_type.is_data_type():
                if target_type != pattern.required_type:
                    raise NotMatchedError(pattern, target, "PlaceholderExpr type has not matched")
        if pattern.require_const and not isinstance(target, Constant):
            raise NotMatchedError(pattern, target, "PlaceholderExpr require const has not matched")
        if pattern.require_non_const and isinstance(target, Constant):
            raise NotMatchedError(pattern, target, "PlaceholderExpr require non-const has not matched")


def match(pattern: Node, target: Node) -> Tuple[Optional[Dict[Node, Any]], str]:
    """
    :return: match, report
    """
    matcher = PatternMatcher()
    return matcher(pattern, target)
