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
from typing import Any, Sequence, Tuple, Type, Optional, List, Union, Dict, Callable, ContextManager
from contextlib import ExitStack
from hidet.ir.node import Node
from hidet.ir.type import TypeNode, DataType, TensorType, FuncType, data_type
from hidet.ir.expr import Expr, Constant, Add, Sub, Multiply, Div, Mod, FloorDiv, LessThan, Equal, LessEqual
from hidet.ir.expr import TensorElement, IfThenElse, Call, Var, LogicalAnd, LogicalOr, BinaryOp, convert, var
from hidet.ir.compute import TensorNode, ScalarNode, ReduceOperation, ReduceCompute
from hidet.ir.stmt import DeclareScope
from hidet.ir.layout import StridesLayout, DataLayout


class PatternNode(Node):
    # A pattern can match a series of exprs/types/other node objects
    pass


class StringPattern(PatternNode):
    pass


class TypePattern(TypeNode, PatternNode):
    pass


class ScalarTypePattern(TypePattern):
    def __init__(self, allowed_types=None):
        self.allowed_types: Optional[List[str]] = allowed_types


class TensorTypePattern(TypePattern):
    def __init__(self, scope=None, scalar_type=None, rank=None, shape=None, layout=None, allow_dynamic_size=False):
        self.rank: Optional[int] = rank
        self.scope: Optional[Union[DeclareScope, List[DeclareScope]]] = scope
        self.scalar_type: Optional[Union[DataType, ScalarTypePattern]] = scalar_type
        self.shape: Optional[List[Expr]] = shape
        self.layout: Optional[DataLayout] = layout
        self.allow_dynamic_size = allow_dynamic_size


class ExprPattern(Expr, PatternNode):
    pass


class AnyExpr(ExprPattern):
    def __init__(self, cls=None, exclude_cls=None):
        self.cls: Optional[Type[Expr]] = cls
        self.exclude_cls: Optional[Type[Expr]] = exclude_cls


class UnionPattern(PatternNode):
    def __init__(self, patterns):
        self.patterns: List[Node] = patterns


class OptionalPattern(PatternNode):
    def __init__(self, pattern):
        self.pattern = pattern


class NotMatchedError(Exception):
    def __init__(self, pattern, target, message=""):
        super().__init__(message)
        self.pattern = pattern
        self.target = target


Matchable = Optional[Union[Node, tuple]]


class MatchContext:
    def __init__(self, matcher: 'PatternMatcher', pattern: Node, target: Node):
        self.matcher = matcher
        self.matched = matcher.matched
        self.dispatch = matcher.dispatch_table()
        self.pattern: Matchable = pattern
        self.target: Matchable = target

    def __enter__(self):
        # pylint: disable=too-many-branches
        if self.pattern is None:
            # None in pattern matches anything
            return
        if self.target is None:
            if isinstance(self.pattern, OptionalPattern):
                self.matched[self.pattern] = None
                return
            else:
                raise NotMatchedError(self.pattern, self.target, 'Expect non-None target')
        assert not isinstance(self.target, list), self.target
        if self.pattern in self.matched:
            if self.matched[self.pattern] is not self.target:
                # we think the constant with the same value as the same object
                lhs, rhs = self.matched[self.pattern], self.target
                if isinstance(lhs, Constant) and isinstance(rhs, Constant):
                    if lhs.value == rhs.value:
                        return
                if isinstance(lhs, tuple) and isinstance(rhs, tuple):
                    # something like (None, None) with the same hash
                    if lhs is not rhs:
                        return
                raise NotMatchedError(self.pattern, self.target, 'Can not match a pattern to two different targets')
            else:
                return

        if not isinstance(self.pattern, PatternNode):
            # put all pattern class that allow to accept other classes
            PatternMatcher.check_type(self.pattern, self.target)
        try:
            self.matched[self.pattern] = self.target
            if isinstance(self.pattern, DataType):
                self.dispatch[DataType](self.matcher, self.pattern, self.target)
            else:
                # noinspection PyArgumentList
                self.dispatch[self.pattern.__class__](self.matcher, self.pattern, self.target)
        except NotMatchedError as e:
            # error from current <pattern, target>
            del self.matched[self.pattern]
            raise e

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type == NotMatchedError:
            # error from inner <pattern, target>
            # delete the matched target for pattern
            # do not return True, propagate the exception:
            #   1. it can be caught by pattern like UnionPattern to try other target,
            #   2. or, it will be caught by the of PatternMatcher.__call__, indicating failure of matching.
            if self.pattern is not None:
                del self.matched[self.pattern]


class PatternMatcher:
    """
    invariant: every time when we enter match(...)
        0 the self.matched[...] stored the matched patterns and targets, and ongoing matching must follow them
        1 if successful, all sub-expressions of pattern have been set in self.matched[...]
        2 if failed, it acted like we have not call this function (we treat self.matched[v] = None
          and v not in self.matched as the same state)
    """

    # pylint: disable=no-self-use
    _dispatch_table: Dict[Type[Node], Callable[[Node, Node], None]] = None

    @staticmethod
    def dispatch_table():
        if PatternMatcher._dispatch_table is None:
            PatternMatcher._dispatch_table = {
                # string
                StringPattern: PatternMatcher.match_StringPattern,
                # expr
                Add: PatternMatcher.match_CommutativeBinary,
                Sub: PatternMatcher.match_Binary,
                Multiply: PatternMatcher.match_CommutativeBinary,
                Div: PatternMatcher.match_Binary,
                Mod: PatternMatcher.match_Binary,
                FloorDiv: PatternMatcher.match_Binary,
                LessThan: PatternMatcher.match_Binary,
                Equal: PatternMatcher.match_Binary,
                LessEqual: PatternMatcher.match_Binary,
                TensorElement: PatternMatcher.match_TensorElement,
                IfThenElse: PatternMatcher.match_IfThenElse,
                Call: PatternMatcher.match_Call,
                Var: PatternMatcher.match_Var,
                Constant: PatternMatcher.match_Constant,
                LogicalAnd: PatternMatcher.match_CommutativeBinary,
                LogicalOr: PatternMatcher.match_CommutativeBinary,
                # type
                DataType: PatternMatcher.match_DataType,
                TensorType: PatternMatcher.match_TensorType,
                # scope
                # Scope: PatternMatcher.match_Scope,
                # layout
                DataLayout: PatternMatcher.match_DataLayout,
                StridesLayout: PatternMatcher.match_StridesLayout,
                # patterns
                AnyExpr: PatternMatcher.match_AnyPattern,
                UnionPattern: PatternMatcher.match_UnionPattern,
                OptionalPattern: PatternMatcher.match_OptionalPattern,
                ScalarTypePattern: PatternMatcher.match_ScalarTypePattern,
                TensorTypePattern: PatternMatcher.match_TensorTypePattern,
                # python containers and types
                str: PatternMatcher.match_String,
                list: PatternMatcher.match_Sequence,
                tuple: PatternMatcher.match_Sequence,
            }
        return PatternMatcher._dispatch_table

    def __init__(self):
        self.matched: Dict[Matchable, Optional[Any]] = {}

    def __call__(self, pattern, target):
        self.matched.clear()
        try:
            with self.match(pattern, target):
                pass
            return self.matched, "Matched"
        except NotMatchedError as e:
            return None, str(e)
            # return None, str(traceback.format_exc())

    def match(
        self, pattern: Optional[Union[Node, Sequence]], target: Optional[Union[Node, Sequence]]
    ) -> ContextManager:
        return MatchContext(self, pattern, target)

    @staticmethod
    def check_type(pattern, target, expect_target_type=None):
        if expect_target_type is None:
            expect_target_type = pattern.__class__
        if not isinstance(target, expect_target_type):
            raise NotMatchedError(
                pattern,
                target,
                "Pattern expect target with type {}, but got type {}".format(expect_target_type, type(target)),
            )

    def check_cond(self, pattern, target, cond, message=""):
        if not cond:
            raise NotMatchedError(pattern, target, message)

    def always_match(self, pattern, target):
        pass

    def match_StringPattern(self, pattern: StringPattern, target: Any):
        if not isinstance(target, str):
            raise NotMatchedError(pattern, target)

    def match_CommutativeBinary(self, pattern: BinaryOp, target: BinaryOp):
        # return self.match_Binary(pattern, target)
        try:
            with ExitStack() as stack:
                stack.enter_context(self.match(pattern.a, target.a))
                stack.enter_context(self.match(pattern.b, target.b))
        except NotMatchedError:
            pass
        else:
            return
        try:
            with ExitStack() as stack:
                stack.enter_context(self.match(pattern.a, target.b))
                stack.enter_context(self.match(pattern.b, target.a))
        except NotMatchedError:
            pass
        else:
            return

        raise NotMatchedError(pattern, target, "Commutative binary op has not matched")

    def match_Binary(self, pattern: BinaryOp, target: BinaryOp):
        with ExitStack() as stack:
            stack.enter_context(self.match(pattern.a, target.a))
            stack.enter_context(self.match(pattern.b, target.b))

    def match_TensorElement(self, pattern: TensorElement, target: TensorElement):
        with ExitStack() as stack:
            stack.enter_context(self.match(pattern.base, target.base))
            stack.enter_context(self.match(pattern.indices, target.indices))

    def match_IfThenElse(self, pattern: IfThenElse, target: IfThenElse):
        with ExitStack() as stack:
            stack.enter_context(self.match(pattern.cond, target.cond))
            stack.enter_context(self.match(pattern.then_expr, target.then_expr))
            stack.enter_context(self.match(pattern.else_expr, target.else_expr))

    def match_Call(self, pattern: Call, target: Call):
        with ExitStack() as stack:
            stack.enter_context(self.match(pattern.func_var, target.func_var))
            stack.enter_context(self.match(pattern.args, target.args))

    def match_Var(self, pattern: Var, target: Var):  # pylint: disable=unused-argument
        if isinstance(pattern.type, FuncType):
            return
        # with self.match(pattern.type, target.type):
        #     pass

    def match_Constant(self, pattern: Constant, target: Constant):
        with self.match(pattern.type, target.type):
            pass
        if pattern.value is None:
            # None matches any const value
            return
        if pattern.value != target.value:
            raise NotMatchedError(pattern, target)

    def match_DataLayout(self, pattern, target):
        if isinstance(target, (StridesLayout, DataLayout)):
            pass
        else:
            raise NotMatchedError(pattern, target)

    def match_StridesLayout(self, pattern: StridesLayout, target: StridesLayout):
        pass

    def match_AnyPattern(self, pattern: AnyExpr, target: Expr):
        # if pattern.type is None, match any expr, otherwise match any expr with specific type
        if pattern.cls and not isinstance(target, pattern.cls):
            raise NotMatchedError(pattern, target)
        if pattern.exclude_cls and isinstance(target, pattern.exclude_cls):
            raise NotMatchedError(pattern, target)

    def match_UnionPattern(self, pattern: UnionPattern, target: Node):
        for p in pattern.patterns:
            success = True
            try:
                with self.match(p, target):
                    pass
            except NotMatchedError:
                success = False
            if success:
                return
        raise NotMatchedError(pattern, target)

    def match_OptionalPattern(self, pattern: OptionalPattern, target: Node):
        if target is None:
            return
        else:
            with self.match(pattern.pattern, target):
                pass

    # def match_Scope(self, pattern: Scope, target: Scope):
    #     if pattern.name is not None and (pattern.name is None or pattern.name != target.name):
    #         raise NotMatchedError(pattern, target)
    #
    def match_DataType(self, pattern: DataType, target: DataType):
        if pattern.name:
            if pattern.name != target.name:
                raise NotMatchedError(pattern, target)

    def match_TensorType(self, pattern: TensorType, target: TensorType):
        with ExitStack() as stack:
            stack.enter_context(self.match(pattern.dtype, target.dtype))
            stack.enter_context(self.match(pattern.shape, target.shape))
            stack.enter_context(self.match(pattern.layout, target.layout))
            # stack.enter_context(self.match(pattern.scope, target.scope))

    def match_ScalarTypePattern(self, pattern: ScalarTypePattern, target: DataType):
        self.check_type(pattern, target, DataType)
        if pattern.allowed_types is not None and target.name not in pattern.allowed_types:
            raise NotMatchedError(pattern, target)

    def match_TensorTypePattern(self, pattern: TensorTypePattern, target: TensorType):
        self.check_type(pattern, target, TensorType)
        with ExitStack() as stack:
            if pattern.rank is not None and len(target.shape) != pattern.rank:
                raise NotMatchedError(pattern, target)
            if pattern.scope is not None:
                if isinstance(pattern.scope, DeclareScope) and pattern.scope.name != target.scope.name:
                    raise NotMatchedError(pattern, target)
                if isinstance(pattern.scope, list) and target.scope.name not in [s.name for s in pattern.scope]:
                    raise NotMatchedError(pattern, target)
            stack.enter_context(self.match(pattern.scalar_type, target.dtype))
            stack.enter_context(self.match(pattern.shape, target.shape))
            stack.enter_context(self.match(pattern.layout, target.layout))
            if not pattern.allow_dynamic_size and any(not isinstance(s, Constant) for s in target.shape):
                raise NotMatchedError(pattern, target)

    def match_Sequence(self, pattern: Sequence, target: Sequence):
        with ExitStack() as stack:
            if len(pattern) != len(target):
                raise NotMatchedError(pattern, target, "length does not match")
            for a, b in zip(pattern, target):
                stack.enter_context(self.match(a, b))

    def match_String(self, pattern: str, target: str):
        if pattern != target:
            raise NotMatchedError(pattern, target)


def reduce_pattern(shape: Sequence[Union[int, Expr]], fcompute, reduce_type: str):
    from hidet.ir.functors import collect  # pylint: disable=import-outside-toplevel

    shape = convert(shape)
    axes = [var() for _ in shape]
    value = convert(fcompute(*axes))
    input_tensors = collect(value, TensorNode, stop_when_found=True)
    input_scalars = collect(value, ScalarNode, stop_when_found=True)
    reduce_operation = ReduceOperation.from_name(reduce_type)
    return ReduceCompute(
        input_tensors, input_scalars, shape, axes, value, reduce_operation, accumulate_dtype=data_type('float32')
    )


def any_const_int():
    return Constant(None, data_type('int32'))


def any_const_ints(num=1):
    return [any_const_int() for _ in range(num)]


def any_const():
    return AnyExpr(Constant)


def int_vars(names):
    return [var(name, dtype='int32') for name in names]


def match(pattern: Node, target: Node) -> Tuple[Optional[Dict[Node, Any]], str]:
    """
    :return: match, report
    """
    matcher = PatternMatcher()
    return matcher(pattern, target)
