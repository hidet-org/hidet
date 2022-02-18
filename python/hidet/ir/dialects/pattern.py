import traceback
from typing import Type, Tuple, Any, ContextManager
from contextlib import ExitStack
from hidet.ir.type import *
from hidet.ir.expr import *
from hidet.ir.dialects.compute import *
from hidet.ir.dialects.lowlevel import *
from hidet.ir.task import *
from hidet.ir.layout import StridesLayout


class PatternNode(Node):
    # A pattern can match a series of exprs/types/other node objects
    pass


class TypePattern(TypeNode, PatternNode):
    pass


class ScalarTypePattern(TypePattern):
    def __init__(self, allowed_types=None):
        self.allowed_types: Optional[List[str]] = allowed_types


class TensorTypePattern(TypePattern):
    def __init__(self, scope=None, scalar_type=None, rank=None, shape=None, layout=None, allow_dynamic_size=False):
        self.rank: Optional[int] = rank
        self.scope: Optional[Union[Scope, List[Scope]]] = scope
        self.scalar_type: Optional[Union[ScalarType, ScalarTypePattern]] = scalar_type
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


class ReduceComputePattern(ExprPattern):
    def __init__(self, allow_dynamic_axis=True):
        self.allow_dynamic_axis = allow_dynamic_axis


class TensorComputePattern(ExprPattern):
    def __init__(self, rank=None, allow_dynamic_axis=True):
        self.rank = rank
        self.allow_dynamic_axis = allow_dynamic_axis


class ScalarExprPattern(ExprPattern):
    def __init__(self, allow_reduce=True, reduce=None):
        self.allow_reduce = allow_reduce
        self.reduce: Optional[ReduceCompute] = reduce


class TaskPattern(PatternNode):
    def __init__(self, compute_pattern=None, required_params=None, required_param_types=None,
                 allow_extra_params=True, allow_tensor_extra_params=True, worker=None):
        self.compute_pattern: Optional[Expr] = compute_pattern
        self.required_params: Optional[List[ComputeNode]] = required_params
        self.required_params_types: Optional[List[TypeNode]] = required_param_types
        self.extra_params: Expr = Expr()  # as a handle to reference the unmatched params
        self.allow_extra_params: bool = allow_extra_params
        self.allow_tensor_extra_params: bool = allow_tensor_extra_params
        self.worker: Optional[Worker] = worker


class NotMatchedError(Exception):
    def __init__(self, pattern, target, message=""):
        super().__init__(message)
        self.pattern = pattern
        self.target = target


class MatchContext:
    def __init__(self, matcher: 'PatternMatcher', pattern: Node, target: Node):
        self.matched = matcher.matched
        self.dispatch = matcher.match_func_dispatcher
        self.pattern = pattern
        self.target = target

    def __enter__(self):
        if isinstance(self.pattern, list):
            # we do not cache list pattern because it is un-hashable
            if not isinstance(self.target, (list, tuple)):
                raise NotMatchedError(self.pattern, self.target)
            # noinspection PyArgumentList
            self.dispatch[self.pattern.__class__](self.pattern, self.target)
            return
        if self.pattern is None:
            # None in pattern matches anything
            return
        if self.target is None:
            if isinstance(self.pattern, OptionalPattern):
                self.matched[self.pattern] = None
                return
            else:
                msg = 'Expect {} but the target is empty.'.format(type(self.pattern))
                raise NotMatchedError(self.pattern, self.target, msg)
        if self.pattern in self.matched:
            if self.matched[self.pattern] is not self.target:
                # we think the constant with the same value as the same object
                lhs, rhs = self.matched[self.pattern], self.target
                if isinstance(lhs, Constant) and isinstance(rhs, Constant):
                    if lhs.value == rhs.value:
                        return
                msg = "Try to match {} and {}, but the prior one has been matched to another target {}".format(type(self.pattern), type(self.target), type(self.matched[self.pattern]))
                raise NotMatchedError(self.pattern, self.target, msg)
            else:
                return

        self.matched[self.pattern] = self.target
        if not isinstance(self.pattern, (PatternNode, list, tuple)):
            # put all pattern class that allow to accept other classes
            # list, tuple accept each other
            PatternMatcher.check_type(self.pattern, self.target)
        # noinspection PyArgumentList
        self.dispatch[self.pattern.__class__](self.pattern, self.target)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type == NotMatchedError:
            # delete the matched target for pattern
            # do not return True, propagate the exception:
            #   1. it can be caught by pattern like UnionPattern to try other target,
            #   2. or, it will be caught by the of PatternMatcher.__call__, indicating failure of matching.
            if not isinstance(self.pattern, list) and self.pattern is not None:
                del self.matched[self.pattern]


Matchable = Optional[Union[Node, list, tuple]]


class PatternMatcher:
    """
    invariant: every time when we enter match(...)
        0 the self.matched[...] stored the matched patterns and targets, and ongoing matching must follow them
        1 if successful, all sub-expressions of pattern have been set in self.matched[...]
        2 if failed, it acted like we have not call this function (we treat self.matched[v] = None and v not in self.matched as the same state)
    """

    def __init__(self):
        self.matched: Dict[Node, Optional[Node]] = {}
        self.match_func_dispatcher = {
            # expr
            Add: self.match_Binary,
            Sub: self.match_Binary,
            Multiply: self.match_Binary,
            Div: self.match_Binary,
            Mod: self.match_Binary,
            FloorDiv: self.match_Binary,
            LessThan: self.match_Binary,
            Equal: self.match_Binary,
            TensorElement: self.match_TensorElement,
            Call: self.match_Call,
            Var: self.match_Var,
            Constant: self.match_Constant,
            # compute dialect expr
            ScalarInput: self.match_ScalarInput,
            TensorInput: self.match_TensorInput,
            TensorCompute: self.match_TensorCompute,
            ReduceCompute: self.match_ReduceCompute,
            # type
            ScalarType: self.match_ScalarType,
            TensorType: self.match_TensorType,
            # scope
            Scope: self.match_Scope,
            # layout
            TaskLayout: self.always_match,
            DataLayout: self.match_DataLayout,
            StridesLayout: self.match_StridesLayout,
            # worker
            Host: self.always_match,
            Grid: self.match_Grid,
            ThreadBlock: self.match_ThreadBlock,
            Warp: self.match_Warp,
            Thread: self.always_match,
            # patterns
            TaskPattern: self.match_TaskPattern,
            AnyExpr: self.match_AnyPattern,
            ReduceComputePattern: self.match_ReduceComputePattern,
            TensorComputePattern: self.match_TensorComputePattern,
            ScalarExprPattern: self.match_ScalarExprPattern,
            UnionPattern: self.match_UnionPattern,
            OptionalPattern: self.match_OptionalPattern,
            ScalarTypePattern: self.match_ScalarTypePattern,
            TensorTypePattern: self.match_TensorTypePattern,
            # python containers
            list: self.match_Sequence,
            tuple: self.match_Sequence
        }

    def __call__(self, pattern, target):
        self.matched.clear()
        try:
            with self.match(pattern, target):
                pass
            return self.matched, "Matched"
        except NotMatchedError as e:
            return None, "Failed"
            # return None, str(traceback.format_exc())

    def match(self, pattern: Optional[Union[Node, Sequence]], target: Optional[Union[Node, Sequence]]) -> ContextManager:
        return MatchContext(self, pattern, target)

    @staticmethod
    def check_type(pattern, target, expect_target_type=None):
        if expect_target_type is None:
            expect_target_type = pattern.__class__
        if not isinstance(target, expect_target_type):
            raise NotMatchedError(pattern, target, "Pattern expect target with type {}, but got type {}".format(expect_target_type, type(target)))

    @staticmethod
    def check_cond(pattern, target, cond, message=""):
        if not cond:
            raise NotMatchedError(pattern, target, message)

    @staticmethod
    def always_match(pattern, target):
        pass

    def match_Binary(self, pattern: BinaryOp, target: BinaryOp):
        with ExitStack() as stack:
            stack.enter_context(self.match(pattern.a, target.a))
            stack.enter_context(self.match(pattern.b, target.b))

    def match_TensorElement(self, pattern: TensorElement, target: TensorElement):
        with ExitStack() as stack:
            stack.enter_context(self.match(pattern.base, target.base))
            stack.enter_context(self.match(pattern.indices, target.indices))

    def match_Call(self, pattern: Call, target: Call):
        with ExitStack() as stack:
            stack.enter_context(self.match(pattern.func_var, target.func_var))
            stack.enter_context(self.match(pattern.args, target.args))

    def match_Var(self, pattern: Var, target: Var):
        with self.match(pattern.type, target.type):
            pass

    def match_Constant(self, pattern: Constant, target: Constant):
        with self.match(pattern.dtype, target.dtype):
            pass
        if pattern.value is None:
            # None matches any const value
            return
        if pattern.value != target.value:
            raise NotMatchedError(pattern, target)

    def match_ScalarInput(self, pattern: ScalarInput, target: ScalarInput):
        with ExitStack() as stack:
            stack.enter_context(self.match(pattern.dtype, target.dtype))

    def match_TensorInput(self, pattern: TensorInput, target: TensorInput):
        with ExitStack() as stack:
            stack.enter_context(self.match(pattern.dtype, target.dtype))
            stack.enter_context(self.match(pattern.shape, target.shape))

    def match_TensorCompute(self, pattern: TensorCompute, target: TensorCompute):
        with ExitStack() as stack:
            stack.enter_context(self.match(pattern.shape, target.shape))
            stack.enter_context(self.match(pattern.axes, target.axes))
            stack.enter_context(self.match(pattern.value, target.value))

    def match_ReduceCompute(self, pattern: ReduceCompute, target: ReduceCompute):
        with ExitStack() as stack:
            stack.enter_context(self.match(pattern.axis, target.axis))
            stack.enter_context(self.match(pattern.shape, target.shape))
            stack.enter_context(self.match(pattern.value, target.value))

    @staticmethod
    def match_DataLayout(pattern, target):
        if isinstance(target, (StridesLayout, DataLayout)):
            pass
        else:
            raise NotMatchedError(pattern, target)

    def match_StridesLayout(self, pattern: StridesLayout, target: StridesLayout):
        pass

    @staticmethod
    def match_AnyPattern(pattern: AnyExpr, target: Expr):
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

    @staticmethod
    def match_Scope(pattern: Scope, target: Scope):
        if pattern.name is not None and (pattern.name is None or pattern.name != target.name):
            raise NotMatchedError(pattern, target)

    def match_ReduceComputePattern(self, pattern: ReduceComputePattern, target: ReduceCompute):
        self.check_type(pattern, target, ReduceCompute)
        if not pattern.allow_dynamic_axis and \
                any(not isinstance(v, Constant) for v in [target.axis.min_value, target.axis.extent]):
            raise NotMatchedError(pattern, target, "does not allow dynamic axis in reduce")

    def match_TensorComputePattern(self, pattern: TensorComputePattern, target: TensorCompute):
        self.check_type(pattern, target, TensorCompute)
        if pattern.rank is not None and len(target.shape) != pattern.rank:
            raise NotMatchedError(pattern, target, "rank does not match")
        if not pattern.allow_dynamic_axis and any(not isinstance(v, Constant) for v in target.shape):
            raise NotMatchedError(pattern, target, "does not allow dynamic axis")

    def match_ScalarExprPattern(self, pattern: ScalarExprPattern, target: Expr):
        self.check_cond(pattern, target, not isinstance(target, (TensorCompute, TensorInput)))
        from hidet.ir.functors import collect
        with ExitStack() as stack:
            reduce_exprs = collect(target, ReduceCompute)
            if len(reduce_exprs) > 1:
                raise NotMatchedError(pattern, target, "more than one reduce, current not supported")
            if len(reduce_exprs) == 1:
                target_reduce = reduce_exprs[0]
            if len(reduce_exprs) == 0:
                target_reduce = None
            stack.enter_context(self.match(pattern.reduce, target_reduce))

    @staticmethod
    def match_ScalarType(pattern: ScalarType, target: ScalarType):
        if pattern.name:
            if pattern.name != target.name:
                raise NotMatchedError(pattern, target)

    def match_TensorType(self, pattern: TensorType, target: TensorType):
        with ExitStack() as stack:
            stack.enter_context(self.match(pattern.scalar_type, target.scalar_type))
            stack.enter_context(self.match(pattern.shape, target.shape))
            stack.enter_context(self.match(pattern.layout, target.layout))
            stack.enter_context(self.match(pattern.scope, target.scope))

    def match_ScalarTypePattern(self, pattern: ScalarTypePattern, target: ScalarType):
        self.check_type(pattern, target, ScalarType)
        if pattern.allowed_types is not None and target.name not in pattern.allowed_types:
            raise NotMatchedError(pattern, target)

    def match_TensorTypePattern(self, pattern: TensorTypePattern, target: TensorType):
        self.check_type(pattern, target, TensorType)
        with ExitStack() as stack:
            if pattern.rank is not None and len(target.shape) != pattern.rank:
                raise NotMatchedError(pattern, target)
            if pattern.scope is not None:
                if isinstance(pattern.scope, Scope) and pattern.scope.name != target.scope.name:
                    raise NotMatchedError(pattern, target)
                if isinstance(pattern.scope, list) and target.scope.name not in [s.name for s in pattern.scope]:
                    raise NotMatchedError(pattern, target)
            stack.enter_context(self.match(pattern.scalar_type, target.scalar_type))
            stack.enter_context(self.match(pattern.shape, target.shape))
            stack.enter_context(self.match(pattern.layout, target.layout))
            if not pattern.allow_dynamic_size and any(not isinstance(s, Constant) for s in target.shape):
                raise NotMatchedError(pattern, target)

    def match_Grid(self, pattern: Grid, target: Grid):
        with ExitStack() as stack:
            stack.enter_context(self.match(pattern.grid_dim, target.grid_dim))
            stack.enter_context(self.match(pattern.block_dim, target.block_dim))

    def match_ThreadBlock(self, pattern: ThreadBlock, target: ThreadBlock):
        with ExitStack() as stack:
            stack.enter_context(self.match(pattern.block_dim, target.block_dim))
            stack.enter_context(self.match(pattern.task_layout, target.task_layout))

    def match_Warp(self, pattern: Warp, target: Warp):
        with ExitStack() as stack:
            stack.enter_context(self.match(pattern.task_layout, target.task_layout))

    def match_TaskPattern(self, pattern: TaskPattern, target: Task):
        self.check_type(pattern, target, Task)
        with ExitStack() as stack:
            stack.enter_context(self.match(pattern.compute_pattern, target.compute))
            if pattern.required_params and pattern.required_params_types:
                assert len(pattern.required_params) == len(pattern.required_params_types)
                for required_param, required_type in zip(pattern.required_params, pattern.required_params_types):
                    matched_param = self.matched[required_param]
                    actual_type = target.params_type[target.params.index(matched_param)]
                    # assert isinstance(matched_param, (ScalarInput, TensorInput)), "as far as now, we only support specify the param type, not the output type"
                    stack.enter_context(self.match(required_type, actual_type))
                matched_params = [self.matched[param] for param in pattern.required_params]
            else:
                matched_params = []
            extra_params = [param for param in target.params if param not in matched_params]
            if not pattern.allow_extra_params and len(extra_params) > 0:
                raise NotMatchedError(pattern, target, "do not allow extra param(s)")
            if not pattern.allow_tensor_extra_params and any(isinstance(p, TensorInput) for p in extra_params):
                raise NotMatchedError(pattern, target, "do not allow extra tensor param(s)")
            stack.enter_context(self.match(pattern.worker, target.worker))

    def match_Sequence(self, pattern: Sequence, target: Sequence):
        with ExitStack() as stack:
            for a, b in zip(pattern, target):
                stack.enter_context(self.match(a, b))


def any_const_int():
    return Constant(None, ScalarType('int32'))


def any_const():
    return AnyExpr(Constant)


def match(pattern: Node, target: Node) -> Tuple[Optional[Dict[Node, Any]], str]:
    """
    :return: match, report
    """
    matcher = PatternMatcher()
    return matcher(pattern, target)
