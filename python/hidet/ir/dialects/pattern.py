import traceback
from typing import Type, Tuple
import contextlib
from hidet.ir.type import *
from hidet.ir.expr import *
from hidet.ir.dialects.compute import *
from hidet.ir.dialects.lowlevel import *
from hidet.ir.task import *


class TypePattern(BaseType):
    pass


class ScalarTypePattern(TypePattern):
    def __init__(self, allowed_types=None):
        self.allowed_types: Optional[List[str]] = allowed_types


class TensorTypePattern(TypePattern):
    def __init__(self, scope=None, scalar_type=None, rank=None, shape=None, strides=None, allow_dynamic_size=False):
        self.rank: Optional[int] = rank
        self.scope: Optional[Union[Scope, List[Scope]]] = scope
        self.scalar_type: Optional[Union[ScalarType, ScalarTypePattern]] = scalar_type
        self.shape: Optional[List[Expr]] = shape
        self.strides: Optional[List[Expr]] = strides
        self.allow_dynamic_size = allow_dynamic_size


class ExprPattern(Expr):
    pass


class AnyExpr(ExprPattern):
    def __init__(self, type=None):
        self.type: Optional[Type[Expr]] = type


class UnionPattern(Node):
    def __init__(self, patterns):
        self.patterns: List[Node] = patterns


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


class TaskPattern(Node):
    def __init__(self, compute_pattern=None, required_params=None, required_param_types=None,
                 allow_extra_params=True, allow_tensor_extra_params=True, worker=None):
        self.compute_pattern: Optional[Expr] = compute_pattern
        self.required_params: Optional[List[ComputeNode]] = required_params
        self.required_params_types: Optional[List[BaseType]] = required_param_types
        self.extra_params: Expr = Expr()  # as a handle to reference the unmatched params
        self.allow_extra_params: bool = allow_extra_params
        self.allow_tensor_extra_params: bool = allow_tensor_extra_params
        self.worker: Optional[Worker] = worker


class NotMatchedError(Exception):
    def __init__(self, pattern, target, message=""):
        super().__init__(message)
        self.pattern = pattern
        self.target = target


class PatternMatcher:
    """
    invariant: every time when we enter match(...)
        0 the self.matched[...] stored the matched patterns and targets, and ongoing matching must follow them
        1 if successful, all sub-expressions of pattern have been set in self.matched[...]
        2 if failed, it acted like we have not call this function (we treat self.matched[v] = None and v not in self.matched as the same state)
    """
    def __init__(self):
        self.matched: Dict[Node, Optional[Node]] = {}

    def __call__(self, pattern, target):
        success = True
        self.matched.clear()
        msg = "Success"
        try:
            with self.match(pattern, target):
                pass
        except NotMatchedError as e:
            msg = str(traceback.format_exc())
            success = False
        if success:
            return self.matched, msg
        else:
            return None, msg

    @contextlib.contextmanager
    def match(self, pattern: Node, target: Node):
        if pattern not in self.matched:
            self.matched[pattern] = None
        if self.matched[pattern] is not None and self.matched[pattern] is not target:
            raise NotMatchedError(pattern, target,
                                  "Try to match {} and {}, but the prior one has been matched to another target {}".format(pattern, target, self.matched[pattern]))
        old = self.matched[pattern]
        try:
            self.matched[pattern] = target
            if not isinstance(pattern, (ExprPattern, TypePattern, TaskPattern, UnionPattern, Scope)):
                # all pattern except ExprPattern sub-classes instances requires that target with the same type as pattern's
                self.check_type(pattern, target)
            if isinstance(pattern, Add):
                self.match_Add(pattern, target)
            elif isinstance(pattern, Sub):
                self.match_Sub(pattern, target)
            elif isinstance(pattern, Multiply):
                self.match_Multiply(pattern, target)
            elif isinstance(pattern, Div):
                self.match_Div(pattern, target)
            elif isinstance(pattern, Mod):
                self.match_Mod(pattern, target)
            elif isinstance(pattern, FloorDiv):
                self.match_FloorDiv(pattern, target)
            elif isinstance(pattern, LessThan):
                self.match_LessThan(pattern, target)
            elif isinstance(pattern, Equal):
                self.match_Equal(pattern, target)
            elif isinstance(pattern, TensorSlice):
                self.match_TensorSlice(pattern, target)
            elif isinstance(pattern, TensorElement):
                self.match_TensorElement(pattern, target)
            elif isinstance(pattern, Cast):
                self.match_Cast(pattern, target)
            elif isinstance(pattern, Dereference):
                self.match_Dereference(pattern, target)
            elif isinstance(pattern, Call):
                self.match_Call(pattern, target)
            elif isinstance(pattern, Var):
                self.match_Var(pattern, target)
            elif isinstance(pattern, Constant):
                self.match_Constant(pattern, target)
            elif isinstance(pattern, ScalarInput):
                self.match_ScalarInput(pattern, target)
            elif isinstance(pattern, TensorInput):
                self.match_TensorInput(pattern, target)
            elif isinstance(pattern, TensorCompute):
                self.match_TensorCompute(pattern, target)
            elif isinstance(pattern, ReduceCompute):
                self.match_ReduceCompute(pattern, target)
            elif isinstance(pattern, AnyExpr):
                self.match_AnyPattern(pattern, target)
            elif isinstance(pattern, ReduceComputePattern):
                self.match_ReduceComputePattern(pattern, target)
            elif isinstance(pattern, TensorComputePattern):
                self.match_TensorComputePattern(pattern, target)
            elif isinstance(pattern, ScalarExprPattern):
                self.match_ScalarExprPattern(pattern, target)

            # general matching
            elif isinstance(pattern, UnionPattern):
                self.match_UnionPattern(pattern, target)

            # scope related matching
            elif isinstance(pattern, RegisterScope):
                self.match_RegisterScope(pattern, target)
            elif isinstance(pattern, Scope):
                self.match_Scope(pattern, target)

            # type related matching
            elif isinstance(pattern, ScalarType):
                self.match_ScalarType(pattern, target)
            elif isinstance(pattern, TensorType):
                self.match_TensorType(pattern, target)
            elif isinstance(pattern, ScalarTypePattern):
                self.match_ScalarTypePattern(pattern, target)
            elif isinstance(pattern, TensorTypePattern):
                self.match_TensorTypePattern(pattern, target)

            # task related matching
            elif isinstance(pattern, Host):
                pass
            elif isinstance(pattern, Grid):
                self.match_Grid(pattern, target)
            elif isinstance(pattern, ThreadBlock):
                self.match_ThreadBlock(pattern, target)
            elif isinstance(pattern, Warp):
                pass
            elif isinstance(pattern, Thread):
                pass
            elif isinstance(pattern, TaskPattern):
                self.match_TaskPattern(pattern, target)
            else:
                raise NotImplementedError(str(type(pattern)))
            yield
        except NotMatchedError as e:
            self.matched[pattern] = old
            raise e

    @staticmethod
    def check_type(pattern, target, expect_target_type=None):
        if expect_target_type is None:
            expect_target_type = pattern.__class__
        if not isinstance(target, expect_target_type):
            raise NotMatchedError(pattern, target,
                                  "Pattern {} expect target {} with type {}, but got type {}".format(pattern, target, expect_target_type, type(target)))

    @staticmethod
    def check_cond(pattern, target, cond, message=""):
        if not cond:
            raise NotMatchedError(pattern, target, message)

    def match_Add(self, pattern: Add, target: Add):
        with self.match(pattern.a, target.a), self.match(pattern.b, target.b):
            pass

    def match_Sub(self, pattern: Sub, target: Sub):
        with self.match(pattern.a, target.a), self.match(pattern.b, target.b):
            pass

    def match_Multiply(self, pattern: Multiply, target: Multiply):
        with self.match(pattern.a, target.a), self.match(pattern.b, target.b):
            pass

    def match_Div(self, pattern: Div, target: Div):
        with self.match(pattern.a, target.a), self.match(pattern.b, target.b):
            pass

    def match_Mod(self, pattern: Mod, target: Mod):
        with self.match(pattern.a, target.a), self.match(pattern.b, target.b):
            pass

    def match_FloorDiv(self, pattern: FloorDiv, target: FloorDiv):
        with self.match(pattern.a, target.a), self.match(pattern.b, target.b):
            pass

    def match_LessThan(self, pattern: LessThan, target: LessThan):
        with self.match(pattern.a, target.a), self.match(pattern.b, target.b):
            pass

    def match_Equal(self, pattern: Equal, target: Equal):
        with self.match(pattern.a, target.a), self.match(pattern.b, target.b):
            pass

    def match_TensorSlice(self, pattern: TensorSlice, target: TensorSlice):
        with self.match(pattern.base, target.base):
            with contextlib.ExitStack() as stack:
                self.check_cond(pattern, target, len(pattern.indices) == len(target.indices))
                for a, b in zip(pattern.indices, target.indices):
                    if a is None and b is None:
                        continue
                    if a is None or b is None:
                        raise NotMatchedError(pattern, target)
                    stack.enter_context(self.match(a, b))
                self.check_cond(pattern, target, len(pattern.starts) == len(target.starts))
                for a, b in zip(pattern.starts, target.starts):
                    stack.enter_context(self.match(a, b))
                self.check_cond(pattern, target, len(pattern.ends) == len(target.ends))
                for a, b in zip(pattern.ends, target.ends):
                    stack.enter_context(self.match(a, b))

    def match_TensorElement(self, pattern: TensorElement, target: TensorElement):
        with self.match(pattern.base, target.base):
            with contextlib.ExitStack() as stack:
                self.check_cond(pattern, target, len(pattern.indices) == len(target.indices))
                for a, b in zip(pattern.indices, target.indices):
                    stack.enter_context(self.match(a, b))

    def match_Cast(self, pattern: Cast, target: Cast):
        with self.match(pattern.expr, target.expr):
            pass    # todo: check target type

    def match_Dereference(self, pattern: Dereference, target: Dereference):
        with self.match(pattern.expr, target.expr):
            pass

    def match_Call(self, pattern: Call, target: Call):
        with self.match(pattern.func_var, target.func_var):
            with contextlib.ExitStack() as stack:
                self.check_cond(pattern, target, len(pattern.args) == len(target.args))
                for a, b in zip(pattern.args, target.args):
                    stack.enter_context(self.match(a, b))

    def match_Var(self, pattern: Var, target: Var):
        with self.match(pattern.type, target.type):
            pass

    def match_Constant(self, pattern: Constant, target: Constant):
        if pattern.dtype is not None:
            if target.dtype is None:
                raise NotMatchedError(pattern, target)
            with self.match(pattern.dtype, target.dtype):
                pass
        if pattern.value is None:
            # None matches any const value
            return
        if pattern.value != target.value:
            raise NotMatchedError(pattern, target)

    def match_ScalarInput(self, pattern: ScalarInput, target: ScalarInput):
        if pattern.dtype:
            with self.match(pattern.dtype, target.dtype):
                pass

    def match_TensorInput(self, pattern: TensorInput, target: TensorInput):
        if pattern.dtype:
            with self.match(pattern.dtype, target.dtype):
                pass

    def match_TensorCompute(self, pattern: TensorCompute, target: TensorCompute):
        self.check_cond(pattern, target, len(pattern.shape) == len(target.shape))
        with contextlib.ExitStack() as stack:
            for a, b in zip(pattern.shape, target.shape):
                stack.enter_context(self.match(a, b))
            for a, b in zip(pattern.axes, target.axes):
                stack.enter_context(self.match(a, b))
            stack.enter_context(self.match(pattern.value, target.value))

    def match_ReduceCompute(self, pattern: ReduceCompute, target: ReduceCompute):
        with contextlib.ExitStack() as stack:
            stack.enter_context(self.match(pattern.axis, target.axis))
            for a, b in zip(pattern.shape, target.shape):
                stack.enter_context(self.match(a, b))
            stack.enter_context(self.match(pattern.value, target.value))

    def match_AnyPattern(self, pattern: AnyExpr, target: Expr):
        # if pattern.type is None, match anything, otherwise match any expr with specific type
        if pattern.type and not isinstance(target, pattern.type):
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

    def match_RegisterScope(self, pattern: RegisterScope, target: RegisterScope):
        # always match
        pass

    def match_Scope(self, pattern: Scope, target: Scope):
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
        from hidet.ir.functors import collect
        self.check_cond(pattern, target, not isinstance(target, (TensorCompute, TensorInput)))
        reduce_exprs = collect(target, ReduceCompute)
        with contextlib.ExitStack() as stack:
            if len(reduce_exprs) > 1:
                raise NotMatchedError(pattern, target, "more than one reduce, current not supported")
            if len(reduce_exprs) == 1:
                if not pattern.allow_reduce:
                    raise NotMatchedError(pattern, target, "reduce found, but not expect")
                elif pattern.reduce is not None:
                    stack.enter_context(self.match(pattern.reduce, reduce_exprs[0]))
            if len(reduce_exprs) == 0:
                if pattern.reduce is not None:
                    raise NotMatchedError(pattern, target, "expect reduce, but not found")

    def match_ScalarType(self, pattern: ScalarType, target: ScalarType):
        if pattern.name:
            if pattern.name != target.name:
                raise NotMatchedError(pattern, target)

    def match_TensorType(self, pattern: TensorType, target: TensorType):
        with contextlib.ExitStack() as stack:
            stack.enter_context(self.match(pattern.scalar_type, target.scalar_type))
            if pattern.shape:
                if target.shape is None:
                    raise NotMatchedError(pattern, target)
                for a, b in zip(pattern.shape, target.shape):
                    stack.enter_context(self.match(a, b))
            if pattern.strides:
                if target.strides is None:
                    raise NotMatchedError(pattern, target)
                for a, b in zip(pattern.strides, target.strides):
                    stack.enter_context(self.match(a, b))
            if pattern.scope:
                stack.enter_context(self.match(pattern.scope, target.scope))

    def match_ScalarTypePattern(self, pattern: ScalarTypePattern, target: ScalarType):
        self.check_type(pattern, target, ScalarType)
        if pattern.allowed_types is not None and target.name not in pattern.allowed_types:
            raise NotMatchedError(pattern, target)

    def match_TensorTypePattern(self, pattern: TensorTypePattern, target: TensorType):
        self.check_type(pattern, target, TensorType)
        with contextlib.ExitStack() as stack:
            if pattern.rank is not None and len(target.shape) != pattern.rank:
                raise NotMatchedError(pattern, target)
            if pattern.scope is not None:
                if isinstance(pattern.scope, Scope) and pattern.scope.name != target.scope.name:
                    raise NotMatchedError(pattern, target)
                if isinstance(pattern.scope, list) and target.scope.name not in [s.name for s in pattern.scope]:
                    raise NotMatchedError(pattern, target)
            if pattern.scalar_type is not None:
                stack.enter_context(self.match(pattern.scalar_type, target.scalar_type))
            if pattern.shape is not None:
                for a, b in zip(pattern.shape, target.shape):
                    stack.enter_context(self.match(a, b))
            if pattern.strides is not None:
                for a, b in zip(pattern.strides, target.strides):
                    stack.enter_context(self.match(a, b))
            if not pattern.allow_dynamic_size and any(not isinstance(s, Constant) for s in target.shape):
                raise NotMatchedError(pattern, target)

    def match_Grid(self, pattern: Grid, target: Grid):
        with contextlib.ExitStack() as stack:
            if pattern.grid_dim:
                stack.enter_context(self.match(pattern.grid_dim, target.grid_dim))
            if pattern.block_dim:
                stack.enter_context(self.match(pattern.block_dim, target.block_dim))

    def match_ThreadBlock(self, pattern: ThreadBlock, target: ThreadBlock):
        with contextlib.ExitStack() as stack:
            if pattern.block_dim:
                stack.enter_context(self.match(pattern.block_dim, target.block_dim))

    def match_TaskPattern(self, pattern: TaskPattern, target: Task):
        self.check_type(pattern, target, Task)
        with contextlib.ExitStack() as stack:
            if pattern.compute_pattern:
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
            if pattern.worker:
                stack.enter_context(self.match(pattern.worker, target.worker))


def any_const_int():
    return Constant(None, ScalarType('int32'))


def any_const():
    return AnyExpr(Constant)


def match(pattern: Node, target: Node) -> Tuple[Optional[Dict[Node, Node]], str]:
    """
    :return: match, report
    """
    matcher = PatternMatcher()
    return matcher(pattern, target)
