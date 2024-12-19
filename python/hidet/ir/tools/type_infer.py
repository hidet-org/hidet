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
from typing import Type, Union
from enum import Enum
from hidet.ir.type import DataType, TensorType, FuncType, PointerType, TensorPointerType, data_type, tensor_pointer_type
from hidet.ir.type import tensor_type, BaseType, ArrayType
from hidet.ir.expr import BinaryExpr, Add, Sub, Multiply, Div, Mod, FloorDiv, LessThan, Equal, IfThenElse
from hidet.ir.expr import TensorSlice, LogicalNot, LogicalOr, LogicalAnd, LessEqual, Let, RightShift, LeftShift
from hidet.ir.expr import BitwiseAnd, Neg, NotEqual, BitwiseXor, Dereference, Reference, Address, BitwiseNot, BitwiseOr
from hidet.ir.expr import Var, Constant, TensorElement, Call, Cast
from hidet.ir.compute import ArgReduceCompute, ReduceCompute, GridCompute, TensorInput, ScalarInput
from hidet.ir.functors import IRFunctor
from hidet.ir.dialects.pattern import PlaceholderExpr
from hidet.ir import dtypes


from hidet.ir.cute.type import TiledTensorType, tiled_tensor
from hidet.ir.cute.expr import CallOp


class OpKind(Enum):
    ARITHMATIC = 0
    COMPARE = 1
    LOGICAL = 2


def is_bool(tp: DataType):
    return isinstance(tp, DataType) and tp.name == 'bool'


class BinaryTypeInfer:
    def infer(self, lhs: BaseType, rhs: BaseType, op: Type[BinaryExpr]):
        if isinstance(lhs, DataType) and isinstance(rhs, DataType):
            return self.dtype(lhs, rhs, op)
        elif lhs.is_pointer() and rhs.is_pointer():
            return self.pointer(lhs, rhs, op)
        elif lhs.is_pointer() and isinstance(rhs, DataType):
            return self.pointer_dtype(lhs, rhs, op)
        elif isinstance(lhs, DataType) and rhs.is_pointer():
            return self.pointer_dtype(rhs, lhs, op)
        elif isinstance(lhs, TiledTensorType) and isinstance(rhs, TiledTensorType):
            from hidet.ir.cute.ops import Arithmetic

            if lhs.scope != rhs.scope:
                return self.fail(lhs, rhs, op)
            dtype = self.infer(lhs.dtype, rhs.dtype, op)
            layout = Arithmetic.infer_layout(lhs, rhs)
            return tiled_tensor(dtype, layout, lhs.scope)
        elif isinstance(lhs, TiledTensorType) and isinstance(rhs, DataType):
            from hidet.ir.cute.ops import Arithmetic

            if not lhs.scope.is_register():
                return self.fail(lhs, rhs, op)
            dtype = self.infer(lhs.dtype, rhs, op)
            layout = Arithmetic.infer_layout(lhs)
            return tiled_tensor(dtype, layout, lhs.scope)
        elif isinstance(lhs, DataType) and isinstance(rhs, TiledTensorType):
            from hidet.ir.cute.ops import Arithmetic

            if not rhs.scope.is_register():
                return self.fail(lhs, rhs, op)
            dtype = self.infer(lhs, rhs.dtype, op)
            layout = Arithmetic.infer_layout(rhs)
            return tiled_tensor(dtype, layout, rhs.scope)
        else:
            return self.fail(lhs, rhs, op)

    def dtype(self, lhs: DataType, rhs: DataType, op: Type[BinaryExpr]):
        self.fail(lhs, rhs, op)

    def pointer(self, lhs: Union[TensorPointerType, PointerType], rhs: Union[TensorPointerType, PointerType], op):
        self.fail(lhs, rhs, op)

    def pointer_dtype(self, pointer: Union[TensorPointerType, PointerType], dtype: DataType, op):
        self.fail(pointer, dtype, op)

    def fail(self, a, b, op):
        raise RuntimeError('can not infer type for: {} {} {}'.format(a, type(op).__name__, b))


class ArithBinaryInfer(BinaryTypeInfer):
    def dtype(self, lhs: DataType, rhs: DataType, op):
        from hidet.ir.dtypes.promotion import promote_type

        return promote_type(lhs, rhs)

    def pointer(self, lhs: Union[TensorPointerType, PointerType], rhs: Union[TensorPointerType, PointerType], op):
        if issubclass(op, Sub):
            return data_type('int32')
        else:
            return self.fail(lhs, rhs, op)

    def pointer_dtype(self, pointer: Union[TensorPointerType, PointerType], dtype: DataType, op):
        if issubclass(op, (Add, Sub)):
            return pointer
        else:
            return self.fail(pointer, dtype, op)


class CompareBinaryInfer(BinaryTypeInfer):
    def dtype(self, lhs: DataType, rhs: DataType, op):
        return dtypes.boolean

    def pointer(self, lhs: Union[TensorPointerType, PointerType], rhs: Union[TensorPointerType, PointerType], op):
        if issubclass(op, (Equal, NotEqual)):
            return dtypes.boolean
        else:
            return self.fail(lhs, rhs, op)

    def pointer_dtype(self, pointer: Union[TensorPointerType, PointerType], dtype: DataType, op):
        if issubclass(op, (Equal, NotEqual)):
            return dtypes.boolean
        else:
            return self.fail(pointer, dtype, op)


class LogicalBinaryInfer(BinaryTypeInfer):
    # and, or
    def dtype(self, lhs: DataType, rhs: DataType, op):
        if is_bool(lhs) and is_bool(rhs):
            return dtypes.boolean
        else:
            return self.fail(lhs, rhs, op)


_arith_binary_infer = ArithBinaryInfer()
_compare_binary_infer = CompareBinaryInfer()
_logical_binary_infer = LogicalBinaryInfer()


class TypeInfer(IRFunctor):
    def visit_Address(self, e: Address):
        base_type = self(e.expr)
        return PointerType(base_type=base_type)

    def visit_Reference(self, e: Reference):
        return self(e.expr)

    def visit_Binary(self, e: BinaryExpr):
        a_type: BaseType = self.visit(e.a)
        b_type: BaseType = self.visit(e.b)
        op = type(e)
        if isinstance(e, (Add, Sub, Multiply, Div, Mod)):
            return _arith_binary_infer.infer(a_type, b_type, op)
        elif isinstance(e, (LessThan, Equal, LessEqual, NotEqual)):
            return _compare_binary_infer.infer(a_type, b_type, op)
        elif isinstance(e, (LogicalAnd, LogicalOr)):
            return _logical_binary_infer.infer(a_type, b_type, op)
        else:
            raise NotImplementedError()

    def visit_Neg(self, e: Neg):
        return self(e.a)

    def visit_Add(self, e: Add):
        return self.visit_Binary(e)

    def visit_Sub(self, e: Sub):
        return self.visit_Binary(e)

    def visit_Multiply(self, e: Multiply):
        return self.visit_Binary(e)

    def visit_Div(self, e: Div):
        return self.visit_Binary(e)

    def visit_Mod(self, e: Mod):
        return self.visit_Binary(e)

    def visit_FloorDiv(self, e: FloorDiv):
        return self.visit_Binary(e)

    def visit_LessThan(self, e: LessThan):
        return self.visit_Binary(e)

    def visit_Equal(self, e: Equal):
        return self.visit_Binary(e)

    def visit_NotEqual(self, e: NotEqual):
        return self.visit_Binary(e)

    def visit_LessEqual(self, e: LessEqual):
        return self.visit_Binary(e)

    def visit_And(self, e: LogicalAnd):
        return self.visit_Binary(e)

    def visit_Or(self, e: LogicalOr):
        return self.visit_Binary(e)

    def visit_Not(self, e: LogicalNot):
        # commenting out thie assertion makes logical_not
        # work for hidet Tensor...
        # assert is_bool(self.visit(e.a))
        return data_type('bool')

    def visit_BitwiseAnd(self, e: BitwiseAnd):
        return self.visit(e.a)

    def visit_BitwiseOr(self, e: BitwiseOr):
        return self.visit(e.a)

    def visit_BitwiseXor(self, e: BitwiseXor):
        return self.visit(e.a)

    def visit_BitwiseNot(self, e: BitwiseNot):
        return self.visit(e.a)

    def visit_LeftShift(self, e: LeftShift):
        return self.visit(e.a)

    def visit_RightShift(self, e: RightShift):
        return self.visit(e.a)

    def visit_TensorElement(self, e: TensorElement):
        base_type = self.visit(e.base)
        if isinstance(base_type, TensorType):
            return base_type.dtype
        elif isinstance(base_type, PointerType):
            return base_type.base_type
        elif isinstance(base_type, TensorPointerType):
            return base_type.tensor_type.dtype
        elif isinstance(base_type, ArrayType):
            return base_type.base_type
        else:
            raise NotImplementedError()

    def visit_TensorSlice(self, e: TensorSlice):
        base_type = self.visit(e.base)
        if isinstance(base_type, TensorPointerType):
            base_type = base_type.tensor_type
        assert isinstance(base_type, TensorType)
        shape = []
        for dim, (index, start, end) in enumerate(zip(e.indices, e.starts, e.ends)):
            if index is not None:
                pass
            else:
                if start is None:
                    start = 0
                if end is None:
                    end = base_type.shape[dim]
                shape.append(end - start)

        # the layout of the slice is not used
        return tensor_pointer_type(dtype=base_type.dtype, shape=shape, layout=None)

    def visit_IfThenElse(self, e: IfThenElse):
        cond_type = self.visit(e.cond)
        true_type = self.visit(e.then_expr)
        false_type = self.visit(e.else_expr)
        assert is_bool(cond_type)

        pointer_types = (PointerType, TensorPointerType)

        if isinstance(true_type, DataType) and isinstance(false_type, DataType):
            if true_type.name != false_type.name:
                msg = 'If-then-else operand 1 and 2 have different types ({} vs {}) {}'.format(true_type, false_type, e)
                raise ValueError(msg)
        elif isinstance(true_type, pointer_types) and isinstance(false_type, pointer_types):
            # pass the check
            pass
        else:
            msg = 'If-then-else operand 1 and 2 have different types ({} vs {}): {}'.format(true_type, false_type, e)
            raise ValueError(msg)

        return true_type

    def visit_Let(self, e: Let):
        self.visit(e.value)
        return self.visit(e.body)

    def visit_Call(self, e: Call):
        func_var = e.func_var
        func_type = func_var.type
        if not isinstance(func_type, FuncType):
            raise ValueError(
                'Type infer failed, expect a function var "{}" but got variable with type "{}"'.format(
                    func_var, func_type
                )
            )
        args_type = [self(arg) for arg in e.args]
        return func_type.ret_type_on(args_type)

    def visit_Cast(self, e: Cast):
        return e.target_type

    def visit_Dereference(self, e: Dereference):
        tp = self.visit(e.expr)
        assert isinstance(tp, PointerType)
        return tp.base_type

    def visit_Var(self, e: Var):
        return e.type

    def visit_Constant(self, e: Constant):
        return e.type

    def visit_ScalarInput(self, node: ScalarInput):
        return node.dtype

    def visit_TensorInput(self, node: TensorInput):
        return node.ttype

    def visit_GridCompute(self, c: GridCompute):
        dtype = self.visit(c.value)
        return tensor_type(dtype, c._shape, c.layout)  # pylint: disable=protected-access

    def visit_ReduceCompute(self, c: ReduceCompute):
        return self.visit(c.value)

    def visit_ArgReduceCompute(self, c: ArgReduceCompute):
        return c.index_dtype

    def visit_PlaceholderExpr(self, e: PlaceholderExpr):
        raise NotImplementedError()

    def visit_CallOp(self, call: CallOp):
        arg_types = [self.visit(arg) for arg in call.op.args if not isinstance(arg, (tuple, list))]
        return call.op.infer_type(arg_types)


def infer_type(expr):
    infer = TypeInfer()
    return infer(expr)
