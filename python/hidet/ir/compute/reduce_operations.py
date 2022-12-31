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
from enum import Enum
from typing import Union
from hidet.ir.type import DataType, data_type
from hidet.ir.expr import Expr, Constant, LogicalAnd, LogicalOr


class ReduceType(Enum):
    Sum = 'sum'
    Product = 'prod'
    Max = 'max'
    Min = 'min'
    Average = 'avg'
    And = 'and'
    Or = 'or'

    def __str__(self):
        return self.value


class ReduceOperation:
    @staticmethod
    def from_name(name: Union[ReduceType, str]) -> ReduceOperation:
        name = ReduceType(name)
        name2operation = {
            ReduceType.Sum: SumReduce,
            ReduceType.Product: ProductReduce,
            ReduceType.Max: MaxReduce,
            ReduceType.Min: MinReduce,
            ReduceType.Average: AverageReduce,
            ReduceType.And: AndReduce,
            ReduceType.Or: OrReduce,
        }
        if name not in name2operation:
            raise ValueError('Can not recognize reduce type {}'.format(name))
        return name2operation[name]()

    def __str__(self):
        return self.__class__.__name__.lower()

    def initial_value(self, dtype: Union[DataType, str]) -> Constant:
        """
        The initial value of the reduction.

        Parameters
        ----------
        dtype: DataType
            The data type of elements to conduct the reduction.

        Returns
        -------
        init_value: Constant
            The initial value of the reduction.
        """
        raise NotImplementedError()

    def combine(self, lhs: Expr, rhs: Expr) -> Expr:
        """
        Reduce two values.

        Parameters
        ----------
        lhs: Expr
            The left hand side value.
        rhs: Expr
            The right hand side value.

        Returns
        -------
        result: Expr
            The result of the reduction.
        """
        raise NotImplementedError()

    def arg_combine(self, lhs_value: Expr, rhs_value: Expr):
        """
        For some reductions like argmin and argmax, we need to combine the arg (index) instead of the value itself.
        This function returns True if the combine(lhs_value, rhs_value) == lhs_value, otherwise False.

        Only need to override this function if the reduction supports arg_reduce (e.g., argmin, argmax).

        Parameters
        ----------
        lhs_value: Expr
            The left hand side value.

        rhs_value: Expr
            The right hand side value.

        Returns
        -------
        result: bool
            True if the combine(lhs_value, rhs_value) == lhs_value, otherwise False.
        """
        raise ValueError('{} reduction does not argument reduce.'.format(str(self)))

    def require_finalize(self) -> bool:
        """
        Whether the reduction requires a finalization step.

        For some reduction, the finalization step is required to get the final result. For example, the average
        reduction requires a finalization step to divide the sum by the size of the reduction.

        Returns
        -------
        result: bool
            True if the reduction requires a finalization step, otherwise False.
        """
        return False

    def finalize(self, acc: Expr, size: Expr) -> Expr:
        """
        Finalize the reduction result.

        Parameters
        ----------
        acc: Expr
            The accumulated value.
        size: Expr
            The number of elements to conduct the reduction.

        Returns
        -------
        result: Expr
            The final result of the reduction.
        """
        return acc


class MinReduce(ReduceOperation):
    def initial_value(self, dtype: Union[DataType, str]) -> Expr:
        if isinstance(dtype, str):
            dtype = data_type(dtype)
        return dtype.max_value

    def combine(self, lhs: Expr, rhs: Expr) -> Expr:
        from hidet.ir import primitives  # pylint: disable=import-outside-toplevel

        return primitives.min(lhs, rhs)

    def arg_combine(self, lhs_value: Expr, rhs_value: Expr):
        from hidet.ir.expr import LessThan  # pylint: disable=import-outside-toplevel

        return LessThan(lhs_value, rhs_value)


class MaxReduce(ReduceOperation):
    def initial_value(self, dtype: Union[DataType, str]) -> Constant:
        if isinstance(dtype, str):
            dtype = data_type(dtype)
        return dtype.min_value

    def combine(self, lhs: Expr, rhs: Expr) -> Expr:
        from hidet.ir import primitives  # pylint: disable=import-outside-toplevel

        return primitives.max(lhs, rhs)

    def arg_combine(self, lhs_value: Expr, rhs_value: Expr):
        from hidet.ir.expr import LessThan  # pylint: disable=import-outside-toplevel

        return LessThan(rhs_value, lhs_value)


class SumReduce(ReduceOperation):
    def initial_value(self, dtype: Union[DataType, str]) -> Constant:
        if isinstance(dtype, str):
            dtype = data_type(dtype)
        return dtype.zero

    def combine(self, lhs: Expr, rhs: Expr) -> Expr:
        return lhs + rhs


class AverageReduce(ReduceOperation):
    def initial_value(self, dtype: Union[DataType, str]) -> Constant:
        if isinstance(dtype, str):
            dtype = data_type(dtype)
        return dtype.zero

    def combine(self, lhs: Expr, rhs: Expr) -> Expr:
        return lhs + rhs

    def require_finalize(self) -> bool:
        return True

    def finalize(self, acc: Expr, size: Expr) -> Expr:
        return acc / size


class AndReduce(ReduceOperation):
    def initial_value(self, dtype: Union[DataType, str]) -> Constant:
        dtype = data_type(dtype)
        assert dtype.name == 'bool', 'AndReduce only support bool type'
        return dtype.one

    def combine(self, lhs: Expr, rhs: Expr) -> Expr:
        return LogicalAnd(lhs, rhs)


class OrReduce(ReduceOperation):
    def initial_value(self, dtype: Union[DataType, str]) -> Constant:
        dtype = data_type(dtype)
        assert dtype.name == 'bool', 'OrReduce only support bool type'
        return dtype.zero

    def combine(self, lhs: Expr, rhs: Expr) -> Expr:
        return LogicalOr(lhs, rhs)


class ProductReduce(ReduceOperation):
    def initial_value(self, dtype: Union[DataType, str]) -> Constant:
        return dtype.one

    def combine(self, lhs: Expr, rhs: Expr) -> Expr:
        return lhs * rhs
