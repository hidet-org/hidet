from __future__ import annotations
from typing import Union
from hidet.ir.type import DataType, data_type
from hidet.ir.expr import Expr, Constant


class ReduceOperation:
    @staticmethod
    def from_name(name: str) -> ReduceOperation:
        name2operation = {'max': Max, 'min': Min, 'sum': Sum, 'avg': Average}
        if name not in name2operation:
            raise ValueError('Can not recognize reduce type {}'.format(name))
        return name2operation[name]()

    def __str__(self):
        return self.__class__.__name__.lower()

    def initial_value(self, dtype: Union[DataType, str]) -> Constant:
        raise NotImplementedError()

    def combine(self, lhs: Expr, rhs: Expr) -> Expr:
        raise NotImplementedError()

    def arg_combine(self, lhs_value: Expr, rhs_value: Expr):
        raise NotImplementedError()

    def require_finalize(self) -> bool:
        raise NotImplementedError()

    def finalize(self, acc: Expr, size: Expr) -> Expr:
        raise NotImplementedError()


class Min(ReduceOperation):
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

    def require_finalize(self) -> bool:
        return False

    def finalize(self, acc: Expr, size: Expr) -> Expr:
        return acc


class Max(ReduceOperation):
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

    def require_finalize(self) -> bool:
        return False

    def finalize(self, acc: Expr, size: Expr) -> Expr:
        return acc


class Sum(ReduceOperation):
    def initial_value(self, dtype: Union[DataType, str]) -> Constant:
        if isinstance(dtype, str):
            dtype = data_type(dtype)
        return dtype.zero

    def combine(self, lhs: Expr, rhs: Expr) -> Expr:
        return lhs + rhs

    def arg_combine(self, lhs_value: Expr, rhs_value: Expr):
        raise ValueError('Sum reduction does not argument reduce.')

    def require_finalize(self) -> bool:
        return False

    def finalize(self, acc: Expr, size: Expr) -> Expr:
        return acc


class Average(ReduceOperation):
    def initial_value(self, dtype: Union[DataType, str]) -> Constant:
        if isinstance(dtype, str):
            dtype = data_type(dtype)
        return dtype.zero

    def combine(self, lhs: Expr, rhs: Expr) -> Expr:
        return lhs + rhs

    def arg_combine(self, lhs_value: Expr, rhs_value: Expr):
        raise ValueError('Average reduction does not argument reduce.')

    def require_finalize(self) -> bool:
        return True

    def finalize(self, acc: Expr, size: Expr) -> Expr:
        return acc / size
