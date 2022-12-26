"""
This submodule exposes the functions defined in the array API standard: https://data-apis.org/array-api
"""
from collections import namedtuple
from hidet.ir.type import DataType

from hidet.ir.dtypes import (
    boolean as bool,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)
from hidet.graph.tensor import ones, asarray


def finfo(dtype: DataType):
    ret_type = namedtuple("finfo", ["bits", "eps", "max", "min", "smallest_normal", "dtype"])
    if dtype == float32:
        return ret_type(
            bits=32,
            eps=1.1920929e-07,
            max=3.4028235e+38,
            min=-3.4028235e+38,
            smallest_normal=1.1754944e-38,
            dtype=float32,
        )
    elif dtype == float64:
        return ret_type(
            bits=64,
            eps=2.220446049250313e-16,
            max=1.7976931348623157e+308,
            min=-1.7976931348623157e+308,
            smallest_normal=2.2250738585072014e-308,
            dtype=float64,
        )
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def iinfo(dtype: DataType):
    ret_type = namedtuple("iinfo", ["bits", "max", "min", "dtype"])
    if not dtype.is_integer():
        raise ValueError(f"Unsupported dtype: {dtype}")
    return ret_type(
        bits=dtype.nbytes * 8,
        max=dtype.max_value.value,
        min=dtype.min_value.value,
        dtype=dtype,
    )
