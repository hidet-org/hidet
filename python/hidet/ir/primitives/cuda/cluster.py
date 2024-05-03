from collections import namedtuple
from typing import Union

from hidet.ir.expr import Expr
from hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from hidet.ir.primitives.vars import lookup_primitive_variable, register_primitive_variable
from hidet.ir.type import DataType, FuncType, PointerType, VoidType, data_type
from hidet.utils.py import initialize
from hidet.ir.dtypes import i32

_cluster_fields = ["thread_rank", "block_rank", "dim_threads", "dim_blocks"]


@initialize()
def register_cuda_cluster_functions():

    for suffix in _cluster_fields:
        register_primitive_variable(name=f"cooperative_groups::this_cluster().{suffix}()", dtype=i32)

    register_primitive_function(
        name="this_cluster.sync",
        func_or_type=FuncType([], VoidType()),
        codegen_name="cooperative_groups::this_cluster().sync",
    )

    for dtype in ['int8', 'uint8', 'uint32', 'uint64', 'int32', 'float16', 'float32', 'bool']:
        dtype = data_type(dtype)

        register_primitive_function(
            name=f"this_cluster.map_shared_rank_{dtype}",
            func_or_type=FuncType([PointerType(dtype), i32], PointerType(dtype)),
            codegen_name="cooperative_groups::this_cluster().map_shared_rank",
        )


def cluster_sync():
    return call_primitive_func("this_cluster.sync", [])


def cluster_map_shared_rank(addr: Expr, rank: Union[Expr, int], dtype: Union[DataType, str]):
    func_name = f"this_cluster.map_shared_rank_{dtype}"
    return call_primitive_func(func_name, [addr, rank])


this_cluster = namedtuple("this_cluster", field_names=_cluster_fields + ["sync", "map_shared_rank"])(
    *[lookup_primitive_variable("cooperative_groups::this_cluster().{}()".format(field)) for field in _cluster_fields],
    cluster_sync,
    cluster_map_shared_rank,
)
