from typing import Union

from hidet.ir.expr import Expr, convert
from hidet.ir.func import Function, FuncType
from hidet.ir.type import ScalarType
from hidet.ir.dialects.lowlevel import VoidType
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.utils import initialize


@initialize()
def register_functions():
    # from hidet.lang import script, u32, asm, attr
    #
    # @script
    # def cuda_nano_sleep(nano_seconds: u32):
    #     attr.func_kind = 'cuda_device'
    #     attr.func_name = 'cuda_nano_sleep'
    #     asm('nanosleep.u32 %0;', inputs=[nano_seconds], is_volatile=True)
    #
    # assert isinstance(cuda_nano_sleep, Function)
    # register_primitive_function(cuda_nano_sleep.name, cuda_nano_sleep)
    register_primitive_function(name='cuda_nano_sleep',
                                func_or_type=FuncType([ScalarType('uint32')], VoidType()),
                                codegen_name='__nanosleep')


def nano_sleep(nano_seconds: Union[Expr, int]):
    """
    Sleep for given nanoseconds.

    Parameters
    ----------
    nano_seconds: int
        The number of nanoseconds to sleep.
    """
    if isinstance(nano_seconds, int):
        nano_seconds = convert(nano_seconds, 'uint32')
    return call_primitive_func('cuda_nano_sleep', [nano_seconds])
