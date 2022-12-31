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
from typing import Union

from hidet.ir.expr import Expr, convert
from hidet.ir.type import VoidType, FuncType, data_type
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
    register_primitive_function(
        name='cuda_nano_sleep', func_or_type=FuncType([data_type('uint32')], VoidType()), codegen_name='__nanosleep'
    )


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
