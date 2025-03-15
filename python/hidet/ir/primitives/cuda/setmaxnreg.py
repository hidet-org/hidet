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
from hidet.utils import initialize
from hidet.ir.expr import Expr
from hidet.ir.stmt import asm
from hidet.ir.func import Function
from hidet.ir.primitives.func import register_primitive_function
from hidet.ir.primitives.cuda.funcs import call_cuda
from hidet.lang import attrs, script


# The syntax of the setmaxnreg looks like this:
# setmaxnreg.action.sync.aligned.u32 imm-reg-count;
# where .action = {.inc, .dec}
# For more details:
# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg
@initialize()
def register_setmaxnreg():
    for action in ["inc", "dec"]:
        # Valid register counts: 24-256, multiples of 8
        for nregs in range(24, 257, 8):
            func_name = "cuda_setmaxnreg_{}_{}".format(action, nregs)
            template_string = f"setmaxnreg.{action}.sync.aligned.u32 {nregs};"

            @script
            def cuda_setmaxnreg():
                attrs.func_name = func_name
                attrs.func_kind = "cuda_internal"
                asm(template=template_string, inputs=[])

            assert isinstance(cuda_setmaxnreg, Function)
            register_primitive_function(name=cuda_setmaxnreg.name, func_or_type=cuda_setmaxnreg)


def setmaxnreg(nregs: Union[Expr, int], action: str):
    """
    Increase the maximum number of registers that can be allocated to a thread.

    Parameters
    ----------
    nregs: Expr or int
        The number of registers to set as maximum. Must be in range 24-256 and a multiple of 8.
    action: str
        The action to perform. Must be either 'inc' or 'dec'.

    Returns
    -------
    ret: Call
        The call expression.
    """
    from hidet.ir.tools.simplifier import simplify_to_int

    if action not in ["inc", "dec"]:
        raise ValueError("action must be 'inc' or 'dec'")

    if isinstance(nregs, Expr):
        nregs = simplify_to_int(nregs)

    if not isinstance(nregs, int):
        raise ValueError("Register count must be a constant integer value")

    if nregs < 24 or nregs > 256 or nregs % 8 != 0:
        raise ValueError(f"Register count must be in range 24-256 and a multiple of 8, got {nregs}")

    func_name = f"setmaxnreg_{action}_{nregs}"
    return call_cuda(func_name=func_name, args=[])
