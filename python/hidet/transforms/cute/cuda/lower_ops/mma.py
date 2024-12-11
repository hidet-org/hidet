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
from typing import List, Union, Optional
from hidet.ir.expr import Expr
from hidet.ir.type import TensorType
from hidet.ir.tools import infer_type

from hidet.ir.cute.ops import Mma

from .registry import OpEmitter, Buffer, register_impl


@register_impl(Mma)
class MmaEmitter(OpEmitter):
    """
    Emitter for Matrix Multiply-Accumulate (MMA) operations, responsible for generating code for
    MMA operations on tensors.
    """

    def emit(self, op: Mma, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        """
        Emit the code for the MMA operation.

        Args:
            op (Mma): The MMA operation.
            args (List[Union[Buffer, Expr]]): The input arguments, expected to be buffers.
            output (Optional[Buffer]): The output buffer, if any.

        Raises:
            AssertionError: If any of the input arguments are not buffers.
        """

        assert all(isinstance(arg, Buffer) for arg in args)
        d: Buffer = args[0]
        a: Buffer = args[1]
        b: Buffer = args[2]
        c: Buffer = args[3]
        annotations = op.annotations
        assert len(annotations) > 0
        inst = annotations["inst"]
        a_rest = annotations["a_rest"]
        b_rest = annotations["b_rest"]
        c_rest = annotations["c_rest"]
        d_rest = annotations["d_rest"]

        m = d_rest[0].shape
        n = d_rest[1].shape
        k = a_rest[1].shape

        def get_pointer(buf: Expr):
            buf_ty = infer_type(buf)
            if isinstance(buf_ty, TensorType):
                return ~buf[0]
            else:
                return buf

        d_ptr, a_ptr, b_ptr, c_ptr = (get_pointer(i) for i in [d.buffer, a.buffer, b.buffer, c.buffer])

        # The code generation follows the structure of a typical matrix multiplication:
        # ```python
        # for m in range(M):
        #     for n in range(N):
        #         for k in range(K):
        #             d[m, n] += a[m, k] * b[k, n] + c[m, n]
        # ```
        # Here, instead of single indices, `m_indices`, `n_indices`, and `k_indices` are tuples of indices
        # for the M, N, and K dimensions, respectively. Instruction selection for the MMA operation is done first.
        # The thread-value layouts' axes are grouped into instruction and rest axes. The rest axes represent the
        # iteration space for the loops. By analyzing the strides of the indices, we can extract the indices
        # corresponding to the M, N, and K dimensions and generate the loop nest for the MMA instruction.
        # Note: this approach can unify the code generation for different MMA instructions.
        with self.for_grid(m) as m_indices:
            with self.for_grid(n) as n_indices:
                with self.for_grid(k) as k_indices:
                    a_addr = a_ptr + a_rest((m_indices, k_indices), base=a.offset)
                    b_addr = b_ptr + b_rest((n_indices, k_indices), base=b.offset)
                    c_addr = c_ptr + c_rest((m_indices, n_indices), base=c.offset)
                    d_addr = d_ptr + d_rest((m_indices, n_indices), base=d.offset)
                    self.append(inst(d_addr, a_addr, b_addr, c_addr))
