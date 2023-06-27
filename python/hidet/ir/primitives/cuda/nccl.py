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
from hidet.ir.expr import Expr
from hidet.ir.stmt import BlackBoxStmt
from hidet.ir.type import DataType

# TODO: we should not put nccl-related types here since hidet.cuda.nccl depends on
#       the existence of nccl library?
from hidet.cuda.nccl import NcclRedOp, dtype_to_nccl


def all_reduce(sendbuff: Expr, recvbuff: Expr, count: Expr, dtype: DataType, op: NcclRedOp, comm_id: int):
    from hidet.ir.primitives.runtime import get_cuda_stream, get_nccl_comm

    comm = get_nccl_comm(comm_id)
    return BlackBoxStmt(
        'ncclAllReduce({}, {}, {}, (ncclDataType_t){}, (ncclRedOp_t){}, '
        '(ncclComm_t){}, (cudaStream_t){});'.format(
            sendbuff, recvbuff, count, int(dtype_to_nccl(dtype)), int(op), comm, get_cuda_stream()
        )
    )
