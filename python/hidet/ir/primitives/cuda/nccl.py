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

from hidet.cuda.nccl import NcclRedOp, dtype_to_nccl
from hidet.ir.primitives.runtime import get_cuda_stream, get_nccl_comm

def all_reduce(sendbuff: Expr, recvbuff: Expr, count: Expr, dtype: DataType, op: NcclRedOp, comm_id: int):
    comm = get_nccl_comm(comm_id)
    return BlackBoxStmt(
        'ncclAllReduce({}, {}, {}, (ncclDataType_t){}, (ncclRedOp_t){}, (ncclComm_t){}, (cudaStream_t){});',
        sendbuff,
        recvbuff,
        count,
        int(dtype_to_nccl(dtype)),
        int(op),
        comm,
        get_cuda_stream(),
    )

def broadcast(sendbuff: Expr, recvbuff: Expr, count: Expr, dtype: DataType, root: Expr, comm_id: int):
    comm = get_nccl_comm(comm_id)
    return BlackBoxStmt(
        'ncclBroadcast({}, {}, {}, (ncclDataType_t){}, {}, (ncclComm_t){}, (cudaStream_t){});',
        sendbuff,
        recvbuff,
        int(dtype_to_nccl(dtype)),
        root,
        comm,
        get_cuda_stream()
    )

def reduce(sendbuff: Expr, recvbuff: Expr, count: Expr, dtype: DataType, op: NcclRedOp, root: Expr, comm_id: int):
    comm = get_nccl_comm(comm_id)
    return BlackBoxStmt(
        'ncclReduce({}, {}, (ncclDataType_t){}, {}, {}, (ncclComm_t){}, (cudaStream_t){});',
        sendbuff,
        recvbuff,
        int(dtype_to_nccl(dtype)),
        int(op),
        root,
        comm,
        get_cuda_stream()
    )

def all_gather(sendbuff: Expr, recvbuff: Expr, sendcount: Expr, dtype: DataType, comm_id:int):
    comm = get_nccl_comm(comm_id)
    return BlackBoxStmt(
        'ncclAllGather({}, {}, {}, (ncclDataType_t){}, (ncclComm_t){}, (cudaStream_t){});',
        sendbuff,
        recvbuff,
        sendcount,
        int(dtype_to_nccl(dtype)),
        comm,
        get_cuda_stream()
    )

def reduce_scatter(sendbuff: Expr, recvbuff: Expr, recvcount: Expr, dtype: DataType, op: NcclRedOp, comm_id: int):
    comm = get_nccl_comm(comm_id)
    return BlackBoxStmt(
        'ncclReduceScatter({}, {}, {}, (ncclDataType_t){}, (ncclRedOp_t){}, (ncclComm_t){}, (cudaStream_t){});',
        sendbuff,
        recvbuff,
        recvcount,
        int(dtype_to_nccl(dtype)),
        int(op),
        comm,
        get_cuda_stream() 
    )