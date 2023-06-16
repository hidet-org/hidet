from ..ffi import NcclUniqueId, NcclCommunicator, nccl_runtime_api

def create_comm(nranks: int, unique_id: NcclUniqueId, rank: int):
    handle = nccl_runtime_api.comm_init_rank(nranks, unique_id, rank)
    return NcclCommunicator(handle)