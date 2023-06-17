import hidet
import multiprocessing
from multiprocessing import Process
import hidet.cuda.nccl
from hidet.cuda.nccl import NcclUniqueId, create_comm, ncclDataType, ncclRedOp
from hidet.ffi.runtime_api import NCCLRuntimeAPI
from hidet.lang import attrs
from hidet.ir.primitives.cuda.nccl import all_reduce
from hidet.ir.type import data_type
from hidet.ir import Task
from hidet.graph.ops.utils import input_like, compute
from hidet.utils import prod
from hidet.drivers import build_task, build_ir_module
from hidet.libinfo import get_nccl_include_dirs, get_nccl_library_search_dirs
from hidet.runtime import load_compiled_module

print("NCCL version:", NCCLRuntimeAPI.get_version())

def run(world_size, rank, shared_id, barrier):
    if rank == 0:
        NCCLRuntimeAPI.get_unique_id(shared_id)
    barrier.wait()
    hidet.cuda.set_device(rank)

    print('initialize', rank)
    comm = create_comm(world_size, shared_id, rank)

    device = f"cuda:{rank}"
    send = hidet.randn([2, 2], device=device)
    recv = hidet.empty([2, 2], device=device)

    dtype = data_type('float32')
    shape = [2, 2] 
    nbytes = dtype.nbytes * prod(shape)

    with hidet.script_module() as script_module:
        @hidet.script
        def launch(send: dtype[shape], recv: dtype[shape]):
            attrs.func_kind = 'public'
            all_reduce(0, send, recv, nbytes, ncclDataType.float32, ncclRedOp.sum)
            
    ir_module = script_module.ir_module()
    ir_module.target = 'cuda'
    ir_module.include_dirs.extend(get_nccl_include_dirs())
    ir_module.linking_dirs.extend(get_nccl_library_search_dirs())
    ir_module.include_headers.append(["nccl.h"])
    ir_module.linking_libs.append(":libnccl.so.2")
    out_dir = f'./.cache/all_reduce_{rank}'
    build_ir_module(ir_module, out_dir, target='cuda')
    compiled_module = load_compiled_module(out_dir)
    compiled_module(send, recv)
    s = hidet.cuda.current_stream() 
    s.synchronize()
    print(recv)

world_size = 4
barrier = multiprocessing.Barrier(world_size)
shared_id = multiprocessing.Value(NcclUniqueId, lock=False)
processes = [Process(target=run, args=(world_size, i, shared_id, barrier)) for i in range(world_size)]

for p in processes:
    p.start()
for p in processes:
    p.join()