"""
Testing script for distributed components for hidet
To debug, set the environment variable NCCL_DEBUG=INFO
"""
import hidet
import multiprocessing
from multiprocessing import Process
import numpy
import argparse

import hidet.cuda.nccl
from hidet.cuda import nccl
from hidet.cuda.nccl import NcclUniqueId, NcclDataType, NcclRedOp, nccl_library_filename
from hidet.ffi import runtime_api
from hidet.lang import attrs
from hidet.ir.primitives.cuda.nccl import all_reduce
from hidet.ir.type import data_type
from hidet.utils import prod
from hidet.drivers import build_ir_module
from hidet.cuda.nccl.libinfo import get_nccl_include_dirs, get_nccl_library_search_dirs
from hidet.runtime import load_compiled_module

print("NCCL version:", nccl.nccl_version())

parser = argparse.ArgumentParser()
parser.add_argument("n_gpus", type=int)
parser.add_argument("reduce_op", choices=['sum', 'prod', 'max', 'min', 'avg'])
args = parser.parse_args()

def run(world_size, rank, shared_id, barrier):
    numpy.random.seed(rank)

    # Initialize unique id
    if rank == 0:
        nccl.init_unique_id(shared_id)

    barrier.wait()
    hidet.cuda.set_device(rank)

    print('initialize', rank)
    # Create NcclCommunicator and set the cuda context
    # this part should be moved into CompiledGraph in the future
    comm = nccl.create_comm(world_size, shared_id, rank)
    comms_array = nccl.comms_to_array([comm])
    runtime_api.set_nccl_comms(comms_array)

    # Initialize send and receive buffer
    device = f"cuda:{rank}"
    send = hidet.randn([2, 2], device=device)
    recv = hidet.empty([2, 2], device=device)

    print(rank, send)

    dtype = data_type('float32')
    shape = [2, 2] 
    nbytes = dtype.nbytes * prod(shape)

    # Define IRModule
    with hidet.script_module() as script_module:
        @hidet.script
        def launch(send: dtype[shape], recv: dtype[shape]):
            attrs.func_kind = 'public'
            all_reduce(0, send, recv, nbytes, NcclDataType.float32, getattr(NcclRedOp, args.reduce_op))

    # Build
    ir_module = script_module.ir_module()
    ir_module.target = 'cuda'
    ir_module.include_dirs.extend(get_nccl_include_dirs())
    ir_module.linking_dirs.extend(get_nccl_library_search_dirs())
    ir_module.include_headers.append(["nccl.h"])
    ir_module.linking_libs.append(":" + nccl_library_filename())
    out_dir = f'./.cache/all_reduce_{rank}'

    build_ir_module(ir_module, out_dir, target='cuda')
    compiled_module = load_compiled_module(out_dir)

    compiled_module(send, recv)
    s = hidet.cuda.current_stream() 
    s.synchronize()
    print(rank, recv)

world_size = args.n_gpus

# Barrier to ensure unique id is created
barrier = multiprocessing.Barrier(world_size)

# Create a unique id object in shared memory
shared_id = multiprocessing.Value(NcclUniqueId, lock=False)

processes = [Process(target=run, args=(world_size, i, shared_id, barrier)) for i in range(world_size)]

for p in processes:
    p.start()
for p in processes:
    p.join()