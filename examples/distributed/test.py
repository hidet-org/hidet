"""
Testing script for distributed components for hidet
To debug, set the environment variable NCCL_DEBUG=INFO
"""
import hidet
import multiprocessing
from multiprocessing import Process
import numpy
import argparse

import hidet
import hidet.cuda.nccl
from hidet.cuda import nccl
from hidet.cuda.nccl import NcclUniqueId,NcclRedOp

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

    device = f"cuda:{rank}"
    x = hidet.randn([2, 3], device=device)
    w = hidet.randn([3, 2], device=device)

    # Create Computation Graph
    x_symb = hidet.symbol_like(x)
    w_symb = hidet.symbol_like(w)
    y_local = hidet.ops.relu(x_symb @ w_symb)
    y_sync = hidet.ops.all_reduce(y_local, getattr(NcclRedOp, args.reduce_op), 0)
    graph = hidet.trace_from([y_local, y_sync], inputs=[x_symb, w_symb])
    opt_graph = hidet.graph.optimize(graph)

    # Create Distributed Graph
    dist_graph = hidet.graph.DistributedFlowGraph(opt_graph, world_size, rank)
    dist_graph.initialize(shared_id)

    y_local, y_sync = dist_graph(x, w)

    s = hidet.cuda.current_stream() 
    s.synchronize()
    print(f"process {rank}\nbefore allreduce:{y_local}\nafter allreduce:{y_sync}")

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