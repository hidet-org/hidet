"""
Testing script for distributed components for hidet
To debug, set the environment variable NCCL_DEBUG=INFO

To install nccl, run

    pip install nvidia-nccl-cu11==2.18.3

Or

    pip install nvidia-nccl-cu12==2.18.3
"""
import hidet
import multiprocessing
from multiprocessing import Process
import numpy
import argparse

import hidet
import hidet.cuda.nccl
from hidet.cuda import nccl
from hidet.cuda.nccl import NcclUniqueId
from hidet.runtime.compiled_graph import GraphDistributedInfo

print("NCCL version:", nccl.nccl_version())

parser = argparse.ArgumentParser()
parser.add_argument("n_gpus", type=int)
parser.add_argument("reduce_op", choices=['sum', 'prod', 'max', 'min', 'avg'])
parser.add_argument("--group_size", type=int, default=0)
args = parser.parse_args()

def run(world_size, rank, shared_id, barrier):
    numpy.random.seed(rank)

    # Initialize unique id
    if rank == 0:
        nccl.init_unique_id(shared_id)

    barrier.wait()
    hidet.cuda.set_device(rank)

    use_group = args.group_size > 1
    if use_group:
        gs = args.group_size
        gn = world_size // gs
        assert world_size % gs == 0
        groups = [list(range(i * gs, (i + 1) * gs)) for i in range(gn)]
    else:
        groups = []
        

    device = f"cuda:{rank}"
    x = hidet.randn([1, 3], device=device)
    w = hidet.randn([3, 2], device=device)

    # Create Computation Graph
    x_symb = hidet.symbol_like(x)
    w_symb = hidet.symbol_like(w)
    y_local = hidet.ops.relu(x_symb @ w_symb)
    y_sync = hidet.ops.all_reduce(y_local, args.reduce_op, comm_id=int(use_group))
    graph = hidet.trace_from([y_local, y_sync], inputs=[x_symb, w_symb])
    opt_graph = hidet.graph.optimize(graph)
    opt_graph.set_dist_attrs(nrank=world_size, rank=rank, groups=groups)
    compiled = opt_graph.build()

    # Create Distributed Graph
    compiled.init_dist(shared_id)

    y_local, y_sync = compiled(x, w)

    s = hidet.cuda.current_stream() 
    s.synchronize()
    print(f"process {rank}\nbefore allreduce:{y_local}\nafter allreduce:{y_sync}\n", end='')

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