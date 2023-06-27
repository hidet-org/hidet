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
import atexit

import hidet
import hidet.cuda.nccl

parser = argparse.ArgumentParser()
parser.add_argument("n_gpus", type=int)
parser.add_argument("reduce_op", choices=['sum', 'prod', 'max', 'min', 'avg'])
args = parser.parse_args()

def run(world_size, rank):
    numpy.random.seed(rank)

    hidet.cuda.set_device(rank)
    hidet.distributed.init_process_group(init_method='file://tmp', world_size=world_size, rank=rank)
    hidet.distributed.set_nccl_comms()

    device = f"cuda:{rank}"
    x = hidet.randn([1, 3], device=device)
    w = hidet.randn([3, 2], device=device)
    
    # Create Computation Graph
    x_symb = hidet.symbol_like(x)
    w_symb = hidet.symbol_like(w)
    y_local = hidet.ops.relu(x_symb @ w_symb)
    y_sync = hidet.ops.all_reduce(y_local, args.reduce_op)
    graph = hidet.trace_from([y_local, y_sync], inputs=[x_symb, w_symb])
    opt_graph = hidet.graph.optimize(graph)
    compiled = opt_graph.build()
    y_local, y_sync = compiled(x, w)

    hidet.cuda.current_stream().synchronize()
    print(f"process {rank}\nbefore allreduce:{y_local}\nafter allreduce:{y_sync}\n", end='')
    atexit._run_exitfuncs()

world_size = args.n_gpus
processes = [Process(target=run, args=(world_size, i)) for i in range(world_size)]

for p in processes:
    p.start()
for p in processes:
    p.join()