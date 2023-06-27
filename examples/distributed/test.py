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

    s = hidet.cuda.current_stream() 
    s.synchronize()
    print(f"process {rank}")

world_size = args.n_gpus
processes = [Process(target=run, args=(world_size, i)) for i in range(world_size)]

for p in processes:
    p.start()
for p in processes:
    p.join()