import pytest
import multiprocessing
from multiprocessing import Process
import os

import hidet
import hidet.distributed

TMP_PATH = './tmp'
def distributed_test(world_size):
    def decorator(func):
        def _f():
            if os.path.exists(TMP_PATH):
                os.remove(TMP_PATH)

            def proc(i):
                hidet.cuda.set_device(i)
                hidet.distributed.init_process_group(init_method=f'file://{TMP_PATH}', world_size=world_size, rank=i)
                hidet.distributed.set_nccl_comms()
                func(i)

            processes = [Process(target=proc, args=(i,)) for i in range(world_size)]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
        return _f
    return decorator