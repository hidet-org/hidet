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
import pytest
import multiprocessing
from multiprocessing import Process
import os
import time
from datetime import timedelta
import random
import numpy

import hidet
import hidet.distributed

TMP_PATH = './tmp'

def test_all_reduce():
    if os.path.exists(TMP_PATH):
        os.remove(TMP_PATH)
    def foo(i):
        device = f'cuda:{i}'
        hidet.cuda.set_device(i)
        hidet.distributed.init_process_group(init_method=f'file://{TMP_PATH}', world_size=2, rank=i)
        hidet.distributed.set_nccl_comms()
        x = hidet.ones([4], device=device) * i
        y = hidet.ops.all_reduce(x, 'avg')
        assert x.shape == y.shape
        assert all(y.cpu().numpy() == 0.5)
    processes = [Process(target=foo, args=(i, )) for i in range(2)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

def test_all_gather():
    if os.path.exists(TMP_PATH):
        os.remove(TMP_PATH)
    def foo(i):
        device = f'cuda:{i}'
        hidet.cuda.set_device(i)
        hidet.distributed.init_process_group(init_method=f'file://{TMP_PATH}', world_size=2, rank=i)
        hidet.distributed.set_nccl_comms()
        x = hidet.ones([4], device=device) * i
        y = hidet.ops.all_gather(x, 2)
        assert numpy.array_equal(y.cpu().numpy(), [[0, 0, 0, 0],[1, 1, 1, 1]])
    processes = [Process(target=foo, args=(i, )) for i in range(2)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

if __name__ == '__main__':
    test_all_reduce()
    test_all_gather()