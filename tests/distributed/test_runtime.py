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

from utils import distributed_test

TMP_PATH = './tmp'

# This testing script assumes we have two GPUs.
WORLD_SIZE = 2


@distributed_test(world_size=WORLD_SIZE)
def test_all_reduce(rank):
    x = hidet.ones([4], device='cuda') * rank
    hidet.distributed.all_reduce(x, 'avg')
    hidet.cuda.synchronize()
    assert all(x.cpu().numpy() == 0.5)


@distributed_test(world_size=WORLD_SIZE)
def test_broadcast(rank):
    x = hidet.ones([4], device='cuda') * rank
    hidet.distributed.broadcast(x, 1)
    hidet.cuda.synchronize()
    assert numpy.array_equal(x.cpu().numpy(), [1, 1, 1, 1])


@distributed_test(world_size=WORLD_SIZE)
def test_reduce(rank):
    x = hidet.ones([4], device='cuda') * rank
    hidet.distributed.reduce(x, 1, 'avg')
    hidet.cuda.synchronize()
    if rank == 0:
        assert all(x.cpu().numpy() == 0)
    elif rank == 1:
        assert all(x.cpu().numpy() == 0.5)


@distributed_test(world_size=WORLD_SIZE)
def test_all_gather(rank):
    if rank == 0:
        x = hidet.ones([3], device='cuda', dtype=hidet.float32)
    elif rank == 1:
        x = hidet.ones([4], device='cuda', dtype=hidet.float32) * 2
    recv = [hidet.empty([3], device='cuda', dtype=hidet.float32), hidet.empty([4], device='cuda', dtype=hidet.float32)]
    hidet.distributed.all_gather(recv, x)
    hidet.cuda.synchronize()
    assert numpy.array_equal(recv[0].cpu().numpy(), [1, 1, 1])
    assert numpy.array_equal(recv[1].cpu().numpy(), [2, 2, 2, 2])


@distributed_test(world_size=WORLD_SIZE)
def test_all_gather_into_tensor(rank):
    x = hidet.ones([4], device='cuda') * rank
    y = hidet.empty([2, 4], device='cuda')
    hidet.distributed.all_gather_into_tensor(y, x)
    hidet.cuda.synchronize()
    assert numpy.array_equal(y.cpu().numpy(), [[1, 1, 1, 1], [0, 0, 0, 0]])


@distributed_test(world_size=WORLD_SIZE)
def test_gather(rank):
    if rank == 0:
        x = hidet.ones([3], device='cuda', dtype=hidet.float32)
        recv = [
            hidet.empty([3], device='cuda', dtype=hidet.float32),
            hidet.empty([4], device='cuda', dtype=hidet.float32),
        ]
        hidet.distributed.gather(x, recv, 0)
        hidet.cuda.synchronize()
        assert numpy.array_equal(recv[0].cpu().numpy(), [1, 1, 1])
        assert numpy.array_equal(recv[1].cpu().numpy(), [2, 2, 2, 2])
    elif rank == 1:
        x = hidet.ones([4], device='cuda', dtype=hidet.float32) * 2
        hidet.distributed.gather(x, None, 0)


@distributed_test(world_size=WORLD_SIZE)
def test_scatter(rank):
    if rank == 0:
        x = hidet.ones([3], device='cuda', dtype=hidet.float32)
        y = hidet.ones([4], device='cuda', dtype=hidet.float32) * 2

        recv = hidet.empty([3], device='cuda', dtype=hidet.float32)
        hidet.distributed.scatter(recv, [x, y], 0)
        hidet.cuda.synchronize()
        assert numpy.array_equal(recv.cpu().numpy(), [1, 1, 1])
    elif rank == 1:
        recv = hidet.empty([4], device='cuda', dtype=hidet.float32)
        hidet.distributed.scatter(recv, None, 0)
        hidet.cuda.synchronize()
        assert numpy.array_equal(recv.cpu().numpy(), [2, 2, 2, 2])


@distributed_test(world_size=WORLD_SIZE)
def test_reduce_scatter_tensor(rank):
    if rank == 0:
        x = hidet.asarray([[1, 2], [3, 4]], device='cuda', dtype=hidet.float32)
    elif rank == 1:
        x = hidet.asarray([[5, 6], [7, 8]], device='cuda', dtype=hidet.float32)
    y = hidet.empty([2], device='cuda')
    hidet.distributed.reduce_scatter_tensor(y, x, 'sum')
    hidet.cuda.synchronize()
    if rank == 0:
        assert numpy.array_equal(y.cpu().numpy(), [6, 8])
    elif rank == 1:
        assert numpy.array_equal(y.cpu().numpy(), [10, 12])


@distributed_test(world_size=WORLD_SIZE)
def test_barrier(rank):
    hidet.distributed.barrier()
    hidet.cuda.synchronize()


@distributed_test(world_size=WORLD_SIZE)
def test_send_recv(rank):
    if rank == 0:
        x = hidet.asarray([[1, 2], [3, 4]], device='cuda', dtype=hidet.float32)
        hidet.distributed.send(x, 1)
        hidet.cuda.synchronize()
    elif rank == 1:
        x = hidet.empty([2, 2], device='cuda', dtype=hidet.float32)
        hidet.distributed.recv(x, 0)
        hidet.cuda.synchronize()
        assert numpy.array_equal(x.cpu().numpy(), [[1, 2], [3, 4]])


if __name__ == '__main__':
    # test_all_reduce()
    # test_broadcast()
    # test_reduce()
    # test_all_gather()
    # test_gather()
    test_scatter()
    # test_all_gather_into_tensor()
    # test_reduce_scatter_tensor()
    # test_barrier()
    test_send_recv()
