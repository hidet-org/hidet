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
from typing import Any, Sequence, Callable, Optional, Iterable
import multiprocessing
import os
import psutil


class JobQueue:
    def __init__(self, func, jobs: Sequence[Any] = tuple()):
        self.func: Callable = func
        self.jobs: Sequence[Any] = jobs


_job_queue: Optional[JobQueue] = None


def _wrapped_func(job_index):
    """
    Wrapper function for parallel_imap.

    We use this function to avoid pickling the jobs.
    """
    assert job_index < len(_job_queue.jobs)

    job = _job_queue.jobs[job_index]
    func = _job_queue.func

    return func(job)


def get_parallel_num_workers(max_num_workers: Optional[int] = None, mem_for_worker: Optional[int] = None):
    num_workers = (
        os.cpu_count() if (max_num_workers is None or max_num_workers == -1) else min(max_num_workers, os.cpu_count())
    )
    if mem_for_worker is not None:
        mem_for_worker *= 1024**3
        limit_by_memory = psutil.virtual_memory().available // mem_for_worker
        limit_by_memory = max(limit_by_memory, 1)
        num_workers = min(num_workers, limit_by_memory)
    return num_workers


def parallel_imap(
    func: Callable, jobs: Sequence[Any], max_num_workers: Optional[int] = None, mem_for_worker: Optional[int] = None
) -> Iterable[Any]:
    global _job_queue
    assert len(jobs) > 1

    if _job_queue is not None:
        raise RuntimeError('Cannot call parallel_map recursively.')

    _job_queue = JobQueue(func, jobs)

    num_workers = get_parallel_num_workers(max_num_workers, mem_for_worker)

    ctx = multiprocessing.get_context('fork')
    # Chunksize is taken from cpython/Lib/multiprocessing/pool.py::_map_async
    chunksize, extra = divmod(len(jobs), num_workers * 4)
    if extra:
        chunksize += 1

    with ctx.Pool(num_workers) as pool:
        yield from pool.imap(_wrapped_func, range(len(jobs)), chunksize=chunksize)

    _job_queue = None


def parallel_map(func: Callable, jobs: Sequence[Any], num_workers: Optional[int] = None) -> Iterable[Any]:
    global _job_queue

    if _job_queue is not None:
        raise RuntimeError('Cannot call parallel_map recursively.')

    _job_queue = JobQueue(func, jobs)

    if num_workers is None:
        num_workers = os.cpu_count()

    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(num_workers) as pool:
        ret = pool.map(_wrapped_func, range(len(jobs)))

    _job_queue = None
    return ret
