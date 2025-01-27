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
from concurrent.futures import ProcessPoolExecutor, as_completed
from hidet.option import compile_server, get_num_local_workers


def get_parallel_num_workers(is_remote_allowed: bool) -> int:
    if is_remote_allowed and compile_server.enabled():
        return compile_server.get_num_workers()

    num_workers = get_num_local_workers()
    assert num_workers > 0, 'Number of workers must be positive.'
    return num_workers


# During model compilation we can run building of different `Task` in parallel.
# Some `Task` are tunable and has a lot of candidates (implementations).
# Hidet compiles all these candidates and choose that is fastest.
# Compilation of every candidate is independent and all compilation can be done in parallel.
#
# Hidet's building/compilation has two parallelization levels(corresponds to description above):
# First Level (`parallel_imap_1stlevel`): Run in parallel different tasks building.
# Second Level (`parallel_imap_2ndlevel`): A nested parallelization level targeting to utilise possible
# parallelisation inside `Task` building (IR Generation, Fusion and Compilation itself).
# The reason why we have two levels is that we have to control how many processes are spawned in the second level
# to prevent overload of the system.


# 1ST LEVEV PARALLELISATION IMPLEMENTATION
_global_func = None

# We need it to be able to process local functions.
# Parallel function should be pickable but local functions are not pickable.
def _wrapper(*args, **kwargs):
    return _global_func(*args, **kwargs)


def parallel_imap_1stlevel(func: Callable, jobs: Sequence[Any], is_remote_allowed: bool = False) -> Iterable[Any]:
    jobs_num = len(jobs)
    assert jobs_num > 0

    num_workers = get_parallel_num_workers(is_remote_allowed)
    num_workers = min(num_workers, jobs_num)

    # num_workers == 1 or len(jobs) == 1
    if num_workers == 1:
        for job in jobs:
            yield func(job)
        return

    global _global_func
    if _global_func is not None:
        raise RuntimeError('Cannot call parallel_imap_1stlevel recursively.')
    _global_func = func

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        submited_jobs = [executor.submit(_wrapper, job) for job in jobs]
        for complited_job in as_completed(submited_jobs):
            yield complited_job.result()

    _global_func = None


# 2ND LEVEV PARALLELISATION IMPLEMENTATION
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


semaphore_local_compilation = multiprocessing.Semaphore(1)
semaphore_remote_compilation = multiprocessing.Semaphore(1)


def parallel_imap_2ndlevel(func: Callable, jobs: Sequence[Any], is_remote_allowed: bool = False) -> Iterable[Any]:
    if is_remote_allowed and compile_server.enabled():
        semaphore = semaphore_remote_compilation
    else:
        semaphore = semaphore_local_compilation

    with semaphore:
        jobs_num = len(jobs)
        assert jobs_num > 0

        num_workers = get_parallel_num_workers(is_remote_allowed)
        num_workers = min(num_workers, jobs_num)

        # num_workers == 1 or len(jobs) == 1
        if num_workers == 1:
            for job in jobs:
                yield func(job)
            return

        global _job_queue

        if _job_queue is not None:
            raise RuntimeError('Cannot call parallel_imap recursively.')

        _job_queue = JobQueue(func, jobs)

        ctx = multiprocessing.get_context('fork')
        # Chunksize is taken from cpython/Lib/multiprocessing/pool.py::_map_async
        chunksize, extra = divmod(len(jobs), num_workers * 4)
        if extra:
            chunksize += 1

        with ctx.Pool(num_workers) as pool:
            yield from pool.imap(_wrapped_func, range(len(jobs)), chunksize=chunksize)

        _job_queue = None
