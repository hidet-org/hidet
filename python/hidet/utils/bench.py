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
from typing import Union, List
import time
import numpy as np


def benchmark_func(run_func, warmup=1, number=5, repeat=5, median=True) -> Union[List[float], float]:
    """Benchmark given function.

    The given function ``run_func`` will be executed :math:`warmup + repeat * number` times. Each :math:`number` times
    of execution will be grouped and conducted together.

    Parameters
    ----------
    run_func: Callable[[], Any]
        Any callable function to be benchmarked.

    warmup: int
        The number of warm-up executions.

    number: int
        The number of executions to be grouped for measurement.

    repeat: int
        The number of repeat times of the group measurement.

    median: bool
        Whether the median latency is returned, instead of the latency.

    Returns
    -------
    ret: Union[float, List[float]]
        - When median == True, a single latency number is returned.
        - When median == False, the latency of each repeat is returned, as a list of floats.
    """
    import nvtx
    import hidet.cuda

    results = []
    with nvtx.annotate('warmup'):
        for _ in range(warmup):
            run_func()
            hidet.cuda.synchronize()
    for i in range(repeat):
        with nvtx.annotate(f'repeat {i}'):
            hidet.cuda.synchronize()
            start_time = time.time()
            for _ in range(number):
                run_func()
            hidet.cuda.synchronize()
            end_time = time.time()
        results.append((end_time - start_time) * 1000 / number)
    if median:
        return float(np.median(results))
    else:
        return results
