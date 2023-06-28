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
from typing import List, Optional, Callable, Tuple, Any, Dict, Union
import time
from dataclasses import dataclass

import numpy as np


# copied from: https://github.com/openai/triton/blob/main/python/triton/testing.py
def do_bench(fn, warmup=25, rep=100, percentiles=(0.2, 0.5, 0.8)):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param percentiles: Performance percentile to return in addition to the median.
    :type percentiles: list[float]
    """

    # Estimate the runtime of the function
    import torch
    fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5
    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]

    cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)])
    if percentiles:
        percentiles = torch.quantile(times, torch.tensor(percentiles)).tolist()
        return tuple(percentiles)
    else:
        return torch.mean(times).item()


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


@dataclass
class BenchData:
    x_vals: List[int]
    x_name: str
    y_name: str
    config: Any
    data: Dict[str, Tuple[List[float], List[float], List[float]]]  # [t_min, t_avg, t_max]

    def show_plot(self, show=True, save_path=None):
        from matplotlib import pyplot as plt

        plt.figure()
        ax = plt.subplot()
        for name, (t_min, t_avg, t_max) in self.data.items():
            p = ax.plot(self.x_vals, t_avg, label=name)
            color = p[0].get_color()
            ax.fill_between(self.x_vals, t_min, t_max, alpha=0.15, color=color)
        ax.legend()
        ax.set_xlabel(self.x_name)
        ax.set_ylabel(self.y_name)
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)
        return self

    def to_dataframe(self):
        import pandas as pd

        columns = list(self.data.keys())
        df = pd.DataFrame(columns=columns, index=self.x_vals)
        for n in columns:
            df[n] = self.data[n][1]  # get t_avg
        return df

    def print_data(self):
        print(self.to_dataframe())


class Bench:
    def __init__(self, x_vals: List[int], x_name: str, config: Any = None):
        self.x_vals = x_vals
        self.x_name = x_name
        self.y_name = 'ms'
        self.byte_fn = None

        self.config = config
        self.bench_fns: List[Tuple[str, Callable]] = []
        self.bench_data: Dict[str, Tuple[List[float], List[float], List[float]]] = {}

    def measure_flops(self, byte_fn: Callable[[Any, int], int]):
        """
        set a function that takes in the config, and the current x_val and returns the number of bytes
        """
        self.byte_fn = byte_fn
        self.y_name = 'flops'

    def bench(self, fn: Callable[[Any, int], Callable[[], None]], name: Optional[str] = None):
        """
        add a function that takes in the config and int and returns a function to be benchmarked
        to the list of functions to be benchmarked.
        If the name argument is None, the the name for this particular line is fn.__name__
        """
        if name is None:
            if hasattr(fn, '__name__'):
                name = fn.__name__
            else:
                raise ValueError("cannot get name of function")
        self.bench_fns.append((name, fn))
        return self

    def run(self):
        """
        run all the functions that needs to be benchmarked, returning BenchData representing
        the collected results
        """
        for i in self.x_vals:
            for name, fn in self.bench_fns:

                if name not in self.bench_data:
                    self.bench_data[name] = ([], [], [])
                t_min, t_avg, t_max = self.bench_data[name]

                bench_fn = fn(self.config, i)
                lo, avg, hi = do_bench(bench_fn)
                if self.byte_fn is not None:
                    lo = self.byte_fn(self.config, i) * 1e-9 / (lo * 1e-3)
                    avg = self.byte_fn(self.config, i) * 1e-9 / (avg * 1e-3)
                    hi = self.byte_fn(self.config, i) * 1e-9 / (hi * 1e-3)
                t_min.append(lo)
                t_avg.append(avg)
                t_max.append(hi)
        return BenchData(self.x_vals, self.x_name, self.y_name, self.config, self.bench_data)
