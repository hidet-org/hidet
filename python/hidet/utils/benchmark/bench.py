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
from dataclasses import dataclass
from scipy import stats
import numpy as np
from tqdm import tqdm
import hidet
import hidet.cuda
from hidet.utils import green, gc_disabled
from hidet.option import is_fix_gpu_frequency_for_tuning
from .gpu_freq import GPUSetFrequencyForBenchmarking
from .utils import create_event, sync, get_empty_kernel_cpu_time_ns, _benchmark_func_internal
from .utils import get_event_time_accuracy_ms, get_cuda_event_duration

# Number of repetitions between time measurements for benchmarking
DEFAULT_NUMBER_FOR_MEASUREMENTS = 5


# copied from: https://github.com/openai/triton/blob/main/python/triton/testing.py
def _do_bench(fn, warmup, rep, percentiles):
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

    cuda_available = hidet.cuda.available()
    hip_available = hidet.hip.available()

    if not cuda_available and not hip_available:
        raise RuntimeError("No GPU found")

    fn()
    sync()
    start_event = create_event()
    end_event = create_event()
    start_event.record()
    for _ in range(5):
        fn()
    end_event.record()
    sync()
    estimate_ms = end_event.elapsed_time(start_event) / 5
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    start_event = [create_event() for i in range(n_repeat)]
    end_event = [create_event() for i in range(n_repeat)]

    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    sync()
    times = np.array([e.elapsed_time(s) for s, e in zip(start_event, end_event)])
    if percentiles:
        percentiles = np.quantile(times, percentiles)
        return tuple(percentiles)
    else:
        return np.mean(times).item()


def do_bench(fn, warmup=25, rep=100, percentiles=(0.2, 0.5, 0.8)):
    with gc_disabled():
        return _do_bench(fn, warmup, rep, percentiles)


# Find the `number` of iterations and `delay` for best accuracy of benchmarking
# of the given function with given arguments.
# `number` is the number of iterations to be grouped for measurement.
#          Number of function calls between time measurements with events.
# `delay`  is the delay in the beginning of measurement to allow a big number of API calls
#          to be asynchronously sent to the device. Delay is required to avoid "measuring" CPU overheads.
def find_number_and_delay_4_benchmark(run_func, *args):
    # Make the first time estimation.
    # Use `number=1`.
    # Use delay equal to the "CPU time of empty kernel" multiplied by 2.
    # This delay must be enough because empty kernel bench and `fn` bench do exactly the same API calls (by design).
    empty_kernel_cpu_time_ns = get_empty_kernel_cpu_time_ns(number=1)
    delay = empty_kernel_cpu_time_ns * 2
    times = _benchmark_func_internal(run_func, *args, repeat=5, number=1, delay=delay)
    time_ms = np.min(times)
    # Drop the overhead of event calls
    time_ms = time_ms - get_cuda_event_duration()
    time_ms = max(1e-3, time_ms)

    # The first estimation of time is good if the function is not small.
    # If the function is small, the first estimation is not accurate.
    # Timer resolution (`get_event_time_accuracy_ms()`) typically is 1us or 0.5us.
    # If function execution time is, for example, 5us, the inaccuracy is too high.
    # Have to increase the number of iterations to get more accurate time.
    # Initial number of iterations is 100 multiplied by the timer resolution.
    number = 100 * get_event_time_accuracy_ms() / time_ms
    number = max(1, int(np.ceil(number)))

    # The next loop is solving the following problem:
    # Delay can't be bigger than 1ms (this is a CUDA limitation).
    # If with the current `number` CPU time for empty kernel is too big (more than 750us),
    # we decrease the `number`.
    while True:
        empty_kernel_cpu_time_ns = get_empty_kernel_cpu_time_ns(number=number)
        if number <= 1:
            break
        if empty_kernel_cpu_time_ns < 750000:
            break
        number = int(number * 0.75)

    # Calculate the delay for the new `number`. Delay can't be bigger than 1ms (CUDA limitation).
    delay = empty_kernel_cpu_time_ns * 2 - 10**6 * time_ms * number
    delay = max(0, int(delay))
    delay = min(1000000, delay)

    return number, delay


def _benchmark_func(run_func, *args, warmup, number, repeat, median) -> Union[List[float], float]:
    """Benchmark given function.

    The given function ``run_func`` will be executed :math:`warmup + repeat * number` times. Each :math:`number` times
    of execution will be grouped and conducted together.

    Parameters
    ----------
    run_func: Callable[[], Any]
        Any callable function to be benchmarked.

    warmup: int
        The number of warm-up executions.
        Default `warmup=3` is good choose. In most cases 3 iterations for warmup is enough.

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

    # The first step.
    # Find number and delay those provide the best accuracy for benchmarking
    if number is None:
        if hidet.cuda.available():
            # This inerface is available only for CUDA
            number, delay = find_number_and_delay_4_benchmark(run_func, *args)
        else:
            number = DEFAULT_NUMBER_FOR_MEASUREMENTS
            delay = 0
    else:
        delay = 0

    # Have warmed up in `find_number_and_delay_4_benchmark`
    warmup = 0

    times = _benchmark_func_internal(run_func, *args, repeat=warmup + repeat, number=number, delay=delay)
    times = times[warmup:]

    if median:
        return float(np.median(times))
    else:
        return times


def benchmark_func(
    run_func, *args, warmup=3, number=DEFAULT_NUMBER_FOR_MEASUREMENTS, repeat=5, median=True
) -> Union[List[float], float]:
    with gc_disabled():
        return _benchmark_func(run_func, *args, warmup=warmup, number=number, repeat=repeat, median=median)


@dataclass
class CandidateData:
    idx: int
    latencies: List[float] = None
    median: float = 0.0
    in_game: bool = True


def _find_best_candidate(candidates: List[Callable[..., None]], pbar, *args):
    P_VALUE_THRESHOLD = 0.01
    num_candidates = len(candidates)
    candidates_data = [CandidateData(idx=idx) for idx, _ in enumerate(candidates)]
    repeats = (7, 31)
    for cur_repeat in repeats:
        for idx, cand in enumerate(candidates):
            if candidates_data[idx].in_game:
                lats = benchmark_func(cand, *args, warmup=3, number=None, repeat=cur_repeat, median=False)
                candidates_data[idx].latencies = lats
                pbar.update(1)

        for cand in candidates_data:
            if cand.in_game:
                cand.median = np.median(cand.latencies)

        # We have samples for every candidate.
        # Start with candidate with minimum median. Likely it'll drop a lot of slower candidates.
        # Just optimisation. The next loop is enough for functionality
        min_lat_cand = min((cand for cand in candidates_data if cand.in_game), key=lambda cand: cand.median)
        min_idx = min_lat_cand.idx
        for i in range(num_candidates):
            if i == min_idx or not candidates_data[i].in_game:
                continue
            _, p_value = stats.ttest_ind(
                candidates_data[min_idx].latencies, candidates_data[i].latencies, alternative='less'
            )
            if p_value < P_VALUE_THRESHOLD:
                candidates_data[i].in_game = False
        # If left only one candidate - good we found the best
        left_candidates = [cand for cand in candidates_data if cand.in_game]

        if len(left_candidates) == 1:
            return (left_candidates[0].idx, [cand.median for cand in candidates_data])

        # Compare all candidates betwee each other. Comparison use T-test
        for i in range(num_candidates):
            if not candidates_data[i].in_game:
                continue
            for j in range(num_candidates):
                if not candidates_data[j].in_game or i == j:
                    continue
                _, p_value = stats.ttest_ind(
                    candidates_data[i].latencies, candidates_data[j].latencies, alternative='less'
                )
                if p_value < P_VALUE_THRESHOLD:
                    candidates[j].in_game = False

        # If left only one candidate - good we found the best
        left_candidates = [cand for cand in candidates_data if cand.in_game]
        if len(left_candidates) == 1:
            return (left_candidates[0].idx, [cand.median for cand in candidates_data])

    # Can not prove that one candidate statistically significant than all other.
    # There are several but we can not order them using above method.
    # Should choose some candidate. Choose one with minimal median
    best = min((cand for cand in candidates_data if cand.in_game), key=lambda cand: cand.median)
    best_idx = best.idx
    latensies = [cand.median for cand in candidates_data]
    return (best_idx, latensies)


def find_best_candidate(candidates: List[Callable[..., None]], name, *args):
    desc = "Finding the best candidates for " + green(name)
    for i in args:
        desc += f" {tuple(i.shape)}"
    if is_fix_gpu_frequency_for_tuning():
        with GPUSetFrequencyForBenchmarking():
            with gc_disabled(), tqdm(desc=desc, ncols=80) as pbar:
                return _find_best_candidate(candidates, pbar, *args)
    else:
        with gc_disabled(), tqdm(desc=desc, ncols=80) as pbar:
            return _find_best_candidate(candidates, pbar, *args)


@dataclass
class BenchData:
    x_vals: List[Any]
    x_name: str
    y_name: str
    kwargs: Dict[str, Any]
    data: Dict[str, Tuple[List[float], List[float], List[float]]]  # [t_min, t_avg, t_max]

    def show_plot(self, show=True, save_path=None, figsize=None, title=None):
        from matplotlib import pyplot as plt

        if all(isinstance(x, (float, int)) for x in self.x_vals):
            x_vals = self.x_vals
        else:
            x_vals = range(1, len(self.x_vals) + 1)

        plt.figure(figsize=figsize)
        ax = plt.subplot()
        for name, (t_min, t_avg, t_max) in self.data.items():
            p = ax.plot(x_vals, t_avg, label=name)
            color = p[0].get_color()
            ax.fill_between(x_vals, t_min, t_max, alpha=0.15, color=color)
        ax.legend()
        ax.set_xlabel(self.x_name)
        ax.set_ylabel(self.y_name)
        if title is not None:
            ax.set_title(title)
        ax.set_xticks(ticks=x_vals, labels=[str(x) for x in self.x_vals])
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
    def __init__(self, x_vals: List[Any], x_name: str, **kwargs):
        self.x_vals = x_vals
        self.x_name = x_name
        self.y_name = 'ms'
        self.byte_fn = None

        self.kwargs: Dict[str, Any] = kwargs
        self.bench_fns: List[Tuple[str, Callable]] = []
        self.bench_data: Dict[str, Tuple[List[float], List[float], List[float]]] = {}

    def measure_flops(self, byte_fn: Callable[[Any], int]):
        """
        set a function that takes in the config, and the current x_val and returns the number of bytes
        """
        self.byte_fn = byte_fn
        self.y_name = 'TFLOP/s'

    def bench(self, fn: Callable[[Any], Callable[[], Any]], name: Optional[str] = None):
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

                bench_fn = fn(i, **self.kwargs)
                lo, avg, hi = do_bench(bench_fn)
                if self.byte_fn is not None:
                    lo = self.byte_fn(i, **self.kwargs) * 1e-12 / (lo * 1e-3)
                    avg = self.byte_fn(i, **self.kwargs) * 1e-12 / (avg * 1e-3)
                    hi = self.byte_fn(i, **self.kwargs) * 1e-12 / (hi * 1e-3)
                t_min.append(lo)
                t_avg.append(avg)
                t_max.append(hi)
        return BenchData(self.x_vals, self.x_name, self.y_name, self.kwargs, self.bench_data)
