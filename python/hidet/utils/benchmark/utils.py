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
import time
from typing import List
import numpy as np
import hidet
import hidet.cuda
import hidet.lang


def create_event():
    cuda_available = hidet.cuda.available()
    hip_available = hidet.hip.available()

    if not cuda_available and not hip_available:
        raise RuntimeError("No GPU found")

    if cuda_available:
        return hidet.cuda.Event(enable_timing=True)
    else:
        return hidet.hip.Event(enable_timing=True)


def sync():
    cuda_available = hidet.cuda.available()
    hip_available = hidet.hip.available()

    if not cuda_available and not hip_available:
        raise RuntimeError("No GPU found")

    if cuda_available:
        hidet.cuda.synchronize()
    else:
        hidet.hip.synchronize()


def get_event_time_accuracy_ms():
    return 1e-3


def _device_sleep():
    if hidet.cuda.available():
        from hidet.lang import attrs, u32
        from hidet.lang.cuda import syncwarp, nano_sleep

        with hidet.lang.script_module() as script_module:

            @hidet.lang.script
            def _sleep(cycle: u32):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = 32
                attrs.cuda.grid_dim = 1
                syncwarp()
                nano_sleep(cycle)

            @hidet.lang.script
            def launch(cycle: u32):
                attrs.func_kind = 'public'
                _sleep(cycle)

        module = script_module.build()
        return module.functions['launch']
    else:
        return lambda x: time.sleep(x * 1e-9)


# Callable kernel that sleep on device `time_ns` nanoseconds: `device_sleep(time_ns)`
device_sleep = _device_sleep()


def _empty_kernel():
    if hidet.cuda.available():
        from hidet.lang import attrs

        with hidet.lang.script_module() as script_module:

            @hidet.lang.script
            def _empty_kernel():
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = 1
                attrs.cuda.grid_dim = 1

            @hidet.lang.script
            def launch():
                attrs.func_kind = 'public'
                _empty_kernel()

        module = script_module.build()
        return module.functions['launch']
    else:
        return lambda: None


# Callable empty kernel
empty_kernel = _empty_kernel()


_cached_cuda_event_duration = None


# Get the duration between 2 event calls
def get_cuda_event_duration():
    global _cached_cuda_event_duration
    if _cached_cuda_event_duration is not None:
        return _cached_cuda_event_duration

    warmup = 5
    repeat = 51
    tmp_start_events = [create_event() for _ in range(warmup)]
    tmp_end_events = [create_event() for _ in range(warmup)]

    start_events = [create_event() for _ in range(repeat)]
    end_events = [create_event() for _ in range(repeat)]

    # Sometimes, especcially on the first run, the sleep duration is not accurate.
    # Repeat the sleep a few times as a workaround.
    for _ in range(3):
        device_sleep(10**6)

    for i in range(warmup):
        tmp_start_events[i].record()
        tmp_end_events[i].record()
    for i in range(repeat):
        start_events[i].record()
        end_events[i].record()

    sync()

    times = [e.elapsed_time(s) for s, e in zip(start_events, end_events)]
    _cached_cuda_event_duration = np.median(times)
    return _cached_cuda_event_duration


_cached_empty_kernel_times = {}


# Get the CPU time requered for calling `number` repeats of empty kernel
# In fact this is a estimation of CPU overhead in `_benchmark_func_internal`
def get_empty_kernel_cpu_time_ns(number: int) -> int:
    global _cached_empty_kernel_times  # pylint: disable=global-variable-not-assigned
    if number in _cached_empty_kernel_times:
        return _cached_empty_kernel_times[number]

    repeat = 51
    delay = 1
    times = []

    start_events = [create_event() for _ in range(repeat)]
    end_events = [create_event() for _ in range(repeat)]

    for i in range(repeat):
        start_time = time.time_ns()
        device_sleep(delay)
        start_events[i].record()
        for _ in range(number):
            empty_kernel()
        end_events[i].record()
        sync()
        end_time = time.time_ns()
        times.append((end_time - start_time))

    _cached_empty_kernel_times[number] = int(np.median(times))
    return _cached_empty_kernel_times[number]


def _benchmark_func_internal(run_func, *args, repeat, number, delay) -> List[float]:
    delay = int(delay)
    start_events = [hidet.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [hidet.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        device_sleep(delay)
        start_events[i].record()
        for _ in range(number):
            run_func(*args)
        end_events[i].record()

    hidet.cuda.synchronize()
    times = [e.elapsed_time(s) / number for s, e in zip(start_events, end_events)]
    return times
