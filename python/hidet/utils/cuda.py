import os
import subprocess
from functools import lru_cache
from subprocess import PIPE
from typing import List, Optional, Union

import pycuda.autoinit
import pycuda.driver


def max_smem_bytes_per_sm(cc=None):
    legacy = True
    if legacy:
        return 48 * 1024
    else:
        if cc is None:
            cc = query_compute_capability()
        data = {
            (6, 0): 64,
            (6, 1): 96,
            (6, 2): 64,
            (7, 0): 96,
            (7, 2): 96,
            (7, 5): 64,
            (8, 0): 164,
            (8, 6): 100,
            (8, 7): 164
        }
        return data[cc] * 1024


def max_smem_bytes_per_block(cc=None):
    legacy = True
    if legacy:
        return 48 * 1024
    else:
        if cc is None:
            cc = query_compute_capability()
        data = {
            (6, 0): 48,
            (6, 1): 48,
            (6, 2): 48,
            (7, 0): 96,
            (7, 2): 96,
            (7, 5): 64,
            (8, 0): 163,
            (8, 6): 99,
            (8, 7): 163
        }
        return data[cc] * 1024


def max_num_regs_per_thread():
    return 255


def max_num_regs_per_block(cc=None):
    if cc is None:
        cc = query_compute_capability()
    data = {
        (6, 0): 64,
        (6, 1): 64,
        (6, 2): 32,
        (7, 0): 64,
        (7, 2): 64,
        (7, 5): 64,
        (8, 0): 64,
        (8, 6): 64
    }
    return data[cc] * 1024


def max_num_regs_per_sm(cc=None):
    return 64 * 1024


@lru_cache(maxsize=128)
def query_compute_capability():
    major, minor = query_gpu('compute_cap').split('.')
    return int(major), int(minor)


def device_synchronize():
    pycuda.driver.Context.synchronize()


def preferred_gpu_clock():
    base_clocks = {
        'NVIDIA GeForce RTX 3070 Laptop GPU': 1560,
        'Tesla V100-SXM2-16GB': 1312,
        'Tesla T4': 585,
    }
    name = query_gpu('gpu_name')
    if name not in base_clocks:
        return base_clocks[name]
    else:
        print('running on a new device: {}'.format(name))
        print('please set the base clock at {}:preferred_gpu_clock()', __file__)
        return int(query_gpu_max_clock() * 0.8)


def lock_gpu_clock(clock: Optional[int] = None):
    if clock is None:
        clock = preferred_gpu_clock()
    command = f'sudo -S nvidia-smi --lock-gpu-clocks={clock}'
    print(f"Running '{command}'...")
    subprocess.run(command.split(), check=True)


def reset_gpu_clock():
    command = 'sudo -S nvidia-smi --reset-gpu-clocks'
    print(f"Running '{command}'")
    subprocess.run(command.split(), check=True)


def query_gpu_current_clock() -> int:
    return int(query_gpu('clocks.current.graphics'))


def query_gpu_temperature() -> int:
    return int(query_gpu('temperature.gpu'))


def query_name() -> str:
    return query_gpu('name')


def query_arch() -> str:
    arch2name = {
        (2, 0): 'Fermi',
        (3, 0): 'Kepler',
        (3, 5): 'Kepler',
        (3, 7): 'Kepler',
        (5, 0): 'Maxwell',
        (5, 2): 'Maxwell',
        (5, 3): 'Maxwell',
        (6, 0): 'Pascal',
        (6, 1): 'Pascal',
        (6, 2): 'Pascal',
        (7, 0): 'Volta',
        (7, 2): 'Volta',
        (7, 5): 'Turing',
        (8, 0): 'Ampere',
        (8, 6): 'Ampere'
    }
    return arch2name[query_compute_capability()]


def query_clocks_throttle_reason() -> str:
    # see 'nvidia-smi --help' and 'nvml.h' for more information
    bitmask = int(query_gpu('clocks_throttle_reasons.active'), base=16)
    bit2reason = {
        1: 'gpu_idle',
        2: 'app_clock_setting',
        4: 'sw_power_cap',
        8: 'hw_slowdown',
        16: 'sync_boost',
        32: 'sw_thermal_slowdown',
        64: 'hw_thermal_slowdown',
        128: 'hw_power_brake_slowdown',
        256: 'display_clock_setting',
    }
    if bitmask == 0:
        return 'no'
    else:
        reasons = []
        for bit, reason in bit2reason.items():
            if (bitmask & bit) != 0:
                reasons.append(reason)
        if len(reasons) == 0:
            raise NotImplementedError()
        return "/".join(reasons)


def query_gpu(names: Union[List[str], str]):
    if not isinstance(names, (list, tuple)):
        names = [names]
    result = subprocess.run(f'nvidia-smi -i 0 --query-gpu={",".join(names)} --format=csv,noheader,nounits'.split(),
                            stdin=PIPE, stdout=PIPE, check=True)
    results = [s.strip() for s in result.stdout.decode('utf-8').split(',')]
    if len(results) == 1:
        return results[0]
    else:
        return results


def query_gpu_max_clock() -> int:
    return int(query_gpu('clocks.max.sm'))


def lock_memory_clock(clock: int):
    command = f'sudo -S nvidia-smi --lock-memory-clocks={clock}'
    print(f"Running '{command}'...")
    subprocess.run(command.split(), check=True)


def reset_memory_clock():
    command = 'sudo -S nvidia-smi --reset-memory-clocks'
    print(f"Running '{command}'")
    subprocess.run(command.split(), check=True)


def query_memory_current_clock() -> int:
    return int(query_gpu('clocks.current.memory'))


def query_memory_max_clock() -> int:
    return int(query_gpu('clocks.max.memory'))


def query_persistent_mode() -> bool:
    result = subprocess.run('nvidia-smi -pm 1'.split(), stdin=PIPE, stdout=PIPE)
    return result.returncode == 0


def turn_on_persistent_mode():
    result = subprocess.run('nvidia-smi -pm 1'.split(), stdin=PIPE, stdout=PIPE)
    if result.returncode != 0:
        # the persistent mode is disabled, use sudo -S to turn it on, passwd is required from shell
        command = 'sudo -S nvidia-smi -pm 1'
        print(f"Running '{command}' to turn on persistent mode...")
        subprocess.run(command.split(), check=True)


class BenchmarkContext:
    def __init__(self, fix_clock=True):
        self.fix_clock = fix_clock

    def __enter__(self):
        if self.fix_clock:
            # sm clock; to make result more stable (trying to avoid gpu throttle)
            turn_on_persistent_mode()
            lock_gpu_clock()
            lock_memory_clock(query_memory_max_clock())

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fix_clock:
            reset_memory_clock()
            reset_gpu_clock()

    @staticmethod
    def get_bench_ratio(clock_ratio=None):
        ratio = os.environ.get('HIDET_BENCH_RATIO')
        if ratio:
            ratio = float(ratio)
        elif clock_ratio is not None:
            ratio = clock_ratio
        else:
            default_clock_ratio = 1.0
            ratio = default_clock_ratio
        return min(max(float(ratio), 0.1), 1.0)


if __name__ == '__main__':
    print(query_compute_capability())
