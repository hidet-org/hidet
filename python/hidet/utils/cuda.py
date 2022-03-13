import os
from functools import lru_cache
import sys
import re
import pycuda.driver
import pycuda.autoinit
import subprocess
from subprocess import PIPE


class Attr:
    COUNT = 'count'
    NAME = 'name'
    TOTAL_MEMORY = 'total_memory'
    COMPUTE_CAPACITY = 'compute_capacity'
    ARCH_NAME = 'arch_name'


_arch2name = {
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


def max_smem_bytes_per_sm(cc=None):
    if cc is None:
        cc = get_compute_capability()
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
    if cc is None:
        cc = get_compute_capability()
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


def max_num_regs_per_thread(cc=None):
    return 255


def max_num_regs_per_block(cc=None):
    if cc is None:
        cc = get_compute_capability()
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
def get_compute_capability():
    # use this function in sub-progress
    python_executable = sys.executable
    result = subprocess.run([python_executable, '-c', 'import pycuda.driver; pycuda.driver.init(); print(pycuda.driver.Device(0).compute_capability());'],
                            stdin=PIPE, stdout=PIPE, check=True)
    pair = result.stdout.decode('utf-8')
    # pair is a string like '(8, 6)'
    major, minor = re.search(r'\((\d*), (\d*)\)', pair).group(1, 2)
    return int(major), int(minor)


def get_attribute(attr_name: str, device_no=0):
    device = pycuda.driver.Device(device_no)
    if attr_name == 'count':
        return device.count()
    elif attr_name == 'name':
        return device.name()
    elif attr_name == 'total_memory':
        return device.total_memory()
    elif attr_name == 'compute_capacity':
        return device.compute_capability()
    elif attr_name == 'arch_name':
        return _arch2name[get_attribute(Attr.COMPUTE_CAPACITY)]
    else:
        name = attr_name.upper()
        return device.get_attribute(getattr(pycuda.driver.device_attribute, name))


def get_attributes(device_no=0):
    attr_names = [
        'count', 'name', 'total_memory', 'compute_capacity', 'arch_name'
    ]
    device = pycuda.driver.Device(device_no)
    attrs = {}
    attrs.update({name: get_attribute(name) for name in attr_names})
    attrs.update({str(name).lower(): value for name, value in device.get_attributes().items()})
    return attrs


def device_synchronize():
    pycuda.driver.Context.synchronize()


def lock_gpu_clock(clock: int):
    command = f'sudo -S nvidia-smi --lock-gpu-clocks={clock}'
    print(f"Running '{command}'...")
    subprocess.run(command.split(), check=True)


def reset_gpu_clock():
    command = 'sudo -S nvidia-smi --reset-gpu-clocks'
    print(f"Running '{command}'")
    subprocess.run(command.split(), check=True)


def query_gpu_current_clock() -> int:
    result = subprocess.run('nvidia-smi -i 0 --query-gpu=clocks.current.graphics --format=csv,noheader,nounits'.split(),
                            stdin=PIPE, stdout=PIPE, check=True)
    return int(result.stdout.decode('utf-8'))


def query_gpu_max_clock() -> int:
    result = subprocess.run('nvidia-smi -i 0 --query-gpu=clocks.max.sm --format=csv,noheader,nounits'.split(),
                            stdin=PIPE, stdout=PIPE, check=True)
    return int(result.stdout.decode('utf-8'))


def lock_memory_clock(clock: int):
    command = f'sudo -S nvidia-smi --lock-memory-clocks={clock}'
    print(f"Running '{command}'...")
    subprocess.run(command.split(), check=True)


def reset_memory_clock():
    command = 'sudo -S nvidia-smi --reset-memory-clocks'
    print(f"Running '{command}'")
    subprocess.run(command.split(), check=True)


def query_memory_current_clock() -> int:
    result = subprocess.run('nvidia-smi -i 0 --query-gpu=clocks.current.memory --format=csv,noheader,nounits'.split(),
                            stdin=PIPE, stdout=PIPE, check=True)
    return int(result.stdout.decode('utf-8'))


def query_memory_max_clock() -> int:
    result = subprocess.run('nvidia-smi -i 0 --query-gpu=clocks.max.memory --format=csv,noheader,nounits'.split(),
                            stdin=PIPE, stdout=PIPE, check=True)
    return int(result.stdout.decode('utf-8'))


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
    def __init__(self, fix_clock=True, clock_ratio=None):
        self.fix_clock = fix_clock
        self.clock_ratio = self.get_bench_ratio(clock_ratio)

    def __enter__(self):
        if self.fix_clock:
            # sm clock; to make result more stable (trying to avoid gpu throttle)
            turn_on_persistent_mode()
            lock_gpu_clock(int(self.clock_ratio * query_gpu_max_clock()))
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
    # for k, v in get_attributes().items():
    #     print("{:>40}: {}".format(k, v))
    turn_on_persistent_mode()
    print('max gpu clock:', query_gpu_max_clock())
    print('max memory clock:', query_memory_max_clock())
    print('current gpu clock:', query_gpu_current_clock())
    print('current memory clock:', query_memory_current_clock())
    with BenchmarkContext():
        print('in')
        print('current gpu clock:', query_gpu_current_clock())
        print('current memory clock:', query_memory_current_clock())
    print('out')
    print('current gpu clock:', query_gpu_current_clock())
    print('current memory clock:', query_memory_current_clock())
    print(BenchmarkContext.get_bench_ratio())
    print(get_compute_capability())
