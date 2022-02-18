import os
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
    def __enter__(self):
        # sm clock; to make result more stable (trying to avoid gpu throttle)
        ratio = self.get_bench_ratio()
        turn_on_persistent_mode()
        lock_gpu_clock(int(ratio * query_gpu_max_clock()))
        lock_memory_clock(query_memory_max_clock())

    def __exit__(self, exc_type, exc_val, exc_tb):
        reset_memory_clock()
        reset_gpu_clock()

    @staticmethod
    def get_bench_ratio(default=0.8):
        ratio = os.environ.get('HIDET_BENCH_RATIO')
        if ratio:
            ratio = float(ratio)
        else:
            ratio = default
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

