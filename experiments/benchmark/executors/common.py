from typing import List, Optional
from hidet import Tensor


class BenchResult:
    def __init__(self, latencies: List[float] = None, outputs: List[Tensor] = None, configs: str = None):
        self.latencies = latencies
        self.outputs: Optional[List[Tensor]] = outputs
        self.configs = configs


def benchmark_run(run_func, warmup, number, repeat) -> List[float]:
    from hidet.utils.nvtx_utils import nvtx_annotate
    from hidet.utils import cuda
    import time
    results = []
    with nvtx_annotate('warmup'):
        for i in range(warmup):
            run_func()
            cuda.device_synchronize()
    for i in range(repeat):
        with nvtx_annotate(f'repeat {i}'):
            cuda.device_synchronize()
            start_time = time.time()
            for j in range(number):
                run_func()
            cuda.device_synchronize()
            end_time = time.time()
        results.append((end_time - start_time) * 1000 / number)
    return results
