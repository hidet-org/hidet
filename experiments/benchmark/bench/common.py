from typing import List, Optional, Tuple
from functools import lru_cache
import numpy as np
import hidet
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


@lru_cache()
def get_onnx_model(name: str, batch_size: int, precision='float32') -> Tuple[str, List[str], List[hidet.Tensor]]:
    from hidet.testing import onnx_models
    precision_dict = {
        'f16': 'float16',
        'f32': 'float32',
        'bf16': 'bfloat16'
    }
    if precision in precision_dict:
        precision = precision_dict[precision]
    return onnx_models.get_onnx_model(name, batch_size, precision=precision)


def run_with_onnx(model_path: str, input_names: List[str], input_tensors: List[hidet.Tensor]) -> List[np.ndarray]:
    import onnxruntime
    onnx_session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])  # use cpu executor for high accuracy
    onnx_outputs = onnx_session.run(None, input_feed={name: tensor.numpy() for name, tensor in zip(input_names, input_tensors)})
    return onnx_outputs

