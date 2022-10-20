from typing import List, Union, Callable, Any
from hidet.graph.tensor import array
import numpy as np


def benchmark_func(run_func, warmup=1, number=5, repeat=5, median=True) -> Union[List[float], float]:
    """ Benchmark given function.

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
    from hidet.utils.nvtx_utils import nvtx_annotate
    from hidet.utils import cuda
    import numpy as np
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
    if median:
        return float(np.median(results))
    else:
        return results


def check_unary(shape, numpy_op, hidet_op, device: str = 'all', dtype: Union[str, np.dtype] = np.float32, atol=0, rtol=0):
    if device == 'all':
        for device in ['cuda', 'cpu']:
            check_unary(shape, numpy_op, hidet_op, device, dtype, atol, rtol)
    # wrap np.array(...) in case shape = []
    data = np.array(np.random.randn(*shape)).astype(dtype)
    numpy_result = numpy_op(data)
    hidet_result = hidet_op(array(data).to(device=device)).cpu().numpy()
    np.testing.assert_allclose(actual=hidet_result, desired=numpy_result, atol=atol, rtol=rtol)


def check_binary(a_shape, b_shape, numpy_op, hidet_op, device: str = 'all', dtype: Union[str, np.dtype] = np.float32, atol=0.0, rtol=0.0):
    if device == 'all':
        for device in ['cuda', 'cpu']:
            print('checking', device)
            check_binary(a_shape, b_shape, numpy_op, hidet_op, device, dtype, atol, rtol)
    a = np.array(np.random.randn(*a_shape)).astype(dtype)
    b = np.array(np.random.randn(*b_shape)).astype(dtype)
    numpy_result = numpy_op(a, b)
    hidet_result = hidet_op(array(a).to(device=device), array(b).to(device=device)).cpu().numpy()
    np.testing.assert_allclose(actual=hidet_result, desired=numpy_result, atol=atol, rtol=rtol)

