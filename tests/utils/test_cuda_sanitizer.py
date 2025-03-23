import os
import pytest
import hidet
from hidet.utils.cuda_sanitizer import sanitizer_run, sanitizer_get_path


def func(b):
    a = hidet.empty([1000], device='cuda')
    a + b


@pytest.mark.skipif(not os.path.exists(sanitizer_get_path()), reason='CUDA Compute Sanitizer is not available.')
def test_nsys_run():
    sanitizer_run(func, b=1)
