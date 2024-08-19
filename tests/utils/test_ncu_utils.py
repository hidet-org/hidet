import os
import pytest
import hidet
from hidet.utils.ncu_utils import ncu_run, ncu_get_path


def func(b):
    a = hidet.empty([1000], device='cuda')
    a + b


@pytest.mark.skipif(not os.path.exists(ncu_get_path()), reason='Nsight Compute is not available.')
@pytest.mark.skip(
    reason='Skip due to the ci error: The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0'
)
def test_nsys_run():
    report = ncu_run(func, b=1)
    # we can visualize the profiling result by calling the `visualize` method.
    # do not test this part as it will open the nsight compute ui and waiting for the user to close it.
    # report.visualize()
