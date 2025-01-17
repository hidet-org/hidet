# Notes on Hidet tests


## Tests enabled on different devices

There are three devices that are (going to be) supported by Hidet:
- NVIDIA GPUs (CUDA Platform)
- AMD GPUs (ROCm/HIP Platform)
- CPU

We need to test of all the model/operator tests on these devices. 

To achieve so, we added a `device` fixture that will be filled with the device that the current CI server supports.

For example, we could add a test like
```python
import pytest
import hidet
import hidet.testing

@pytest.mark.parameterize('n', [1, 10, 100])
def test_add(n, device):
    assert device in ['cpu', 'cuda', 'hip']
    a = hidet.randn([n], device=device)
    b = hidet.randn([n], device=device)
    c = hidet.ops.add(a, b)
    torch_c = a.torch() + b.torch()
    assert hidet.utils.assert_close(c, torch_c)
```

In the CUDA CI server, the device will be `cuda`; and in the HIP CI server, the device will be `hip`. 

See the `tests/conftest.py` for details where the `device` fixture is defined.

## Tests specific to some device

Some tests are specific to some devices. For example, we might test some primitive function for CUDA platform. For those
tests, we can use the `pytest.mark.requires_cuda` and `pytest.mark.requires_hip` decorators to make sure that the test 
is only run on the specific device.

For example, we could add a test like
```python
import pytest
import hidet
import hidet.testing

@pytest.mark.requires_cuda
def test_add_cuda():
    a = hidet.randn([10], device='cuda')
    b = hidet.randn([10], device='cuda')
    c = hidet.ops.add(a, b)
    torch_c = a.torch() + b.torch()
    assert hidet.utils.assert_close(c, torch_c)
```
that will be only run on the CUDA CI server.


## Trigger tests on AMD GPU

Currently, we run the AMD GPU tests on a temporary CI server, and we need to trigger the CI by putting the following 
code snippet in the PR body or comment
```
$ hidet-ci amdgpu
```
This is temporary, and we will test AMD GPUs just like NVIDIA GPUs.
