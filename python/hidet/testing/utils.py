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
from typing import Union, Sequence
import numpy as np
from hidet.graph.tensor import asarray


def check_unary(
    shape, numpy_op, hidet_op, device: str = 'all', dtype: Union[str, np.dtype] = np.float32, atol=0, rtol=0
):
    if device == 'all':
        for dev in ['cuda', 'cpu']:
            check_unary(shape, numpy_op, hidet_op, dev, dtype, atol, rtol)
        return
    # wrap np.array(...) in case shape = []
    data = np.array(np.random.randn(*shape)).astype(dtype)
    numpy_result = numpy_op(data)
    hidet_result = hidet_op(asarray(data).to(device=device)).cpu().numpy()
    np.testing.assert_allclose(actual=hidet_result, desired=numpy_result, atol=atol, rtol=rtol)


def check_binary(
    a_shape,
    b_shape,
    numpy_op,
    hidet_op,
    device: str = 'all',
    dtype: Union[str, np.dtype] = np.float32,
    atol=0.0,
    rtol=0.0,
):
    if device == 'all':
        for dev in ['cuda', 'cpu']:
            print('checking', dev)
            check_binary(a_shape, b_shape, numpy_op, hidet_op, dev, dtype, atol, rtol)
        return
    a = np.array(np.random.randn(*a_shape)).astype(dtype)
    b = np.array(np.random.randn(*b_shape)).astype(dtype)
    numpy_result = numpy_op(a, b)
    hidet_result = hidet_op(asarray(a).to(device=device), asarray(b).to(device=device)).cpu().numpy()
    np.testing.assert_allclose(actual=hidet_result, desired=numpy_result, atol=atol, rtol=rtol)


def check_torch_unary(
    shape: Sequence[int], torch_func, hidet_func, device: str = 'all', dtype: str = 'float32', atol=0.0, rtol=0.0
):
    if device == 'all':
        for dev in ['cuda', 'cpu']:
            check_torch_unary(shape, torch_func, hidet_func, dev, dtype, atol, rtol)
        return
    import torch
    import hidet

    torch_data = torch.randn(*shape, dtype=getattr(torch, dtype)).to(device=device)
    hidet_data = hidet.from_torch(torch_data)
    torch_result: torch.Tensor = torch_func(torch_data)
    hidet_result: hidet.Tensor = hidet_func(hidet_data)
    np.testing.assert_allclose(
        actual=hidet_result.cpu().numpy(), desired=torch_result.cpu().numpy(), atol=atol, rtol=rtol
    )


def check_torch_binary(
    a_shape: Sequence[int],
    b_shape: Sequence[int],
    torch_func,
    hidet_func,
    device: str = 'all',
    dtype: str = 'float32',
    atol=0.0,
    rtol=0.0,
):
    if device == 'all':
        for dev in ['cuda', 'cpu']:
            check_torch_binary(a_shape, b_shape, torch_func, hidet_func, dev, dtype, atol, rtol)
        return
    import torch
    import hidet

    torch_a = torch.randn(*a_shape, dtype=getattr(torch, dtype)).to(device=device)
    torch_b = torch.randn(*b_shape, dtype=getattr(torch, dtype)).to(device=device)
    hidet_a = hidet.from_torch(torch_a)
    hidet_b = hidet.from_torch(torch_b)
    torch_result: torch.Tensor = torch_func(torch_a, torch_b)
    hidet_result: hidet.Tensor = hidet_func(hidet_a, hidet_b)
    np.testing.assert_allclose(
        actual=hidet_result.cpu().numpy(), desired=torch_result.cpu().numpy(), atol=atol, rtol=rtol
    )
