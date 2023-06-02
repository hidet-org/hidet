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
from typing import Union, Sequence, Tuple
import numpy as np
from hidet import symbol, trace_from
from hidet.graph.tensor import asarray


def check_unary(shape, numpy_op, hidet_op, device: str = 'all', dtype=np.float32, atol=0, rtol=0):
    if device == 'all':
        for dev in ['cuda', 'cpu']:
            check_unary(shape, numpy_op, hidet_op, dev, dtype, atol, rtol)
        return
    # wrap np.array(...) in case shape = []
    data = np.array(np.random.randn(*shape)).astype(dtype)
    numpy_result = numpy_op(data)
    hidet_result = hidet_op(asarray(data).to(device=device)).cpu().numpy()
    np.testing.assert_allclose(actual=hidet_result, desired=numpy_result, atol=atol, rtol=rtol)


def check_unary_dynamic(
    # when type(shape[i]) == int, we take it as static,
    # when type(shape[i]) == Tuple[str, int], we take it as dynamic,
    #   where the value of the integer is the shape of the testing value
    shape: Sequence[Union[int, Tuple[str, int]]],
    numpy_op,
    hidet_op,
    device: str = 'all',
    dtype=np.float32,
    atol=0,
    rtol=0,
):
    if device == 'all':
        for dev in ['cuda', 'cpu']:
            check_unary_dynamic(shape, numpy_op, hidet_op, dev, dtype, atol, rtol)
        return
    concrete_shape = [(i if isinstance(i, int) else i[1]) for i in shape]
    symbolic_shape = [(i if isinstance(i, int) else i[0]) for i in shape]
    data = np.array(np.random.randn(*concrete_shape)).astype(dtype)
    hidet_data = asarray(data).to(device=device)
    numpy_result = numpy_op(data)
    sym = symbol(symbolic_shape, dtype=hidet_data.dtype, device=hidet_data.device)
    out = hidet_op(sym)
    func = trace_from(out, sym).build()
    hidet_result = func(hidet_data).cpu().numpy()
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


def check_ternary(
    a_shape, b_shape, c_shape, numpy_op, hidet_op, dtype: Union[str, np.dtype] = np.float32, atol=0.0, rtol=0.0
):
    np.random.seed(1)
    a = np.array(np.random.randn(*a_shape)).astype(dtype)
    b = np.array(np.random.randn(*b_shape)).astype(dtype)
    c = np.array(np.random.randn(*c_shape)).astype(dtype)

    c = np.abs(c)

    numpy_result = numpy_op(a, b, c)
    import hidet as hi

    hidet_args = [hi.asarray(v).cuda() for v in [a, b, c]]
    hidet_result = hidet_op(*hidet_args).cpu().numpy()
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


def check_torch_ternary(
    a_shape: Sequence[int],
    b_shape: Sequence[int],
    c_shape: Sequence[int],
    torch_func,
    hidet_func,
    device: str = 'all',
    dtype: str = 'float32',
    atol=0.0,
    rtol=0.0,
):
    if device == 'all':
        for dev in ['cuda', 'cpu']:
            check_torch_ternary(a_shape, b_shape, c_shape, torch_func, hidet_func, dev, dtype, atol, rtol)
        return
    import torch
    import hidet

    torch_a = torch.randn(*a_shape, dtype=getattr(torch, dtype)).to(device=device)
    torch_b = torch.randn(*b_shape, dtype=getattr(torch, dtype)).to(device=device)
    torch_c = torch.randn(*c_shape, dtype=getattr(torch, dtype)).to(device=device)
    hidet_a = hidet.from_torch(torch_a)
    hidet_b = hidet.from_torch(torch_b)
    hidet_c = hidet.from_torch(torch_c)
    torch_result: torch.Tensor = torch_func(torch_a, torch_b, torch_c)
    hidet_result: hidet.Tensor = hidet_func(hidet_a, hidet_b, hidet_c)
    np.testing.assert_allclose(
        actual=hidet_result.cpu().numpy(), desired=torch_result.cpu().numpy(), atol=atol, rtol=rtol
    )
