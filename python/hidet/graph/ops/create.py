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
from typing import Union, Optional, Sequence
import warnings

from hidet.ir import dtypes
from hidet.ir.expr import Constant, Expr, Int, if_then_else
from hidet.ir.type import DataType, data_type
from hidet.runtime.device import Device, instantiate_device
from .utils import Task, Operator, Tensor, compute


class FullTask(Task):
    def __init__(
        self, shape: Sequence[int], value: Union[int, float, bool, Constant, Expr], dtype: Union[DataType, str]
    ):
        dtype: DataType = data_type(dtype)
        value: Constant = dtype(value) if isinstance(value, (int, float, bool)) else value
        const_output = compute(name='c', shape=list(shape), fcompute=lambda *indices: value)
        super().__init__(
            name='full',
            inputs=[],
            outputs=[const_output],
            inverse_map={},
            attributes={'shape': shape, 'value': value, 'dtype': dtype.name},
        )


class ArangeTask(Task):
    def __init__(self, start, stop, step, dtype: DataType):
        count = dtypes.int32((stop - start) // step)
        super().__init__(
            name='arange',
            inputs=[],
            outputs=[compute(name='c', shape=[count], fcompute=lambda idx: dtype(start + step * idx))],
            attributes={'start': start, 'stop': stop, 'step': step, 'dtype': dtype.name},
        )


class TriTask(Task):
    def __init__(self, n: Int, m: Int, k: Int, dtype: DataType):
        if m is None:
            m = n
        out = compute(name='out', shape=[n, m], fcompute=lambda i, j: if_then_else(j <= i + k, dtype.one, dtype.zero))
        super().__init__(name='tri', inputs=[], outputs=[out], attributes={'n': n, 'm': m, 'k': k, 'dtype': dtype.name})


class LinSpaceTask(Task):
    def __init__(self, start, stop, num, endpoint: bool, dtype: DataType):
        start: Constant = dtype(start)
        stop: Constant = dtype(stop)
        num: int = int(num)
        if endpoint:
            step = (stop - start) / (num - 1)
        else:
            step = (stop - start) / num

        c = compute(name='c', shape=[num], fcompute=lambda idx: dtype(start + step * idx))

        super().__init__(
            name='linspace',
            inputs=[],
            outputs=[c],
            attributes={
                'start': start.value,
                'stop': stop.value,
                'num': num,
                'endpoint': endpoint,
                'dtype': dtype.name,
            },
        )


class LinSpaceOp(Operator):
    def __init__(self, start, stop, num, *, dtype: DataType, device, endpoint=True):
        device = instantiate_device(device)
        super().__init__(
            inputs=[],
            attributes={'dtype': dtype, 'device': device},
            task=LinSpaceTask(start, stop, num, endpoint, dtype),
        )


class ArangeOp(Operator):
    def __init__(self, start, stop, step, dtype, device):
        if stop is None:
            stop = start
            start = 0
        if dtype is None:
            dtype = self.infer_dtype(start, stop, step)
        dtype = data_type(dtype)
        device = instantiate_device(device)
        super().__init__(
            inputs=[],
            attributes={'start': start, 'stop': stop, 'step': step, 'dtype': dtype, 'device': device},
            task=ArangeTask(start, stop, step, dtype),
        )

    def infer_dtype(self, start, stop, step):
        from hidet.ir.expr import convert
        from hidet.ir.tools import infer_type
        from hidet.ir.dtypes import promote_type

        dtype = None
        for v in [start, stop, step]:
            v = convert(v)
            assert isinstance(v, Expr)
            v_dtype = infer_type(v)
            assert isinstance(v_dtype, DataType)
            dtype = v_dtype if dtype is None else promote_type(dtype, v_dtype)
        return dtype


class FullOp(Operator):
    def __init__(
        self,
        shape: Sequence[int],
        value: Union[float, int, bool, Constant, Tensor],
        dtype: Optional[DataType] = None,
        device: Union[Device, str] = 'cpu',
    ):
        shape = [int(v) for v in shape]
        device: Device = instantiate_device(device)

        if isinstance(value, Tensor):
            if value.is_symbolic():
                raise NotImplementedError('Currently, we do not support symbolic tensor as value in full op')
            value = value.item()

        if dtype is None:
            if isinstance(value, int):
                dtype = dtypes.int64
            elif isinstance(value, float):
                dtype = dtypes.float32
            elif isinstance(value, bool):
                dtype = dtypes.boolean
            elif isinstance(value, Constant):
                assert isinstance(value.type, DataType)
                dtype = value.type
            else:
                raise ValueError(f'Unknown type for value {value}')

        super().__init__(
            inputs=[],
            attributes={'shape': shape, 'value': value, 'dtype': dtype, 'device': device},
            task=FullTask(shape=shape, value=value, dtype=dtype),
        )


class TriOp(Operator):
    def __init__(
        self,
        n: Int,
        m: Optional[Int] = None,
        k: Int = 0,
        dtype: Optional[DataType] = None,
        device: Union[Device, str] = 'cpu',
    ):
        if m is None:
            m = n
        if dtype is None:
            dtype = dtypes.float32
        device = instantiate_device(device)
        super().__init__(
            inputs=[],
            attributes={'n': n, 'm': m, 'k': k, 'dtype': dtype, 'device': device},
            task=TriTask(n, m, k, dtype),
        )


def full(
    shape: Sequence[int],
    value: Union[float, int, bool, Constant, Tensor],
    dtype: Optional[Union[DataType, str]] = None,
    device: Union[Device, str] = 'cpu',
) -> Tensor:
    return FullOp(shape, value, data_type(dtype) if dtype is not None else dtype, device).outputs[0]


def arange(start, /, stop=None, step=1, *, dtype=None, device='cpu') -> Tensor:
    return ArangeOp(start, stop, step, dtype=dtype, device=device).outputs[0]


def linspace(start, stop, /, num, *, dtype=None, device='cpu', endpoint=True) -> Tensor:
    if dtype is None:
        dtype = dtypes.float32
    dtype = data_type(dtype)
    if dtype.is_integer():
        warnings.warn('linspace with integer dtype is not supported, changed to float32')
        dtype = dtypes.float32
    return LinSpaceOp(start, stop, num, dtype=dtype, device=device, endpoint=endpoint).outputs[0]


def tri(
    n: Int, m: Optional[Int] = None, k: Int = 0, dtype: DataType = dtypes.float32, device: Union[str, Device] = 'cpu'
) -> Tensor:
    if m is None:
        m = n
    return TriOp(n, m, k, dtype, device).outputs[0]
