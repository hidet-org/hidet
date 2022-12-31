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
from hidet.ir.expr import Constant
from hidet.ir.type import DataType, data_type
from hidet.runtime.device import Device, instantiate_device
from .utils import Task, Operator, Tensor, compute


class FullTask(Task):
    def __init__(self, shape: Sequence[int], value: Union[int, float, bool, Constant], dtype: Union[DataType, str]):
        dtype: DataType = data_type(dtype)
        value: Constant = dtype(value)
        const_output = compute(name='c', shape=list(shape), fcompute=lambda *indices: value)
        super().__init__(
            name='full',
            inputs=[],
            outputs=[const_output],
            inverse_map={},
            attributes={'shape': shape, 'value': value.value, 'dtype': dtype.name},
        )


class ArangeTask(Task):
    def __init__(self, start, stop, step, dtype: DataType):
        start: Constant = dtype(start)
        stop: Constant = dtype(stop)
        step: Constant = dtype(step)
        count = int((stop.value - start.value) // step.value)
        super().__init__(
            name='arange',
            inputs=[],
            outputs=[compute(name='c', shape=[count], fcompute=lambda idx: dtype(start + step * idx))],
            attributes={'start': start.value, 'stop': stop.value, 'step': step.value, 'dtype': dtype.name},
        )


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
            name='linspace',
            inputs=[],
            task=LinSpaceTask(start, stop, num, endpoint, dtype),
            attributes={'dtype': dtype, 'device': device},
        )


class ArangeOp(Operator):
    def __init__(self, start, stop, step, dtype, device):
        if stop is None:
            stop = start
            start = 0
        if dtype is None:
            if all(isinstance(v, int) for v in [start, stop, step]):
                dtype = dtypes.int32
            else:
                dtype = dtypes.float32
        device = instantiate_device(device)
        super().__init__(
            name='arange',
            inputs=[],
            task=ArangeTask(start, stop, step, dtype),
            attributes={'start': start, 'stop': stop, 'step': step, 'dtype': dtype, 'device': device},
        )


class FullOp(Operator):
    def __init__(
        self,
        shape: Sequence[int],
        value: Union[float, int, bool, Constant],
        dtype: Optional[DataType] = None,
        device: Union[Device, str] = 'cpu',
    ):
        shape = [int(v) for v in shape]
        device: Device = instantiate_device(device)
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
            name='constant',
            inputs=[],
            task=FullTask(shape=shape, value=value, dtype=dtype),
            attributes={'shape': shape, 'value': value, 'dtype': dtype, 'device': device},
        )


def full(
    shape: Sequence[int],
    value: Union[float, int, bool, Constant],
    dtype: Optional[Union[DataType, str]] = None,
    device: Union[Device, str] = 'cpu',
) -> Tensor:
    return FullOp(shape, value, data_type(dtype), device).get_output(0)


def arange(start, /, stop=None, step=1, *, dtype=None, device='cpu') -> Tensor:
    return ArangeOp(start, stop, step, dtype=dtype, device=device).get_output(0)


def linspace(start, stop, /, num, *, dtype=None, device='cpu', endpoint=True) -> Tensor:
    if dtype is None:
        dtype = dtypes.float32
    dtype = data_type(dtype)
    if dtype.is_integer():
        warnings.warn('linspace with integer dtype is not supported, changed to float32')
        dtype = dtypes.float32
    return LinSpaceOp(start, stop, num, dtype=dtype, device=device, endpoint=endpoint).get_output(0)
