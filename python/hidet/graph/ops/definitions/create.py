from typing import Union, Optional, Sequence

from hidet.ir import dtypes
from hidet.ir.expr import Constant
from hidet.ir.type import DataType, data_type
from .utils import Task, Operator, Tensor, compute
from hidet.runtime.device import Device


class FullTask(Task):
    def __init__(
            self, shape: Sequence[int], value: Union[int, float, bool, Constant], dtype: Union[DataType, str]
    ):
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
    def __init__(
            self, start: Union[int, float], stop: Union[int, float], step: Union[int, float] = 1,
            dtype: Optional[DataType] = None, device: Optional[str] = None
    ):
        pass


class FullOp(Operator):
    def __init__(
            self,
            shape: Sequence[int],
            value: Union[float, int, bool, Constant],
            dtype: Optional[DataType] = None,
            device: str = 'cpu',
    ):
        shape = [int(v) for v in shape]
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
            task=FullTask(shape=shape, value=value, dtype=dtype),
            name='constant',
            attributes={'shape': shape, 'value': value, 'dtype': dtype, 'device': device},
        )


def full(
        shape: Sequence[int],
        value: Union[float, int, bool, Constant],
        dtype: Optional[Union[DataType, str]] = None,
        device: Union[Device, str] = 'cpu',
) -> Tensor:
    return FullOp(shape, value, data_type(dtype), device).get_output(0)
