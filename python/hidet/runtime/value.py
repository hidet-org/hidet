from typing import Union
from hidet.ir.expr import Constant
from hidet.ir.task import Task
from pycuda.gpuarray import GPUArray
from pycuda import gpuarray
import pycuda.autoinit
import numpy as np

from hidet.ir.type import TensorType, ScalarType, tensor_type, scalar_type


class Value:
    pass


class TensorValue(Value):
    def __init__(self, type: TensorType, array):
        self.type = type
        # storage is an allocation in global memory that can be converted to int to get the address
        self.array: Union[np.ndarray, GPUArray] = array

    @staticmethod
    def empty(shape, scalar_type, scope, strides=None):
        array = np.ndarray(shape=shape, dtype=scalar_type, strides=strides)
        return TensorValue.from_numpy(array, scope)

    @staticmethod
    def zeros(shape, scalar_type, scope, strides=None):
        array = np.ndarray(shape=shape, dtype=scalar_type, strides=strides)
        flattened: np.ndarray = array.ravel()
        for i in range(flattened.size):
            flattened[i] = 0.0
        return TensorValue.from_numpy(array, scope)

    @staticmethod
    def randn(shape, scalar_type, scope, strides=None, seed=0):
        array = np.ndarray(shape=shape, dtype=scalar_type, strides=strides)
        flattened: np.ndarray = array.ravel()
        for i in range(flattened.size):
            seed = (seed * 5 + 1) % 7
            flattened[i] = float(seed)
        return TensorValue.from_numpy(array, scope)

    @staticmethod
    def full(shape, scalar_type, scope, strides=None, fill_value=0):
        array = np.ndarray(shape=shape, dtype=scalar_type, strides=strides)
        flattened: np.ndarray = array.ravel()
        for i in range(flattened.size):
            flattened[i] = fill_value
        return TensorValue.from_numpy(array, scope)

    @staticmethod
    def from_numpy(array: np.ndarray, scope):
        type = tensor_type(scope, str(array.dtype), array.shape, array.strides)
        if scope == 'host':
            return TensorValue(type, array)
        else:
            return TensorValue(type, gpuarray.to_gpu(array))

    def to_cuda(self):
        if isinstance(self.array, GPUArray):
            return self
        else:
            return TensorValue.from_numpy(self.array, 'global')

    def to_cpu(self):
        if isinstance(self.array, np.ndarray):
            return self
        else:
            return TensorValue.from_numpy(self.to_numpy(), 'host')

    def to_numpy(self):
        if isinstance(self.array, GPUArray):
            return self.array.get()
        else:
            return self.array

    def __str__(self):
        return str(self.array)


class ScalarValue(Value):
    def __init__(self, type: ScalarType, value):
        self.type = type
        self.value = value

    @staticmethod
    def from_python(value):
        if isinstance(value, int):
            return ScalarValue(ScalarType('int32'), value)
        elif isinstance(value, float):
            return ScalarValue(ScalarType('float32'), value)
        elif isinstance(value, bool):
            return ScalarValue(ScalarType('bool'), value)
        else:
            raise NotImplementedError()

    def __str__(self):
        return str(self.value)


def randn(shape, scalar_type: str, scope: str, strides=None, seed=0):
    return TensorValue.randn(shape, scalar_type, scope, strides, seed)


def full(shape, scalar_type: str, scope: str, strides=None, fill_value=1):
    return TensorValue.full(shape, scalar_type, scope, strides, fill_value)


def empty(shape, scalar_type: str, scope: str, strides=None):
    return TensorValue.empty(shape, scalar_type, scope, strides)


def zeros(shape, scalar_type: str, scope: str, strides=None):
    return TensorValue.zeros(shape, scalar_type, scope, strides)


def scalar(value):
    return ScalarValue.from_python(value)


def dummy_inputs_from_task(task: Task):
    inputs = []
    for idx, param_type in enumerate(task.params_type):
        assert isinstance(param_type, TensorType)
        assert all(isinstance(s, Constant)for s in param_type.shape)
        stype = param_type.scalar_type.name
        scope = param_type.scope.name
        shape = [int(s) for s in param_type.shape]
        # strides = [int(s) for s in param_type.strides]
        inputs.append(randn(shape, stype, scope, seed=idx))
    return inputs
