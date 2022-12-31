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
from hidet.ir import dtypes
from hidet.ir.type import DataType


def dtype_from_onnx(onnx_dtype) -> DataType:
    import onnx

    dtype_map = {
        onnx.TensorProto.DOUBLE: dtypes.float64,
        onnx.TensorProto.FLOAT: dtypes.float32,
        onnx.TensorProto.FLOAT16: dtypes.float16,
        onnx.TensorProto.BFLOAT16: dtypes.bfloat16,
        onnx.TensorProto.INT64: dtypes.int64,
        onnx.TensorProto.INT32: dtypes.int32,
        onnx.TensorProto.INT16: dtypes.int16,
        onnx.TensorProto.INT8: dtypes.int8,
        onnx.TensorProto.UINT64: dtypes.uint64,
        onnx.TensorProto.UINT32: dtypes.uint32,
        onnx.TensorProto.UINT16: dtypes.uint16,
        onnx.TensorProto.UINT8: dtypes.uint8,
        onnx.TensorProto.BOOL: dtypes.boolean,
    }
    return dtype_map[onnx_dtype]


def dtype_to_onnx(dtype: DataType):
    import onnx

    dtype_map = {
        dtypes.float64: onnx.TensorProto.DOUBLE,
        dtypes.float32: onnx.TensorProto.FLOAT,
        dtypes.float16: onnx.TensorProto.FLOAT16,
        dtypes.bfloat16: onnx.TensorProto.BFLOAT16,
        dtypes.int64: onnx.TensorProto.INT64,
        dtypes.int32: onnx.TensorProto.INT32,
        dtypes.int16: onnx.TensorProto.INT16,
        dtypes.int8: onnx.TensorProto.INT8,
        dtypes.uint64: onnx.TensorProto.UINT64,
        dtypes.uint32: onnx.TensorProto.UINT32,
        dtypes.uint16: onnx.TensorProto.UINT16,
        dtypes.uint8: onnx.TensorProto.UINT8,
        dtypes.boolean: onnx.TensorProto.BOOL,
    }
    return dtype_map[dtype.name]
