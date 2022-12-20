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
