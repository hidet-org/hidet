from hidet.ir.type import DataType, data_type


def numeric_promotion(a_dtype: DataType, b_dtype: DataType) -> DataType:
    if a_dtype.is_float() and b_dtype.is_float():
        return a_dtype if a_dtype.nbytes >= b_dtype.nbytes else b_dtype
    elif a_dtype.is_integer() and b_dtype.is_integer():
        return a_dtype if a_dtype.nbytes >= b_dtype.nbytes else b_dtype
    elif a_dtype.is_integer() and b_dtype.is_float():
        return b_dtype
    elif a_dtype.is_float() and b_dtype.is_integer():
        return a_dtype
    else:
        raise ValueError('Cannot do numeric promotion for {} and {}.'.format(a_dtype, b_dtype))


def numeric_promotion_for_all(*dtypes: DataType) -> DataType:
    if len(dtypes) == 0:
        raise ValueError('No data type is provided.')
    elif len(dtypes) == 1:
        return dtypes[0]
    else:
        return numeric_promotion(numeric_promotion_for_all(*dtypes[:-1]), dtypes[-1])


def from_numpy_dtype(np_dtype):
    import numpy as np

    if np_dtype == np.float32:
        return data_type('float32')
    elif np_dtype == np.int32:
        return data_type('int32')
    elif np_dtype == np.int64:
        return data_type('int64')
    else:
        raise ValueError("Unrecognized numpy data type: '{}'".format(np_dtype))
