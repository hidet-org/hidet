from typing import Union

import numpy as np
import hidet as hi


def check_unary(shape, numpy_op, hidet_op, dtype: Union[str, np.dtype] = np.float32, atol=0, rtol=0):
    # wrap np.array(...) in case shape = []
    data = np.array(np.random.randn(*shape)).astype(dtype)
    numpy_result = numpy_op(data)
    hidet_result = hidet_op(hi.array(data).cuda()).cpu().numpy()
    np.testing.assert_allclose(actual=hidet_result, desired=numpy_result, atol=atol, rtol=rtol)


def check_binary(a_shape, b_shape, numpy_op, hidet_op, dtype: Union[str, np.dtype] = np.float32, atol=0.0, rtol=0.0):
    a = np.array(np.random.randn(*a_shape)).astype(dtype)
    b = np.array(np.random.randn(*b_shape)).astype(dtype)
    numpy_result = numpy_op(a, b)
    hidet_result = hidet_op(hi.array(a).cuda(), hi.array(b).cuda()).cpu().numpy()
    np.testing.assert_allclose(actual=hidet_result, desired=numpy_result, atol=atol, rtol=rtol)

