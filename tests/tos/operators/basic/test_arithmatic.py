import pytest
import hidet
import numpy as np
from hidet.tos import operators as ops


def check_binary(a_shape, b_shape, dtype, op, hidet_op=None, a_positive=False, b_positive=False):
    a = np.random.rand(*a_shape).astype(dtype)
    b = np.random.rand(*b_shape).astype(dtype)
    a = np.abs(a) if a_positive else a
    b = np.abs(b) if b_positive else b
    numpy_c = op(a, b)
    if hidet_op is None:
        hidet_op = op
    hidet_c = hidet_op(hidet.array(a).cuda(), hidet.array(b).cuda()).cpu().numpy()
    np.testing.assert_allclose(actual=hidet_c, desired=numpy_c, atol=1e-5, rtol=1e-5)


def check_unary(shape, dtype, numpy_op, hidet_op, positive=False):
    a = np.random.rand(*shape).astype(dtype)
    a = np.abs(a) if positive else a
    numpy_b = numpy_op(a)
    hidet_b = hidet_op(hidet.array(a).cuda()).cpu().numpy()
    np.testing.assert_allclose(actual=hidet_b, desired=numpy_b, atol=1e-5, rtol=1e-5)


binary_op_shapes = [[[1], [200]],
                    [[100, 200], [1, 200]],
                    [[200, 1], [200]]]

unary_op_shapes = [[1], [100], [200]]


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_add(a_shape, b_shape):
    check_binary(a_shape, b_shape, np.float32, lambda a, b: a + b)


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_sub(a_shape, b_shape):
    check_binary(a_shape, b_shape, np.float32, lambda a, b: a - b)


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_multiply(a_shape, b_shape):
    check_binary(a_shape, b_shape, np.float32, lambda a, b: a * b)


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_divide(a_shape, b_shape):
    check_binary(a_shape, b_shape, np.float32, lambda a, b: a / b)


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_pow(a_shape, b_shape):
    check_binary(a_shape, b_shape, np.float32, np.power, ops.pow, a_positive=True)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_sqrt(shape):
    check_unary(shape, np.float32, np.sqrt, ops.sqrt, positive=True)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_tanh(shape):
    check_unary(shape, np.float32, np.tanh, ops.tanh)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_erf(shape):
    check_unary(shape, np.float32, np.tanh, ops.tanh)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_rsqrt(shape):
    check_unary(shape, np.float32, lambda v: np.reciprocal(np.sqrt(v)), ops.rsqrt, positive=True)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_neg(shape):
    check_unary(shape, np.float32, np.negative, ops.neg)


if __name__ == '__main__':
    pytest.main([__name__])

