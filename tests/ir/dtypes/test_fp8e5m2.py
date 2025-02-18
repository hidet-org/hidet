import pytest
from hidet import ops, graph, symbol, full, trace_from

DTYPE = "f8e5m2"
TOLERANCE = 0.5

"""
Tests all fp8e5m2 custom operations. Does not test fma, min, max.
For fp8e5m2 numbers the normalized representation is
    value = (1 + f/4) * 2^(E - bias)
with f ∈ {0, 1, 2, 3} and rounding done with tie-to-even.
"""


@pytest.mark.requires_cuda
def test_ir_float8_e5m2_binary_arithmetic():
    """
    Tests a chain of binary arithmetic operations:
      add, subtract, negative, multiply, divide.

    For example, given inputs [9, 96, 112, 10, 4]:
      - 9.0 converts to 8.0 in fp8e5m2.
      - 96 and 112 are exactly representable.

    Then:
      o  = add(8, 96) = 104  → 104/64 = 1.625 rounds (tie-to-even) to 1.50 so 1.50*64 = 96.
      o2 = 96 - 112 = -16.
      o3 = -(-16) = 16.
      o4 = 16 * 10 = 160.
      o5 = 160 / 4 = 40.
    """
    shape = (2, 2)
    inputs = [9, 96, 112, 10, 4]
    expected = [96.0, -16.0, 16.0, 160.0, 40.0]
    symbolic_inputs = [symbol(shape, dtype=DTYPE, device='cuda') for _ in range(len(inputs))]

    o = ops.add(symbolic_inputs[0], symbolic_inputs[1])
    o2 = ops.subtract(o, symbolic_inputs[2])
    o3 = ops.negative(o2)
    o4 = ops.multiply(o3, symbolic_inputs[3])
    o5 = ops.divide(o4, symbolic_inputs[4])

    g = graph.optimize(trace_from([o, o2, o3, o4, o5], inputs=symbolic_inputs)).build()
    res = g.run_async([full(shape, inputs[i], dtype=DTYPE, device='cuda') for i in range(len(inputs))])
    res = [c.to(dtype='float32') for c in res]

    for i in range(len(expected)):
        assert res[i][0][0].item() == expected[i], f"res[{i}][0][0].item() = {res[i][0][0].item()} != {expected[i]}"


@pytest.mark.requires_cuda
def test_ir_float8_e5m2_unary_functions():
    """
    Tests several unary operations in a chain:
      abs, sin, cos, tanh, exp.

    For fp8e5m2 we have (for example):
      - abs(-2.0) → 2.0.
      - sin(0.5) in real arithmetic is ~0.47943 but rounds to 0.4375.
      - cos(1.5) ~ 0.07074 rounds to 0.078125.
      - tanh(2.0) ~ 0.96403 rounds to 0.875.
      - exp(2.0) ~ 7.38906 rounds to 7.0.
    """
    shape = (2, 2)
    inputs = [-2.0, 0.5, 1.5, 2.0, 2.0]
    expected = [2.0, 0.4375, 0.078125, 0.875, 7.0]
    symbolic_inputs = [symbol(shape, dtype=DTYPE, device='cuda') for _ in range(len(inputs))]

    o = ops.abs(symbolic_inputs[0])
    o2 = ops.sin(symbolic_inputs[1])
    o3 = ops.cos(symbolic_inputs[2])
    o4 = ops.tanh(symbolic_inputs[3])
    o5 = ops.exp(symbolic_inputs[4])

    g = graph.optimize(trace_from([o, o2, o3, o4, o5], inputs=symbolic_inputs)).build()
    data = [full(shape, val, dtype=DTYPE, device='cuda') for val in inputs]
    res = g.run_async(data)
    res = [r.to(dtype='float32') for r in res]

    for i in range(len(expected)):
        diff = abs(res[i][0][0].item() - expected[i])
        assert diff < TOLERANCE, f"res[{i}][0][0].item() = {res[i][0][0].item()} (diff {diff}) != {expected[i]}"


@pytest.mark.requires_cuda
def test_ir_float8_e5m2_pow():
    """
    Tests pow(a,b).

    With inputs 2.0 and 3.0 (both exactly representable), we have:
      pow(2.0, 3.0) = 8.0.
    """
    shape = (2, 2)
    inputs = [2.0, 3.0]
    expected = [8.0]
    symbolic_inputs = [symbol(shape, dtype=DTYPE, device='cuda') for _ in range(len(inputs))]

    o = ops.pow(symbolic_inputs[0], symbolic_inputs[1])
    g = graph.optimize(trace_from([o], inputs=symbolic_inputs)).build()
    data = [full(shape, val, dtype=DTYPE, device='cuda') for val in inputs]
    res = g.run_async(data)
    res = [r.to(dtype='float32') for r in res]

    for i in range(len(expected)):
        assert res[i][0][0].item() == expected[i], f"res[{i}][0][0].item() = {res[i][0][0].item()} != {expected[i]}"


@pytest.mark.requires_cuda
def test_ir_float8_e5m2_increment_decrement():
    """
    Tests increment and decrement (simulated by x+1 and x-1).

    For example, with input 9.0 (which rounds to 8.0) and -4.0:
      - inc = 8.0 + 1 = 9.0, which rounds back to 8.0.
      - dec = -4.0 - 1 = -5.0.
    """
    shape = (2, 2)
    inputs = [9.0, -4.0]
    expected = [8.0, -5.0]
    symbolic_inputs = [symbol(shape, dtype=DTYPE, device='cuda') for _ in range(len(inputs))]

    inc = ops.add(symbolic_inputs[0], 1.0)
    dec = ops.subtract(symbolic_inputs[1], 1.0)
    g = graph.optimize(trace_from([inc, dec], inputs=symbolic_inputs)).build()
    data = [full(shape, val, dtype=DTYPE, device='cuda') for val in inputs]
    res = g.run_async(data)
    res = [r.to(dtype='float32') for r in res]

    for i in range(len(expected)):
        assert res[i][0][0].item() == expected[i], f"res[{i}][0][0].item() = {res[i][0][0].item()} != {expected[i]}"


@pytest.mark.requires_cuda
def test_ir_float8_e5m2_comparators():
    """
    Tests comparison operators (==, <, <=, >, >=, !=).
    (Outputs may be stored as booleans or as 0/1 numbers.)

    With inputs 2.0 and 3.0 (with 3.0 represented as 1.5*2), we expect:
      eq  = (2.0 == 3.0) → 0
      lt  = (2.0 <  3.0) → 1
      le  = (2.0 <= 3.0) → 1
      gt  = (2.0 >  3.0) → 0
      ge  = (2.0 >= 3.0) → 0
      ne  = (2.0 != 3.0) → 1
    """
    shape = (2, 2)
    inputs = [2.0, 3.0]
    expected = [0, 1, 1, 0, 0, 1]
    sym_a = symbol(shape, dtype=DTYPE, device='cuda')
    sym_b = symbol(shape, dtype=DTYPE, device='cuda')

    eq = ops.equal(sym_a, sym_b)
    lt = ops.less(sym_a, sym_b)
    le = ops.less_equal(sym_a, sym_b)
    gt = ops.greater(sym_a, sym_b)
    ge = ops.greater_equal(sym_a, sym_b)
    ne = ops.not_equal(sym_a, sym_b)

    g = graph.optimize(trace_from([eq, lt, le, gt, ge, ne], inputs=[sym_a, sym_b])).build()
    data = [full(shape, val, dtype=DTYPE, device='cuda') for val in inputs]
    res = g.run_async(data)
    res = [r.to(dtype='int32') for r in res]

    for i in range(len(expected)):
        val = res[i][0][0].item()
        assert val == expected[i], f"Comparator {i}: got {val} != {expected[i]}"


@pytest.mark.requires_cuda
def test_ir_float8_e5m2_more_functions():
    """
    Tests a few additional unary math ops:
      erf, exp, sqrt, rsqrt, log, round, floor, ceil.

    For fp8e5m2 the inputs are first rounded:
      - 0.5 → 0.5
      - 2.0 → 2.0
      - 4.0 → 4.0
      - 2.7182818 → 2.5 (since 2.718... rounds to 1.25*2)
      - 1.9 → 1.75
      - -1.2 → -1.25
      - 1.2 → 1.25.

    Then, for example:
      o1 = erf(0.5) ≈ 0.5205 becomes 0.5.
      o2 = exp(2.0) ≈ 7.3891 becomes 7.0.
      o3 = sqrt(4.0) = 2.0.
      o4 = rsqrt(4.0) = 0.5.
      o5 = log(2.7182818) is computed as log(2.5) ≈ 0.9163, which rounds to 0.875.
      o6 = round(1.9) from 1.75 → 2.0.
      o7 = floor(-1.2) from -1.25 → -2.0.
      o8 = ceil(1.2) from 1.25 → 2.0.
    """
    shape = (2, 2)
    inputs = [0.5, 2.0, 4.0, 4.0, 2.7182818, 1.9, -1.2, 1.2]
    expected = [
        0.5,  # erf(0.5)
        7.0,  # exp(2.0)
        2.0,  # sqrt(4.0)
        0.5,  # rsqrt(4.0)
        0.875,  # log(2.7182818) computed from 2.5
        2.0,  # round(1.9) from 1.75
        -2.0,  # floor(-1.2) from -1.25
        2.0,  # ceil(1.2) from 1.25
    ]
    symbolic_inputs = [symbol(shape, dtype=DTYPE, device='cuda') for _ in range(len(inputs))]

    o1 = ops.erf(symbolic_inputs[0])
    o2 = ops.exp(symbolic_inputs[1])
    o3 = ops.sqrt(symbolic_inputs[2])
    o4 = ops.rsqrt(symbolic_inputs[3])
    o5 = ops.log(symbolic_inputs[4])
    o6 = ops.round(symbolic_inputs[5])
    o7 = ops.floor(symbolic_inputs[6])
    o8 = ops.ceil(symbolic_inputs[7])

    g = graph.optimize(trace_from([o1, o2, o3, o4, o5, o6, o7, o8], inputs=symbolic_inputs)).build()
    data = [full(shape, val, dtype=DTYPE, device='cuda') for val in inputs]
    res = g.run_async(data)
    res = [r.to(dtype='float32') for r in res]

    for i in range(len(expected)):
        diff = abs(res[i][0][0].item() - expected[i])
        assert diff < TOLERANCE, f"res[{i}][0][0].item() = {res[i][0][0].item()} (diff {diff}) != {expected[i]}"
