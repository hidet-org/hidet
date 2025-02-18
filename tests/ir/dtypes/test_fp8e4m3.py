import pytest
from hidet import ops, graph, symbol, full, trace_from

DTYPE = "f8e4m3"
TOLERANCE = 0.2

"""
Tests all float8_e4m3 custom operations. Does not test fma, min, max.
"""


@pytest.mark.requires_cuda
def test_ir_float8_e4m3_binary_arithmetic():
    """
    Tests a chain of binary arithmetic operations:
      add, subtract, negative, multiply, divide
    """
    shape = (2, 2)
    inputs = [9, 96, 112, 10, 4]
    expected = [104.0, -8.0, 8.0, 80.0, 20.0]
    symbolic_inputs = [symbol(shape, dtype=DTYPE, device='cuda') for i in range(len(inputs))]

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
def test_ir_float8_e4m3_unary_functions():
    """
    Tests several unary operations in a chain:
      abs, sin, cos, tanh, exp
    """
    shape = (2, 2)
    inputs = [-2.0, 0.5, 1.5, 2.0, 2.0]
    #   o  = abs(-2.0)      = 2.0
    #   o2 = sin(0.5)       ≈ 0.479425539
    #   o3 = cos(1.5)       ≈ 0.0707372017
    #   o4 = tanh(2.0)      ≈ 0.96402758
    #   o5 = exp(2.0)       ≈ 7.389056
    expected = [
        2.0,
        0.479425549,  # rounded sin(0.5)
        0.0707372017,  # rounded cos(1.5)
        0.96402758,  # rounded tanh(2.0)
        7.389056,  # exp(2.0)
    ]

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
        assert (
            abs(res[i][0][0].item() - expected[i]) < TOLERANCE
        ), f"res[{i}][0][0].item() = {res[i][0][0].item()} != {expected[i]}"


@pytest.mark.requires_cuda
def test_ir_float8_e4m3_pow():
    """
    Tests pow: pow(a,b)
    """
    shape = (2, 2)
    inputs = [2.0, 3.0]
    #   o  = pow(2.0, 3.0) = 8.0
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
def test_ir_float8_e4m3_increment_decrement():
    """
    Tests increment and decrement, simulated by (x + 1) and (x - 1).
    """
    shape = (2, 2)
    inputs = [9.0, -4.0]
    #   inc = 9.0 + 1 = 10.0
    #   dec = -4.0 - 1 = -5.0
    expected = [10.0, -5.0]

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
def test_ir_float8_e4m3_comparators():
    """
    Tests comparison ops (==, <, <=, >, >=, !=).
    We'll store them as bool in outputs or as numeric 0/1 (depending on how your ops.* returns).
    """
    shape = (2, 2)
    inputs = [2.0, 3.0]
    #   eq  = (2.0 == 3.0) -> 0
    #   lt  = (2.0 <  3.0) -> 1
    #   le  = (2.0 <= 3.0) -> 1
    #   gt  = (2.0 >  3.0) -> 0
    #   ge  = (2.0 >= 3.0) -> 0
    #   ne  = (2.0 != 3.0) -> 1
    expected = [0, 1, 1, 0, 0, 1]

    sym_a, sym_b = symbol(shape, dtype=DTYPE, device='cuda'), symbol(shape, dtype=DTYPE, device='cuda')

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
def test_ir_float8_e4m3_more_functions():
    """
    Tests a few more unary math ops in one go:
      erf, exp2, sqrt, rsqrt, log, round, floor, ceil
    """
    shape = (2, 2)
    inputs = [0.5, 2.0, 4.0, 4.0, 2.7182818, 1.9, -1.2, 1.2]

    #   o1 = erf(0.5)       ~ 0.520499877
    #   o2 = exp(2.0)       ~ 7.389
    #   o3 = sqrt(4.0)      = 2.0
    #   o4 = rsqrt(4.0)     = 0.5
    #   o5 = log(2.7182818) ~ 1.0 (since input ~ e)
    #   o6 = round(1.9)     = 2.0
    #   o7 = floor(-1.2)    = -2.0
    #   o8 = ceil(1.2)      = 2.0
    expected = [
        0.520499877,  # erf(0.5)
        7.389,  # exp2(2.0)
        2.0,  # sqrt(4.0)
        0.5,  # rsqrt(4.0)
        1.0,  # log(e)
        2.0,  # round(1.9)
        -2.0,  # floor(-1.2)
        2.0,  # ceil(1.2)
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
        assert (
            abs(res[i][0][0].item() - expected[i]) < TOLERANCE
        ), f"res[{i}][0][0].item() = {res[i][0][0].item()} != {expected[i]}"
