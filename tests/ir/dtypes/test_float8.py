import pytest
import tabulate
import torch
import hidet
import math


def float8_e4m3_to_float(n: int) -> float:
    """
    Cast float8_e4m3 to float64 (python's float use 64-bit double precision)

    Parameters
    ----------
    n: int
        The float8_e4m3 number stored as an uint8 integer

    Returns
    -------
    ret: float
        The float value castted from the float8_e4m3 number
    """
    sign = -1 if n & 0b10000000 else 1
    n = n & 0b01111111
    exponent = (n >> 3) & 0b1111
    mantissa = n & 0b111
    if exponent == 0b1111 and mantissa == 0b111:
        return float('NaN')
    elif exponent > 0:
        return sign * (0b1000 + mantissa) * 2 ** (exponent - 10)
    else:
        return sign * mantissa * 2 ** (-9)


def float8_e5m2_to_float(n: int) -> float:
    """
    Cast float8_e5m2 to float64 (python's float use 64-bit double precision)

    Parameters
    ----------
    n: int
        The float8_e5m2 number stored as an uint8 integer

    Returns
    -------
    ret: float
        The float value castted from the float8_e5m2 number
    """
    sign = -1 if n & 0b10000000 else 1
    exponent = (n >> 2) & 0b11111
    mantissa = n & 0b11
    if exponent == 0b11111:
        if mantissa == 0b11:
            return sign * float('Inf')
        else:
            return sign * float('NaN')
    elif exponent > 0:
        return sign * (0b100 + mantissa) * 2 ** (exponent - 17)
    else:
        return sign * mantissa * 2 ** (-16)


float8_e4m3_table = [float8_e4m3_to_float(n) for n in range(256)]


@pytest.mark.requires_cuda
def test_float8_e4m3_from_float32():
    from hidet.lang import attrs
    from hidet.lang.types import float32, float8_e4m3

    with hidet.script_module() as script_module:

        @hidet.script
        def float32_to_float8_e4m3(dst: ~float8_e4m3, src: ~float32):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 1

            dst[0] = float8_e4m3(src[0])

    func = script_module.build()

    rows = []
    for i in range(256):
        src = torch.asarray([float8_e4m3_table[i]], dtype=torch.float32, device='cuda')
        dst = torch.empty([1], dtype=torch.float8_e4m3fn, device='cuda')
        func(dst, src)
        actual = dst.to(torch.float32).item()  # use pytorch's conversion as the ground truth
        desire = float8_e4m3_table[i]
        identical = actual == desire or (math.isnan(actual) and math.isnan(desire))
        rows.append([i, actual, desire, identical])
    headers = ['i', 'actual', 'desire', 'identical']
    print(tabulate.tabulate(rows, headers=headers))
    assert all(row[-1] for row in rows)


@pytest.mark.requires_cuda
def test_float8_e4m3_to_float32():
    from hidet.lang import attrs
    from hidet.lang.types import float32, float8_e4m3

    with hidet.script_module() as script_module:

        @hidet.script
        def float8_to_float32_e4m3(dst: ~float32, src: ~float8_e4m3):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 1

            dst[0] = float32(src[0])

    func = script_module.build()

    rows = []
    for i in range(256):
        src_uint8 = torch.asarray([i], dtype=torch.uint8, device='cuda')
        src = src_uint8.view(torch.float8_e4m3fn)
        dst = torch.empty([1], dtype=torch.float32, device='cuda')
        func(dst, src)
        actual = dst.item()
        desire = float8_e4m3_table[i]
        identical = actual == desire or (math.isnan(actual) and math.isnan(desire))
        rows.append([i, actual, desire, identical])
    headers = ['i', 'actual', 'desire', 'identical']
    print(tabulate.tabulate(rows, headers=headers))
    assert all(row[-1] for row in rows)
