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
import pytest
import torch
import hidet
from hidet.testing import device_to_torch
from hidet.testing.torch_utils import check_module, FunctionalModule


def test_as_torch_tensor():
    """
    test __torch_func__ protocol
    """
    a = hidet.randn([32, 32], dtype='float16')
    b = torch.abs(a)
    c = hidet.ops.abs(a)
    torch.testing.assert_close(b, c.torch())


def test_torch_reshape_tuple_arg(device):
    a = torch.randn(4741, 2, 2)
    func = FunctionalModule(op=lambda x: x.reshape((4741, 4)))
    check_module(func, args=[a], atol=0, rtol=0, device=device)


@pytest.mark.parametrize(
    'input1, input2',
    [
        [torch.tensor([7, -7, 8], dtype=torch.int16), torch.tensor([2, 3, -2], dtype=torch.int32)],
        [torch.tensor([71, -72], dtype=torch.int32), torch.tensor([2.5, -2.0], dtype=torch.float32)],
        [torch.tensor([101, 103], dtype=torch.int64), 55],
        [torch.tensor([101, 103], dtype=torch.int64), 55.0],
        [torch.randn([3, 3, 4]) * 100, torch.randn([3, 1, 1]) * 75],
        [torch.tensor([101, 103], dtype=torch.float32), 66],
    ],
)
def test_torch_div(input1, input2, device):
    input1 = input1.cuda() if isinstance(input1, torch.Tensor) else input1
    input2 = input2.cuda() if isinstance(input2, torch.Tensor) else input2
    func = FunctionalModule(op=lambda x, y: torch.div(x, y))
    func_floor = FunctionalModule(op=lambda x, y: torch.div(x, y, rounding_mode='floor'))
    func_floor_divice = FunctionalModule(op=lambda x, y: torch.floor_divide(x, y))
    func_op_floordiv = FunctionalModule(op=lambda x, y: x // y)
    check_module(func, args=[input1, input2], atol=1e-5, rtol=1e-5, device=device)
    check_module(func_floor, args=[input1, input2], atol=1e-5, rtol=1e-5, device=device)
    check_module(func_floor_divice, args=[input1, input2], atol=1e-5, rtol=1e-5, device=device)
    check_module(func_op_floordiv, args=[input1, input2], atol=1e-5, rtol=1e-5, device=device)


@pytest.mark.parametrize('shape,expanded_shape', [([2, 1], [2, 11]), ([2, 3, 4], [2, 3, 4]), ([1], [6])])
def test_expand_as(shape, expanded_shape, device):
    check_module(
        FunctionalModule(op=lambda x, y: x.expand_as(y)),
        args=[torch.randn(shape), torch.randn(expanded_shape)],
        atol=0,
        rtol=0,
        device=device,
    )


@pytest.mark.parametrize('shape, new_shape', [[[2, 3, 4], [6, 4]], [[2, 3, 4], [12, 2]]])
def test_view_as(shape, new_shape, device):
    check_module(
        FunctionalModule(op=lambda x: x.view_as(torch.randn(new_shape))),
        args=[torch.randn(shape)],
        atol=0,
        rtol=0,
        device=device,
    )


@pytest.mark.parametrize('shape', [[2, 3]])
def test_tensor_sigmod(shape, device):
    check_module(
        FunctionalModule(op=lambda x: x.sigmoid_()), args=[torch.randn(shape)], atol=1e-5, rtol=1e-5, device=device
    )


@pytest.mark.parametrize(
    'shape,src_shape',
    [
        ([2, 3, 4], [2, 3, 4]),
        ([2, 3, 4], [2, 3, 1]),
        ([2, 3, 4], [2, 1, 1]),
        ([2, 3, 4], [1, 1]),
        ([2, 3, 4], [1]),
        ([5, 3, 4, 1], [3, 1, 1]),
        ([5, 3, 4, 1], [4, 1]),
    ],
)
def test_torch_copy(shape, src_shape, device):
    check_module(
        FunctionalModule(op=lambda x, y: x.copy_(y)),
        args=[torch.randn(shape), torch.randn(src_shape)],
        atol=0,
        rtol=0,
        device=device,
    )


@pytest.mark.parametrize(
    'shape, repeats, dim',
    [
        ([2, 3, 4], 2, 1),
        ([2, 3, 4], 3, 2),
        ([2, 3, 4], 4, 0),
        ([2, 3, 4], 1, None),
        ([2, 3, 4], 3, None),
        ([2, 3, 4], 1, 2),
        ([3], 3, None),
        ([4], 1, None),
    ],
)
def test_torch_repeat_interleave(shape, repeats, dim, device):
    check_module(
        FunctionalModule(op=lambda x: x.repeat_interleave(repeats, dim)),
        args=[torch.randn(shape)],
        atol=0,
        rtol=0,
        device=device,
    )


@pytest.mark.parametrize('shape, dim', [([2, 3, 4], 0), ([2, 3, 4], 1), ([2, 3, 1], 2), ([1], 0), ([3, 1], 1)])
def test_torch_unbind(shape, dim, device):
    check_module(FunctionalModule(op=lambda x: x.unbind(dim)), args=[torch.randn(shape)], atol=0, rtol=0, device=device)


@pytest.mark.parametrize('shape, dim', [([2, 3, 4], 0), ([2, 3, 4], 1), ([2, 3, 1], 2), ([1], 0), ([3, 1], 1)])
def test_torch_var(shape, dim, device):
    check_module(
        FunctionalModule(op=lambda x: torch.var(x, dim=dim)),
        args=[torch.randn(shape)],
        atol=1e-5,
        rtol=1e-5,
        device=device,
    )
    check_module(
        FunctionalModule(op=lambda x: torch.var(x, dim=dim, unbiased=False)),
        args=[torch.randn(shape)],
        atol=1e-5,
        rtol=1e-5,
        device=device,
    )
    check_module(
        FunctionalModule(op=lambda x: torch.var(x, dim=dim, correction=0)),
        args=[torch.randn(shape)],
        atol=1e-5,
        rtol=1e-5,
        device=device,
    )


@pytest.mark.parametrize('embed_dim', [512])
@pytest.mark.parametrize('num_heads', [8])
@pytest.mark.parametrize('batch_first', [False, True])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('target_len, src_len', [[77, 77]])
@pytest.mark.parametrize('have_mask', [True])
@pytest.mark.parametrize('is_causal', [False])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_torch_multihead_attention(
    embed_dim, num_heads, batch_first, batch_size, target_len, src_len, have_mask, is_causal, dtype, device
):
    torch_attention = torch.nn.MultiheadAttention(
        embed_dim, num_heads, batch_first=batch_first, device='cuda', dtype=dtype
    )
    query_shape = [target_len, batch_size, embed_dim] if not batch_first else [batch_size, target_len, embed_dim]

    query = torch.randn(query_shape, dtype=dtype, device='cuda')
    key = query
    value = query

    if have_mask:
        mask = torch.full((target_len, src_len), float('-inf'), dtype=dtype, device='cuda').triu(1)
    else:
        mask = None
    if not have_mask:
        is_causal = False

        # same as above, but just check the first element in the output tuple
    check_module(
        model=FunctionalModule(op=lambda *args: torch_attention(*args)[0]),
        args=[query, key, value, None, False, mask, False, is_causal],
        atol=1e-2,
        rtol=1e-2,
        device=device,
    )


@pytest.mark.parametrize('d_model', [512])
@pytest.mark.parametrize('nhead', [8])
@pytest.mark.parametrize('dim_feedforward', [2048])
@pytest.mark.parametrize('dropout', [0.0])
@pytest.mark.parametrize('activation', [torch.nn.functional.relu])
@pytest.mark.parametrize('batch_first', [False])
@pytest.mark.parametrize('norm_first', [True])
@pytest.mark.parametrize('src_shape', [[77, 32, 512]])
@pytest.mark.parametrize('need_mask', [True])
@pytest.mark.parametrize('mask_shape', [[77, 77]])
@pytest.mark.parametrize('is_causal', [True])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_torch_transformer_encoder(
    d_model,
    nhead,
    dim_feedforward,
    dropout,
    activation,
    batch_first,
    norm_first,
    src_shape,
    need_mask,
    mask_shape,
    is_causal,
    dtype,
    device,
):
    torch_device = device_to_torch(device)
    torch_layer = torch.nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        batch_first=batch_first,
        norm_first=norm_first,
        device=torch_device,
        dtype=dtype,
    )

    src = torch.randn(src_shape, dtype=dtype, device=torch_device)
    mask = torch.full(mask_shape, float('-inf'), dtype=dtype, device=torch_device).triu(1) if need_mask else None

    if not need_mask:
        is_causal = False

    torch_encoder = torch.nn.TransformerEncoder(torch_layer, num_layers=12)

    # Change the atol to 5e-2 since the test is quite flaky here...
    # for atol=1e-2 sometimes the test fails with way less than 1% of mismatch
    check_module(model=torch_encoder, args=[src, mask, None, is_causal], atol=5e-2, rtol=1e-2, device=device)


@pytest.mark.parametrize('shape', [[2, 3]])
def test_bitwise_and(shape, device):
    a = torch.randint(low=0, high=10, size=shape, device='cuda')
    b = torch.randint(low=0, high=10, size=shape, device='cuda')
    check_module(FunctionalModule(op=lambda x, y: x & y), args=[a, b], atol=0, rtol=0, device=device)
    check_module(FunctionalModule(op=lambda x, y: torch.bitwise_and(x, y)), args=[a, b], atol=0, rtol=0, device=device)
    check_module(FunctionalModule(op=lambda x, y: x.bitwise_and(y)), args=[a, b], atol=0, rtol=0, device=device)


@pytest.mark.parametrize('shape', [[2, 3]])
def test_bitwise_or(shape, device):
    a = torch.randint(low=0, high=10, size=shape, device='cuda')
    b = torch.randint(low=0, high=10, size=shape, device='cuda')
    check_module(FunctionalModule(op=lambda x, y: x | y), args=[a, b], atol=0, rtol=0, device=device)
    check_module(FunctionalModule(op=lambda x, y: torch.bitwise_or(x, y)), args=[a, b], atol=0, rtol=0, device=device)
    check_module(FunctionalModule(op=lambda x, y: x.bitwise_or(y)), args=[a, b], atol=0, rtol=0, device=device)


@pytest.mark.parametrize('shape', [[2, 3]])
def test_logical_and(shape, device):
    a = torch.randint(low=0, high=2, size=shape, device='cuda')
    b = torch.randint(low=0, high=10, size=shape, device='cuda')

    check_module(FunctionalModule(op=lambda x, y: torch.logical_and(x, y)), args=[a, b], atol=0, rtol=0, device=device)

    check_module(FunctionalModule(op=lambda x, y: x.logical_and(y)), args=[a, b], atol=0, rtol=0, device=device)


@pytest.mark.parametrize('shape', [[2, 3]])
def test_logical_or(shape, device):
    a = torch.randint(low=0, high=2, size=shape, device='cuda')
    b = torch.randint(low=0, high=10, size=shape, device='cuda')

    check_module(FunctionalModule(op=lambda x, y: torch.logical_or(x, y)), args=[a, b], atol=0, rtol=0, device=device)

    check_module(FunctionalModule(op=lambda x, y: x.logical_or(y)), args=[a, b], atol=0, rtol=0, device=device)


@pytest.mark.parametrize('shape', [[2, 3]])
def test_bitwise_not(shape, device):
    a = torch.randint(low=0, high=10, size=shape, device='cuda')
    check_module(FunctionalModule(op=lambda x: ~x), args=[a], atol=0, rtol=0, device=device)

    check_module(FunctionalModule(op=lambda x: torch.bitwise_not(x)), args=[a], atol=0, rtol=0, device=device)

    check_module(FunctionalModule(op=lambda x: x.bitwise_not()), args=[a], atol=0, rtol=0, device=device)


@pytest.mark.parametrize('shape', [[2, 3]])
def test_logical_not(shape, device):
    a = torch.randint(low=0, high=2, size=shape, device='cuda')

    check_module(FunctionalModule(op=lambda x: torch.logical_not(x)), args=[a], atol=0, rtol=0, device=device)

    check_module(FunctionalModule(op=lambda x: x.logical_not()), args=[a], atol=0, rtol=0, device=device)


@pytest.mark.parametrize('shape', [[2, 3]])
def test_bitwise_xor(shape, device):
    a = torch.randint(low=0, high=10, size=shape, device='cuda')
    b = torch.randint(low=0, high=10, size=shape, device='cuda')
    check_module(FunctionalModule(op=lambda x, y: x ^ y), args=[a, b], atol=0, rtol=0, device=device)

    check_module(FunctionalModule(op=lambda x, y: torch.bitwise_xor(x, y)), args=[a, b], atol=0, rtol=0, device=device)

    check_module(FunctionalModule(op=lambda x, y: x.bitwise_xor(y)), args=[a, b], atol=0, rtol=0, device=device)


@pytest.mark.parametrize('shape', [[2, 3]])
def test_logical_xor(shape, device):
    a = torch.randint(low=0, high=2, size=shape, device='cuda')
    b = torch.randint(low=0, high=10, size=shape, device='cuda')

    check_module(FunctionalModule(op=lambda x, y: torch.logical_xor(x, y)), args=[a, b], atol=0, rtol=0, device=device)

    check_module(FunctionalModule(op=lambda x, y: x.logical_xor(y)), args=[a, b], atol=0, rtol=0, device=device)


def test_tensor_not(device):
    torch_device = device_to_torch(device)
    a = torch.tensor(3, device=torch_device)
    b = torch.tensor([0], device=torch_device)

    func = lambda x: not x
    func_compiled = torch.compile(func, backend='hidet', mode=None)

    # Cannot use check_module since the output is not a tensor
    assert not func_compiled(a)
    assert func_compiled(b)


@pytest.mark.parametrize('shape, negative_slope', [([2, 2, 2], 0.1), ([2], 0.9)])
def test_torch_leaky_relu(shape, negative_slope, device):
    a = torch.randn(shape, device='cuda')
    check_module(
        FunctionalModule(op=lambda x: torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)),
        args=[a],
        atol=1e-5,
        rtol=1e-5,
        device=device,
    )

    leaky_relu_mod = torch.nn.LeakyReLU(negative_slope=negative_slope)
    check_module(leaky_relu_mod, args=[a], atol=1e-5, rtol=1e-5, device=device)


@pytest.mark.parametrize('shape', [[2, 3], [1, 1], [400, 1]])
def test_torch_T(shape, device):
    a = torch.randint(low=0, high=10, size=shape, device='cuda')
    # The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release.
    check_module(FunctionalModule(op=lambda x: x.T), args=[a], atol=0, rtol=0, device=device)


@pytest.mark.parametrize('shape', [[2, 3], [4, 9, 77, 5]])
def test_torch_new_empty(shape, device):
    a: torch.Tensor = torch.randint(low=0, high=10, size=shape, device='cuda')
    check_module(FunctionalModule(op=lambda x: x.new_empty((3, 2, 1))), args=[a], atol=0, rtol=0, device=device)
    check_module(
        FunctionalModule(op=lambda x: x.new_empty(3, 2, 1, device=device)), args=[a], atol=0, rtol=0, device=device
    )


@pytest.mark.parametrize(
    'input_shape, offsets_shape, weight_shape', [([223], [156], [333, 444]), ([26], [1], [33, 33])]
)
@pytest.mark.parametrize('mode_name', ['mean', 'sum'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_torch_embedding_bag(input_shape, offsets_shape, weight_shape, mode_name, dtype, device):

    input_data = torch.randint(0, weight_shape[0], input_shape, dtype=torch.int64)
    offsets_data = torch.randint(0, input_shape[0], offsets_shape, dtype=torch.int64)
    offsets_data, _ = torch.sort(offsets_data)
    offsets_data = torch.unique(offsets_data, sorted=True)

    weight_data = torch.randn(weight_shape, dtype=dtype)

    torch_embedding_bag_module = torch.nn.EmbeddingBag.from_pretrained(weight_data, mode=mode_name)

    atol = 1e-4 if dtype == torch.float32 else 1e-2

    check_module(torch_embedding_bag_module, args=[input_data, offsets_data], atol=atol, rtol=1e-4, device=device)


@pytest.mark.parametrize(
    "equation, operand_shapes",
    [
        ["bhwc,hkc->bhwk", [[400, 14, 14, 80], [14, 14, 80]]],
        ["bhwc,wkc->bhwk", [[400, 14, 14, 80], [14, 14, 80]]],
        ["abcd,cd->ab", [[10, 20, 30, 40], [30, 40]]],
        ["i, j -> ij", [[160], [26]]],
        ["..., f -> ... f", [[160], [26]]],
        ["...ij, j -> ...i", [[5, 5, 10, 20], [20]]],
        ["...ij, ...ij -> ...", [[5, 10, 15], [5, 10, 15]]],
    ],
)
def test_torch_einsum(equation, operand_shapes, device):
    operands_torch = [torch.randn(shape, device='cuda') for shape in operand_shapes]

    atol = 5e-2
    if equation == 'abcd,cd->ab':
        atol = 1e-1

    check_module(
        FunctionalModule(op=lambda *args: torch.einsum(equation, *args)),
        args=operands_torch,
        atol=atol,
        rtol=1e-4,
        device=device,
    )


def test_scatter_add_compile(device):
    # This operator was already tested in tests/operators/test_inplace_operator.py.
    # Just to add one more additional test here to ensure the bug mentioned in #429 is gone.
    if device != 'cuda':
        pytest.skip('This test is only for CUDA')
    input_tensor = torch.zeros((6, 6), dtype=torch.float32, device='cuda')

    index_tensor = torch.tensor([[4, 1, 4, 4, 2], [0, 1, 4, 5, 2], [5, 1, 3, 4, 2]]).to(dtype=torch.int64).cuda()

    input_tensor_clone = input_tensor.clone()
    src = torch.tensor([[0, 5, 3, 6, 5], [9, 6, 8, 8, 4], [7, 4, 5, 4, 7]]).to(dtype=torch.float32).cuda()

    dim = 1

    check_module(
        FunctionalModule(op=lambda x, y, z: x.scatter_add_(dim, y, z)),
        args=[input_tensor_clone, index_tensor, src],
        atol=0,
        rtol=0,
        dynamic=False,
        device=device,
    )


def test_linear_odd_dims(device):
    # As mentioned in #509, the we cannot use `matmul_nt` when the n or k dimension is odd.
    # This test is to ensure those corner cases are handled correctly.
    if device != 'cuda':
        pytest.skip('This test is only for CUDA')
    x = torch.randn(10, 20, device='cuda', dtype=torch.half)
    w = torch.randn(1, 20, device='cuda', dtype=torch.half)

    check_module(
        FunctionalModule(op=lambda x, w: torch.nn.functional.linear(x, w)),
        args=[x, w],
        atol=1e-2,
        rtol=1e-2,
        dynamic=False,
        device=device,
    )


if __name__ == '__main__':
    pytest.main([__file__])
