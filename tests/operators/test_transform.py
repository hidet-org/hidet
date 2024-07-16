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
from typing import Optional, List
import pytest
import numpy as np
import torch

import hidet
import hidet as hi
from hidet import ops
from hidet import symbol, trace_from
from hidet.ir.utils import broadcast_shape
from hidet.utils import prod
from hidet.graph.tensor import asarray


def check_transform(shape, numpy_op, hidet_op, dtype=np.float32, atol=0, rtol=0):
    # wrap np.array(...) in case shape = []
    data = np.array(np.random.randn(*shape)).astype(dtype)
    numpy_result = numpy_op(data)
    hidet_result = hidet_op(hi.asarray(data).cuda()).cpu().numpy()
    np.testing.assert_allclose(actual=hidet_result, desired=numpy_result, atol=atol, rtol=rtol)


def check_transform_torch(shape, torch_op, hidet_op, dtype=np.float32, atol=0, rtol=0):
    data = torch.asarray(np.array(np.random.randn(*shape)).astype(dtype))
    torch_result = torch_op(data)
    hidet_result = hidet_op(hi.asarray(data).cuda()).cpu().numpy()
    np.testing.assert_allclose(actual=hidet_result, desired=torch_result.cpu().numpy(), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "shape, new_shape",
    [
        [[100, 200, 3], [100, 600]],
        [[123, 321], [321, 123]],
        [[123, 321], [-1, 123]],
        [[123, 321], [123 * 321]],
        [[1, 123, 321, 1, 1], [1, 123, 1, 321, 1]],
        [[1], []],
    ],
)
def test_reshape(shape, new_shape):
    check_transform(shape, lambda x: np.reshape(x, new_shape), lambda x: ops.reshape(x, new_shape))


def test_rearrange():
    # Do not test rearrange separately because there is not a corresponding op in numpy,
    # and rearrange has been tested in squeeze, unsqueeze, flatten, transpose because those operators
    # are special cases of rearrange.
    pass


@pytest.mark.parametrize("shape, dims", [[[1, 3, 1, 4], [0, 2]], [[2, 9, 9, 1], [3]], [[1, 1, 1, 1], [0, 1, 2, 3]]])
def test_squeeze(shape, dims):
    check_transform(shape, lambda x: np.squeeze(x, axis=tuple(dims)), lambda x: ops.squeeze(x, dims))


@pytest.mark.parametrize("shape, dims", [[[3, 4], [0, 2]], [[2, 9, 9], [3]], [[], [0, 1, 2, 3]]])
def test_unsqueeze(shape, dims: List[int]):
    check_transform(shape, lambda x: np.expand_dims(x, dims), lambda x: ops.unsqueeze(x, dims))


@pytest.mark.parametrize(
    "shape, start_dim, end_dim",
    [[[33, 44, 55], 0, -1], [[33, 44, 55], 0, 1], [[33, 44, 55], 0, 2], [[33, 44, 55], 1, 2]],
)
def test_flatten(shape, start_dim: int, end_dim: Optional[int]):
    rank = len(shape)
    if start_dim < 0:
        start_dim += rank
    if end_dim < 0:
        end_dim += len(shape)
    flattened_shape = shape[:start_dim] + [prod(shape[start_dim : end_dim + 1])] + shape[end_dim + 1 :]
    check_transform(shape, lambda x: np.reshape(x, flattened_shape), lambda x: ops.flatten(x, start_dim, end_dim))


@pytest.mark.parametrize(
    "shape, axes",
    [[[33, 44, 55], [0, 1, 2]], [[33, 44, 55], [0, 2, 1]], [[33, 44, 55], [2, 1, 0]], [[33, 44, 55], [1, 2, 0]]],
)
def test_transpose(shape, axes):
    check_transform(shape, lambda x: np.transpose(x, axes), lambda x: ops.transpose(x, axes))


@pytest.mark.parametrize("shape", [[33, 44], [1, 100], [100, 1], [10, 20], [20, 10], [100, 200], [2000, 3000]])
def test_transpose_2d(shape):
    check_transform(shape, lambda x: np.transpose(x), lambda x: ops.transpose(x))


@pytest.mark.parametrize(
    "shapes, dtype, axis",
    [
        [[[33, 44, 55], [1, 44, 55], [32, 44, 55]], 'float32', 0],
        [[[33, 1, 55], [33, 8, 55], [33, 111, 55]], 'float32', 1],
        [[[33, 1, 55], [33, 8, 55], [33, 111, 55]], 'float32', -2],
    ],
)
def test_concat(shapes, dtype, axis):
    data_list = [np.random.randn(*shape).astype(dtype) for shape in shapes]
    numpy_result = np.concatenate(data_list, axis)
    hidet_result = ops.concat([hi.asarray(data).cuda() for data in data_list], axis).cpu().numpy()
    np.testing.assert_allclose(actual=hidet_result, desired=numpy_result, rtol=0, atol=0)


@pytest.mark.parametrize("shape, src_type, dst_type", [[[33, 44, 55], "int64", "float32"]])
def test_cast(shape, src_type, dst_type):
    check_transform(shape, lambda x: x.astype(dst_type), lambda x: ops.cast(x, dst_type), dtype=src_type)


@pytest.mark.parametrize("shape, indices_shape, axis", [[[1234, 512], [128], 0], [[12, 34, 56], [2, 2], 1]])
def test_take(shape, indices_shape, axis):
    dim_extent = shape[axis]
    indices = np.random.randint(0, dim_extent - 1, indices_shape).astype(np.int64)
    check_transform(shape, lambda x: np.take(x, indices, axis), lambda x: ops.take(x, hi.asarray(indices).cuda(), axis))


@pytest.mark.parametrize("shape, indices_shape, axis", [[[1234, 512], [2100, 512], 0], [[12, 34, 56], [12, 1, 56], 1]])
def test_gather(shape, indices_shape, axis):
    dim_extent = shape[axis]
    indices = np.random.randint(0, dim_extent - 1, indices_shape).astype(np.int64)
    check_transform_torch(
        shape,
        lambda x: torch.gather(x, axis, torch.asarray(indices)),
        lambda x: ops.gather(x, hi.asarray(indices).cuda(), axis),
    )


@pytest.mark.parametrize(
    "shape, starts, ends, axes, strides",
    [
        [[100, 100, 100], [0, 0, 0], [10, 20, 30], [0, 1, 2], [1, 1, 1]],
        [[100, 100, 100], [5, 6, 7], [10, 20, 30], [0, 1, 2], [1, 1, 1]],
        [[100, 100, 100], [5, 6, 7], [10, 20, 30], [0, 1, 2], [1, 2, 3]],
    ],
)
def test_strided_slice(shape, starts, ends, axes, strides):
    slice_obj = [slice(None, None) for _ in range(len(shape))]
    for start, end, axis, stride in zip(starts, ends, axes, strides):
        slice_obj[axis] = slice(start, end, stride)
    check_transform(shape, lambda x: x[tuple(slice_obj)], lambda x: ops.strided_slice(x, starts, ends, axes, strides))


@pytest.mark.parametrize(
    "shape, broadcast_shape", [[[1, 1, 1], [33, 44, 55]], [[1, 22, 5], [33, 22, 5]], [[1, 55, 1], [33, 55, 44]]]
)
def test_broadcast(shape, broadcast_shape):
    check_transform(shape, lambda x: x + np.zeros(broadcast_shape), lambda x: ops.broadcast(x, broadcast_shape))


def test_symbolic_broadcast():
    """
    Test broadcasting semantics with symbolic shapes.
    """

    n = hidet.symbol_var("n")
    m = hidet.symbol_var("m")

    # When strict broadcasting check is disabled, pairs of symbolic dimensions are assumed to be equal and no error
    # is raised
    with hidet.option.context():
        hidet.option.debug_strict_broadcast_check(False)
        broadcast_shape([n, m], [m, n])
        broadcast_shape([n], [m, m])

    with hidet.option.context():
        hidet.option.debug_strict_broadcast_check(True)

        # Broadcasting between these shapes with the strict broadcasting check enabled will raise an error
        with pytest.raises(ValueError):
            broadcast_shape([n, m], [m, n])
        with pytest.raises(ValueError):
            broadcast_shape([n], [m, m])

        # If one dimension is 1, the broadcast result takes on the other dimension, even if it is symbolic
        assert broadcast_shape([n, 1], [1, 2]) == [n, 2]
        assert broadcast_shape([1], [n, n]) == [n, n]

        # Pairs of symbolic dimensions don't necessarily have to be the same if one of them can be resolved to 1.
        assert broadcast_shape([m // m], [n]) == [n]
        assert broadcast_shape([n - n + 1], [3]) == [3]

        # In the case where exactly one dimension is symbolic, the symbolic dimension is assumed to be 1.
        assert broadcast_shape([2, 3], [n, n]) == [2, 3]

        # This should never work without further conditions on n and m
        with pytest.raises(ValueError):
            broadcast_shape([n], [m])


def numpy_getitem(data, item):
    return data[item]


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        [[1, 1000], [2, 3]],
        [[16, 1000], [1, 2]],
        [[1, 1000, 1, 1], [10, 2, 3]],
        [[16, 1000, 1, 1], [3, 2, 4, 2]],
        [[1, 128, 128, 128], [2, 2]],
        [[1, 128, 128, 128], [2]],
        [[129], [2]],
        [[129], [2, 3]],
    ],
)
def test_getitem_nd(a_shape, b_shape):
    for device in ['cuda', 'cpu']:
        a = np.array(np.random.randn(*a_shape)).astype('float32')
        b = np.array(np.random.randint(low=0, high=a_shape[0], size=b_shape)).astype('int32')
        atol = 0
        rtol = 0

        numpy_result = numpy_getitem(a, b)
        a = asarray(a).to(device=device)
        b = asarray(b).to(device=device)

        hidet_result = a[b].cpu().numpy()
        np.testing.assert_allclose(actual=hidet_result, desired=numpy_result, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "shapes, indexing",
    [
        [[2, 3], "ij"],
        [[2, 3], "xy"],
        [[2, 3, 4], "ij"],
        [[2, 3, 4], "xy"],
        [[2, 3, 4, 6], "ij"],
        [[2, 3, 4, 6], "xy"],
        [[0, 2], "ij"],
        [[4, 0], "xy"],
        [[6, 0, 2], "ij"],
        [[2, 0, 3, 0], ""],
        [[2, 1], ""],
    ],
)
def test_meshgrid(shapes, indexing):
    dtype = torch.float32
    atol = 0
    rtol = 0
    tensors = []
    for shape in shapes:
        tensors.append(torch.rand([] if shape == 0 else shape, dtype=dtype))
    tensors_hi = [hi.asarray(t).cuda() for t in tensors]
    grid_torch = torch.meshgrid(*tensors, indexing=indexing) if indexing else torch.meshgrid(*tensors)
    grid_hi = ops.meshgrid(*tensors_hi, indexing=indexing) if indexing else ops.meshgrid(*tensors_hi)
    for i in range(len(grid_torch)):
        np.testing.assert_allclose(
            actual=grid_hi[i].cpu().numpy(), desired=grid_torch[i].cpu().numpy(), atol=atol, rtol=rtol
        )


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        [[('a', 1), ('b', 1000)], [('c', 2), ('d', 3)]],
        [[('a', 16), ('b', 1000)], [('c', 1), ('d', 2)]],
        [[('a', 1), ('b', 1000), ('c', 1), ('d', 1)], [('e', 10), ('f', 2), ('g', 3)]],
        [[('a', 16), ('b', 1000), ('c', 1), ('d', 1)], [('e', 3), ('f', 2), ('g', 4), ('h', 2)]],
        [[('a', 1), ('b', 128), ('c', 128), ('d', 128)], [('e', 2), ('f', 2)]],
        [[('a', 1), ('b', 128), ('c', 128), ('d', 128)], [('e', 2)]],
        [[('a', 129)], [('b', 2)]],
        [[('a', 129)], [('b', 2), ('c', 3)]],
    ],
)
def test_getitem_nd_dynamic(a_shape, b_shape):
    for dev in ['cuda', 'cpu']:
        a_concrete_shape = [(i if isinstance(i, int) else i[1]) for i in a_shape]
        a_symbolic_shape = [(i if isinstance(i, int) else i[0]) for i in a_shape]

        b_concrete_shape = [(i if isinstance(i, int) else i[1]) for i in b_shape]
        b_symbolic_shape = [(i if isinstance(i, int) else i[0]) for i in b_shape]
        a = np.array(np.random.randn(*a_concrete_shape)).astype('float32')
        b = np.array(np.random.randint(low=0, high=a_concrete_shape[0], size=b_concrete_shape)).astype('int32')
        numpy_result = a[b]
        a_hidet = asarray(a).to(device=dev)
        b_hidet = asarray(b).to(device=dev)
        sym_a = symbol(a_symbolic_shape, dtype=a_hidet.dtype, device=a_hidet.device)
        sym_b = symbol(b_symbolic_shape, dtype=b_hidet.dtype, device=b_hidet.device)
        sym_result = sym_a[sym_b]

        func = trace_from(sym_result, [sym_a, sym_b])
        hidet_result = func(a_hidet, b_hidet).cpu().numpy()
        np.testing.assert_allclose(actual=hidet_result, desired=numpy_result, atol=0, rtol=0)


@pytest.mark.parametrize(
    "a_shape, b_shape, c_shape",
    [
        [[10, 150], [2, 3], [1, 3]],
        [[16, 130], [1, 2], [1, 1]],
        [[10, 140], [10, 3], [10, 3]],
        [[16, 136], [1, 2], [4, 2]],
        [[128, 128], [2, 2], [2, 2]],
        [[10, 128], [2, 1], [1, 1]],
        [[129, 138], [1, 2], [1, 1]],
        [[129, 138], [2, 3], [2, 3]],
    ],
)
def test_getitem_advanced(a_shape, b_shape, c_shape):
    for device in ['cuda', 'cpu']:
        a = np.array(np.random.randn(*a_shape)).astype('float32')
        b = np.array(np.random.randint(low=0, high=10, size=b_shape)).astype('int32')
        c = np.array(np.random.randint(low=0, high=128, size=c_shape)).astype('int32')
        atol = 0
        rtol = 0

        numpy_result = a[b, c]
        a = asarray(a).to(device=device)
        b = asarray(b).to(device=device)
        c = asarray(c).to(device=device)

        hidet_result = a[b, c].cpu().numpy()
        np.testing.assert_allclose(actual=hidet_result, desired=numpy_result, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "a_shape, b_shape, c_shape",
    [
        [[('a', 10), ('b', 150), 10], [2, 3], [1, 3]],
        [[('a', 16), ('b', 130)], [1, 2], [1, 1]],
        [[('a', 10), ('b', 140), 10], [10, 3], [10, 3]],
        [[('a', 16), ('b', 136), 11, 1], [1, 2], [4, 2]],
        [[('a', 128), ('b', 128)], [2, 2], [2, 2]],
        [[('a', 10), ('b', 128)], [2, 1], [1, 1]],
        [[('a', 129), ('b', 138)], [1, 2], [1, 1]],
        [[('a', 129), ('b', 138)], [2, 3], [2, 3]],
    ],
)
def test_getitem_advanced_dynamic(a_shape, b_shape, c_shape):
    for dev in ['cuda', 'cpu']:
        a_concrete_shape = [(i if isinstance(i, int) else i[1]) for i in a_shape]
        a_symbolic_shape = [(i if isinstance(i, int) else i[0]) for i in a_shape]

        b_concrete_shape = [(i if isinstance(i, int) else i[1]) for i in b_shape]
        b_symbolic_shape = [(i if isinstance(i, int) else i[0]) for i in b_shape]

        c_concrete_shape = [(i if isinstance(i, int) else i[1]) for i in c_shape]
        c_symbolic_shape = [(i if isinstance(i, int) else i[0]) for i in c_shape]

        a = np.array(np.random.randn(*a_concrete_shape)).astype('float32')
        b = np.array(np.random.randint(low=0, high=10, size=b_concrete_shape)).astype('int32')
        c = np.array(np.random.randint(low=0, high=128, size=c_concrete_shape)).astype('int32')

        numpy_result = a[b, c]
        a_hidet = asarray(a).to(device=dev)
        b_hidet = asarray(b).to(device=dev)
        c_hidet = asarray(c).to(device=dev)

        sym_a = symbol(a_symbolic_shape, dtype=a_hidet.dtype, device=a_hidet.device)
        sym_b = symbol(b_symbolic_shape, dtype=b_hidet.dtype, device=b_hidet.device)
        sym_c = symbol(c_symbolic_shape, dtype=c_hidet.dtype, device=c_hidet.device)
        sym_result = sym_a[sym_b, sym_c]

        func = trace_from(sym_result, [sym_a, sym_b, sym_c])
        hidet_result = func(a_hidet, b_hidet, c_hidet).cpu().numpy()
        np.testing.assert_allclose(actual=hidet_result, desired=numpy_result, atol=0, rtol=0)


def test_adv_indexing_with_slices():
    def tests(x, ind1, ind2):
        y1 = x[ind1, :]
        y2 = x[ind1, ...]
        y3 = x[..., ind1, :]
        y4 = x[:, :, ind1, :]

        y5 = x[ind1, :, ind2]
        y6 = x[ind1, ..., ind2]
        y7 = x[..., ind1, :, ind2]
        y8 = x[:, :, ind1, ind2]

        y9 = x[ind1, ind2, :]
        y10 = x[ind1, ind2, ...]
        y11 = x[ind1, ind2, :]
        y12 = x[:, ind2, ind1, :]

        y13 = x[:, :, ind1, :10]
        return [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13]

    x = hidet.randn(shape=(10, 11, 12, 13), dtype='float32', device='cuda')
    ind1 = hidet.randint(low=0, high=10, shape=(1, 1)).cuda()
    ind2 = hidet.randint(low=0, high=10, shape=(1, 2)).cuda()

    outs1 = tests(x, ind1, ind2)

    x, ind1, ind2 = [t.cpu().numpy() for t in [x, ind1, ind2]]

    outs2 = tests(x, ind1, ind2)
    [np.testing.assert_allclose(actual=ho.cpu().numpy(), desired=no) for ho, no in zip(outs1, outs2)]


@pytest.mark.parametrize(
    "input_shape, repeats, dim", [([2, 3, 4], 2, 0), ([1, 2, 9], 3, 1), ([1, 3, 4], 4, 2), ([1, 2, 3], 3, None)]
)
def test_repeat_interleave(input_shape, repeats, dim):
    input_tensor = torch.randn(input_shape)
    input_tensor_hidet = hidet.from_torch(input_tensor)
    output_tensor = torch.repeat_interleave(input_tensor, repeats, dim=dim)
    output_tensor_hidet = ops.repeat_interleave(input_tensor_hidet, repeats, dim=dim)

    np.testing.assert_allclose(output_tensor.numpy(), output_tensor_hidet.numpy(), atol=0, rtol=0)


@pytest.mark.parametrize(
    "shape, kernel_size, dilation, padding, stride",
    [
        ([1, 3, 10, 10], 2, 1, 0, 1),
        ([2, 3, 99, 99], 3, 2, 1, 3),
        ([3, 1, 10, 9], 3, 4, 5, 6),
        ([3, 4, 5, 7], 3, 1, 4, 2),
        ([3, 44, 41], 2, 3, 0, 5),
        ([4, 24, 122], 4, 2, 1, 2),
        ([3, 1, 45, 33], 3, 4, 5, 6),
        ([4, 10, 13], 2, 1, 0, 1),
    ],
)
def test_unfold(shape, kernel_size, dilation, padding, stride):
    check_transform_torch(
        shape,
        lambda x: torch.nn.functional.unfold(x, kernel_size, dilation, padding, stride),
        lambda x: ops.im2col(x, kernel_size, dilation, padding, stride),
    )


if __name__ == '__main__':
    pytest.main([__file__])
