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
from typing import List, Sequence, Tuple
from hidet.graph.ops import permute_dims, reshape
from .utils import Tensor
from .matmul import matmul


def multiply_sum(left: Tensor, right: Tensor, sum_dims: Sequence[int], keepdim: bool = False) -> Tensor:
    # computes (left * right).sum(sum_dims) by means of permutation and batched matmul
    if len(sum_dims) == 0:
        return matmul(left, right)

    assert len(left.shape) == len(
        right.shape
    ), "sumproduct_pair: The number of dimensions of the two operands must match"
    dim = len(left.shape)

    # lro: dimensions that appear in left, right and output
    # similarly: lo: left and output, ro: right and output
    # also we need to keep track of relevant sizes for reshaping
    lro, lo, ro = [], [], []
    lro_size, lo_size, ro_size, sum_size = 1, 1, 1, 1

    for i in range(dim):
        sl = left.shape[i] != 1
        sr = right.shape[i] != 1
        if i in sum_dims:
            if sl and sr:
                assert left.shape[i] == right.shape[i], "Non-broadcast dimensions must match"
                sum_size *= left.shape[i]
            elif sl:
                raise NotImplementedError("sl and !sr: should only happen with more than 2 operands passed to einsum")
            elif sr:
                raise NotImplementedError("!sl and sr: should only happen with more than 2 operands passed to einsum")
        elif sl and sr:
            assert left.shape[i] == right.shape[i], "Non-broadcast dimensions must match"
            lro.append(i)
            lro_size *= left.shape[i]
        elif sl:
            lo.append(i)
            lo_size *= left.shape[i]
        elif sr:
            ro.append(i)
            ro_size *= right.shape[i]
        else:
            assert False, "This should never be reached"

    out_ndims = len(lro) + len(lo) + len(sum_dims) + len(ro)
    out_shape = []

    # prepare to permute the inputs & outputs
    for dimension in lro:
        out_shape.append(left.shape[dimension])
    for dimension in lo:
        out_shape.append(left.shape[dimension])
    for dimension in sum_dims:
        out_shape.append(1)
    for dimension in ro:
        out_shape.append(right.shape[dimension])

    left_perm = lro[:]
    left_perm.extend(lo)
    left_perm.extend(sum_dims)
    left_perm.extend(ro)

    right_perm = lro[:]
    right_perm.extend(sum_dims)
    right_perm.extend(ro)
    right_perm.extend(lo)

    output_perm = [-1] * out_ndims

    idx = 0
    for i in lro:
        output_perm[i] = idx
        idx += 1
    for i in lo:
        output_perm[i] = idx
        idx += 1
    for i in sum_dims:
        output_perm[i] = idx
        idx += 1
    for i in ro:
        output_perm[i] = idx
        idx += 1

    left = permute_dims(left, left_perm)
    left = reshape(left, [lro_size, lo_size, sum_size])
    right = permute_dims(right, right_perm)
    right = reshape(right, [lro_size, sum_size, ro_size])

    result = matmul(left, right)
    result = reshape(result, out_shape)
    result = permute_dims(result, output_perm)

    del left
    del right

    # squeeze the dimensions that were summed over if needed
    if not keepdim:
        rst_shape = result.shape
        new_shape = []
        for i in range(len(rst_shape)):
            if i not in sum_dims:
                new_shape.append(rst_shape[i])
        result = reshape(result, new_shape)

    return result


def einsum_different_ranks(lhs: str, rhs: str, operands: Sequence[Tensor]):
    num_ops = len(operands)

    NLETTERS = 26
    NLABELS = NLETTERS * 2
    ELLIPSIS_ID = NLABELS

    def label_to_subscript(label: str) -> int:
        if label.isupper():
            return ord(label) - ord('A')
        else:
            return ord(label) - ord('a') + NLETTERS

    def subscript_to_label(subscript: int) -> str:
        if subscript < NLETTERS:
            return chr(subscript + ord('A'))
        else:
            return chr(subscript - NLETTERS + ord('a'))

    op_labels = [[] for _ in range(num_ops)]
    i = 0
    curr_op = 0  # the current operand we are processing

    ellipsis_met = False

    while i < len(lhs):
        label = lhs[i]
        if label == ',':
            # move to the next operand
            curr_op += 1
            ellipsis_met = False
            assert curr_op < num_ops, "einsum: the equation specifies more operands than provided"
        elif label.isalpha():
            op_labels[curr_op].append(label_to_subscript(label))
        elif label == '.':
            # raise NotImplementedError("einsum: ellipsis not supported in hidet yet")
            assert not ellipsis_met, "einsum: only one ellipsis is allowed for each operand"
            assert lhs[i + 1] == '.' and lhs[i + 2] == '.', "einsum: found '.' that is not part of an ellipsis"
            ellipsis_met = True
            op_labels[curr_op].append(ELLIPSIS_ID)
            i += 2
        else:
            raise ValueError(f"einsum: invalid character {label} in equation")
        i += 1

    assert curr_op == num_ops - 1, "einsum: the equation specifies fewer operands than provided"
    label_count = [0 for _ in range(NLABELS)]

    # max number of dims covered by the ellipsis
    ellipsis_ndims = 0

    for idx in range(num_ops):
        operand = operands[idx]
        labels = op_labels[idx]
        ndims = len(operand.shape)

        nlabels = len(labels)

        have_ellipsis = False

        for label in labels:
            if label == ELLIPSIS_ID:
                have_ellipsis = True
                nlabels -= 1
                ellipsis_ndims = max(ellipsis_ndims, ndims - nlabels)
            else:
                label_count[label] += 1

        if have_ellipsis:
            assert ellipsis_ndims <= ndims, "einsum: ellipsis covers more dimensions than the operand has"
        else:
            assert (
                nlabels == ndims
            ), "einsum: without ellipsis,  the number of labels must match the number of dimensions in the operand"

    # The mapping from labels to the index into the finally permuted shape
    label_perm_map = [-1 for _ in range(NLABELS)]
    perm_idx = 0

    # starting index of dims covered by ellipsis, in the shape after permutation
    ellipsis_idx = 0
    ellipsis_in_output = False

    # parsing the output
    i = 0
    while i < len(rhs):
        label = rhs[i]
        if label.isalpha():
            idx_label = label_to_subscript(label)
            if label_count[idx_label] == 0:
                raise ValueError(f"einsum: output label {label} does not appear in any input")
            if label_perm_map[idx_label] != -1:
                raise ValueError(f"einsum: label {label} appears more than once in the output")
            label_perm_map[idx_label] = perm_idx
            perm_idx += 1
        elif label == '.':
            assert not ellipsis_in_output, "einsum: only one ellipsis is allowed in the output"
            assert (
                i + 2 < len(rhs) and rhs[i + 1] == '.' and rhs[i + 2] == '.'
            ), "einsum: found '.' that is not part of an ellipsis"
            i += 2
            ellipsis_idx = perm_idx
            perm_idx += ellipsis_ndims
            ellipsis_in_output = True
        else:
            raise ValueError(f"einsum: invalid character {label} in the output of the equation")
        i += 1

    out_ndims = perm_idx

    # if ellipsis is not part of the output, we need to add the dimensions covered by it
    if not ellipsis_in_output:
        ellipsis_idx = perm_idx
        perm_idx += ellipsis_ndims

    for ilabel in range(NLABELS):
        if label_count[ilabel] > 0 and label_perm_map[ilabel] == -1:
            label_perm_map[ilabel] = perm_idx
            perm_idx += 1

    # Next: we check the size, unsqueeze the missing dimensions in each input operand,
    # permute their dimensions to align them before multiplying & summing
    # over specified dimensions(with sumproduct_pair),
    # and finally permute the dimensions of the result to match the output
    label_size = [1 for _ in range(NLABELS)]

    ellipsis_sizes = [1 for _ in range(ellipsis_ndims)]

    dim_counts = [0 for _ in range(perm_idx)]

    updated_operands = []

    for idx_op in range(num_ops):
        op = operands[idx_op]
        permutation = [-1 for _ in range(perm_idx)]
        dim = 0
        for s in op_labels[idx_op]:
            # Here we assume that `s` cannot be Ellipsis for now --- NO LONGER HOLDS!

            if s == ELLIPSIS_ID:
                # go through the dimensions covered by the ellipsis
                op_ndims = len(op.shape)
                ndim = op_ndims - (len(op_labels[idx_op]) - 1)
                j = ellipsis_ndims - ndim
                while j < ellipsis_ndims:
                    if op.shape[dim] != 1:
                        assert (
                            ellipsis_sizes[j] == 1 or ellipsis_sizes[j] == op.shape[dim]
                        ), f"einsum: dimension {dim} covered by ellipsis in operand {idx_op} has mismatched size"
                        ellipsis_sizes[j] = op.shape[dim]
                        dim_counts[ellipsis_idx + j] += 1
                    permutation[ellipsis_idx + j] = dim
                    dim += 1
                    j += 1

            elif permutation[label_perm_map[s]] == -1:
                if op.shape[dim] != 1:
                    assert (
                        label_size[s] == 1 or label_size[s] == op.shape[dim]
                    ), f"einsum: mismatched sizes for label {subscript_to_label(s)}"
                    label_size[s] = op.shape[dim]
                    dim_counts[label_perm_map[s]] += 1
                permutation[label_perm_map[s]] = dim
                dim += 1
            else:
                # TODO: repeated label: need to take diagonal(as per the PyTorch einsum documentation),
                # not supported for the time being to save some time
                raise NotImplementedError("einsum: repeated labels not supported yet")

        for i, val in enumerate(permutation):
            if val == -1:
                op = op.unsqueeze(dim)
                permutation[i] = dim
                dim += 1
        updated_operands.append(permute_dims(op, permutation))

    while len(updated_operands) > 1:
        i, j = 0, 1
        a = updated_operands[i]
        b = updated_operands[j]
        updated_operands.pop(j)
        updated_operands.pop(i)

        # dims over which summation will be performed
        # those are the dimensions that are not 1 in both a and b, and does not appear in the output
        sum_dims = []

        a_shape, b_shape = a.shape, b.shape
        for dim in range(out_ndims, perm_idx):
            if a_shape[dim] != 1 and b_shape[dim] != 1:
                dim_counts[dim] -= 1
                if dim_counts[dim] == 1:
                    sum_dims.append(dim)
                    dim_counts[dim] = 0
            elif dim_counts[dim] == 1:
                if a_shape[dim] != 1:
                    raise NotImplementedError(
                        "einsum this should only happen with more than 2 operands passed to einsum"
                    )
                elif b_shape[dim] != 1:
                    raise NotImplementedError(
                        "einsum: this should only happen with more than 2 operands passed to einsum"
                    )

        updated_operands.insert(0, multiply_sum(a, b, sum_dims, keepdim=True))

    # elimate the dimensions that were summed over
    if perm_idx > out_ndims:
        # For the time being, we can safely assume that num_ops > 1,
        # as we only support einsum with 2 operands; and the same code
        # can also be used for >2 operands.
        # TODO: but it might change in the future, as single-operand einsum is indeed supported in PyTorch
        rst = updated_operands[0]
        rst_shape = rst.shape
        new_shape = rst_shape[:out_ndims]
        return reshape(rst, new_shape)
    else:
        return updated_operands[0]


# ToDo: Actually fully implement einsum, supporting same usage as Numpy and Torch
# For the time being, the cases not supported:
# 1. repeated labels in the operands
# 2. More than 2 operands
# 3. Equations without '->'

# Do ad-hoc pattern matching: only support simple cases such as matrix multiply
def einsum(equation: str, operands: Sequence[Tensor]):
    if isinstance(operands[0], (Tuple, List)):
        operands = operands[0]
    if len(operands) != 2:
        raise NotImplementedError('einsum currently only supports 2 operands')

    a = operands[0]
    b = operands[1]
    equation = equation.replace(' ', '')

    if '->' not in equation:
        raise NotImplementedError(
            "einsum currently only supports equations with explicit output dimensions specified by '->'"
        )
    lhs, rhs = equation.split('->')
    a_subs, b_subs = lhs.split(',')

    if len(rhs) != len(a_subs) or len(a_subs) != len(b_subs):
        return einsum_different_ranks(lhs, rhs, operands)

    a_batch, a_dims = a_subs[:-2], a_subs[-2:]
    b_batch, b_dims = b_subs[:-2], b_subs[-2:]
    c_batch, c_dims = rhs[:-2], rhs[-2:]

    if a_batch != b_batch or a_batch != c_batch:
        raise NotImplementedError('einsum currently only supports batched matmul')

    if a_dims[1] == b_dims[0]:
        c = matmul(a, b)
    elif a_dims[1] == b_dims[1]:
        c = matmul(a, b.transpose(-1, -2))
    elif a_dims[0] == b_dims[0]:
        c = matmul(a.transpose(-1, -2), b)
    elif a_dims[0] == b_dims[1]:
        c = matmul(a.transpose(-1, -2), b.transpose(-1, -2))
    else:
        raise NotImplementedError('einsum currently only supports batched matmul')

    transpose_c = (c_dims[0] == b_dims[0] or c_dims[0] == b_dims[1]) and (
        c_dims[1] == a_dims[0] or c_dims[1] == a_dims[1]
    )

    if transpose_c:
        return c.transpose(-1, -2)
    else:
        return c
