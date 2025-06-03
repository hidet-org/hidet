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
from typing import Tuple, List, Optional

from hidet.ir.expr import is_constant, var

from hidet.ir.cute import shape_div, common_reshape, group, coalesce, make_layout, TensorLayout


def get_last_dim_strides(tile_shape: Tuple[int, ...], global_layout: TensorLayout):
    """
    Example and explanation for TMA tensor dimension calculation:

    Suppose we have a global tensor with layout ((64, 4), (64, 4)): ((1, 4096), (64, 16384)).
    Each parenthesis corresponds to a dimension of the global tensor, so this tensor is 2D with shape (256, 256).
    If we tile this tensor with tile shape (64, 64), what is the number of dimensions of the TMA tensor?

    The answer is 4. Explanation:
    - The first dimension (256) is split into (64, 4) by tiling, but the resulting sub-dimensions are not contiguous
      (i.e., their strides do not allow merging into a single dimension). Thus, both sub-dimensions are kept, and
      the last dimension stride (from (1, 4096)) is added to indicate how to access the next tile in this dimension.
    - The same logic applies to the second dimension.
    - As a result, the TMA tensor has 4 dimensions, and we add the last dimension strides to each dimension.

    In contrast, if the global layout is (256, 256): (256, 1) and we split the first dimension into (64, 4),
    the resulting sub-dimensions are contiguous (their strides allow merging). In this case, there is no need to
    add last dimension strides, and the TMA tensor does not gain extra dimensions.

    Summary:
    - If tiling splits a dimension into non-contiguous sub-dimensions, add last dimension strides for TMA tensor.
    - If sub-dimensions are contiguous, do not add extra strides or dimensions.

    I have another case, there won't be any last dimension strides, for example, the global layout is
    (256, 256): (256, 1). If we split the first dimension into (64, 4), then the strides for the two
    sub-dimensions are contiguous. So in this case, there won't be any additional dimension accounted for
    the TMA tensor. So we don't add last dimension strides to the TMA tensor.

    Parameters:
        tile_shape: The shape of the tile.
        global_layout: The layout of the global tensor.

    Returns:
        A list of last dimension strides for each dimension after tiling. If a dimension is split into non-contiguous
        sub-dimensions, the stride for the last sub-dimension is included; otherwise, None is used. Returns None if
        the shape or tiling is not supported for TMA.
    """
    last_dim_strides = []
    for shp, l in zip(tile_shape, global_layout):
        l = coalesce(l)
        flat_shape = l.shape_tuple
        flat_stride = l.stride_tuple
        current = shp
        for s in flat_shape[:-1]:
            # Note: if the shape is not a constant, we cannot
            # determine the dimensions of the TMA tensor. Thus
            # we cannot use TMA and return None.
            if not is_constant(s):
                return None
            # Note: for cases where the shape cannot be divided
            # by the shapes in the global layout, we have to
            # insert additional dimensions to TMA tensors.
            # currently, we just disable these cases for simplicity.
            # TODO: support this in the future.
            if current % s != 0:
                return None
            current = current // s
        if current == 1:
            last_dim_strides.append(flat_stride[-1])
        else:
            last_dim_strides.append(None)
    return last_dim_strides


def coalesce_per_dim(layout: TensorLayout, shape: Tuple[int, ...], last_dim_strides: List[Optional[int]]):
    """
    Coalesce the layout per dimension. Add the last dimension strides to each dimension if the last dimension strides
    are not None.

    Parameters:
        layout: The layout of the tensor.
        shape: The shape of the tensor.
        last_dim_strides: The last dimension strides for each dimension.
    Returns:
        The layout with each dimension coalesced.

    Example:
    >>> layout = TensorLayout((16, 16, 16, 16), (1, 16, 256, 4096))
    >>> shape = (256, 256)
    >>> last_dim_strides = [4096, 16384]
    >>> coalesce_per_dim(layout, shape, last_dim_strides)
    TensorLayout(((256, 1), (256, 1)), ((1, 4096), (256, 16384)))
    """
    layout = coalesce(layout)
    layouts = []
    for s, d in zip(shape, last_dim_strides):
        cur, layout = group(layout, s)
        if d is not None:
            cur_shape = cur.shape_tuple + (1,)
            cur_stride = cur.stride_tuple + (d,)
            cur = TensorLayout(cur_shape, cur_stride)
        layouts.append(cur)
    return make_layout(*layouts)


def common_reshape_per_dim(gmem_layout: TensorLayout, smem_layout: TensorLayout):
    """
    Given two layouts, compute the common shape for each dimension.

    For TMA tensors, both the global memory and shared memory must be treated as tensors with matching shapes.
    However, dimensions in either the global or shared memory layout may have been merged. This function
    reverts such merged dimensions to recover a common shape for both layouts, ensuring compatibility for TMA
    operations.

    Parameters:
        gmem_layout: The layout of the global memory tensor.
        smem_layout: The layout of the shared memory tensor.

    Returns:
        A tuple of layouts, each with the common shape for every dimension.
    """
    gmems = []
    smems = []
    for g, s in zip(gmem_layout, smem_layout):
        g, s = common_reshape(g, s)
        gmems.append(g)
        smems.append(s)
    return make_layout(*gmems), make_layout(*smems)


def coalesce_gmem_shape_and_smem_shape(gmem_shape, smem_shape, gmem_stride, smem_stride):
    """
    Merge contiguous dimensions in global and shared memory shapes and strides to minimize the number of
    dimensions for the TMA tensor.

    Note:
    - The merged dimensions must be contiguous in both the global and shared memory.

    Parameters:
        gmem_shape: The shape of the global memory tensor.
        smem_shape: The shape of the shared memory tensor.
        gmem_stride: The stride of the global memory tensor.
        smem_stride: The stride of the shared memory tensor.

    Returns:
        A tuple containing:
            - The coalesced global memory shape
            - The coalesced global memory stride
            - The coalesced shared memory shape
            - The coalesced shared memory stride
            - The list of dimension groupings for the TMA tensor
    """
    gmem_shape_ = []
    gmem_stride_ = []
    smem_shape_ = []
    smem_stride_ = []
    dims = []
    cur_dims = []
    for i, (u, v, s, t) in enumerate(zip(gmem_shape, gmem_stride, smem_shape, smem_stride)):
        if len(gmem_shape_) == 0:
            gmem_shape_.append(u)
            gmem_stride_.append(v)
            smem_shape_.append(s)
            smem_stride_.append(t)
            cur_dims.append(i)
        else:
            curr_shape = gmem_shape_[-1]
            # curr_box_shape = smem_shape_[-1]
            gmem_curr_stride = gmem_stride_[-1]
            smem_curr_stride = smem_stride_[-1]
            if (
                all(is_constant(e) for e in [v, gmem_curr_stride])
                and v == curr_shape * gmem_curr_stride
                and t == curr_shape * smem_curr_stride
            ):
                gmem_shape_[-1] *= u
                smem_shape_[-1] *= s
                cur_dims.append(i)
            else:
                dims.append(cur_dims)
                cur_dims = []
                cur_dims.append(i)
                gmem_shape_.append(u)
                gmem_stride_.append(v)
                smem_shape_.append(s)
                smem_stride_.append(t)
    if len(cur_dims) > 0:
        dims.append(cur_dims)
    return gmem_shape_, gmem_stride_, smem_shape_, smem_stride_, dims


def split_shapes(gmem_shape, smem_shape, gmem_stride, smem_stride, max_element_per_dim):
    """
    This function splits the dimensions to satisfy the constraint that the maximum element per dimension
    of the TMA tensor.

    Parameters:
        gmem_shape: The shape of the global memory tensor.
        smem_shape: The shape of the shared memory tensor.
        gmem_stride: The stride of the global memory tensor.
        smem_stride: The stride of the shared memory tensor.
        max_element_per_dim: The maximum element per dimension of the TMA tensor.

    Returns:
        A tuple containing:
            - The coalesced global memory shape
            - The coalesced global memory stride
            - The coalesced shared memory shape
            - The coalesced shared memory stride
            - The list of dimension groupings for the TMA tensor
    """
    gmem_shape_ = []
    smem_shape_ = []
    gmem_stride_ = []
    smem_stride_ = []
    split_dims = {}
    for i, (u, v, s, t) in enumerate(zip(gmem_shape, gmem_stride, smem_shape, smem_stride)):
        if u > max_element_per_dim:
            remain = u
            cur_gmem_stride = v
            cur_smem_stride = t
            split_shape_list = []
            while remain > max_element_per_dim:
                from hidet.utils.py import gcd

                cur_shape = gcd(remain, max_element_per_dim)
                split_shape_list.append(cur_shape)
                gmem_shape_.append(cur_shape)
                smem_shape_.append(cur_shape)
                gmem_stride_.append(cur_gmem_stride)
                smem_stride_.append(cur_smem_stride)
                remain = shape_div(remain, cur_shape)
                cur_gmem_stride = cur_gmem_stride * cur_shape
                cur_smem_stride = cur_smem_stride * cur_shape
            if remain > 1:
                split_shape_list.append(remain)
                gmem_shape_.append(remain)
                smem_shape_.append(remain)
                gmem_stride_.append(cur_gmem_stride)
                smem_stride_.append(cur_smem_stride)
            split_dims[i] = tuple(split_shape_list)
        else:
            gmem_shape_.append(u)
            gmem_stride_.append(v)
            smem_shape_.append(s)
            smem_stride_.append(t)
    return gmem_shape_, gmem_stride_, smem_shape_, smem_stride_, split_dims


def swap(arr, i, j):
    """
    Swap the elements at index i and j in the array.
    """
    arr[i], arr[j] = arr[j], arr[i]


def sort_dims(smem_stride, smem_shape, gmem_stride, gmem_shape, index):
    """
    Sort the dimensions in ascending order of shared memory stride.

    Note:
        The sorted dimensions must be contiguous in shared memory. This is required because, for TMA tensors,
        shared memory dimensions must be contiguous. The number of TMA tensor dimensions can only be determined
        after sorting the shared memory dimensions.

    Parameters:
        smem_stride: The stride of the shared memory tensor.
        smem_shape: The shape of the shared memory tensor.
        gmem_stride: The stride of the global memory tensor.
        gmem_shape: The shape of the global memory tensor.
        index: The indices of the dimensions.

    Returns:
        A tuple containing:
            - The sorted shared memory stride
            - The sorted shared memory shape
            - The sorted global memory stride
            - The sorted global memory shape
            - The permutation of the dimensions
    """
    max_smem_stride = max(filter(lambda x: is_constant(x), smem_stride))
    sorted_DS = sorted(
        filter(
            lambda x: not is_constant(x[0]) or x[0] > 0, zip(smem_stride, smem_shape, gmem_stride, gmem_shape, index)
        ),
        key=lambda x: x[0] if is_constant(x[0]) else max_smem_stride + x[2],
    )
    smem_stride_sorted, smem_shape_sorted, gmem_stride_sorted, gmem_shape_sorted, permute = zip(*sorted_DS)
    smem_stride_sorted = list(smem_stride_sorted)
    smem_shape_sorted = list(smem_shape_sorted)
    gmem_stride_sorted = list(gmem_stride_sorted)
    gmem_shape_sorted = list(gmem_shape_sorted)
    permute = list(permute)
    # find the first dimension that has the max smem stride
    l = 0
    current_gmem_stride = -1
    for r in range(len(smem_stride_sorted)):
        sd = smem_stride_sorted[r]
        gd = gmem_stride_sorted[r]
        gs = gmem_shape_sorted[r]
        if sd == max_smem_stride:
            current_gmem_stride = gd * gs
            l = r + 1
        if not is_constant(sd):
            if gd == current_gmem_stride:
                current_gmem_stride = gd * gs
                for i in range(r, l, -1):
                    swap(smem_stride_sorted, i, i - 1)
                    swap(smem_shape_sorted, i, i - 1)
                    swap(gmem_stride_sorted, i, i - 1)
                    swap(gmem_shape_sorted, i, i - 1)
                    swap(permute, i, i - 1)
                l = l + 1
    return smem_stride_sorted, smem_shape_sorted, gmem_stride_sorted, gmem_shape_sorted, permute


def make_contiguous_stride(shape, stride):
    """
    Fill in the unknown strides (which is marked as var('v')) with the contiguous strides.

    Parameters:
        shape: The shape of the tensor.
        stride: The stride of the tensor.

    Returns:
        A tuple containing:
            - The shape of the tensor
            - The stride of the tensor
    """
    current_index = 1
    for i, (s, d) in enumerate(zip(shape, stride)):
        if not is_constant(d):
            stride[i] = current_index
            current_index *= s
        else:
            current_index = s * d
    return shape, stride


def construct_memory_constraint(
    sorted_shape: Tuple[int, ...], smem_layout: TensorLayout, permute: List[int], extra_memory_hint: TensorLayout
):
    """
    Constructs the memory constraint for the Tensor Memory Accelerator (TMA) tensor.

    During TMA tensor layout derivation, shared memory dimensions are reordered by ascending stride.
    To maintain logical consistency with other memory constraints, this function:
      1. Reverts dimension reordering by applying a permutation
      2. Reshapes the layout to match the user-provided memory hint
      3. Marks final undeterminedshared memory strides with symbolic variables ('v')

    Parameters:
        sorted_shape (tuple): Shape of the TMA tensor (after stride-based reordering)
        smem_layout (TensorLayout): Shared memory tensor layout
        permute (tuple[int]): Permutation to reverse dimension reordering
        extra_memory_hint (tuple): User-provided shape hint for shared memory

    Returns:
        TensorLayout: Memory constraint for the TMA tensor
    """
    remain = smem_layout
    layouts = []
    for s in sorted_shape:
        cur, remain = group(remain, s)
        if cur is None or remain is None:
            return None
        layouts.append(cur)
    layouts_ = [None] * len(layouts)
    for i in range(len(layouts)):
        j = permute[i]
        layouts_[j] = layouts[i]
    layout = coalesce(make_layout(*layouts_))
    full_shape = extra_memory_hint.shape_tuple
    result_shapes = []
    result_strides = []
    shape = list(layout.shape_tuple)
    stride = list(layout.stride_tuple)
    idx = 0
    for s in full_shape:
        cur_shapes = []
        cur_strides = []
        remaining = s
        while remaining > 1 and idx < len(shape):
            if shape[idx] < remaining:
                if remaining % shape[idx] != 0:
                    return None
                cur_shapes.append(shape[idx])
                cur_strides.append(stride[idx])
                remaining = shape_div(remaining, shape[idx])
                idx += 1
            elif shape[idx] > remaining:
                if shape[idx] % remaining != 0:
                    return None
                cur_shapes.append(remaining)
                cur_strides.append(stride[idx])
                shape[idx] = shape_div(shape[idx], remaining)
                stride[idx] = stride[idx] * remaining
                remaining = 1
            else:
                cur_shapes.append(remaining)
                cur_strides.append(stride[idx])
                remaining = 1
                idx += 1
        if remaining != 1:
            cur_shapes.append(remaining)
            cur_strides.append(var('v'))
        result_shapes.append(tuple(cur_shapes))
        result_strides.append(tuple(cur_strides))
    return TensorLayout(tuple(result_shapes), tuple(result_strides))
