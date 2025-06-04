import math
import operator
import re
import copy
import itertools
import functools

from typing import List, Union
from hidet.graph.tensor import Tensor
from hidet.graph import ops


class RearrangePattern:
    """Class to parse and validate the rearrange pattern."""

    RE_TOKEN = re.compile(r'\([^)]*\)|\S+')

    def __init__(self, pattern: str, ndims: int) -> None:
        self.ellipsis = False
        self.ellipsis_parenthesized = False
        self.ellipsis_start_id = 0
        self.ellipsis_dims = ndims

        self.build_composition(pattern)
        self.ellipsis_dims = self.ellipsis_dims if self.ellipsis else 0
        self._flat = tuple(itertools.chain.from_iterable(self.composition))

    def build_composition(self, pattern: str) -> None:
        """Validate the pattern and check for errors."""
        id_map = {}
        comp: List[List[str]] = []
        id_val = 0
        tokens = self.RE_TOKEN.findall(pattern)
        self.ellipsis_dims -= len(tokens)

        for group in tokens:
            if group[0] == '(':
                inner_comp: List[str] = []
                for inner in group[1:-1].split():
                    if inner == '...':
                        self.ellipsis = True
                        self.ellipsis_parenthesized = True
                        id_map[Ellipsis] = 0
                        inner_comp.append(Ellipsis)
                        continue
                    if inner == '1':  # Ignore inner 1
                        continue
                    check, msg = self.check_name(inner)
                    if not check:
                        raise ValueError(msg)
                    if inner in id_map:
                        raise ValueError(
                            f"Invalid pattern: {pattern}. Duplicate identifier \
                                {inner}."
                        )
                    id_map[inner] = id_val
                    id_val += 1
                    inner_comp.append(inner)
                if inner_comp:
                    comp.append(inner_comp)
            else:
                if group == '1':
                    # Add a null character in place of 1 so flatten can be used
                    # for shape equality later
                    comp.append(['\0'])
                    continue
                if group == '...':
                    if self.ellipsis:  # Ellipsis already found
                        raise ValueError(
                            f"Invalid pattern: {pattern}. Only one ellipsis \
                                (...) is allowed in the pattern."
                        )
                    self.ellipsis = True
                    self.ellipsis_start_id = id_val
                    id_map[Ellipsis] = 0  # Add ellipsis to id_map for set diff
                    id_val += self.ellipsis_dims
                    comp.append([Ellipsis])
                    continue
                check, msg = self.check_name(group)
                if not check:
                    raise ValueError(msg)
                if group in id_map:
                    raise ValueError(f"Invalid pattern: {pattern}. Duplicate identifier {group}.")
                id_map[group] = id_val
                id_val += 1
                comp.append([group])

        self.id_map = id_map
        self.composition = comp

    def flatten(self):
        """Flatten the composition to a single list of identifiers."""
        return self._flat

    @staticmethod
    def check_name(name: str) -> tuple[bool, str]:
        """Check if the axis name is a valid identifier."""
        if not str.isidentifier(name) and name != '1':
            return (
                False,
                f"Invalid identifier: {name}. Identifier must be a \
                valid Python identifier.",
            )
        if name[0] == '_' or name[-1] == '_':
            return (
                False,
                f"Invalid identifier: {name}. Identifier cannot \
                start or end with underscore.",
            )
        return True, ''


@functools.lru_cache(maxsize=128)
def _get_rearrange_pattern_raw(pattern: str, ndims: int) -> RearrangePattern:
    """
    Get the rearrange pattern and check for errors. Wrapped in a function to
    allow for caching since LLMs use many of the same operations frequently.
    """
    return RearrangePattern(pattern, ndims)


def _get_rearrange_pattern(pattern: str, ndims: int) -> RearrangePattern:
    """
    Mutable version of _get_rearrange_pattern_raw. If the original pattern is
    modified, the LRU cache retains the mofified version. To prevent this, we
    return a deep copy of the original pattern and cache the original.
    """
    raw = _get_rearrange_pattern_raw(pattern, ndims)
    return copy.deepcopy(raw)


def build_plan(left: RearrangePattern, right: RearrangePattern) -> List[List[int]]:
    """Build the plan for rearranging the tensor."""
    ellipsis_dims = left.ellipsis_dims
    id_map = left.id_map
    comps = right.composition
    if not right.ellipsis:  # No ellipsis in the right side, just use the id_map
        return [[id_map[name] for name in comp] if comp != ['\0'] else [] for comp in comps]
    plan: List[List[int]] = []
    for comp in comps:
        inner: List[int] = []
        if comp == ['\0']:  # Unsqueeze
            plan.append([])
            continue
        for axis in comp:
            if axis == Ellipsis:
                if right.ellipsis_parenthesized:
                    inner.extend(list(range(left.ellipsis_start_id, left.ellipsis_start_id + ellipsis_dims + 1)))
                else:
                    plan.extend(
                        [[j] for j in range(left.ellipsis_start_id, left.ellipsis_start_id + ellipsis_dims + 1)]
                    )
            else:
                inner.append(id_map[axis])
        plan.append(inner)
    return plan


def get_ellipsis_dims(x: Tensor, left: RearrangePattern) -> List[int]:
    """Get the ellipsis dimensions from the tensor."""
    if not left.ellipsis:
        return []
    start = next(i for i, grp in enumerate(left.composition) if grp and grp[0] == Ellipsis)
    end = start + left.ellipsis_dims + 1
    return list(x.shape[start:end])


def reshape_to_pattern(
    x: Tensor, left: RearrangePattern, right: RearrangePattern, axes_lengths: dict, ellipsis_dims: List[int]
) -> Tensor:
    """
    Reshape the tensor from the left pattern to the right pattern.

    Parameters
    ----------
    x : Tensor
        The input tensor to be reshaped.
    left : RearrangePattern
        The rearrange pattern for the left side of the expression.
    right : RearrangePattern
        The rearrange pattern for the right side of the expression.
    axes_lengths : dict
        A dictionary mapping axis names to their lengths.
    ellipsis_dims : List[int]
        The dimensions of the ellipsis in the left pattern.

    Returns
    -------
    Tensor
        The reshaped tensor.
    """
    axes_lengths = {k: v if k not in axes_lengths else axes_lengths[k] for k, v in zip(left.flatten(), x.shape)}
    target_shape = []
    for elt in right.composition:
        if not right.ellipsis_parenthesized and elt[0] == Ellipsis:
            # If the ellipsis is not parenthesized, we need to add the ellipsis
            # dimensions to the target shape
            target_shape.extend(ellipsis_dims)
        else:
            # Otherwise, we just need to multiply the inner groups together to
            # get the top level shape. Ellipsis will be a key in axes_lengths
            target_shape.append(functools.reduce(operator.mul, (axes_lengths[name] for name in elt), 1))
    return ops.reshape(x, target_shape)


def einops_rearrange(x: Tensor, pattern: str, **axes_lengths) -> Tensor:
    """
    Rearrange the tensor according to the pattern.

    Parameters
    ----------
    x : Tensor
        The input tensor to be rearranged.
    pattern : str
        The rearrangement pattern in the form of 'input_pattern ->
        output_pattern'. There can be ellipsis (...) in the pattern, and the
        number of identifiers must be the same on both sides of the pattern.
        Example: 'b c d -> b (c d)'.
    axes_lengths : dict
        A dictionary mapping axis names to their lengths. This is used to
        infer the lengths of the axes in the pattern. A maximum of one missing
        length may be present per group on the left side of the pattern.
        Example: ('(b c) d -> b c d', {'b': 1}).

    Returns
    -------
    Tensor
        The rearranged tensor.

    Examples
    --------
    b c d -> b (c d) : Rearranges the tensor from shape (b, c, d) to (b, c*d).
    b c d ... -> b (c d ...) : Rearranges the tensor from shape (b, c, d, ...) to (b, c*d, ...).
    (b c) d -> b c d, {b: 1} : Rearranges the tensor from shape (b, c, d) to (b, c, d).
    """
    input_pattern, _, output_pattern = pattern.partition(' -> ')

    if not input_pattern or not output_pattern:
        raise ValueError(
            f"Invalid pattern: {pattern}. Pattern must be in the form \
                    'input_pattern -> output_pattern'."
        )

    dims = len(x.shape)
    left = _get_rearrange_pattern(input_pattern, dims)
    right = _get_rearrange_pattern(output_pattern, dims)
    ellipsis_dim = left.ellipsis_dims

    if left.ellipsis_parenthesized:
        raise ValueError(
            f"Invalid pattern: {pattern}. Ellipsis (...) found in the left \
                side, but cannot be parenthesized."
        )
    if left.ellipsis and dims < len(left.composition):
        raise ValueError(
            f"Invalid pattern: {pattern}. Ellipsis (...) found, but number \
                of dimensions in tensor is less than number of dimensions \
                in pattern."
        )
    if left.id_map.keys() ^ right.id_map.keys():
        raise ValueError(
            f"Identifiers only on one side of expression (should be on \
                both: {left.id_map.keys() ^ right.id_map.keys()}), \
                        pattern: {pattern}."
        )
    if dims != len(left.composition) + ellipsis_dim:
        raise ValueError(
            f"Invalid pattern: {pattern}. The number of dimensions in the \
                    pattern do not match the number of dimensions in the \
                    tensor."
        )

    # Check the set differences of groups in the left and right patterns
    # The right pattern must contain all the groups in the left pattern
    # If not, we need to reshape the tensor so the build plan can be applied
    can_reshape = left.flatten() == right.flatten()
    altered_groups = {tuple(g) for g in left.composition if len(g) > 1} - {
        tuple(g) for g in right.composition if len(g) > 1
    }
    need_reshape = len(altered_groups) > 0

    # Get the ellipsis dimensions if needed
    ell_dims = []
    if can_reshape or need_reshape:
        ell_dims = get_ellipsis_dims(x, left)
        if right.ellipsis_parenthesized:  # Collapse the ellipsis dimensions
            axes_lengths[Ellipsis] = math.prod(ell_dims)

    # Fill in the axes_lengths dictionary with the lengths of the axes
    # If needed, fill in the atomic shape for the tensor
    atomic_shape = []
    for comp, dim in zip(left.composition, x.shape):
        if comp == ['\0'] and dim != 1:
            raise ValueError(
                f"Invalid pattern: {pattern}. Expected \
                                dimension of size 1 but got {dim}."
            )

        if need_reshape:
            if not can_reshape and comp[0] == Ellipsis:
                atomic_shape.extend(ell_dims)
                continue
            prod = 1
            missing: Union[None, str] = None
            for n in comp:
                if n not in axes_lengths and n != Ellipsis:
                    if missing:
                        raise ValueError(f"Cannot infer lengths for {missing} in group {comp}")
                    missing = n
                else:
                    prod *= axes_lengths.get(n, 1)
            if missing is not None:
                axes_lengths[missing] = dim // prod
            if not can_reshape:
                atomic_shape.extend([axes_lengths.get(n, 1) for n in comp])

    if can_reshape:
        # Check if the input and output patterns are the same when flattened
        # If they are, we can just reshape the tensor to the target shape
        return reshape_to_pattern(x, left, right, axes_lengths, ell_dims)
    elif need_reshape:
        # Convert the tensor to a shape that matches the pattern
        # Only reshape here if data movement will occur otherwise
        # Also change the left composition to match the expected shape
        i = 0
        while i < len(left.composition):
            item = left.composition[i]
            if tuple(item) in altered_groups:
                left.composition[i : i + 1] = [[x] for x in item]
                i += len(item)
            else:
                i += 1
        x = ops.reshape(x, atomic_shape)
    return ops.rearrange(x, build_plan(left, right))
