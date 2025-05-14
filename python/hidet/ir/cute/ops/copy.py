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
from typing import Union, List, Dict, Optional

from hidet.ir.cute.layout import ComposedTensorLayout, TensorLayout, TiledTensorLayout, auto_layout, is_auto_layout
from hidet.ir.expr import Expr, is_constant
from hidet.ir.cute.expr import Op, CConst
from hidet.ir.cute.type import tiled_tensor, TiledTensorType, logical_encoding
from hidet.ir.cute.algorithm import TiledCopy, is_auto_copy
from hidet.ir.type import BaseType, void, DataType
from hidet.ir.dtypes import u32, u64, boolean


class Mask(Op):
    """
    Create a mask tensor for copies.

    Attributes:
        extents (List[Expr]): extents of the dimensions of the tensor to be copied
        tiled_copy (TiledCopy): The tiled copy that represents the task mapping of the copy operation.

    Methods:
        reforward(args: List[Expr], attrs_update: Dict[str, CConst] = None,
                  annotations_update: Dict[str, CConst] = None) -> 'Mask':
            Creates a new instance of the Mask class with updated attributes and annotations.

        infer_type(arg_types: List[BaseType]) -> BaseType:
            Infers the type of the result of the mask operation based on input types.

    Example:
        Consider a copy operation that copies a tile of size (16, 16) from a source tensor to a destination tensor.
        The shape of the input tensor is (12, 12). Then, we should create a mask tensor for the copy operation to
        disable the elements that are out of bounds. The mask tensor can be created as follows:
        ```python
        mask = Mask([12, 12], tiled_copy)
        ```
    """

    def __init__(self, extents: List[Expr], tiled_copy: TiledCopy):
        """
        Initializes the Mask class with the given extents and tiled copy operation.

        Args:
            extents (List[Expr]): List of extents for the mask.
            tiled_copy (TiledCopy): The tiled copy operation associated with this mask.
        """

        super().__init__(args=extents, attrs={"tiled_copy": tiled_copy})
        self.extents: List[Expr] = extents
        assert is_auto_copy(tiled_copy) or len(extents) == len(tiled_copy.copy_atom.shape)
        self.tiled_copy = tiled_copy

    def reforward(
        self, args: List[Expr], attrs_update: Dict[str, CConst] = None, annotations_update: Dict[str, CConst] = None
    ):
        """
        Creates a new instance of the Mask class with updated attributes and annotations.

        Args:
            args (List[Expr]): List of input expressions (extents).
            attrs_update (Dict[str, CConst], optional): Dictionary of attributes to update. Defaults to None.
            annotations_update (Dict[str, CConst], optional): Dictionary of annotations to update. Defaults to None.

        Returns:
            Mask: A new instance of the Mask class with updated attributes and annotations.
        """
        attrs = self.attrs.copy()
        annotations = self.annotations.copy()
        if attrs_update is not None:
            attrs.update(attrs_update)
        if annotations_update is not None:
            annotations.update(annotations_update)
        ret = self.__class__(args, **attrs)
        ret.annotations = annotations
        return ret

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        """
        Infers the type of the result of the mask operation based on input types.

        Args:
            arg_types (List[BaseType]): List of input types.

        Returns:
            BaseType: The inferred type of the result.
        """
        # FIXME:
        # Note that we can't infer the output type when the instruction selection pass hasn't been done.
        # So the "-1" here means the register count hasn't been determined, which can't be used to
        # generate code.
        annotations = self.annotations
        if is_auto_copy(self.tiled_copy):
            return tiled_tensor(dtype=u32, layout=auto_layout, scope="register")
        elif "rest_layout" not in annotations:
            return tiled_tensor(dtype=u32, layout=TensorLayout(-1), scope="register")
        rest_layout = annotations["rest_layout"]
        nr_masks = rest_layout.size()
        nr_regs = (nr_masks + u32.nbytes * 8 - 1) // (u32.nbytes * 8)
        return tiled_tensor(dtype=u32, layout=TensorLayout(nr_regs), scope="register")


def mask(tiled_copy: TiledCopy, extents: List[Expr]):
    return Mask(extents, tiled_copy).make_call()


class Copy(Op):
    """
    Copy operation that copies a tile from a source tensor to a destination tensor.

    Attributes:
        tiled_copy (TiledCopy): The tiled copy operation.
        src (Expr): The source tensor expression.
        dst (Expr): The destination tensor expression.
        mask (Optional[Expr]): Optional mask expression for the copy.
        mbarrier (Optional[Expr]): Optional mbarrier expression for the copy.

    Methods:
        reforward(args: List[Expr], attrs_update: Dict[str, CConst] = None,
            annotations_update: Dict[str, CConst] = None) -> 'Copy':
            Creates a new instance of the Copy class with updated attributes and annotations.

        write_memory_op() -> bool:
            Checks if the copy operation involves writing to memory.

        infer_type(arg_types: List[BaseType]) -> BaseType:
            Infers the type of the result of the copy operation based on input types.
    """

    def __init__(
        self,
        tiled_copy: TiledCopy,
        src: Expr,
        dst: Expr,
        mask_: Optional[Expr] = None,
        mbarrier: Optional[Expr] = None,
        evict: Optional[str] = None,
    ):
        super().__init__(args=[src, dst, mask_, mbarrier], attrs={"tiled_copy": tiled_copy, "evict": evict})
        self.tiled_copy: TiledCopy = tiled_copy
        self.src: Expr = src
        self.dst: Expr = dst
        self.mask: Optional[Expr] = mask_
        self.mbarrier: Optional[Expr] = mbarrier
        self.evict: Optional[str] = evict

    def resolve_logical_encoding(self):
        if is_auto_copy(self.tiled_copy):
            raise RuntimeError(
                "Cannot resolve the logical encoding for tensors because the tiled_copy"
                f"hasn't been specified.(got:{self.tiled_copy.str_indented()})"
            )
        shape, src_tv = self.tiled_copy.src_tv_layout()
        _, dst_tv = self.tiled_copy.dst_tv_layout()
        return logical_encoding(shape, src_tv), logical_encoding(shape, dst_tv)

    def reforward(
        self, args: List[Expr], attrs_update: Dict[str, CConst] = None, annotations_update: Dict[str, CConst] = None
    ):
        attrs = self.attrs.copy()
        annotations = self.annotations.copy()
        if attrs_update is not None:
            attrs.update(attrs_update)
        if annotations_update is not None:
            annotations.update(annotations_update)
        assert "tiled_copy" in attrs
        assert "evict" in attrs
        ret = self.__class__(attrs["tiled_copy"], *args, evict=attrs["evict"])
        ret.annotations = annotations
        return ret

    def write_memory_op(self) -> bool:
        from hidet.ir.tools import infer_type

        dst_ty = infer_type(self.dst)
        return dst_ty.scope.is_memory()

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        src_ty = arg_types[0]
        dst_ty = arg_types[1]
        mask_ty = arg_types[2]
        mbarrier_ty = arg_types[3]
        if not (
            all(isinstance(ty, TiledTensorType) for ty in [src_ty, dst_ty])
            and all(ty is void or isinstance(ty, TiledTensorType) for ty in [mask_ty, mbarrier_ty])
        ):
            raise TypeError(f"Type mismatch. (got:src({src_ty}),dst({dst_ty}),mask({mask_ty}),mbarrier({mbarrier_ty}))")
        if is_auto_copy(self.tiled_copy):
            return void
        elif any(ty is not void and is_auto_layout(ty.layout) for ty in [src_ty, dst_ty, mask_ty, mbarrier_ty]):
            return void
        elif not (
            all(isinstance(ty.layout, (TensorLayout, ComposedTensorLayout)) for ty in [src_ty, dst_ty])
            and all(ty is void or isinstance(ty.layout, TensorLayout) for ty in [mask_ty, mbarrier_ty])
        ):
            raise TypeError(
                f"Invalid layout. (got:src({src_ty.layout}),dst({dst_ty.layout}),"
                f"mask({mask_ty.layout}),mbarrier({mbarrier_ty.layout}))"
            )
        elif src_ty.scope.is_global() or dst_ty.scope.is_global():
            # as long as the total element of non-zero-stride dimensions are
            # equal, we can perform the copy. The zero-stride dimensions can be
            # redistributed arbitrarily.
            src_count = src_ty.layout.count()
            dst_count = dst_ty.layout.count()
            if src_count != dst_count:
                raise TypeError(f"Tensor count mismatch. (got:src({src_count}),dst({dst_count}))")
        return void


def copy(
    tiled_copy: TiledCopy,
    src: Expr,
    dst: Expr,
    mask_: Optional[Expr] = None,
    mbarrier: Optional[Expr] = None,
    evict: Optional[str] = None,
):
    return Copy(tiled_copy, src, dst, mask_, mbarrier, evict).make_call()


class Atomic(Op):
    """
    Performs an atomic operation on a memory region (a tile) using a register tensor (a tile in register files).

    Attributes:
        src (Expr): Source expression, representing the register tensor.
        dst (Expr): Destination expression, representing the memory region.
        mask (Optional[Expr]): Optional mask expression, representing the mask tensor.
    """

    def __init__(self, src: Expr, dst: Expr, mask_: Optional[Expr] = None):
        """
        Initializes the Atomic operation with the given source, destination, and optional mask expressions.

        Args:
            src (Expr): Source expression, representing the register tensor.
            dst (Expr): Destination expression, representing the memory region.
            mask_ (Optional[Expr]): Optional mask expression, representing the mask tensor.
        """
        super().__init__(args=[src, dst] + ([mask_] if mask_ is not None else []), attrs={})
        self.src: Expr = src
        self.dst: Expr = dst
        self.mask: Optional[Expr] = mask_

    def write_memory_op(self) -> bool:
        """
        Indicates that this operation writes to memory.

        Returns:
            bool: Always returns True, indicating a memory write operation.
        """
        return True

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        """
        Infers the type of the operation based on the types of its arguments.

        Args:
            arg_types (List[BaseType]): List of argument types.

        Returns:
            BaseType: The inferred type of the operation.

        Raises:
            TypeError: If there is a type mismatch between the arguments or if the input/output types are not
            as expected.
        """
        src_ty = arg_types[0]
        dst_ty = arg_types[1]
        mask_ty = arg_types[2] if len(arg_types) >= 3 else void
        if not (
            isinstance(src_ty, TiledTensorType)
            and isinstance(dst_ty, TiledTensorType)
            and (mask_ty is void or isinstance(mask_ty, TiledTensorType))
        ):
            raise TypeError(f"Type mismatch. (got:src({src_ty}),dst({dst_ty}),mask({mask_ty}))")
        if not dst_ty.scope.is_memory():
            raise TypeError(f"ouput of atomic operation should be a memory region.got({dst_ty})")
        if not src_ty.scope.is_register():
            raise TypeError(f"input of atomic operation should be a register tensor.got({src_ty})")
        if is_auto_layout(src_ty.layout):
            return void
        else:
            dst_shape = dst_ty.layout.shape
            assert dst_shape is not None
            if not isinstance(src_ty.layout, TiledTensorLayout):
                raise TypeError(f"input layout of atomic operation should be TiledTensorLayout.got({src_ty.layout})")
            src_shape = src_ty.layout.shape()
            if src_shape != dst_shape:
                raise TypeError(f"Shape mismatch.(src:{src_shape},dst:{dst_shape})")
            return void


class AtomicAdd(Atomic):
    pass


def cute_atomic_add(src: Expr, dst: Expr, mask_: Optional[Expr] = None):
    return AtomicAdd(src, dst, mask_).make_call()


class MBarriers(Op):
    def __init__(self, num_barriers: int):
        super().__init__(args=[], attrs={"num_barriers": num_barriers})
        self.num_barriers: int = num_barriers

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        if not is_constant(self.num_barriers):
            raise TypeError(f"num_barriers should be a constant. got({self.num_barriers})")
        nr_regs = self.num_barriers
        return tiled_tensor(dtype=u64, layout=TensorLayout(nr_regs), scope="shared")


def make_mbarriers(num_barriers: int):
    return MBarriers(num_barriers).make_call()


class MBarrierArrive(Op):
    def __init__(self, mbarrier: Expr, count: Expr):
        super().__init__(args=[mbarrier, count], attrs={})
        self.mbarrier: Expr = mbarrier
        self.count: Expr = count

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        mbarrier_ty = arg_types[0]
        count_ty = arg_types[1]
        if not isinstance(mbarrier_ty, TiledTensorType):
            raise TypeError(f"mbarrier should be a tiled tensor. got({mbarrier_ty})")
        if not isinstance(count_ty, DataType):
            raise TypeError(f"count should be a scalar. got({count_ty})")
        if not count_ty.is_integer():
            raise TypeError(f"count should be an integer. got({count_ty})")
        return void


def mbarrier_arrive(mbarrier: Expr, count: Union[Expr, int] = 0):
    if isinstance(count, int):
        count = u32(count)
    return MBarrierArrive(mbarrier, count).make_call()


class MBarrierTryWait(Op):
    def __init__(self, mbarrier: Expr, phase: Expr):
        super().__init__(args=[mbarrier, phase], attrs={})
        self.mbarrier: Expr = mbarrier
        self.phase: Expr = phase

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        mbarrier_ty = arg_types[0]
        phase_ty = arg_types[1]
        if not isinstance(mbarrier_ty, TiledTensorType):
            raise TypeError(f"mbarrier should be a tiled tensor. got({mbarrier_ty})")
        if not isinstance(phase_ty, DataType):
            raise TypeError(f"phase should be a scalar. got({phase_ty})")
        if not phase_ty.is_boolean():
            raise TypeError(f"phase should be a boolean. got({phase_ty})")
        return tiled_tensor(dtype=u32, layout=TensorLayout(1), scope="register")


def mbarrier_try_wait(mbarrier: Expr, phase: Expr):
    if isinstance(phase, bool):
        phase = boolean(phase)
    return MBarrierTryWait(mbarrier, phase).make_call()


class MBarrierWait(Op):
    def __init__(self, mbarrier: Expr, phase: Expr):
        super().__init__(args=[mbarrier, phase], attrs={})
        self.mbarrier: Expr = mbarrier
        self.phase: Expr = phase

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        mbarrier_ty = arg_types[0]
        phase_ty = arg_types[1]
        if not isinstance(mbarrier_ty, TiledTensorType):
            raise TypeError(f"mbarrier should be a tiled tensor. got({mbarrier_ty})")
        if not isinstance(phase_ty, DataType):
            raise TypeError(f"phase should be a scalar. got({phase_ty})")
        if not phase_ty.is_boolean():
            raise TypeError(f"phase should be a boolean. got({phase_ty})")
        return void


def mbarrier_wait(mbarrier: Expr, phase: Expr):
    if isinstance(phase, bool):
        phase = boolean(phase)
    return MBarrierWait(mbarrier, phase).make_call()
