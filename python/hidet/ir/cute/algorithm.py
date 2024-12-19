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
from typing import Union, Tuple, List, Optional
from hidet.ir.cute.layout import CopyAtom, MmaAtom, Level, chain, compact_coshape
from hidet.ir.cute import TiledTensorLayout


class TiledCopy:
    def __init__(self, copy_atom: CopyAtom, levels: List[Level] = None):
        if levels is None:
            levels = []
        self.copy_atom = copy_atom
        self.levels = sorted(levels, key=lambda x: x.level.value)

    @staticmethod
    def from_tiled_tensor_layout(layout: TiledTensorLayout):
        copy_atom = CopyAtom.from_tv_atom(layout.atom)
        return TiledCopy(copy_atom, layout.levels)

    @property
    def shape(self):
        shape, _ = chain(
            self.copy_atom.shape,
            self.copy_atom.src_thrval_layout,
            self.copy_atom.repeat_shape,
            self.copy_atom.repeat_layout,
            self.levels,
        )
        return shape

    def src_tv_layout(self):
        shape, src_thrval_layout = chain(
            self.copy_atom.shape,
            self.copy_atom.src_thrval_layout,
            self.copy_atom.repeat_shape,
            self.copy_atom.repeat_layout,
            self.levels,
        )
        return shape, src_thrval_layout

    def dst_tv_layout(self):
        shape, dst_thrval_layout = chain(
            self.copy_atom.shape,
            self.copy_atom.dst_thrval_layout,
            self.copy_atom.repeat_shape,
            self.copy_atom.repeat_layout,
            self.levels,
        )
        return shape, dst_thrval_layout

    def str_indented(self, depth: int = 0):
        indent = " " * (depth * 2)
        prev_indent = " " * (max(0, depth - 1) * 2)
        return (
            "{"
            + f"\n{indent}atom: {self.copy_atom.str_indented(depth+1)}, \n{indent}levels:["
            + ", ".join([f"{level.str_indented(depth+1)}" for level in self.levels])
            + f"]\n{prev_indent}"
            + "}"
        )


class AutoCopy(TiledCopy):
    def __init__(self, tile_shape: Optional[Tuple[int, ...]] = None):
        super().__init__(None)
        self.tile_shape = tile_shape

    @property
    def shape(self):
        return self.tile_shape

    def src_tv_layout(self):
        raise NotImplementedError()

    def dst_tv_layout(self):
        raise NotImplementedError()

    def str_indented(self, depth: int = 0):
        shape_str = str(self.shape) if self.shape is not None else "auto"
        indent = " " * (depth * 2)
        prev_indent = " " * (max(0, depth - 1) * 2)
        return "{" + f"\n{indent}shape: {shape_str}, \n{indent}atom: auto, \n{indent}levels: auto\n{prev_indent}" + "}"


def auto_copy(shape: Optional[Tuple[int, ...]] = None):
    return AutoCopy(shape)


def is_auto_copy(tiled_copy: Union[AutoCopy, TiledCopy]):
    return isinstance(tiled_copy, AutoCopy)


class TiledMma:
    def __init__(self, mma_atom: MmaAtom, levels: List[Level] = None):
        if levels is None:
            levels = []
        self.mma_atom = mma_atom
        self.levels = sorted(levels, key=lambda x: x.level.value)
        shape, a_thrval_layout = chain(
            self.mma_atom.shape_mk(), self.mma_atom.a_thrval_layout, *self.mma_atom.repeat_mk(), self.levels_mk()
        )
        shape = compact_coshape(shape, a_thrval_layout)
        self.a_shape = shape
        self.a_thrval_layout = a_thrval_layout
        shape, b_thrval_layout = chain(
            self.mma_atom.shape_nk(), self.mma_atom.b_thrval_layout, *self.mma_atom.repeat_nk(), self.levels_nk()
        )
        shape = compact_coshape(shape, b_thrval_layout)
        self.b_shape = shape
        self.b_thrval_layout = b_thrval_layout
        shape, c_thrval_layout = chain(
            self.mma_atom.shape_mn(),
            self.mma_atom.c_thrval_layout,
            self.mma_atom.repeat_shape,
            self.mma_atom.repeat_layout,
            self.levels,
        )
        self.c_shape = shape
        self.c_thrval_layout = c_thrval_layout

    def levels_mk(self):
        return [lvl.level_mk() for lvl in self.levels]

    def levels_nk(self):
        return [lvl.level_nk() for lvl in self.levels]

    def a_tv_layout(self):
        return self.a_shape, self.a_thrval_layout

    def b_tv_layout(self):
        return self.b_shape, self.b_thrval_layout

    def c_tv_layout(self):
        return self.c_shape, self.c_thrval_layout

    def d_tv_layout(self):
        return self.c_tv_layout()

    def str_indented(self, depth: int = 0):
        indent = " " * (depth * 2)
        prev_indent = " " * (max(0, depth - 1) * 2)
        return (
            "{"
            + f"\n{indent}atom: {self.mma_atom.str_indented(depth+1)}, \n{indent}levels:["
            + ", ".join([f"{level.str_indented(depth+1)}" for level in self.levels])
            + f"]\n{prev_indent}"
            + "}"
        )

    def __str__(self):
        return self.str_indented()


class AutoMma(TiledMma):
    def __init__(self, shape_mnk: Optional[Tuple[int, ...]] = None):
        super().__init__(None)
        self.shape_mnk = shape_mnk

    def levels_mk(self):
        raise NotImplementedError()

    def levels_nk(self):
        raise NotImplementedError()

    def a_tv_layout(self):
        raise NotImplementedError()

    def b_tv_layout(self):
        raise NotImplementedError()

    def c_tv_layout(self):
        raise NotImplementedError()

    def d_tv_layout(self):
        raise NotImplementedError()

    def str_indented(self, depth: int = 0):
        shape_str = str(self.shape_mnk) if self.shape_mnk is not None else "auto"
        indent = " " * (depth * 2)
        prev_indent = " " * (max(0, depth - 1) * 2)
        return "{" + f"\n{indent}shape: {shape_str}, \n{indent}atom: auto, \n{indent}levels: auto\n{prev_indent}" + "}"


def auto_mma(shape: Optional[Tuple[int, ...]] = None):
    return AutoMma(shape)


def is_auto_mma(tiled_mma: Union[AutoMma, TiledMma]):
    return isinstance(tiled_mma, AutoMma)
