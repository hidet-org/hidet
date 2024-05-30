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
from typing import Union, Tuple, List
from hidet.ir.cute.layout import Label, Atom, Level, chain
from hidet.ir.cute import TensorLayout
from hidet.ir.cute.layout import label_names


class CopyAtom(Atom):
    def __init__(
        self,
        level: Union[str, Label],
        shape: Tuple[int],
        src_thrval_layout: TensorLayout,
        dst_thrval_layout: TensorLayout = None,
        repeat_shape: Tuple[int] = None,
        repeat_layout: TensorLayout = None,
    ):
        super().__init__(level, shape, repeat_shape, repeat_layout)
        if dst_thrval_layout is None:
            dst_thrval_layout = src_thrval_layout
        self.src_thrval_layout = src_thrval_layout
        self.dst_thrval_layout = dst_thrval_layout

    def str_indented(self, depth: int = 0):
        indent = ' ' * (depth * 2)
        prev_indent = ' ' * (max(0, depth - 1) * 2)
        return (
            "{"
            + f"\n{indent}level: {label_names[self.level]}, \n{indent}shape: {self.shape}, "
            + f"\n{indent}src: {self.src_thrval_layout}, \n{indent}dst: {self.dst_thrval_layout}"
            + f"\n{indent}repeat_shape: {self.repeat_shape}, \n{indent}repeat_layout: {self.repeat_layout}"
            + f"\n{prev_indent}"
            + "}"
        )


class TiledCopy:
    def __init__(self, copy_atom: CopyAtom, levels: List[Level] = None):
        if levels is None:
            levels = []
        self.copy_atom = copy_atom
        self.levels = levels

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
        indent = ' ' * (depth * 2)
        prev_indent = ' ' * (max(0, depth - 1) * 2)
        return (
            "{"
            + f"\n{indent}atom: {self.copy_atom.str_indented(depth+1)}, \n{indent}levels:["
            + ", ".join([f"{level.str_indented(depth+1)}" for level in self.levels])
            + f"]\n{prev_indent}"
            + "}"
        )
