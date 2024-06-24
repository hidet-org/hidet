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
from typing import List
from hidet.ir.cute.layout import CopyAtom, Level, chain
from hidet.ir.cute import TensorLayout, TiledTensorLayout, filter
from hidet.ir.cute.int_tuple import rank


class TiledCopy:
    def __init__(self, copy_atom: CopyAtom, levels: List[Level] = None):
        if levels is None:
            levels = []
        self.copy_atom = copy_atom
        self.levels = levels

    @staticmethod
    def from_tiled_tensor_layout(layout: TiledTensorLayout):
        copy_atom = CopyAtom.from_tv_atom(layout.atom)
        return TiledCopy(copy_atom, layout.levels)

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

    def val_remain_layout(self):
        from hidet.ir.cute.int_tuple import depth

        shape, src_thrval_layout = self.src_tv_layout()
        atom_shape = self.copy_atom.shape
        if atom_shape == shape:
            val = self.copy_atom.src_thrval_layout[1]
            assert depth(val.stride_tuple) == 1
            if rank(val.stride) == 1:
                ret = TensorLayout((1,), val.stride_tuple)
            else:
                ret = val[1:]
        else:
            ret = src_thrval_layout[1]
        return filter(ret, False)

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
