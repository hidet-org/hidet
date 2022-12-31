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
from typing import Iterable
from collections import defaultdict


class Namer:
    def __init__(self):
        self.name_id_clock = defaultdict(int)
        self.obj_name = {}
        self.clear()

    def __call__(self, x):
        return self.get_name(x)

    def clear(self):
        self.name_id_clock.clear()
        self.obj_name.clear()
        # add keywords in target language
        keywords = ['const']
        for kw in keywords:
            self.name_id_clock[kw] = 0

    def get_name(self, e, hint=None):
        from hidet.ir.expr import Var
        from hidet.ir.compute import ScalarNode, TensorNode
        from hidet.graph.tensor import Tensor

        if e in self.obj_name:
            return self.obj_name[e]
        if hint:
            orig_name = hint
        elif isinstance(e, Var) and e.hint is not None:
            orig_name = e.hint
        elif isinstance(e, (ScalarNode, TensorNode)):
            orig_name = e.name
        else:
            alias = {ScalarNode: 'scalar', TensorNode: 'tensor', Var: 'v', Tensor: 'x'}
            orig_name = alias[type(e)] if type(e) in alias else type(e).__name__

        if orig_name in self.name_id_clock:
            name = orig_name
            while name in self.name_id_clock:
                self.name_id_clock[orig_name] += 1
                name = orig_name + '_' + str(self.name_id_clock[orig_name])
        else:
            self.name_id_clock[orig_name] = 0
            name = orig_name

        self.obj_name[e] = name
        return name

    @staticmethod
    def unique_name_among(name: str, existed_names: Iterable[str]) -> str:
        name_set = set(existed_names)
        if name not in name_set:
            return name
        else:
            i = 1
            while name + '_' + str(i) in name_set:
                i += 1
            return name + '_' + str(i)
