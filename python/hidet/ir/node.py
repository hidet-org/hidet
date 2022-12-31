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
from typing import Mapping, Type, Any, List


class Node:
    _dispatch_index = {None: 0}

    def __str__(self):
        from hidet.ir.functors.printer import astext  # pylint: disable=import-outside-toplevel

        return astext(self)

    def __repr__(self):
        return str(self)

    def __int__(self):
        return None

    @classmethod
    def class_index(cls):
        if not hasattr(cls, '_class_index'):
            setattr(cls, '_class_index', len(Node._dispatch_index))
            Node._dispatch_index[cls] = getattr(cls, '_class_index')
        return getattr(cls, '_class_index')

    @staticmethod
    def dispatch_table(mapping: Mapping[Type['Node'], Any]) -> List[Any]:
        table = []
        for cls, target in mapping.items():
            idx = cls.class_index()
            while idx >= len(table):
                table.append(None)
            table[idx] = target
        return table
