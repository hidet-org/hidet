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
from __future__ import annotations
from typing import List, Dict


class Target:
    _supported_targets = ['cuda', 'hip', 'cpu']

    def __init__(self, name: str, flags: List[str], attrs: Dict[str, str]):
        if name not in self._supported_targets:
            raise ValueError('Does not support target {}, candidates {}.'.format(name, self._supported_targets))
        self.name = name
        self.flags: List[str] = flags
        self.attrs: Dict[str, str] = attrs

        self._check()

    @staticmethod
    def from_string(target_string: str) -> Target:
        items: List[str] = target_string.strip().split()
        name, items = items[0], items[1:]
        flags = []
        attrs = {}
        for item in items:
            if item.startswith('--'):
                key, value = item[2:].split('=')
                attrs[key] = value
            elif item.startswith('-'):
                flags.append(item)
            else:
                raise ValueError('Cannot recognize target item "{}".'.format(item))
        return Target(name, flags, attrs)

    def _check(self):
        if self.name == 'cpu':
            valid_flags = []
            valid_attrs = ['arch']
        elif self.name == 'cuda':
            valid_flags = []
            valid_attrs = ['arch', 'cpu_arch']  # e.g., '--arch=sm_80', '--cpu_arch=x86-64'
        elif self.name == 'hip':
            valid_flags = []
            valid_attrs = []
        else:
            raise ValueError('Cannot recognize target "{}".'.format(self.name))
        for flag in self.flags:
            if flag not in valid_flags:
                raise ValueError('Invalid flag "{}" for target "{}".'.format(flag, self.name))
        for attr in self.attrs:
            if attr not in valid_attrs:
                raise ValueError('Invalid attribute "{}" for target "{}".'.format(attr, self.name))
