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
import os


def get_size(path: str) -> int:
    if not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    size = 0
    for entry in os.scandir(path):
        size += get_size(entry.path)
    return size


def nbytes2str(nbytes: int) -> str:
    for uint in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if nbytes < 128:
            if isinstance(nbytes, int):
                return '{} {}'.format(nbytes, uint)
            else:
                return '{:.2f} {}'.format(nbytes, uint)
        nbytes /= 1024
    return '{:.2f} PiB'.format(nbytes)
