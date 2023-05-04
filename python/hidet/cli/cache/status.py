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
from tabulate import tabulate
import click
import hidet
from .utils import nbytes2str, get_size


@click.command(name='status', help='Show cache status.')
def hidet_cache_status():
    cache_dir: str = hidet.option.get_cache_dir()
    if not os.path.exists(cache_dir):
        print('Cache directory does not exist: {}'.format(cache_dir))
        return
    table = []
    print('Hidet cache directory: {}'.format(cache_dir))
    for entry in os.scandir(cache_dir):
        relpath = os.path.relpath(entry.path, cache_dir)
        if entry.is_dir():
            relpath += '/'
        table.append([relpath, get_size(entry.path)])
    table.sort(key=lambda x: x[1], reverse=True)
    table = [[a, nbytes2str(b)] for a, b in table]
    print(tabulate(table, headers=['Path', 'Size'], colalign=('right', 'right')))
