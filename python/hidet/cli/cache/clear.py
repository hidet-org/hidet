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
import shutil
import click
import hidet
from .utils import get_size, nbytes2str


@click.command(name='clear', help='Clear the hidet cache.')
@click.option('--all', is_flag=True, default=False, help='Clear all the cache.')
def hidet_cache_clear(all: bool):
    if all:
        cache_dir: str = hidet.option.get_cache_dir()
        print('Clearing hidet cache: {}'.format(cache_dir))
        freed_space: int = get_size(cache_dir)
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
    else:
        op_cache_dir: str = os.path.join(hidet.option.get_cache_dir(), 'ops')
        print('Clearing hidet ops cache: {}'.format(op_cache_dir))
        freed_space: int = get_size(op_cache_dir)
        if os.path.exists(op_cache_dir):
            shutil.rmtree(op_cache_dir, ignore_errors=True)

    print('Freed space: {}'.format(nbytes2str(freed_space)))
