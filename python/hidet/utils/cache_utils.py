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
import hidet.option


def cache_dir(*items: str) -> str:
    root = hidet.option.get_cache_dir()
    ret = os.path.abspath(os.path.join(root, *items))
    os.makedirs(ret, exist_ok=True)
    return ret


def cache_file(*items: str) -> str:
    root_dir = cache_dir('./')
    ret = os.path.abspath(os.path.join(root_dir, *items))
    os.makedirs(os.path.dirname(ret), exist_ok=True)
    return ret


def clear_cache_dir(*items: str):
    root = hidet.option.get_cache_dir()
    dir_to_clear = os.path.abspath(os.path.join(root, *items))
    print('Clearing hidet cache dir: {}'.format(dir_to_clear))
    shutil.rmtree(dir_to_clear, ignore_errors=True)


def clear_op_cache():
    clear_cache_dir('ops')
