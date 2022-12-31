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


def hidet_cache_dir(category='./') -> str:
    root = hidet.option.get_cache_dir()
    ret = os.path.abspath(os.path.join(root, category))
    os.makedirs(ret, exist_ok=True)
    return ret


def hidet_cache_file(*items: str) -> str:
    root_dir = hidet_cache_dir('./')
    ret_path = os.path.abspath(os.path.join(root_dir, *items))
    os.makedirs(os.path.dirname(ret_path), exist_ok=True)
    return ret_path


def hidet_clear_op_cache():
    op_cache = hidet_cache_dir('ops')
    print('Clearing operator cache: {}'.format(op_cache))
    shutil.rmtree(op_cache, ignore_errors=True)
