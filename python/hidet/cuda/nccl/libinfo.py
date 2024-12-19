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


def _get_nccl_dirs():
    import site

    res = [os.path.join(root, 'nvidia', 'nccl') for root in site.getsitepackages()]
    res += [os.path.join(site.getusersitepackages(), 'nvidia', 'nccl')]
    return res


def get_nccl_include_dirs():
    return [os.path.join(root, 'include') for root in _get_nccl_dirs()]


def get_nccl_library_search_dirs():
    return [os.path.join(root, 'lib') for root in _get_nccl_dirs()]
