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

"""
**Please note that most features under this module only support 1-D partitioning**, because it is
mainly designed for single-machine-multi-GPU settings. This module can automatically search intra-op
partition plan (data-parallel and tensor-parallel), while pipeline-parallel will be processed in
other modules.

We are planning to extend it to 2-D (multi-machine-multi-GPU) in the future.
"""

from .partition import partition
