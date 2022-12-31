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
from hidet.graph.frontend.torch import availability as torch_availability
from .bench import bench_group

if not torch_availability.dynamo_available():
    raise RuntimeError(
        'PyTorch version is less than 2.0. Please upgrade PyTorch to 2.0 or higher to enable torch dynamo'
        'which is required by the benchmark scripts.'
    )
