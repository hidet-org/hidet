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
import hidet
import pytest


def test_profile_config():
    a = hidet.randn([1, 10, 10], device='cuda')
    b = hidet.randn([1, 10, 10], device='cuda')
    hidet.option.search_space(1)
    hidet.option.bench_config(1, 1, 1)
    c = hidet.ops.batch_matmul(a, b)
    hidet.option.search_space(0)


if __name__ == '__main__':
    pytest.main(__file__)
