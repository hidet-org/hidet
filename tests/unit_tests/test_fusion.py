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


def test_fusion_v1():
    @hidet.jit(opt=True)
    def func(a: hidet.Tensor, b: hidet.Tensor):
        c = hidet.ops.equal(a, b)
        d = hidet.ops.logical_not(c)
        e = d.astype('int32')
        f = hidet.ops.cumsum(e, dim=1)
        g = f * e
        h = g.astype('int64')
        i = h + 1
        return i

    a = hidet.zeros([1, 9], dtype='int64')
    b = hidet.zeros([], dtype='int64')
    func(a, b)
