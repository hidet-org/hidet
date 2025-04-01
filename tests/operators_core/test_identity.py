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
import pytest
import numpy as np
import hidet
from hidet.testing.utils import check_3_execution_paths


def test_remove_identity_ops_1():
    shape = [2, 4, 8]

    def hidet_ops(x):
        o = hidet.ops.flatten(x)
        o = hidet.ops.reshape(o, [2, 4, 8])
        o = hidet.ops.unsqueeze(o, [1])
        return o

    def numpy_ops(x):
        yy = x.flatten().reshape(2, 4, 8)
        yy = np.expand_dims(yy, axis=1)
        return yy

    check_3_execution_paths(shape, hidet_ops, numpy_ops, device='cuda')


def test_remove_identity_ops_2():
    shape = [2, 4, 8]

    def hidet_ops(x):
        o = hidet.ops.sum(x, [0])
        o = hidet.ops.reshape(o, [2, 2, 8])
        o = hidet.ops.unsqueeze(o, [1])
        return o

    def numpy_ops(x):
        yy = np.sum(x, axis=0).reshape((2, 2, 8))
        yy = np.expand_dims(yy, axis=1)
        return yy

    check_3_execution_paths(shape, hidet_ops, numpy_ops, device='cuda')


def test_remove_identity_ops_3():
    shape = [2, 4, 8]

    def hidet_ops(x):
        o = x
        o = hidet.ops.flatten(o)
        o = hidet.ops.reshape(o, shape)
        o = hidet.ops.sum(o, [0])
        o = hidet.ops.unsqueeze(o, [-1])
        o = hidet.ops.squeeze(o, [-1])
        o = hidet.ops.sum(o, [0])
        o = hidet.ops.reshape(o, [2, 4])
        o = hidet.ops.unsqueeze(o, [1])
        return o

    def numpy_ops(x):
        yy = np.sum(x, axis=0)
        yy = np.sum(yy, axis=0).reshape((2, 4))
        yy = np.expand_dims(yy, axis=1)
        return yy

    check_3_execution_paths(shape, hidet_ops, numpy_ops, device='cuda')


if __name__ == '__main__':
    test_remove_identity_ops_1()
    test_remove_identity_ops_2()
    test_remove_identity_ops_3()
