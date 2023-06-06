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
from typing import Union, List, Tuple

import pytest
import numpy as np

import hidet as hi
from hidet import ops

from hidet.testing import check_unary, check_unary_dynamic


def numpy_softmax(data, axis):
    data = np.exp(data - np.max(data, axis, keepdims=True))
    data = data / np.sum(data, axis, keepdims=True)
    return data


@pytest.mark.parametrize(
    "shape, axis",
    [[[1, 1000], 1], [[16, 1000], 1], [[1, 1000, 1, 1], 1], [[16, 1000, 1, 1], 1], [[1, 128, 128, 128], 2]],
)
def test_softmax(shape, axis):
    check_unary(
        shape, lambda x: numpy_softmax(x, axis), lambda x: ops.softmax(x, axis), dtype='float32', atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize(
    "shape, axis",
    [[[1, ("x", 1000)], 1], [[("x", 16), 1000], 1], 
     [[1, ("x", 1000), ('y', 1), 1], 1], [[("x", 16), ("y", 1000), ("z", 1), ("w", 1)], 1], 
     [[1, ("x", 128), ("y", 128), ("z", 128)], 2]],
)
def test_softmax_dynamic(shape, axis):
    check_unary_dynamic(
        shape, lambda x: numpy_softmax(x, axis), lambda x: ops.softmax(x, axis), dtype='float32', atol=1e-5, rtol=1e-5
    )
