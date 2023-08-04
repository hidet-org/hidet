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
from hidet.distributed.partition.rule import op_shard_rule_search


def test_matmul():
    A = hidet.symbol((32, 64))
    B = hidet.symbol((64, 32))
    AB = A @ B
    rules = op_shard_rule_search(AB.op, 4)
    assert len(rules) == 5


def test_conv():
    A = hidet.symbol((32, 64, 224, 224))
    B = hidet.symbol((64, 64, 3, 3))
    AB = hidet.ops.conv2d(A, B)
    rules = op_shard_rule_search(AB.op, 4)
    assert len(rules) == 5


def test_reshape():
    x = hidet.symbol((4, 8, 4))
    y = x.reshape((32, -1))
    rules = op_shard_rule_search(y.op, 4)
    assert len(rules) == 2

    y = x.reshape((-1, 32))
    rules = op_shard_rule_search(y.op, 4)
    assert len(rules) == 2


def test_transpose():
    x = hidet.symbol((4, 8, 12, 16))
    y = x.transpose(1, 3)
    rules = op_shard_rule_search(y.op, 4)
    assert len(rules) == 5


def test_resnet():
    import hidet.testing

    model = hidet.testing.models.resnet.resnet18()
    x = hidet.symbol([8, 3, 224, 224])
    flow_graph = hidet.trace_from(model(x))
    visited = set()

    for op in flow_graph.nodes:
        if str(op) in visited:
            continue
        visited.add(str(op))
        rules = op_shard_rule_search(op, 4)
        assert len(rules) > 0


def test_pad():
    x = hidet.symbol((4, 4, 4))
    y = hidet.ops.pad(x, (1, 1, 0, 0, 0, 0))
    rules = op_shard_rule_search(y.op, 4)
    assert len(rules) == 2
