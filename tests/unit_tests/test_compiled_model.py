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
import numpy.testing
import hidet


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_load_save(device: str):
    # construct graph
    x = hidet.symbol([2, 3], device=device)
    w1 = hidet.randn([3, 4], device=device)
    w2 = hidet.randn([4, 5], device=device)
    y = hidet.ops.matmul(hidet.ops.matmul(x, w1), w2)

    # get computation graph
    graph = hidet.trace_from(y)

    # optimize the graph
    graph = hidet.graph.optimize(graph)

    # build the graph
    compiled_graph = graph.build()

    # save the model
    compiled_graph.save('./model.hidet')

    # load the model
    loaded_compiled_graph = hidet.load_compiled_graph('./model.hidet')

    # compare the results
    xx = hidet.randn([2, 3], device=device)
    y1 = graph(xx)
    y2 = compiled_graph(xx)
    y3 = loaded_compiled_graph(xx)

    numpy.testing.assert_allclose(y1.cpu().numpy(), y2.cpu().numpy())
    numpy.testing.assert_allclose(y1.cpu().numpy(), y3.cpu().numpy())
