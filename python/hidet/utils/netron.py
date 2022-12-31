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
from typing import List, Union
import json
from collections import defaultdict


class Model:
    def __init__(self, graph, description="", author="", company="", license="", domain="", source=""):
        self.graphs: List[Graph] = [graph]
        self.description: str = description
        self.author: str = author
        self.company: str = company
        self.license: str = license
        self.domain: str = domain
        self.source: str = source
        self.format: str = 'netron'

    def export(self):
        return {
            'graphs': [graph.export() for graph in self.graphs],
            'description': self.description,
            'author': self.author,
            'company': self.company,
            'license': self.license,
            'domain': self.domain,
            'source': self.source,
            'format': self.format,
        }


class Graph:
    def __init__(self, inputs, outputs, nodes, name=""):
        self.inputs: List[Parameter] = inputs
        self.outputs: List[Parameter] = outputs
        self.nodes: List[Node] = nodes
        self.name: str = name

    def export(self):
        return {
            'name': self.name,
            'inputs': [param.export() for param in self.inputs],
            'outputs': [param.export() for param in self.outputs],
            'nodes': [node.export() for node in self.nodes],
        }


class Parameter:
    def __init__(self, name, argument, visible=True):
        self.name: str = name
        self.arguments: List[Argument] = [argument]
        self.visible: bool = visible

    def export(self):
        return {'name': self.name, 'arguments': [arg.export() for arg in self.arguments], 'visible': self.visible}


class Argument:
    def __init__(self, name, data_type, shape: Union[str, List[int]], has_initializer=False, scalar_value=None):
        self.name: str = name
        self.data_type: str = data_type
        self.shape: Union[str, List[int]] = shape
        self.has_initializer: bool = has_initializer
        self.scalar_value = scalar_value

    def export(self):
        ret = {
            'name': self.name,
            'type': {
                "string": '{}{}'.format(self.data_type, self.shape),
                "shape": {'dimensions': self.shape},
                "dataType": self.data_type,
            },
        }
        if self.has_initializer:
            ret['initializer'] = {'kind': 'Initializer'}
            if len(self.shape) == 0 and self.scalar_value is not None:
                ret['initializer']['value'] = str(self.scalar_value)
            else:
                ret['initializer']['value'] = '<>'
        return ret


class Node:
    # category influence the color in netron
    categories = {
        'layer': ['Conv2d', 'Matmul'],
        'constant': [],
        'activation': ['Relu'],
        'pool': ['MaxPool2d', 'AvgPool2d'],
        'normalization': [],
        'dropout': [],
        'transform': ['Squeeze', 'Unsqueeze', 'Add', 'Sub', 'Multiply', 'Rsqrt'],
        'custom': [],
    }

    def __init__(self, name, type_name, inputs, outputs, attributes, category=None, description=''):
        self.name: str = name
        self.type_name: str = type_name
        self.inputs: List[Parameter] = inputs
        self.outputs: List[Parameter] = outputs
        self.attributes: List[Attribute] = attributes
        self.description: Union[List[str], str] = description.split('\n')
        self.category = category
        if self.category is None:
            if self.type_name.startswith('Fused') or ' ' in self.type_name:
                # fused op, use the color of 'dropout'
                self.category = 'dropout'
            elif self.type_name.startswith('Conv2dGemm') or self.type_name.startswith('Conv2dWinograd'):
                self.category = 'custom'
            else:
                for cat, ops in self.categories.items():
                    if type_name in ops:
                        self.category = cat
                        break

    def export(self):
        return {
            'name': self.name,
            'type': {'name': self.type_name, 'category': self.category},
            'inputs': [param.export() for param in self.inputs],
            'outputs': [param.export() for param in self.outputs],
            'attributes': [attr.export() for attr in self.attributes],
            'description': self.description,
        }


class Attribute:
    def __init__(self, name, type_name: str, value: str, visible=True, description=""):
        self.name: str = name
        self.type_name: str = type_name
        self.value: str = value
        self.visible: bool = visible
        self.description: str = description

    def export(self):
        return {
            'name': self.name,
            'type': self.type_name,
            'value': self.value,
            'visible': self.visible,
            'description': self.description,
        }


def type_string_of(value):
    if isinstance(value, (list, tuple)):
        if len(value) > 0:
            return 'Sequence[{}]'.format(type(value[0]).__name__)
        else:
            return 'Sequence[]'
    else:
        return str(type(value).__name__)


def dump(flow_graph, fp):
    from hidet import FlowGraph

    assert isinstance(flow_graph, FlowGraph)
    flow_graph.update_nodes()
    tensor2argument = {}
    node2idx = defaultdict(int)

    inputs = []
    outputs = []
    nodes = []
    for idx, tensor in enumerate(flow_graph.inputs):
        name = 'input:{}'.format(idx)
        argument = Argument(name, data_type=tensor.dtype.name, shape=tensor.shape, has_initializer=False)
        tensor2argument[tensor] = argument
        inputs.append(Parameter(name, argument))

    constant_cnt = 0
    for node in flow_graph.nodes:
        node_type = node.name
        node2idx[node_type] += 1
        node_name = '{}{}'.format(node_type, node2idx[node_type])
        for idx, tensor in enumerate(node.inputs):
            if tensor.storage is None:  # not a constant
                continue
            if tensor in tensor2argument:  # constant shared by multiple nodes
                continue
            name = 'const:{}'.format(constant_cnt)
            constant_cnt += 1
            scalar_value = str(tensor.cpu().numpy()) if len(tensor.shape) == 0 and tensor.storage else None
            tensor2argument[tensor] = Argument(
                name, data_type=tensor.dtype.name, shape=tensor.shape, has_initializer=True, scalar_value=scalar_value
            )
        for idx, tensor in enumerate(node.outputs):
            name = '{}:{}'.format(node_name, idx)
            tensor2argument[tensor] = Argument(
                name, data_type=tensor.dtype.name, shape=tensor.shape, has_initializer=False
            )
        nodes.append(
            Node(
                name=node_name,
                type_name=node_type,
                inputs=[Parameter(str(idx), tensor2argument[tensor]) for idx, tensor in enumerate(node.inputs)],
                outputs=[Parameter(str(idx), tensor2argument[tensor]) for idx, tensor in enumerate(node.outputs)],
                attributes=[Attribute(name, type_string_of(value), str(value)) for name, value in node.attrs.items()],
                description="{}".format(str(node.task)),
            )
        )
    for idx, tensor in enumerate(flow_graph.outputs):
        if tensor in tensor2argument:
            outputs.append(Parameter('output:{}'.format(idx), tensor2argument[tensor]))
        else:
            outputs.append(
                Parameter(
                    'output:{}'.format(idx),
                    Argument(
                        'output:{}'.format(idx), data_type=tensor.dtype.name, shape=tensor.shape, has_initializer=False
                    ),
                )
            )
    graph = Graph(inputs, outputs, nodes, name="")
    model = Model(graph, source='Hidet', description='Converted from FlowGraph')

    json.dump(model.export(), fp, indent=2)
