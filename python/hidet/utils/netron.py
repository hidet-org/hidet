from typing import List, Union
import json
from collections import defaultdict


class Model:
    def __init__(self, graph, description="", author="", company="", license="", domain="", source=""):
        self.graphs: List[Graph] = [graph]
        self.description = description
        self.author = author
        self.company = company
        self.license = license,
        self.domain = domain
        self.source = source
        self.format = 'netron'

    def export(self):
        return {
            'graphs': [graph.export() for graph in self.graphs],
            'description': self.description,
            'author': self.author,
            'company': self.company,
            'license': self.license,
            'domain': self.domain,
            'source': self.source,
            'format': self.format
        }


class Graph:
    def __init__(self, inputs, outputs, nodes, name=""):
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = nodes
        self.name = name

    def export(self):
        return {
            'name': self.name,
            'inputs': [param.export() for param in self.inputs],
            'outputs': [param.export() for param in self.outputs],
            'nodes': [node.export() for node in self.nodes]
        }


class Parameter:
    def __init__(self, name, argument, visible=True):
        self.name = name
        self.arguments: List[Argument] = [argument]
        self.visible = visible

    def export(self):
        return {
            'name': self.name,
            'arguments': [arg.export() for arg in self.arguments],
            'visible': self.visible
        }


class Argument:
    def __init__(self, name, data_type, shape: Union[str, List[int]], has_initializer=False):
        self.name = name
        self.data_type = data_type
        self.shape = shape
        self.has_initializer = has_initializer

    def export(self):
        ret = {
            'name': self.name,
            'type': {
                "string": '{}{}'.format(self.data_type, self.shape),
                "shape": {'dimensions': self.shape},
                "dataType": self.data_type
            }
        }
        if self.has_initializer:
            ret['initializer'] = {'kind': 'Initializer'}
        return ret


class Node:
    def __init__(self, name, type_name, inputs, outputs, attributes):
        self.name: str = name
        self.type_name: str = type_name
        self.inputs: List[Parameter] = inputs
        self.outputs: List[Parameter] = outputs
        self.attributes: List[Attribute] = attributes

    def export(self):
        return {
            'name': self.name,
            'type': {
                'name': self.type_name
            },
            'inputs': [param.export() for param in self.inputs],
            'outputs': [param.export() for param in self.outputs],
            'attributes': [attr.export() for attr in self.attributes]
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
            'description': self.description
        }


def dump(flow_graph, fp):
    from hidet import FlowGraph
    assert isinstance(flow_graph, FlowGraph)
    tensor2argument = {}
    node2idx = defaultdict(int)

    inputs = []
    outputs = []
    nodes = []
    for idx, tensor in enumerate(flow_graph.inputs):
        name = 'input:{}'.format(idx)
        argument = Argument(name, data_type=tensor.dtype, shape=tensor.shape, has_initializer=False)
        tensor2argument[tensor] = argument
        inputs.append(Parameter(name, argument))

    constant_cnt = 0
    for node in flow_graph.nodes:
        node_type = node.__class__.__name__[:-2]
        node2idx[node_type] += 1
        node_name = '{}{}'.format(node_type, node2idx[node_type])
        for idx, tensor in enumerate(node.inputs):
            if tensor.storage is None:  # not a constant
                continue
            if tensor in tensor2argument:   # constant shared by multiple nodes
                continue
            name = 'const:{}'.format(constant_cnt)
            constant_cnt += 1
            tensor2argument[tensor] = Argument(name, data_type=tensor.dtype, shape=tensor.shape, has_initializer=True)
        for idx, tensor in enumerate(node.outputs):
            name = '{}:{}'.format(node_name, idx)
            tensor2argument[tensor] = Argument(name, data_type=tensor.dtype, shape=tensor.shape, has_initializer=False)
        nodes.append(Node(
            name=node_name,
            type_name=node_type,
            inputs=[Parameter(str(idx), tensor2argument[tensor]) for idx, tensor in enumerate(node.inputs)],
            outputs=[Parameter(str(idx), tensor2argument[tensor]) for idx, tensor in enumerate(node.outputs)],
            attributes={}
        ))
    for idx, tensor in enumerate(flow_graph.outputs):
        outputs.append(Parameter('output:{}'.format(idx), tensor2argument[tensor]))
    graph = Graph(inputs, outputs, nodes, name="")
    model = Model(graph, source='Hidet', description='Converted from FlowGraph')

    json.dump(model.export(), fp, indent=2)



