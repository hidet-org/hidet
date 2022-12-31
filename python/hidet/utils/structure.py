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
from __future__ import annotations

from typing import TypeVar, Dict, List, Tuple, Callable

GraphNode = TypeVar('GraphNode')


class DirectedGraph:
    """Directed graph.

    A directed graph representation.
    """

    def __init__(self):
        self.adj_list: Dict[GraphNode, List[GraphNode]] = {}

    def __contains__(self, item: GraphNode):
        return item in self.adj_list

    def astext(self, node2str: Callable[[GraphNode], str] = object.__str__) -> str:
        node_name: Dict[GraphNode, str] = {node: node2str(node) for node in self.adj_list}
        max_length = max(len(v) for v in node_name.values())
        buf: List[str] = []
        for u in self.adj_list:
            head = '{{:>{}}}: '.format(max_length).format(node_name[u])
            tail = ', '.join([node_name[v] for v in self.adj_list[u]])
            buf.append(head + tail)
        return '\n'.join(buf)

    @staticmethod
    def from_edges(edges: List[Tuple[GraphNode, GraphNode]]) -> DirectedGraph:
        """Create a directed graph from edges.

        The edges should be a list of (src, dst) tuples, and each tuple represents an edge.

        Parameters
        ----------
        edges: List[Tuple[GraphNode, GraphNode]]
            The edges of the directed graph to be created.

        Returns
        -------
        ret: DirectedGraph
            The created directed graph.
        """
        graph = DirectedGraph()
        for u, v in edges:
            graph.add_edge(u, v)
        return graph

    def has_node(self, node: GraphNode) -> bool:
        """Whether the node has been added to the graph.

        Parameters
        ----------
        node: GraphNode
            The node to be checked.

        Returns
        -------
        ret: bool
            True if the node has been added.
        """
        return node in self.adj_list

    def has_edge(self, src: GraphNode, dst: GraphNode) -> bool:
        """Whether there is an edge (src, dst) in the graph.

        Parameters
        ----------
        src: GraphNode
            The source node of the edge.
        dst: GraphNode
            The destination node of the edge.

        Returns
        -------
        ret: bool
            True if the edge (src, dst) is in the graph.
        """
        return src in self.adj_list and dst in self.adj_list[src]

    def add_node(self, node: GraphNode):
        """Add a node.

        Parameters
        ----------
        node: GraphNode
            The node to be added to the graph.
        """
        if not self.has_node(node):
            self.adj_list[node] = []

    def add_edge(self, src: GraphNode, dst: GraphNode):
        """Add an edge.

        The node `src` and `dst` will be added to the graph if they have not been added.

        Parameters
        ----------
        src: GraphNode
            The source of the edge.

        dst: GraphNode
            The destination of the edge.
        """
        if not self.has_node(src):
            self.add_node(src)
        if not self.has_node(dst):
            self.add_node(dst)
        self.adj_list[src].append(dst)

    def topological_order(self) -> List[GraphNode]:
        """Get a topological order of the nodes in the directed graph.

        Returns
        -------
        ret: List[GraphNode]
            The nodes in the topological order.

        Raises
        ------
        ValueError
            If the directed graph is cyclic (i.e., there is a loop in the graph).
        """
        in_degree: Dict[GraphNode, int] = {node: 0 for node in self.adj_list}
        for u in self.adj_list:
            for v in self.adj_list[u]:
                in_degree[v] += 1

        qu: List[GraphNode] = []
        for node, degree in in_degree.items():
            if degree == 0:
                qu.append(node)

        order: List[GraphNode] = []
        while len(qu) > 0:
            u = qu.pop()
            order.append(u)
            for v in self.adj_list[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    qu.append(v)

        if len(order) != len(self.adj_list):
            raise ValueError('Loop detected during generating topological order for a directed graph.')

        return order
