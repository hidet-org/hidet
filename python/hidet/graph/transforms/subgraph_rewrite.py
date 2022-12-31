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
# pylint: disable=unused-import
from typing import List, Optional, Dict, Tuple, Set

from hidet.graph.ir import functors
from hidet.graph.ir.flow_graph import FlowGraph, Operator, Tensor
from hidet.graph.transforms import GraphPass, PassContext
from hidet.graph.ir.functors import analyze_usage, graph_collect
from hidet.utils import strict_zip
from .fold_const import fold_const_pass
from .graph_patterns import SubgraphRewriteRule, TensorPattern, OperatorPattern, MatchDict, Usage, graph_pattern_match
from .graph_patterns.base import registered_rewrite_rules, register_rewrite_rule


class SubgraphRewritePass(GraphPass):
    """
    A pattern transform can be conducted only if
    1. The pattern source matched the actual tensor and its spanned subregion.
    2. The intermediate tensor in the matched region should not be used. Only the output tensors can be used
       by not matched operators in original graph.
       For example, if pattern a -> b -> c matched x -> y -> z. We need to make sure y has not been
       used by other operators in the original graph.

    Time complexity of this implementation: O(num_applies * num_operators * num_patterns * pattern_size).
    """

    max_num_transforms = 1000

    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        graph = functors.clone(graph)
        fold_const = fold_const_pass()
        for _ in range(self.max_num_transforms):
            updated, graph = self.try_transform(graph, registered_rewrite_rules)
            graph = fold_const.process_graph(graph)
            if not updated:
                graph.update_nodes()
                return graph
        print('Exceeded maximum number of transforms {}, stop early.'.format(self.max_num_transforms))
        graph.update_nodes()
        return graph

    @staticmethod
    def match_pattern(graph_pattern: SubgraphRewriteRule, start_tensor: Tensor, usage: Usage) -> Optional[MatchDict]:
        source_output_tensors = graph_pattern.source()

        matched = graph_pattern_match(source_output_tensors[0], target=start_tensor, usage=usage)
        if matched is None:
            return None

        for source_output_tensor in source_output_tensors:
            if source_output_tensor not in matched:
                msg = 'The source pattern is not connected. Current we do not support disconnected patterns.'
                raise NotImplementedError(msg)

        return matched

    @staticmethod
    def check_usage_requirement(matched: MatchDict, usage: Usage, graph_pattern: SubgraphRewriteRule) -> bool:
        source_output_pattern_tensors: List[TensorPattern] = graph_pattern.source()

        # actual tensor -> pattern tensor
        tensor_map: Dict[Tensor, TensorPattern] = {v: k for k, v in matched.items() if isinstance(v, Tensor)}

        # find out all inner tensors (all matched tensors that are not matched by output tensors, nor input tensors)
        inner_tensors: List[Tensor] = []
        for actual_tensor in tensor_map:
            pattern_tensor = tensor_map[actual_tensor]
            # input tensor in pattern
            if pattern_tensor.trace is None:
                continue
            # output tensor in pattern
            if pattern_tensor in source_output_pattern_tensors:
                continue

        # check whether all inner tensors are only used by matched operators
        matched_operators: Set[Operator] = {v for v in matched.values() if isinstance(v, Operator)}
        for inner_tensor in inner_tensors:
            uses: List[Tuple[Optional[Operator], int]] = usage[inner_tensor]
            if any(use[0] not in matched_operators for use in uses):
                # used by not matched operator
                return False
        return True

    @staticmethod
    def try_transform(graph: FlowGraph, graph_patterns: List[SubgraphRewriteRule]) -> Tuple[bool, FlowGraph]:
        patterns: List[SubgraphRewriteRule] = graph_patterns
        usage: Usage = analyze_usage(graph)
        all_tensors: List[Tensor] = graph_collect(graph, Tensor)

        for graph_pattern in patterns:
            # print(graph_pattern.name)
            for start_tensor in all_tensors:
                # condition 1
                matched = SubgraphRewritePass.match_pattern(graph_pattern, start_tensor, usage)
                if matched is None:
                    continue

                # condition 2
                success = SubgraphRewritePass.check_usage_requirement(matched, usage, graph_pattern)
                if not success:
                    continue

                # generate target subgraph
                target_output_tensors: Optional[List[Tensor]] = graph_pattern.target(matched)
                if target_output_tensors is None:
                    # matched graph pattern can not be applied to this subgraph
                    continue

                # apply the graph transform
                if PassContext.current().configs['verbose']:
                    print('Applying transform: {}'.format(graph_pattern.name))
                source_output_pattern_tensors = graph_pattern.source()
                source_output_tensors = [matched[t] for t in source_output_pattern_tensors]
                for source_tensor, target_tensor in strict_zip(source_output_tensors, target_output_tensors):
                    for use in usage[source_tensor]:
                        op, idx = use
                        if op is None:
                            graph.outputs[idx] = target_tensor
                        else:
                            op.inputs[idx] = target_tensor
                return True, graph
        return False, graph


def subgraph_rewrite_pass() -> GraphPass:
    return SubgraphRewritePass()
