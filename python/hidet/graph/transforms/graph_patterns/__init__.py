from typing import List
from .base import TensorPattern, OperatorPattern, GraphPattern, MatchDict, Usage, graph_pattern_match
from .arithmetic_patterns import arithmetic_patterns
from .transform_patterns import transform_patterns
from .conv2d_patterns import conv2d_patterns
from .matmul_patterns import matmul_patterns


def all_graph_patterns() -> List[GraphPattern]:
    return arithmetic_patterns() + transform_patterns() + conv2d_patterns() + matmul_patterns()
