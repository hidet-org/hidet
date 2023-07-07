from typing import List
from ..base import SubgraphRewriteRule

from .embedding import SymmetricEmbeddingQuantizePattern, symmetric_embedding_quantize_patterns
from .linear import SymmetricLinearQuantizePatternL, SymmetricLinearQuantizePatternR, symmetric_linear_quantize_patterns


def default_quant_patterns(rules: List[SubgraphRewriteRule] = [], quant_type: str = 'int8', dims=0):
    symmetric_linear_quantize_patterns(rules, quant_type=quant_type, dims=dims)
    symmetric_embedding_quantize_patterns(rules, quant_type=quant_type)
    return rules
