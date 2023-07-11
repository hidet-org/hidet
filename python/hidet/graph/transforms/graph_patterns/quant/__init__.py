from typing import List, Union
from ..base import SubgraphRewriteRule

from .embedding import symmetric_embedding_quantize_patterns
from .linear import symmetric_linear_quantize_patterns, matmul_specialization_rules


def default_patterns() -> List[SubgraphRewriteRule]:
    """
    Returns a list of default quantization patterns. Where the quant type is int8
    """
    quant_type = 'int8'
    dims = 0
    rules = symmetric_linear_patterns(quant_type=quant_type, dims=dims)
    rules += symmetric_embedding_patterns(quant_type=quant_type)
    return rules


def symmetric_linear_patterns(quant_type: str = 'int8', dims: Union[int, List[int]] = 0) -> List[SubgraphRewriteRule]:
    """
    Adds channel-wise symmetric weight quantization to linear layers.
    A subgraph is deemed a linear layer if it is a matmul op with a constant weight tensor.

    Parameters
    ----------
    quant_type : str
        Quantization type. One of ['int8', 'int16']
    dims : Union[int, List[int]]
        Axis to quantize over, where each axis has independent quantization parameters.
    """
    rules = symmetric_linear_quantize_patterns(quant_type=quant_type, dims=dims)
    rules.extend(matmul_specialization_rules())
    return rules


def symmetric_embedding_patterns(quant_type: str = 'int8') -> List[SubgraphRewriteRule]:
    """
    Adds channel-wise symmetric weight quantization to embedding layers.
    A subgraph is deemed an embedding layer if it is a TakeOp with a constant weight tensor.

    Parameters
    ----------
    quant_type : str
        Quantization type. One of ['int8', 'int16']
    """
    rules = symmetric_embedding_quantize_patterns(quant_type=quant_type)
    return rules
