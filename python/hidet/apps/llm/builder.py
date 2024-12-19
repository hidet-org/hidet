"""
Builds a LLM app.
"""
from typing import Optional, Type, List
from transformers import PretrainedConfig, AutoConfig
from hidet.ir.dtypes import int32, int64, DataType
from hidet.graph import FlowGraph
from hidet.apps.llm.app import LLM
from hidet.graph.tensor import symbol, Tensor
from hidet.apps.llm.modeling import registry, PretrainedModelForCausalLM
from hidet.apps.llm.nn.attention import PagedAttnState
from hidet.runtime.compiled_app import create_compiled_app
from hidet.runtime.compiled_graph import CompiledGraph
from hidet.utils.py import release_unused_resources


def _load_pretrained_config(model: str, revision: Optional[str]) -> PretrainedConfig:
    try:
        return AutoConfig.from_pretrained(model, revision=revision)
    except ValueError as e:
        raise e


def _get_model_class(config: PretrainedConfig) -> Type[PretrainedModelForCausalLM]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        model_class = registry.load_model_class(arch)
        if model_class is not None:
            return model_class
    supported = '\n'.join(registry.supported_architectures())
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. \n" + f"Supported architectures: \n"
        f"{supported}"
    )


def _build_prefill_graph(
    model: PretrainedModelForCausalLM, device: str, block_size: int, kernel_search_space: int
) -> CompiledGraph:
    import hidet  # pylint: disable=redefined-outer-name

    num_layers: int = model.num_attention_layers()
    num_heads: int = model.num_attention_heads()
    head_size: int = model.attention_head_size()
    dtype: DataType = model.dtype()

    # create the input tensors
    input_ids: Tensor = symbol(['bs', 'seq'], dtype=int32, device=device)
    position_ids: Tensor = symbol(['bs', 'seq'], dtype=int32, device=device)
    cache_slots: Tensor = symbol(['bs', 'seq'], dtype=int64, device=device)
    seq_lengths: Tensor = symbol(['bs'], dtype=int32, device=device)
    key_caches: List[Tensor] = [
        symbol(['blocks', num_heads, head_size, block_size], dtype=dtype, device=device) for _ in range(num_layers)
    ]
    value_caches: List[Tensor] = [
        symbol(['blocks', num_heads, head_size, block_size], dtype=dtype, device=device) for _ in range(num_layers)
    ]

    # run the model
    attn_states = [
        PagedAttnState(
            is_prefill=True,
            seq_lengths=seq_lengths,
            key_cache=key_cache,
            value_cache=value_cache,
            cache_slots=cache_slots,
        )
        for key_cache, value_cache in zip(key_caches, value_caches)
    ]
    hidden_states = model.forward(input_ids=input_ids, position_ids=position_ids, attn_states=attn_states)

    # create the flow graph
    inputs: List[Tensor] = [input_ids, position_ids, cache_slots, seq_lengths, *key_caches, *value_caches]
    updated_key_caches: List[Tensor] = [attn_state.key_cache for attn_state in attn_states]
    updated_value_caches: List[Tensor] = [attn_state.value_cache for attn_state in attn_states]
    outputs: List[Tensor] = [hidden_states, *updated_key_caches, *updated_value_caches]
    graph: FlowGraph = hidet.trace_from(outputs, inputs=inputs)

    # # optimize the flow graph
    # graph: FlowGraph = hidet.graph.optimize(graph)

    # build the flow graph into a CompiledGraph
    compiled_graph = graph.build(space=kernel_search_space)

    return compiled_graph


def _build_decode_graph(
    model: PretrainedModelForCausalLM, device: str, block_size: int, kernel_search_space: int
) -> CompiledGraph:
    import hidet  # pylint: disable=redefined-outer-name

    num_layers: int = model.num_attention_layers()
    num_heads: int = model.num_attention_heads()
    head_size: int = model.attention_head_size()
    dtype: DataType = model.dtype()

    # create the input tensors
    input_ids: Tensor = symbol(['bs', 1], dtype=int32, device=device)
    position_ids: Tensor = symbol(['bs', 1], dtype=int32, device=device)
    cache_slots: Tensor = symbol(['bs', 'seq'], dtype=int64, device=device)
    seq_lengths: Tensor = symbol(['bs'], dtype=int32, device=device)
    max_context_length: Tensor = symbol([], dtype=int32, device=device)
    cache_blocks: Tensor = symbol(['bs', 'cache_blocks'], dtype=int32, device=device)
    key_caches: List[Tensor] = [
        symbol(['blocks', num_heads, head_size, block_size], dtype=dtype, device=device) for _ in range(num_layers)
    ]
    value_caches: List[Tensor] = [
        symbol(['blocks', num_heads, head_size, block_size], dtype=dtype, device=device) for _ in range(num_layers)
    ]

    # run the model
    attn_states = [
        PagedAttnState(
            is_prefill=False,
            seq_lengths=seq_lengths,
            key_cache=key_cache,
            value_cache=value_cache,
            cache_slots=cache_slots,
            cache_blocks=cache_blocks,
        )
        for key_cache, value_cache in zip(key_caches, value_caches)
    ]
    hidden_states = model.forward(input_ids=input_ids, position_ids=position_ids, attn_states=attn_states)

    # create the flow graph
    inputs: List[Tensor] = [
        input_ids,
        position_ids,
        cache_slots,
        seq_lengths,
        max_context_length,
        cache_blocks,
        *key_caches,
        *value_caches,
    ]
    updated_key_caches: List[Tensor] = [attn_state.key_cache for attn_state in attn_states]
    updated_value_caches: List[Tensor] = [attn_state.value_cache for attn_state in attn_states]
    outputs: List[Tensor] = [hidden_states, *updated_key_caches, *updated_value_caches]
    graph: FlowGraph = hidet.trace_from(outputs, inputs=inputs)

    # # optimize the flow graph
    # graph: FlowGraph = hidet.graph.optimize(graph)
    with open('./decode_graph.json', 'w') as f:
        hidet.utils.netron.dump(graph, f)

    # build the flow graph into a CompiledGraph
    compiled_graph = graph.build(space=kernel_search_space)

    return compiled_graph


def create_llm(
    name: str,
    tokenizer: Optional[str] = None,
    revision: Optional[str] = None,
    dtype: Optional[str] = None,
    block_size: int = 16,
    default_memory_capacity: Optional[int] = None,
    device: str = 'cuda',
    kernel_search_space: int = 0,
) -> LLM:
    """
    Builds a LLM app.

    Parameters
    ----------
    name: str
        The name or path of the huggingface model.

    tokenizer: Optional[str]
        The name or path of the huggingface tokenizer.

    revision: Optional[str]
        The revision of the model.

    dtype: Optional[str]
        The dtype of the model.

    block_size: int
        The block size of the cache page table. Default: 16. Candidate values: 8, 16, 24

    default_memory_capacity: Optional[int]
        The default memory capacity (in bytes) used by this app. None indicates use as much memory as possible.
        Default: None.

    device: str
        The device of the app. Default: 'cuda'. Candidate values: 'cuda'.

    kernel_search_space: int
        The kernel search space. Default: 0. Candidate values: 0, 1, 2.

    Returns
    -------
    app: LLM
        The built LLM app.
    """
    # load the huggingface config according (model, revision) pair
    config: PretrainedConfig = _load_pretrained_config(name, revision)

    # get the corresponding hidet model class
    model_class = _get_model_class(config)

    # create the hidet model and load the pretrained weights from huggingface
    model: PretrainedModelForCausalLM = model_class.from_pretrained(name, device=device, dtype=dtype, revision=revision)

    # build the prefill graph
    print('build prefill graph')
    prefill_graph = _build_prefill_graph(
        model, device=device, block_size=block_size, kernel_search_space=kernel_search_space
    )

    # build the decode graph
    print('build decode graph')
    decode_graph = _build_decode_graph(
        model, device=device, block_size=block_size, kernel_search_space=kernel_search_space
    )

    print('finish building graphs')
    llm = LLM(
        compiled_app=create_compiled_app(
            graphs={'prefill': prefill_graph, 'decode': decode_graph},
            modules={},
            tensors={'embedding': model.embedding()},  # [hidden_size, vocab_size]
            attributes={
                'cache_dtype': model.dtype().name,
                'num_layers': model.num_attention_layers(),
                'num_heads': model.num_attention_heads(),
                'head_size': model.attention_head_size(),
                'block_size': block_size,
                'tokenizer': tokenizer if tokenizer is not None else name,
            },
            name='llm',
        ),
        memory_capacity=default_memory_capacity,
    )
    release_unused_resources()
    return llm


def save_llm(app: LLM, path: str):
    """
    Saves a LLM app to the given path.

    Parameters
    ----------
    app: LLM
        The LLM app to save.

    path: str
        The path to save the LLM app.
    """
    raise NotImplementedError()


def load_llm(path: str) -> LLM:
    """
    Loads a LLM app from the given path.

    Parameters
    ----------
    path: str
        The path to load the LLM app.

    Returns
    -------
    app: LLM
        The loaded LLM app.
    """
    raise NotImplementedError()
