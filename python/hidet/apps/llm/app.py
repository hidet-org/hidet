"""

The LLM app is an application built on top of the hidet compiler, used to perform the completion task of the LLM model.

Inside the LLM app, there are two main computation graphs: prefill and decode.

The prefill graph takes the following inputs and outputs:
    inputs:
        input_ids: int32 [bs, seq_len]
        position_ids: int32 [bs, seq_len]
        cache_slots: int64 [bs, seq_len]
        seq_lengths: int32 [bs]
        *key_caches: dtype [num_blocks, num_heads, head_size, block_size]   (num_layers)
        *value_caches: dtype [num_blocks, num_heads, head_size, block_size]  (num_layers)
    outputs:
        hidden_states: dtype [bs, seq_len, hidden_size]
        *key_caches: dtype [num_blocks, num_heads, head_size, block_size]  (num_layers)
        *value_caches: dtype [num_blocks, num_heads, head_size, block_size]  (num_layers)
    (Note: the key_caches in the inputs and outputs share the same memory, similarly for value_caches)

The decode graph takes the following inputs and outputs:
    inputs:
        input_ids: int32 [bs, 1]
        position_ids: int32 [bs, 1]
        cache_slots: int64 [bs, 1]
        seq_lengths: int32 [bs]
        max_context_length: int32
        cache_blocks: int32 [bs, max_num_cache_blocks]
        *key_caches: dtype [num_blocks, num_heads, head_size, block_size]   (num_layers)
        *value_caches: dtype [num_blocks, num_heads, head_size, block_size]  (num_layers)
    outputs:
        hidden_states: dtype [bs, 1, hidden_size]
        *key_caches: dtype [num_blocks, num_heads, head_size, block_size]  (num_layers)
        *value_caches: dtype [num_blocks, num_heads, head_size, block_size]  (num_layers)
    (Note: the key_caches in the inputs and outputs share the same memory, similarly for value_caches)


The LLM app supports two operations:
1. add a sequence to the app
2. perform a step of scheduling and running, which will select a batch of sequences to run and return the outputs
   of the selected sequences.

Acknowledgement:
    - We adopt the page attention mechanism proposed in vLLM: https://github.com/vllm-project/vllm
"""
from typing import List, Optional
import dataclasses
from hidet.runtime.compiled_app import CompiledApp
from hidet.ir.type import data_type
from hidet.graph.tensor import Tensor
from hidet.apps.llm.sampler import SamplingParams
from hidet.apps.llm.sequence import Sequence, SequenceScheduler, SequenceOutput
from hidet.apps.llm.cache import CacheTableManager
from hidet.apps.llm.sampler import Sampler, SamplerOutput
from hidet.apps.llm.tokenizer import Tokenizer
from hidet.utils.dataclass import from_dict
from .utils import tensor_pad, tensor


@dataclasses.dataclass
class Attributes:
    cache_dtype: str
    num_layers: int
    num_heads: int
    head_size: int
    block_size: int
    tokenizer: str  # currently, we use the tokenizer from huggingface


class LLM:
    def __init__(self, compiled_app: CompiledApp, memory_capacity: Optional[int] = None):
        super().__init__()
        self.compiled_app: CompiledApp = compiled_app
        self.attributes: Attributes = from_dict(Attributes, compiled_app.attributes)
        self.sampler: Sampler = Sampler(embedding=self.compiled_app.tensors['embedding'])
        self.tokenizer: Tokenizer = Tokenizer(self.attributes.tokenizer)
        self.cache: CacheTableManager = CacheTableManager(
            dtype=data_type(self.attributes.cache_dtype),
            capacity=memory_capacity,
            num_layers=self.attributes.num_layers,
            num_heads=self.attributes.num_heads,
            head_size=self.attributes.head_size,
            block_size=self.attributes.block_size,
        )
        self.scheduler: SequenceScheduler = SequenceScheduler(self.cache)

        self.cache_inputs: List[Tensor] = [kv[0] for kv in self.cache.gpu_cache.cache] + [
            kv[1] for kv in self.cache.gpu_cache.cache
        ]

    def _prefill(self, sequences: List[Sequence]) -> Tensor:
        # prepare the inputs in the list format
        input_ids_list: List[List[int]] = []
        position_ids_list: List[List[int]] = []
        cache_slots_list: List[List[int]] = []
        seq_lengths_list: List[int] = []
        for seq in sequences:
            input_ids_list.append(seq.prompt_tokens)
            position_ids_list.append(list(range(len(seq.prompt_tokens))))

            block_size = self.cache.block_size
            slots = []
            for i in range(len(seq.prompt_tokens)):
                virtual_block: int = seq.blocks[i // block_size]
                gpu_block: int = self.cache.mapping[virtual_block][1]
                slot: int = gpu_block * block_size + i % block_size
                slots.append(slot)
            cache_slots_list.append(slots)
            seq_lengths_list.append(len(seq.prompt_tokens))

        # convert them into hidet tensors
        input_ids: Tensor = tensor_pad(input_ids_list)
        position_ids: Tensor = tensor_pad(position_ids_list)
        cache_slots: Tensor = tensor_pad(cache_slots_list, pad_value=-1, dtype='int64')
        seq_lengths: Tensor = tensor(seq_lengths_list)

        # run the prefill graph
        prefill_graph = self.compiled_app.graphs['prefill']
        inputs = [input_ids, position_ids, cache_slots, seq_lengths, *self.cache_inputs]
        outputs: List[Tensor] = prefill_graph.run_async(inputs)
        hidden_states: Tensor = outputs[0]  # [bs, seq_len, hidden_size]

        return hidden_states

    def _decode(self, sequences: List[Sequence]) -> Tensor:
        # prepare the inputs in the list format
        input_ids_list: List[List[int]] = []
        position_ids_list: List[List[int]] = []
        cache_slots_list: List[List[int]] = []
        seq_lengths_list: List[int] = []
        max_context_length: int = 0
        cache_blocks: List[List[int]] = []
        for seq in sequences:
            num_tokens = len(seq.prompt_tokens) + len(seq.output_tokens)
            input_ids_list.append([seq.output_tokens[-1]])
            position_ids_list.append([num_tokens - 1])

            block_size = self.cache.block_size
            last_token = num_tokens - 1
            virtual_block: int = seq.blocks[last_token // block_size]
            gpu_block: int = self.cache.mapping[virtual_block][1]
            slot: int = gpu_block * block_size + last_token % block_size
            cache_slots_list.append([slot])
            seq_lengths_list.append(num_tokens)
            cache_blocks.append([self.cache.mapping[virtual_block][1] for virtual_block in seq.blocks])
            max_context_length = max(max_context_length, num_tokens)

        # convert them into hidet tensors
        input_ids: Tensor = tensor_pad(input_ids_list)
        position_ids: Tensor = tensor_pad(position_ids_list)
        cache_slots: Tensor = tensor_pad(cache_slots_list, dtype='int64')
        seq_lengths: Tensor = tensor(seq_lengths_list)
        max_context_length: Tensor = tensor(max_context_length)
        cache_blocks: Tensor = tensor_pad(cache_blocks)

        # run the decode graph
        decode_graph = self.compiled_app.graphs['decode']
        inputs = [
            input_ids,
            position_ids,
            cache_slots,
            seq_lengths,
            max_context_length,
            cache_blocks,
            *self.cache_inputs,
        ]
        outputs: List[Tensor] = decode_graph.run_async(inputs)
        hidden_states: Tensor = outputs[0]  # [bs, 1, hidden_size]

        return hidden_states

    def _post_process(self, sampler_outputs: List[SamplerOutput]) -> List[SequenceOutput]:
        sequence_outputs: List[SequenceOutput] = []
        for sequence, sampler_output in zip(self.scheduler.running, sampler_outputs):
            sequence.append_token(sampler_output.token)

            sequence_outputs.append(
                SequenceOutput(
                    sequence_id=sequence.sequence_id,
                    prompt=sequence.prompt,
                    output_text=self.tokenizer.decode(sequence.output_tokens) if sequence.is_finished() else '',
                    prompt_tokens=sequence.prompt_tokens,
                    output_tokens=sequence.output_tokens,
                    status=sequence.status,
                )
            )

        # update the scheduler status (e.g., some sequences may be finished)
        self.scheduler.post_running_update()

        return sequence_outputs

    def add_sequence(self, sequence_id: int, prompt: str, sampling_params: SamplingParams):
        seq = Sequence(sequence_id, prompt, sampling_params)

        # tokenize the prompt
        seq.prompt_tokens = self.tokenizer.encode(seq.prompt)

        self.scheduler.add_sequence(seq)

    def step(self) -> List[SequenceOutput]:
        # schedule for the next step, got the sequences to run
        sequences: List[Sequence] = self.scheduler.schedule()

        # run the sequences and get the next token for each sequence
        if all(len(seq.output_tokens) == 0 for seq in sequences):
            # prefill
            hidden_states = self._prefill(sequences)
        elif all(len(seq.output_tokens) > 0 for seq in sequences):
            # decode
            hidden_states = self._decode(sequences)
        else:
            raise ValueError("Some sequences are prefilling and some are decoding.")

        # sample the next token given the hidden states
        sampler_outputs: List[SamplerOutput] = self.sampler.sample(sequences, hidden_states)

        # append the next token for each sequence, detokenize the output text (when finished)
        sequence_outputs: List[SequenceOutput] = self._post_process(sampler_outputs)

        return sequence_outputs
