from typing import List
from enum import Enum
from .sampler import SamplingParams
from .cache import CacheTableManager, BlockDevice


class SequenceState(Enum):
    NEW = 'new'
    WAITING = 'waiting'
    RUNNING = 'running'
    FINISHED_STOPPED = 'finished_stopped'
    FINISHED_LENGTH = 'finished_length'


class Sequence:
    def __init__(self, sequence_id: int, prompt: str, sampling_params: SamplingParams):
        self.sequence_id: int = sequence_id
        self.prompt: str = prompt
        self.sampling_params: SamplingParams = sampling_params

        self.prompt_tokens: List[int] = []
        self.output_tokens: List[int] = []
        self.blocks: List[int] = []
        self.status: SequenceState = SequenceState.NEW

    def append_token(self, token: int):
        self.output_tokens.append(token)

        # update status
        if token in self.sampling_params.stop_token_ids:
            self.status = SequenceState.FINISHED_STOPPED
        elif len(self.output_tokens) >= self.sampling_params.max_tokens:
            self.status = SequenceState.FINISHED_LENGTH

    def is_finished(self) -> bool:
        return self.status in [SequenceState.FINISHED_STOPPED, SequenceState.FINISHED_LENGTH]


class SequenceOutput:
    def __init__(
        self,
        sequence_id: int,
        prompt: str,
        output_text: str,
        prompt_tokens: List[int],
        output_tokens: List[int],
        status: SequenceState,
    ):
        self.sequence_id: int = sequence_id
        self.prompt: str = prompt
        self.output_text: str = output_text
        self.prompt_tokens: List[int] = prompt_tokens
        self.output_tokens: List[int] = output_tokens
        self.status: SequenceState = status

    def is_finished(self) -> bool:
        return self.status in [SequenceState.FINISHED_STOPPED, SequenceState.FINISHED_LENGTH]


class SequenceScheduler:
    def __init__(self, cache: CacheTableManager):
        self.cache: CacheTableManager = cache
        self.new: List[Sequence] = []
        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []
        self.swapped: List[Sequence] = []
        self.block_size: int = self.cache.block_size

    def add_sequence(self, sequence: Sequence):
        self.new.append(sequence)

    def schedule(self) -> List[Sequence]:
        """
        Schedule the sequences to run in the next step.

        Current strategy:
        1. if there are new requests, and there are enough free blocks
           1) put all the running sequences to waiting list
           2) allocate virtual and gpu blocks for the new sequences (allocate for as many as possible sequences)
           3) put the new sequences to running list
        2. if there are enough free blocks for the waiting and running sequences for the next token
           1) allocate block for all sequences that have no slots left
           2) put all the sequences in running and waiting list to the running list
        3. evict the sequences that arrived at the late time to swapped list
           1) sort the sequences in waiting and running list by the sequence id
           2) evict the sequences that arrived at the late time to swapped list, and free the gpu blocks
           3) put the remaining sequences to running list
        """
        if not self.new and not self.waiting and not self.running:
            return []

        # case 1
        if self.new:
            num_free_blocks: int = self.cache.gpu_cache.num_free_blocks()
            allocated_blocks: int = 0
            to_add: List[Sequence] = []
            for seq in self.new:
                require_blocks = len(seq.prompt_tokens) // self.block_size + 1
                if allocated_blocks + require_blocks <= num_free_blocks:
                    to_add.append(seq)
                    allocated_blocks += require_blocks
                else:
                    break
            if to_add:
                while self.running:
                    seq = self.running.pop()
                    seq.status = SequenceState.WAITING
                    self.waiting.append(seq)
                while to_add:
                    seq = to_add.pop()
                    seq.status = SequenceState.RUNNING
                    self.running.append(seq)
                self.new = self.new[len(self.running) :]
                # allocate blocks
                virtual_blocks = self.cache.alloc_virtual_blocks(allocated_blocks)
                gpu_blocks = self.cache.alloc_gpu_blocks(allocated_blocks)
                for seq in self.running:
                    require_blocks = len(seq.prompt_tokens) // self.block_size + 1
                    while require_blocks > 0:
                        virtual_block = virtual_blocks.pop()
                        gpu_block = gpu_blocks.pop()
                        seq.blocks.append(virtual_block)
                        self.cache.map_block(virtual_block, BlockDevice.GPU, gpu_block)
                        require_blocks -= 1
                return self.running

        # case 2
        if len(self.running) + len(self.waiting) > 0:
            num_free_blocks: int = self.cache.gpu_cache.num_free_blocks()
            seqs_require_new_block: List[Sequence] = []
            for seq in self.running + self.waiting:
                if ((len(seq.prompt_tokens) + len(seq.output_tokens)) % self.block_size) == 0:
                    seqs_require_new_block.append(seq)
            if len(seqs_require_new_block) <= num_free_blocks:
                virtual_blocks = self.cache.alloc_virtual_blocks(len(seqs_require_new_block))
                gpu_blocks = self.cache.alloc_gpu_blocks(len(seqs_require_new_block))
                for seq, virtual_block, gpu_block in zip(seqs_require_new_block, virtual_blocks, gpu_blocks):
                    seq.blocks.append(virtual_block)
                    self.cache.map_block(virtual_block, BlockDevice.GPU, gpu_block)
                while self.waiting:
                    seq = self.waiting.pop()
                    seq.status = SequenceState.RUNNING
                    self.running.append(seq)
                return self.running
        # case 3
        raise NotImplementedError('case 3 not implemented yet')

    def post_running_update(self):
        for sequence in self.running:
            if sequence.is_finished():
                # free the virtual and gpu blocks
                gpu_blocks = self.cache.get_mapped_blocks(sequence.blocks)
                self.cache.unmap_blocks(sequence.blocks)
                self.cache.free_gpu_blocks(gpu_blocks)
                self.cache.free_virtual_blocks(sequence.blocks)

        self.running = [sequence for sequence in self.running if not sequence.is_finished()]
