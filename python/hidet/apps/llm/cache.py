from typing import Tuple, List, Dict, Optional
import hidet.cuda
from hidet.ir.type import data_type, DataType
from hidet.graph.tensor import Tensor, empty
from hidet.runtime.storage import current_memory_pool

KVCache = Tuple[Tensor, Tensor]


class BlockDevice:
    CPU = 0
    GPU = 1


class CacheTable:
    def __init__(
        self,
        dtype: DataType,
        device: str,
        memory_capacity: int,
        num_layers: int,
        num_heads: int,
        head_size: int,
        block_size: int,
    ):
        self.dtype: DataType = dtype
        self.device: str = device
        self.memory_capacity: int = memory_capacity
        self.num_layers: int = num_layers
        self.num_heads: int = num_heads
        self.head_size: int = head_size
        self.block_size: int = block_size

        self.num_blocks: int = self._calc_num_blocks(memory_capacity)

        self.cache: List[KVCache] = self._allocate_cache_table()
        self._free_blocks: List[int] = list(reversed(range(self.num_blocks)))

    def _calc_num_blocks(self, memory_capacity: int) -> int:
        element_size: int = data_type(self.dtype).nbytes
        size_per_block: int = self.num_heads * self.head_size * self.block_size * element_size
        return memory_capacity // (size_per_block * 2 * self.num_layers)

    def _allocate_cache_table(self) -> List[KVCache]:
        cache: List[KVCache] = []

        if self.device == 'virtual':
            return []

        for _ in range(self.num_layers):
            key_cache_shape = [self.num_blocks, self.num_heads, self.head_size, self.block_size]
            value_cache_shape = [self.num_blocks, self.num_heads, self.head_size, self.block_size]
            cache.append(
                (
                    empty(shape=key_cache_shape, dtype=self.dtype, device=self.device),
                    empty(shape=value_cache_shape, dtype=self.dtype, device=self.device),
                )
            )
        return cache

    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def alloc_blocks(self, num_blocks: int) -> List[int]:
        if num_blocks > len(self._free_blocks):
            raise RuntimeError(
                f'Not enough free blocks for {self.device}, requested: {num_blocks}, current: {len(self._free_blocks)}'
            )

        if num_blocks == 0:
            return []

        allocated = self._free_blocks[-num_blocks:]
        del self._free_blocks[-num_blocks:]
        return allocated

    def free_blocks(self, blocks: List[int]):
        self._free_blocks.extend(blocks)


class CacheTableManager:
    def __init__(
        self, dtype: DataType, capacity: Optional[int], num_layers: int, num_heads: int, head_size: int, block_size: int
    ):
        self.dtype: DataType = dtype
        self.num_layers: int = num_layers
        self.num_heads: int = num_heads
        self.head_size: int = head_size
        self.block_size: int = block_size
        self.capacity: int = self._capacity(capacity)

        self.virtual_cache = CacheTable(
            dtype, 'virtual', 2 * self.capacity, num_layers, num_heads, head_size, block_size
        )
        self.cpu_cache = CacheTable(dtype, 'cpu', self.capacity, num_layers, num_heads, head_size, block_size)
        self.gpu_cache = CacheTable(dtype, 'cuda', self.capacity, num_layers, num_heads, head_size, block_size)

        # the mapping from virtual block number to the physical block number
        # {virtual_block_number: (block_device, physical_block_number)}
        self.mapping: Dict[int, Tuple[int, int]] = {}

    def _capacity(self, capacity: Optional[int], percentage: float = 0.9) -> int:
        if capacity is None:
            # clear the reserved memory in the current memory pool
            current_memory_pool('cuda').clear()

            # query the available memory
            free, total = hidet.cuda.memory_info()  # pylint: disable=unused-variable
            capacity = int(free * percentage)

        return capacity

    def alloc_virtual_blocks(self, num_blocks: int) -> List[int]:
        return self.virtual_cache.alloc_blocks(num_blocks)

    def alloc_cpu_blocks(self, num_blocks: int) -> List[int]:
        return self.cpu_cache.alloc_blocks(num_blocks)

    def alloc_gpu_blocks(self, num_blocks: int) -> List[int]:
        return self.gpu_cache.alloc_blocks(num_blocks)

    def free_virtual_blocks(self, blocks: List[int]):
        self.virtual_cache.free_blocks(blocks)

    def free_cpu_blocks(self, blocks: List[int]):
        self.cpu_cache.free_blocks(blocks)

    def free_gpu_blocks(self, blocks: List[int]):
        self.gpu_cache.free_blocks(blocks)

    def unmap_blocks(self, virtual_blocks: List[int]):
        for block in virtual_blocks:
            del self.mapping[block]

    def get_mapped_blocks(self, virtual_blocks: List[int]) -> List[int]:
        return [self.mapping[block][1] for block in virtual_blocks]

    def map_block(self, virtual_block: int, physical_device: int, physical_block: int):
        assert virtual_block not in self.mapping
        self.mapping[virtual_block] = (physical_device, physical_block)
