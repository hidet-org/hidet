from .device import available, device_count, synchronize, compute_capability, properties
from .device import set_device, current_device, device

# TODO: ADD malloc_async and free_async once we support streams
from .memory import malloc, free, malloc_host, free_host

# TODO: ADD memcpy_async and memset_async once we support streams
from .memory import memcpy, memset, memory_info
