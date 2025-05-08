# Script to ensure proper CUDA cleanup before process termination
import torch
import time
import gc
import sys

print("=== Starting CUDA cleanup and synchronization ===")
sys.stdout.flush()  # Ensure output is visible in logs

# Force garbage collection first
gc.collect()

# Explicitly empty CUDA cache
torch.cuda.empty_cache()

# Synchronize all CUDA operations
print("Synchronizing CUDA devices...")
for i in range(torch.cuda.device_count()):
    torch.cuda.synchronize(i)

# Small delay to allow driver operations to complete
time.sleep(2)

# Print memory stats for debugging
print(f"CUDA memory allocated: {torch.cuda.memory_allocated()} bytes")
print(f"CUDA memory reserved: {torch.cuda.memory_reserved()} bytes")
print("=== CUDA cleanup complete ===")
sys.stdout.flush()