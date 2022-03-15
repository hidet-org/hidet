#!/bin/bash

. "$(dirname "$0")/query_gpu_clocks.sh"

echo "Setting GPU to maximum clocks, requires sudo privilege."
MAX_SM_CLOCK=$(nvidia-smi -i 0 --query-gpu=clocks.max.sm --format=csv,noheader,nounits)
MAX_MEM_CLOCK=$(nvidia-smi -i 0 --query-gpu=clocks.max.memory --format=csv,noheader,nounits)
sudo -S nvidia-smi --lock-memory-clocks="$MAX_MEM_CLOCK" > /dev/null
sudo -S nvidia-smi --lock-gpu-clocks="$MAX_SM_CLOCK" > /dev/null

. "$(dirname "$0")/query_gpu_clocks.sh"
