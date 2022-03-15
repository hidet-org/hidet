#!/bin/bash

. "$(dirname "$0")/query_gpu_clocks.sh"

echo "Resetting GPU to clocks, requires sudo privilege."
sudo -S nvidia-smi --reset-memory-clocks > /dev/null
sudo -S nvidia-smi --reset-gpu-clocks > /dev/null

. "$(dirname "$0")/query_gpu_clocks.sh"
