#!/bin/bash

echo "     Current sm clock: $(nvidia-smi -i 0 --query-gpu=clocks.current.sm --format=csv,noheader,nounits) MHz"
echo " Current memory clock: $(nvidia-smi -i 0 --query-gpu=clocks.current.memory --format=csv,noheader,nounits) MHz"
