#!/bin/bash

echo "        Current sm clock: $(nvidia-smi -i 0 --query-gpu=clocks.current.sm --format=csv,noheader,nounits) MHz"
echo "    Current memory clock: $(nvidia-smi -i 0 --query-gpu=clocks.current.memory --format=csv,noheader,nounits) MHz"
echo " Current gpu temperature: $(nvidia-smi -i 0 --query-gpu=temperature.gpu --format=csv,noheader,nounits) C"
echo "    GPU throttle reasons: $(nvidia-smi -i 0 --query-gpu=clocks_throttle_reasons.active --format=csv,noheader,nounits)"
#nvidia-smi -i 0 --query-gpu=clocks_throttle_reasons.supported --format=csv
#nvidia-smi -i 0 --query-gpu=clocks_throttle_reasons.active --format=csv
#nvidia-smi -i 0 --query-gpu=clocks_throttle_reasons.gpu_idle --format=csv
#nvidia-smi -i 0 --query-gpu=clocks_throttle_reasons.sw_power_cap --format=csv
#nvidia-smi -i 0 --query-gpu=clocks_throttle_reasons.hw_slowdown --format=csv
#nvidia-smi -i 0 --query-gpu=clocks_throttle_reasons.hw_thermal_slowdown --format=csv
#nvidia-smi -i 0 --query-gpu=clocks_throttle_reasons.hw_power_brake_slowdown --format=csv
#nvidia-smi -i 0 --query-gpu=clocks_throttle_reasons.sync_boost --format=csv



