#!/usr/bin/env bash

# expects $INPUT_GPU_L4, $INPUT_GPU_H100, $INPUT_GPU_A10, $INPUT_GPU_A100
echo "Selecting GPU types based on input or github event"
echo "Triggered by event: $GITHUB_EVENT_NAME"
if [ "$GITHUB_EVENT_NAME" == "workflow_dispatch" ]; then
  echo "Using manual dispatch inputs"
  GPU_L4="$INPUT_GPU_L4"
  GPU_H100="$INPUT_GPU_H100"
  GPU_A10="$INPUT_GPU_A10"
  GPU_A100="$INPUT_GPU_A100"
elif [ "$GITHUB_EVENT_NAME" == "pull_request" ]; then
  echo "Using pull request defaults"
  GPU_L4="true"
  GPU_H100="true"
  GPU_A10="false"
  GPU_A100="false"
elif [ "$GITHUB_EVENT_NAME" == "push" ]; then
  echo "Using push defaults"
  GPU_L4="true"
  GPU_H100="true"
  GPU_A10="false"
  GPU_A100="false"
elif [ "$GITHUB_EVENT_NAME" == "schedule" ]; then
  echo "Using scheduled run defaults"
  GPU_L4="false"
  GPU_H100="false"
  GPU_A10="true"
  GPU_A100="true"
else
  echo "Unknown event type. Exiting."
  exit 1
fi

# Write the parameters to GITHUB_OUTPUT so they can be used in later steps
echo "gpu_l4=$GPU_L4" >> $GITHUB_OUTPUT
echo "gpu_h100=$GPU_H100" >> $GITHUB_OUTPUT
echo "gpu_a10=$GPU_A10" >> $GITHUB_OUTPUT
echo "gpu_a100=$GPU_A100" >> $GITHUB_OUTPUT