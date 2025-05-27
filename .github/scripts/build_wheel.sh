#!/bin/bash


# stop immediately if a command exits with a non-zero status.
set -e

# print the executed commands
set -x

# use ./scripts/wheel/build_wheel.sh to build the wheel
bash scripts/wheel/build_wheel.sh

echo $(pwd)
WHEEL=$(find scripts/wheel/built_wheel -maxdepth 1 -name '*.whl')
WHEEL_FILENAME=$(basename "$WHEEL")

echo "wheel_path=./scripts/wheel/built_wheel" >> "$GITHUB_OUTPUT"
echo "wheel_name=${WHEEL_FILENAME}" >> "$GITHUB_OUTPUT"
