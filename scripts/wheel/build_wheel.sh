#!/bin/bash
#
# Build a wheel:
#
# $ bash scripts/build_wheel.sh
#
# would generate a .whl file in the scripts directory.
#

set -e  # exit immediately if a command exits with a non-zero status.

###############################################################################
# This script builds a wheel for the current platform and Python version.
###############################################################################


# work in the same directory of this script
CURRENT_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(cd -- "$CURRENT_SCRIPT_DIR/../.." &> /dev/null && pwd)
cd $CURRENT_SCRIPT_DIR

pip3 install torch
pip3 install nvidia-cuda-runtime-cu12
pip3 install triton

# copy the built libraries and headers to python module
cp $ROOT_DIR/pyproject.toml ./
cp $ROOT_DIR/CMakeLists.txt ./
cp $ROOT_DIR/config.cmake ./
cp $ROOT_DIR/setup.py ./
cp $ROOT_DIR/MANIFEST.in ./
cp $ROOT_DIR/README.md ./
cp -r $ROOT_DIR/python ./
cp -r $ROOT_DIR/include ./
cp -r $ROOT_DIR/src ./
cp -r $ROOT_DIR/docs ./

# update version if needed
if [ $# -eq 1 ]; then
  echo "Updating version to $1"
  python3 $CURRENT_SCRIPT_DIR/update_version.py --version $1
fi

# build wheel
mkdir -p built_wheel;
cd built_wheel; pip3 wheel --no-deps ..; cd ..

# remove all intermediate directories
rm -rf ./python
rm -rf ./src
rm -rf ./docs
rm -rf ./include
rm -rf ./build
rm ./pyproject.toml
rm ./setup.py
rm ./MANIFEST.in
rm ./CMakeLists.txt
rm ./config.cmake
rm ./README.md

if [ "$GITHUB_ACTIONS" == "true" ]; then
  echo "Running in GitHub Actions environment"
  echo $(pwd)
  WHEEL=$(find $CURRENT_SCRIPT_DIR/built_wheel -maxdepth 1 -name '*.whl')
  WHEEL_FILENAME=$(basename "$WHEEL")

  echo "wheel_path=./scripts/wheel/built_wheel" >> "$GITHUB_OUTPUT"
  echo "wheel_name=${WHEEL_FILENAME}" >> "$GITHUB_OUTPUT"
fi
