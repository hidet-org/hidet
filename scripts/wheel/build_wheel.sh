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

# create a new build directory
rm -rf build; mkdir build;

# build
cd build; cmake $ROOT_DIR; make -j4; cd ..

# copy the built libraries and headers to python module
cp $ROOT_DIR/setup.py ./setup.py
cp $ROOT_DIR/MANIFEST.in ./MANIFEST.in
cp $ROOT_DIR/README.md ./README.md
cp -r $ROOT_DIR/python ./
cp -r $ROOT_DIR/include ./python/hidet
cp -r $CURRENT_SCRIPT_DIR/build/lib ./python/hidet

# update version if needed
if [ $# -eq 1 ]; then
  echo "Updating version to $1"
  python3 $CURRENT_SCRIPT_DIR/update_version.py --version $1
fi

# build wheel
mkdir -p built_wheel;
cd built_wheel; pip wheel --no-deps ..; cd ..

# remove all intermediate directories
rm -rf ./python
rm -rf ./build
rm ./setup.py
rm ./MANIFEST.in
rm ./README.md
