#!/bin/bash

###############################################################################
# This script builds a wheel for the current platform and Python version.
###############################################################################

# work in the same directory of this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd $SCRIPT_DIR

# create a new build directory
rm -rf build; mkdir build;

# build
cd build; cmake ../..; make -j4; cd ..

# copy the built libraries and headers to python module
cp -r ./build/lib ../python/hidet
cp -r ../include ../python/hidet

# build wheel
pip wheel --no-deps ..

# remove all intermediate directories
rm -rf ../python/hidet/hidet.egg-info
rm -rf ../python/hidet/lib
rm -rf ../python/hidet/include
rm -rf ./build
