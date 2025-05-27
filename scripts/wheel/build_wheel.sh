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

# clean the build cache
rm -rf ${ROOT_DIR}/build
rm -rf ${CURRENT_SCRIPT_DIR}/built_wheel

# build wheel
mkdir -p built_wheel;
cd built_wheel; pip3 wheel --no-deps $ROOT_DIR; cd ..
