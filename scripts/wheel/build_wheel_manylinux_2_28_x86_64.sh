#!/bin/bash

# work in the same directory of this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd $SCRIPT_DIR

# get the root directory of the project
HIDET_DIR=$(cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd)
echo $HIDET_DIR

# build the docker image
ls ${SCRIPT_DIR}/dockerfiles/manylinux_2_28_x86_64/
docker build -t hidet-manylinux_2_28_x86_64-build ${SCRIPT_DIR}/dockerfiles/manylinux_2_28_x86_64/

# run the docker image
docker run --rm -v $HIDET_DIR:/io hidet-manylinux_2_28_x86_64-build bash /io/scripts/wheel/build_wheel.sh $1
