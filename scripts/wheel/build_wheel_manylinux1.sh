#!/bin/bash

# work in the same directory of this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd $SCRIPT_DIR

# get the root directory of the project
HIDET_DIR=$(cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd)

# build the docker image
bash ./dockerfiles/manylinux1/build_image.sh

echo $HIDET_DIR

# run the docker image
docker run --rm -v $HIDET_DIR:/io hidet-manylinux1-build bash /io/scripts/wheel/build_wheel.sh $1
