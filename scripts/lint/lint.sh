#!/bin/bash

# work in the same directory of this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd $SCRIPT_DIR

# run pylint
python -m pylint --rcfile ./pylintrc -j $(nproc) ../../python/hidet
