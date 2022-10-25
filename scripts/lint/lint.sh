#!/bin/bash

# work in the same directory of this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd $SCRIPT_DIR

# run pylint
PYTHON_ROOT=$(realpath ../../python/hidet)
PYTHON_TEST_ROOT=$(realpath ../../tests)
python -m pylint $PYTHON_ROOT --rcfile ./pylintrc
#python -m pylint $PYTHON_TEST_ROOT --rcfile ./pylintrc
