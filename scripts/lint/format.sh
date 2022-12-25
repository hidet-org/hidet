#!/bin/bash

# work in the same directory of this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd $SCRIPT_DIR

# run black formatter
python -m black --skip-string-normalization --skip-magic-trailing-comma --line-length 120 ../../python/hidet ../../tests
python -m black --skip-string-normalization --skip-magic-trailing-comma --line-length 90 ../../gallery
