#!/bin/bash

WHEEL=$1

set -e  # exit immediately if a command exits with a non-zero status.

# twine upload --repository testpypi $WHEEL
twine upload $WHEEL

