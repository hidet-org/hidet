#!/bin/bash

set -e

# work in the same directory of this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd $SCRIPT_DIR


# Function to format Python files
format_python() {
    # run black formatter for python
    python -m black --skip-string-normalization --skip-magic-trailing-comma --line-length 120 ../../python/hidet ../../tests
    python -m black --skip-string-normalization --skip-magic-trailing-comma --line-length 100 ../../gallery
}

# Function to format C++ files
format_cpp() {
    # run clang-format for cpp
    find ../../src ../../include -iname '*.h' -o -iname '*.cpp' | xargs clang-format -style=file -i
}

# Function to show help message
show_help() {
    echo "Usage: fmt.sh [cpp|python|all]"
    echo "Run formatters for the project"
    echo ""
    echo "Options:"
    echo "  python       Format Python files only with black."
    echo "  cpp          Format C++ files only with clang-format."
    echo "  all          Format both C++ and Python. When no option is given, this one is the default one."
    echo "  -h, --help   Show this help message and exit."
}

# Check if $1 is empty, use "all" by default
if [ -z "$1" ]; then
    set -- "all"
fi

# Check the first argument to the script
case "$1" in
    python)
        format_python
        ;;
    cpp)
        format_cpp
        ;;
    all)
        format_python
        format_cpp
        ;;
    -h|--help|*)
        show_help
        ;;
esac
