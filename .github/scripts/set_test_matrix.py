"""
Sets the strategy matrix for the functional ci tests.
This mimics the discovery strategy used by pytest for files inside the tests/ folder
and shards them based on the top level parent folders. 

Expects to be executed in a GHA envirionment, with GITHUB_OUTPUT context available.
"""
import glob
import json
import os
from pathlib import Path

patterns = ('test_*.py', '*_test.py') # the tuple of file types
files_matched = []
for pattern in patterns:
    files_matched.extend(glob.glob(f"tests/**/{pattern}", recursive=True))

testing_paths = []
for path in files_matched:
    current_path = Path(path)
    testing_paths.append("/".join(current_path.parts[:2]))

include = []

for path in list(set(testing_paths)):
    include.append({
        "path": path
    })

matrix = {
    "include": include
}

matrix_str = json.dumps(matrix)
name = 'matrix'
value = matrix_str
with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
    print(f'{name}={value}', file=fh)