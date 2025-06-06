import subprocess
import hidet
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--expected', type=int, required=True, help='Expected number of local workers')
args = parser.parse_args()

actual = hidet.option.get_num_local_workers()
expected = args.expected

if actual != expected:
    print(f'Expected {expected}, but got {actual}.', file=sys.stderr)
    sys.exit(1)
