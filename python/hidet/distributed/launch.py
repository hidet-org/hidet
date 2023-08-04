# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=consider-using-with
import os
import subprocess
import argparse

FILE_STORE_PATH = "./tmp"

if os.path.exists(FILE_STORE_PATH):
    os.remove(FILE_STORE_PATH)

if os.path.exists(FILE_STORE_PATH + '.lock'):
    os.remove(FILE_STORE_PATH + '.lock')

parser = argparse.ArgumentParser()
parser.add_argument("nprocs", type=int)
parser.add_argument("script", type=str)
parser.add_argument('script_args', nargs=argparse.REMAINDER)
args = parser.parse_args()

procs = []
for rank in range(args.nprocs):
    env = os.environ.copy()
    env['WORLD_SIZE'] = str(args.nprocs)
    env['RANK'] = str(rank)
    env['INIT_METHOD'] = f"file://{FILE_STORE_PATH}"
    procs.append(subprocess.Popen(['python', args.script] + args.script_args, env=env))

for proc in procs:
    proc.wait()
