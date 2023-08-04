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

import hidet
import hidet.testing
import argparse
import os

# Run this script with `python -m hidet.distributed.launch [num_gpus] resnet.py [out_dir]`

def run(world_size, rank, out_dir):
    hidet.cuda.set_device(rank)
    hidet.distributed.init_process_group(init_method=os.environ['INIT_METHOD'], world_size=world_size, rank=rank)
    flow_graph = hidet.distributed.load_partition(out_dir, rank)
    x = hidet.zeros([32, 3, 224, 224], device='cuda')
    opt_graph = hidet.graph.optimize(flow_graph)
    compiled = opt_graph.build()
    print(compiled(x))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--recompile", action='store_true')
    args = parser.parse_args()
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])

    if rank == 0 and args.recompile:
        model = hidet.testing.models.resnet.resnet18()
        x_zero = hidet.zeros([32, 3, 224, 224])
        y_truth = model(x_zero)
        print(y_truth)
        x = hidet.symbol([32, 3, 224, 224])
        flow_graph = hidet.trace_from(model(x))
        hidet.distributed.partition(flow_graph, {'ngpus': world_size, 'mem_budget': 24 * 1024 * 1024, 'search_max_seconds': 300}, args.out_dir)

    run(world_size, rank, args.out_dir)