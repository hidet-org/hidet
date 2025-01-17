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
import pytest
import hidet
from hidet.testing.torch_utils import bench_model
import os


def configure_hidet_options():
    """Set global options for hidet."""
    hidet.option.search_space(2)
    hidet.option.debug_cache_tuning(True)
    hidet.option.save_lower_ir(True)


def run_hidet_benchmark(device):
    """Run hidet benchmark with a sample model."""
    x_shape = [1, 3, 224, 224]
    w_shape = [64, 3, 3, 3]

    x = hidet.symbol(x_shape, dtype='float16', device=device)
    w = hidet.randn(w_shape, dtype='float16', device=device)

    o = hidet.ops.conv2d(x, w)
    g = hidet.trace_from(o, inputs=[x, w])
    g = hidet.graph.optimize(g)
    g = g.cuda_graph()

    g.run_async()


def check_fuse_ir_candidates():
    """Check the correctness of fuse_ir candidates in the cache."""
    for dirpath, dirname, filenames in os.walk("."):
        if ".hidet_cache" in dirpath and 'fuse_ir' in dirname and 'candidates.txt' in filenames:
            candidate_file = os.path.join(dirpath, 'candidates.txt')
            num_candidate = sum(1 for _ in open(candidate_file)) - 1

            fused_candidate_dir = os.path.join(dirpath, 'fuse_ir')
            num_fused_candidate = len(next(os.walk(fused_candidate_dir))[1])

            assert num_candidate == num_fused_candidate


def test_save_lower_ir(device):
    """Main test for saving lower IR and benchmarking."""
    configure_hidet_options()

    with hidet.graph.PassContext() as ctx:
        ctx.set_reduce_precision('float16')

        run_hidet_benchmark(device)

        check_fuse_ir_candidates()


if __name__ == '__main__':
    pytest.main([__file__])
