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
from hidet.ir import IRModule
from hidet.ir.dtypes import float32
from hidet.ir.stmt import DeclareScope
from hidet.graph.ops.definitions.reduce import ReduceTask
from hidet.lang.constructs.type import tensor
from hidet.ir.primitives.cpu.avx import avx_f32x8_add, avx_f32x8_load, avx_f32x8_setzero, avx_f32x8_store, avx_f32x8_broadcast
from hidet.graph.ops.definitions.utils import tune


@tune.space(2, 'nthreads', [4, 8, 16, 32, 64, 96])
@tune.space(1, 'nthreads', [8, 16])
def cpu_schedule_reduce(task: ReduceTask, nthreads=16) -> IRModule:
    # TODO: just try a sum or smth
    if task.reduce_type == "sum":
        from hidet.lang import grid
        x = task.inputs[0]
        shape = task.inputs[0].shape
        # m, n = shape
        print(x.ttype.dtype)
        if len(shape) != 1:
            print("only try for 1d array for now")
            return NotImplemented
        avx_loop_limit = shape[0] // 8 + 1
        with hidet.script_module() as module:
            @hidet.script
            def avx_sum(a_ptr: ~float32, sum_vec_ptr: ~float32):
                sum_vec_ptr = avx_f32x8_add(sum_vec_ptr, a_ptr)

            @hidet.script
            def sum_kernel(a: ~float32, b: ~float32):
                a_ptr = a  # pointer to iterate through the vector
                sum_vec_ptr = avx_f32x8_setzero()  # 0 to 7
                tot = 0.0
                for i in range(avx_loop_limit):
                    data = avx_f32x8_load(a_ptr)
                    sum_vec_ptr = avx_f32x8_add(sum_vec_ptr, data)
                    a_ptr += 8
                arr = tensor(scope=DeclareScope.Default, dtype=float32, shape=[8])
                avx_f32x8_store(arr, sum_vec_ptr)
                for i in range(8):
                    tot += arr[i]
                b[0] = tot

            sum_kernel.kind = "cpu_kernel"
            ir_module = module.ir_module()
            return ir_module
