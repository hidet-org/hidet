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

from hidet.ir import Task


class EmptyTask(Task):
    def __init__(self, name):
        super().__init__(name, [], [])

    def implement(self, target, working_dir):
        from hidet.lang.types import u32, i32, f16
        from hidet.lang import attrs

        with hidet.script_module() as script_module:

            @hidet.script
            def kernel(out: f16[4], input: f16[4]):
                attrs.func_kind = "cpu_kernel"

                for i in range(4):
                    out[i] = input[i]

            @hidet.script
            def launch(out: f16[4], input: f16[4]):
                attrs.func_kind = "public"

                kernel(out, input)

        ir_module = script_module.ir_module()
        ir_module.task = self
        return ir_module


def test():
    # A CPU kernel with user defined launch function can lead to errors
    task = EmptyTask('empty')
    from hidet.transforms.inline_function import inline_function_pass

    mod = task.implement('cpu', '.')
    transforms = [inline_function_pass()]
    for t in transforms:
        mod = t(mod)
    mod.build()


if __name__ == "__main__":
    test()
