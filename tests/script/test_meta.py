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


def test_args():
    from hidet.lang import attrs, meta, printf, int32

    with hidet.script_module() as script_module:

        @hidet.script
        def launch(args: meta.types([int, bool, float, int32]), second_args: int, thrid_args: meta.types([int32])):
            attrs.func_kind = 'public'

            printf("%d\n", args[0])
            printf("%d\n", args[1])
            printf("%f\n", args[2])
            printf("%d\n", args[3])
            printf("%d\n", second_args)
            printf("%d\n", thrid_args[0])

    module = script_module.build()
    module(1, True, 0.1, 2, 3, 4)


def test_meta_range():
    from hidet.lang import attrs, meta, printf

    with hidet.script_module() as script_module:

        @hidet.script
        def launch():
            attrs.func_kind = 'public'

            for i in meta.range(10):
                for j in range(i):
                    printf("%d ", j)
                printf("\n")

    module = script_module.build()
    module()


def test_if_then_else():
    from hidet.lang import attrs, tensor

    dims = [2, 3]

    with hidet.script_module() as script_module:

        @hidet.script
        def launch():
            attrs.func_kind = 'public'

            a = tensor('default', 'float32', shape=[2, 3, 4, 5])
            indices = [0, 1, 2, 3]
            updated_indices = [indices[i] if i in dims else 0 for i in range(len(indices))]
            a[updated_indices] = 1.0

            updated_indices = [indices[i] if i not in dims else 0 for i in range(len(indices))]
            a[updated_indices] = 2.0

    cm = script_module.build()
    cm()
