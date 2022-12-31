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
from hidet.ir.stmt import BlackBoxStmt


def printf(format_string, *args):
    """
    usage:
    printf("%d %d\n", expr_1, expr_2)
    """
    format_string = format_string.replace('\n', '\\n')
    if len(args) > 0:
        arg_string = ', '.join(['{}'] * len(args))
        template_string = f'printf("{format_string}", {arg_string});'
    else:
        template_string = f'printf("{format_string}");'
    # if '\n' in format_string:
    #     raise ValueError('Please use printf(r"...\\n") instead of printf("...\\n").')
    return BlackBoxStmt(template_string, *args)
