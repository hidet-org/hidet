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
from typing import Optional
from hidet.ir.stmt import BlackBoxStmt, Stmt


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


def comment(comment_string: str, style: Optional[str] = None) -> Stmt:
    """
    Generate a comment statement.

    usage:
    > comment("This is a comment.")
    // This is a comment.

    > comment("This is a comment.\nThis is the second line.")
    /*
     * This is a comment.
     * This is the second line.
     */

    > comment("This is a comment.", style='//')
    // This is a comment.

    > comment("This is a comment.", style='/*')
    /*
     * This is a comment.
     */

    > comment("This is a comment.\nThis is the second line.", style='//')
    // This is a comment.
    // This is the second line.
    """
    lines = comment_string.split("\n")

    if style is None:
        if len(lines) > 1:
            style = "/*"
        else:
            style = "//"

    if style not in ["//", "/*"]:
        raise ValueError('Invalid style: "{}", candidates: "//", "/*".'.format(style))

    if style == "/*":
        content = "\n".join(["/*"] + [" * " + line for line in lines] + [" */"])
    else:
        content = "\n".join(["// " + line for line in lines])
    return BlackBoxStmt(template_string=content)


def __builtin_assume(arg):
    return BlackBoxStmt('__builtin_assume({});', arg)
