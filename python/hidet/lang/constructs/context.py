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
from typing import Optional, Any

from hidet.ir.stmt import Stmt


class HidetContext:
    """
    Custom context manager used in Hidet Script to support the syntax of `with ... as ...`.

    with HidetContext() as bind_value:
        body(bind_value)

    with be transformed to

    post_process(body(bind_value))
    """

    def bind_value(self) -> Optional[Any]:
        return None

    def post_process(self, body: Stmt) -> Stmt:
        raise NotImplementedError()
