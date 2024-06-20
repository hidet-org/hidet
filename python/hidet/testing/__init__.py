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
from hidet.utils import benchmark_func

from . import models
from . import utils
from .utils import check_unary, check_unary_dynamic, check_binary, check_binary_dynamic
from .utils import check_ternary, check_torch_unary
from .utils import check_torch_binary, check_torch_binary_with_inputs, check_torch_binary_dynamic, check_torch_ternary
