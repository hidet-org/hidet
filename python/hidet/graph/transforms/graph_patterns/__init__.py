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
from .base import TensorPattern, OperatorPattern, SubgraphRewriteRule, MatchDict, Usage, graph_pattern_match
from .base import register_rewrite_rule, op_pattern, registered_rewrite_rules
from .arithmetic_patterns import arithmetic_patterns
from .transform_patterns import transform_patterns
from .conv2d_patterns import conv2d_patterns
from .matmul_patterns import matmul_patterns
