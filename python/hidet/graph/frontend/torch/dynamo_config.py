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
import warnings
from typing import Optional


def dynamo_config_warning():
    warnings.warn(
        'Configuration by dynamo_config is depricated. '
        'Please pass the options via mode and options arguments of torch.compile() '
    )


class DynamoConfig:
    def __init__(self):
        self._use_cuda_graph: bool = True
        self._use_tensor_core: bool = False
        self._print_input_graph: bool = False
        self._dump_graph_ir: Optional[str] = None
        self._correctness_report: bool = False
        self._steal_weights: bool = False

    def __getitem__(self, item: str):
        assert isinstance(item, str)
        return getattr(self, f"_{item}")

    def reset(self):
        """
        Reset the configuration to the default values
        """
        self._use_cuda_graph: bool = True
        self._use_tensor_core: bool = False
        self._print_input_graph: bool = False
        self._dump_graph_ir: Optional[str] = None
        self._correctness_report: bool = False
        self._steal_weights: bool = False

    def search_space(self, level: int = 2):
        """
        The schedule search space for the operator kernel tuning
        Candidates are: ``0``, ``1``, ``2``

        - ``0``:
            Use the default schedule, without tuning.
        - ``1``:
            Tune the schedule in a small search space. Usually takes less than one minute to tune a kernel.
        - ``2``:
            Tune the schedule in a large search space. Usually achieves the best performance, but takes longer time.

        Parameters
        ----------
        level: int
            The search space level.
        """
        dynamo_config_warning()

    def use_tensor_core(self, flag=True):
        """
        Whether to use tensor core
        """
        self._use_tensor_core = flag
        return self

    def use_fp16(self, flag=True):
        """
        Whether to use float16 data type
        """
        dynamo_config_warning()

    def use_fp16_reduction(self, flag=True):
        """
        Whether to use float16 data type for reduction
        """
        dynamo_config_warning()

    def use_attention(self, flag=False):
        """
        Whether to use fused attention schedule
        """
        dynamo_config_warning()

    def use_cuda_graph(self, flag=True):
        """
        Whether to use cuda graph
        """
        self._use_cuda_graph = flag
        return self

    def print_input_graph(self, flag=True):
        """
        Whether to print the input graph
        """
        self._print_input_graph = flag
        return self

    def dump_graph_ir(self, output_dir: str):
        """
        Whether to dump the graph ir

        Parameters
        ----------
        output_dir: str
            The output directory to dump the graph ir.
        """
        self._dump_graph_ir = output_dir
        return self

    def correctness_report(self, flag=True):
        """
        Whether to check correctness and print report error
        """
        self._correctness_report = flag
        return self

    def steal_weights(self, flag=True):
        """
        Whether to clear pytorch weights in certain layers after
        converting them to Hidet tensors. This will save some GPU memory usage.
        """
        self._steal_weights = flag
        return self


dynamo_config = DynamoConfig()
