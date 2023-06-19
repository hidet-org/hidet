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
# pylint: disable=protected-access

from typing import List, Union

from hidet.cuda.nccl import NcclCommunicator, NcclUniqueId, create_comm, comms_to_array
from hidet.ffi import runtime_api
from .flow_graph import FlowGraph
from .tensor import Tensor


class DistributedFlowGraph:
    def __init__(self, g: FlowGraph, nranks: int, rank: int):
        self._g = g
        self._nranks = nranks
        self._rank = rank
        self._comms: List[NcclCommunicator] = []

    def initialize(self, unique_id: NcclUniqueId):
        """
        This is the default initialization function.
        Should be replaced by a customized one if the compiler gives non-trivial schedule.
        """
        comm = create_comm(self._nranks, unique_id, self._rank)
        self._comms = [comm]

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        comms_array = comms_to_array(self._comms)
        # We need an explicit variable to ensure the comms_array will not be garbage-collected.
        runtime_api.set_nccl_comms(comms_array)
        return self._g.forward(inputs)

    def __call__(self, *inputs: Tensor) -> Union[List[Tensor], Tensor]:
        comms_array = comms_to_array(self._comms)
        # We need an explicit variable to ensure the comms_array will not be garbage-collected.
        runtime_api.set_nccl_comms(comms_array)
        return self._g(*inputs)
