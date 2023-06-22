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
from .ffi import nccl_available, nccl_version, nccl_library_filename
from .comm import (
    create_comm,
    NcclUniqueId,
    NcclDataType,
    NcclRedOp,
    comms_to_array,
    init_unique_id,
    dtype_to_nccl,
    NcclCommunicator,
    str_to_nccl_op,
    NCCL_SPLIT_NOCOLOR,
)
