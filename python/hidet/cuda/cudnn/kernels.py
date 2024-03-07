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
from typing import Union
from hidet.ir.dtypes import DataType
from .ffi import cudnnDataType
from . import ffi
from .utils import as_pointer, as_cudnn_type


def conv2d(
    n: int,
    c: int,
    h: int,
    w: int,
    k: int,
    r: int,
    s: int,
    p: int,
    q: int,
    ptr_x,
    ptr_w,
    ptr_y,
    tx: Union[int, DataType],
    tw: Union[int, DataType],
    ty: Union[int, DataType],
    compute_type: Union[int, cudnnDataType],
    pad_dim1: int,
    pad_dim2: int,
    str_dim1: int,
    str_dim2: int,
    dil_dim1: int,
    dil_dim2: int,
):
    """
    Calculates the 2D convolution of tensor x with filter w, stores the result in tensor y.

    Parameters
    ----------
    n: int
        Batch number.
    c: int
        Number of channels in the input tensor x.
    h: int
        Height of the input tensor x.
    w: int
        Width of the input tensor x.
    k: int
        Number of channels in the output tensor y.
    r: int
        Height of the filter w.
    s: int
        Width of the filter w.
    p: int
        Height of the output tensor y.
    q: int
        Width of the output tensor y.
    ptr_x: hidet.Tensor or int
        Input tensor x, can be either a Tensor or an integer (the address of the tensor).
    ptr_w: hidet.Tensor or int
        Weight tensor w, can be either a Tensor or an integer (the address of the tensor).
    ptr_y: hidet.Tensor or int
        Output tensor y, can be either a Tensor or an integer (the address of the tensor).
    tx: Union[int, DataType]
        Type of elements in tensor x.
    tw: Union[int, DataType]
        Type of elements in tensor w.
    ty: Union[int, DataType]
        Type of elements in tensor y.
    compute_type: Union[int, cudnnDataType]
        The compute type of the operation.
        For cuDNN, there's no such thing as a cudnnComputeType_t type.
        As per the official example, the computeType is defined in terms of cudnnDataType_t
    pad_dim1: int
        The value to use for padding along the height dimension
    pad_dim2: int
        The value to use for padding along the width dimension
    str_dim1: int
        The stride to use for the height dimension
    str_dim2: int
        The stride to use for the width dimension
    dil_dim1: int
        The dilation to use for the height dimension
    dil_dim2: int
        The dilation to use for the width dimension
    """
    ffi.conv2d(
        n,
        c,
        h,
        w,
        k,
        r,
        s,
        p,
        q,
        as_pointer(ptr_x),
        as_pointer(ptr_w),
        as_pointer(ptr_y),
        as_cudnn_type(tx),
        as_cudnn_type(tw),
        as_cudnn_type(ty),
        compute_type,
        pad_dim1,
        pad_dim2,
        str_dim1,
        str_dim2,
        dil_dim1,
        dil_dim2,
    )
