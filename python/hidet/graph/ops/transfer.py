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
from typing import List, Union, Tuple

from hidet.ir.type import DataType
from hidet.ir.expr import Expr
from hidet.ir.module import IRModule
from hidet.ir.task import Target
from hidet.utils import prod
from hidet.runtime.device import Device, instantiate_device
from .utils import Task, TensorNode, Operator, Tensor, compute, input_like


class TransferTask(Task):
    def __init__(self, x: TensorNode, src_device: Device, dst_device: Device):
        allowed_devices = ['cpu', 'cuda']
        if src_device not in allowed_devices:
            raise RuntimeError(f'Unsupported source device {src_device}, candidate devices are {allowed_devices}')
        if dst_device not in allowed_devices:
            raise RuntimeError(f'Unsupported destination device {dst_device}, candidate devices are {allowed_devices}')

        y = compute('out', x.shape, lambda *indices: x[indices])
        self.src_device: Device = src_device
        self.dst_device: Device = dst_device
        super().__init__('transfer', inputs=[x], outputs=[y], attributes={
            'src': src_device,
            'dst': dst_device
        })

    def implement(self, target: Union[Target, str], working_dir: str) -> List[IRModule]:
        import hidet
        from hidet.ir.primitives.cuda import memcpy_async
        from hidet.lang import attrs

        dtype: DataType = self.inputs[0].type.dtype
        shape: Tuple[Expr, ...] = self.inputs[0].shape
        nbytes = dtype.nbytes * prod(shape)
        kind = f'{self.src_device.type}_to_{self.dst_device.type}'

        if (self.src_device.type == 'cuda' and self.src_device.id > 0) or (self.dst_device.type == 'cuda' and self.dst_device.id > 0):
            raise NotImplementedError('The transfer between non-default CUDA devices is not supported yet.')

        with hidet.script_module() as script_module:

            @hidet.script
            def launch(x: dtype[shape], y: dtype[shape]):
                attrs.func_kind = 'public'

                memcpy_async(y, x, count=nbytes, kind=kind)

        return [script_module.ir_module()]


class TransferOp(Operator):
    def __init__(self, x: Tensor, device: Device):
        super().__init__(
            inputs=[x],
            attributes={'device': device},
            task=TransferTask(input_like(x, 'x'), src_device=x.device, dst_device=device)
        )

def transfer(x: Tensor, dst_device: Union[str, Device]) -> Tensor:
    dst_device: Device = instantiate_device(dst_device)
    if dst_device == x.device:
        # no need to transfer if the destination device is the same as the source device
        return x
    return TransferOp(x, dst_device).outputs[0]
