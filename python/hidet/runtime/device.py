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


class Device:
    def __init__(self, device_type: str, device_index: Optional[int] = None):
        self.kind: str = device_type
        self.id: Optional[int] = device_index

    def __repr__(self) -> str:
        if self.id is None:
            return 'device({})'.format(self.kind)
        return 'device({}, {})'.format(self.kind, self.id)

    def __str__(self) -> str:
        if self.id is None:
            return self.kind
        return '{}:{}'.format(self.kind, self.id)

    def __eq__(self, other):
        if not isinstance(other, Device):
            raise ValueError('Cannot compare Device with {}'.format(type(other)))
        return self.kind == other.kind and self.id == other.id

    def __hash__(self):
        return hash((self.kind, self.id))

    def __enter__(self):
        import hidet.cuda
        import hidet.hip

        if self.is_cuda():
            setattr(self, '_prev_device', hidet.cuda.current_device())
            hidet.cuda.set_device(self.id)
        elif self.is_hip():
            setattr(self, '_prev_device', hidet.hip.current_device())
            hidet.hip.set_device(self.id)

    def __exit__(self, exc_type, exc_val, exc_tb):
        import hidet.cuda
        import hidet.hip

        if self.is_cuda():
            hidet.cuda.set_device(getattr(self, '_prev_device'))
        elif self.is_hip():
            hidet.hip.set_device(getattr(self, '_prev_device'))

    def is_cpu(self) -> bool:
        return self.kind == 'cpu'

    def is_cuda(self) -> bool:
        return self.kind == 'cuda'

    def is_vcuda(self) -> bool:
        return self.kind == 'vcuda'

    def is_hip(self) -> bool:
        return self.kind == 'hip'

    @property
    def target(self) -> str:
        if self.kind == 'vcuda':
            return 'cuda'
        return self.kind


def device(device_type: str, device_index: Optional[int] = None):
    if ':' in device_type:
        if device_index is not None:
            raise RuntimeError(
                'device_type must not contain ":" to specify device_index when device_index is '
                f'specified explicitly as a separate argument: ({device_type}, {device_index})'
            )
        items = device_type.split(':')
        if len(items) != 2:
            raise ValueError(f'Invalid device_type: {device_type}')
        device_type, device_index = items
        if not device_index.isdigit():
            raise ValueError(f'Invalid device_index: {device_index}')
        device_index = int(device_index)

    if device_type not in ['cpu', 'cuda', 'vcuda', 'hip']:
        raise ValueError(f'Invalid device_type: {device_type}, must be "cpu" "cuda", "vcuda", or "hip"')

    if device_index is not None and not isinstance(device_index, int):
        raise ValueError(f'Invalid device_index: {device_index}, must be an integer')

    return Device(device_type, device_index)


def instantiate_device(dev) -> Device:
    """
    Instantiate a device from a device string or a device object.

    This function will be used to get a concrete device object from a user given device string or device object. When a
    device object is given, but it does not have a device index (in case of CUDA or HIP device), the device index will
    be set to the current CUDA/HIP device.

    Parameters
    ----------
    dev: str or Device
        The device string or a device object.

    Returns
    -------
    device: Device
        The instantiated device object.
    """
    import hidet.cuda
    import hidet.hip

    if isinstance(dev, str):
        dev = device(dev)
    elif isinstance(dev, Device):
        dev = Device(dev.kind, dev.id)  # make a copy
    else:
        raise ValueError(
            f'Invalid device: {dev}, must be a device string (e.g., "cuda") or a Device object (e.g., '
            f'hidet.device("cuda", 0)).'
        )
    if dev.kind == 'cpu':
        dev.id = None  # CPU device does not have a device index
        return dev
    elif dev.kind in ['cuda', 'vcuda']:
        if dev.id is None:
            dev.id = hidet.cuda.current_device()
        return dev
    elif dev.kind == 'hip':
        if dev.id is None:
            dev.id = hidet.hip.current_device()
        return dev
    else:
        assert False
