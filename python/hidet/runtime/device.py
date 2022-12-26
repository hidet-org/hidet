from typing import Optional
from hidet.cuda import current_device


class Device:
    def __init__(self, device_type: str, device_index: Optional[int] = None):
        self.type: str = device_type
        self.id: Optional[int] = device_index

    def __eq__(self, other):
        if not isinstance(other, Device):
            raise ValueError('Cannot compare Device with {}'.format(type(other)))
        return self.type == other.type and self.id == other.id

    def __repr__(self) -> str:
        if self.id is None:
            return 'device({})'.format(self.type)
        return 'device({}, {})'.format(self.type, self.id)

    def __str__(self) -> str:
        if self.id is None:
            return self.type
        return '{}:{}'.format(self.type, self.id)

    def is_cpu(self) -> bool:
        return self.type == 'cpu'

    def is_cuda(self) -> bool:
        return self.type == 'cuda'


def device(device_type: str, device_index: Optional[int] = None):
    if ':' in device_type:
        if device_index is not None:
            raise RuntimeError('device_type must not contain ":" to specify device_index when device_index is '
                               f'specified explicitly as a separate argument: ({device_type}, {device_index})')
        items = device_type.split(':')
        if len(items) != 2:
            raise ValueError(f'Invalid device_type: {device_type}')
        device_type, device_index = items
        if not device_index.isdigit():
            raise ValueError(f'Invalid device_index: {device_index}')
        device_index = int(device_index)

    if device_type not in ['cpu', 'cuda']:
        raise ValueError(f'Invalid device_type: {device_type}, must be "cpu" or "cuda"')

    if device_index is not None and not isinstance(device_index, int):
        raise ValueError(f'Invalid device_index: {device_index}, must be an integer')

    return Device(device_type, device_index)


def instantiate_device(dev) -> Device:
    """
    Instantiate a device from a device string or a device object.

    This function will be used to get a concrete device object from a user given device string or device object. When a
    device object is given, but it does not have a device index (in case of CUDA device), the device index will be set
    to the current CUDA device.

    Parameters
    ----------
    dev: str or Device
        The device string or a device object.

    Returns
    -------
    device: Device
        The instantiated device object.
    """
    if isinstance(dev, str):
        dev = device(dev)
    elif isinstance(dev, Device):
        dev = Device(dev.type, dev.id)  # make a copy
    else:
        raise ValueError(f'Invalid device: {dev}, must be a device string (e.g., "cuda") or a Device object (e.g., '
                         f'hidet.device("cuda", 0)).')
    if dev.type == 'cpu':
        dev.id = None  # CPU device does not have a device index
        return dev
    elif dev.type == 'cuda':
        if dev.id is None:
            dev.id = current_device()
        return dev
    else:
        assert False
