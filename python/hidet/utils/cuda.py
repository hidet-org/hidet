import pycuda.driver
import pycuda.autoinit


class Attr:
    COUNT = 'count'
    NAME = 'name'
    TOTAL_MEMORY = 'total_memory'
    COMPUTE_CAPACITY = 'compute_capacity'
    ARCH_NAME = 'arch_name'


_arch2name = {
    (2, 0): 'Fermi',
    (3, 0): 'Kepler',
    (3, 5): 'Kepler',
    (3, 7): 'Kepler',
    (5, 0): 'Maxwell',
    (5, 2): 'Maxwell',
    (5, 3): 'Maxwell',
    (6, 0): 'Pascal',
    (6, 1): 'Pascal',
    (6, 2): 'Pascal',
    (7, 0): 'Volta',
    (7, 2): 'Volta',
    (7, 5): 'Turing',
    (8, 0): 'Ampere',
    (8, 6): 'Ampere'
}


def get_attribute(attr_name: str, device_no=0):
    device = pycuda.driver.Device(device_no)
    if attr_name == 'count':
        return device.count()
    elif attr_name == 'name':
        return device.name()
    elif attr_name == 'total_memory':
        return device.total_memory()
    elif attr_name == 'compute_capacity':
        return device.compute_capability()
    elif attr_name == 'arch_name':
        return _arch2name[get_attribute(Attr.COMPUTE_CAPACITY)]
    else:
        name = attr_name.upper()
        return device.get_attribute(getattr(pycuda.driver.device_attribute, name))


def get_attributes(device_no=0):
    attr_names = [
        'count', 'name', 'total_memory', 'compute_capacity', 'arch_name'
    ]
    device = pycuda.driver.Device(device_no)
    attrs = {}
    attrs.update({name: get_attribute(name) for name in attr_names})
    attrs.update({str(name).lower(): value for name, value in device.get_attributes().items()})
    return attrs


def device_synchronize():
    pycuda.driver.Context.synchronize()


if __name__ == '__main__':
    for k, v in get_attributes().items():
        print("{:>40}: {}".format(k, v))
