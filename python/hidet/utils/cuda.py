import pycuda.driver
import pycuda.autoinit


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
    else:
        name = attr_name.upper()
        return device.get_attribute(getattr(pycuda.driver.device_attribute, name))


def get_attributes(device_no=0):
    attr_names = [
        'count', 'name', 'total_memory', 'compute_capacity',
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
