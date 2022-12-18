from .availability import available, dynamo_available

if available():
    from .interpreter import ImportedTorchModule, from_torch
    from . import register_functions
    from . import register_modules
    from . import register_methods
    from .utils import dtype_from_torch

if dynamo_available():
    from . import dynamo_backends
    from .dynamo_backends import onnx2hidet_backend, hidet_backend

    dynamo_backends.register_dynamo_backends()
