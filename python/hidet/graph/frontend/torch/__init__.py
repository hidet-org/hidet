from .availability import available, dynamo_available

if available():
    from .interpreter import Interpreter, from_torch
    from . import register_functions
    from . import register_modules
    from . import register_methods

if dynamo_available():
    from . import dynamo_backends
    from .dynamo_backends import onnx2hidet_backend, hidet_backend, dynamo_config

    dynamo_backends.register_dynamo_backends()
