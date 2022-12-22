"""
Optimize PyTorch Model
======================

Hidet provides a backend to pytorch dynamo to optimize PyTorch models. To use this backend, you need to specify 'hidet'
as the backend when calling :func:`torch.compile` such as

.. code-block:: python

    # optimize the model with hidet provided backend 'hidet'
    model_hidet = torch.compile(model, backend='hidet')

That's it! The first run of hidet optimized model would take tens of minutes to optimize the model and tune the
schedules. Hidet also provides some configurations to control the optimization of hidet backend, as well as correctness
checking tools.

Configure hidet backend of pytorch dynamo
-----------------------------------------

"""
import torch
import torch._dynamo as dynamo
import hidet
from hidet.testing import benchmark_func

x = torch.randn(1, 3, 224, 224).cuda()
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True).cuda().eval()

# torch dynamo requires to call the reset function before we use another backend
dynamo.reset()

# optimize the model with hidet provided backend 'hidet'
# the first run of hidet optimized model would take 20 to 30 minutes to optimize the model and tune the execution
# schedule for each convolution/matrix multiplication in the model. Usually, the optimizing time depends on the number
# of these operators in the model and the performance of your CPU. It takes about 1 minute to tune such a kernel on
# an i9-12900k CPU.
with torch.no_grad():
    # configure the search space of operator kernel tuning.
    # 0: Use the default schedule, without tuning. [Default]
    # 1: Tune the schedule in a small search space. Usually takes some time to tune a kernel.
    # 2: Tune the schedule in a large search space. Usually achieves the best performance, but takes longer time.
    hidet.torch.dynamo_config.search_space(1)
    model_hidet = torch.compile(model, backend='hidet')
    model_hidet(x)

    # benchmark the performance of the optimized model
    hidet_latency: float = benchmark_func(lambda: model_hidet(x), warmup=3, number=10, repeat=10)

# print(' inductor: {} ms'.format(inductor_latency))
print('    hidet: {} ms'.format(hidet_latency))
