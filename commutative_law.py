# %%
import numpy as np
from numpy import testing
import torch

DEVICE='cuda'

p0 = torch.tensor(4, device=DEVICE, dtype=torch.int8)
p1 = torch.tensor(6, device=DEVICE, dtype=torch.int8)

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v3_0 = p0
        self.v5_0 = p1

    def forward(self, *args):
        v3_0 = self.v3_0
        v5_0 = self.v5_0
        mul = torch.mul(v5_0, v3_0)
        mul_1 = torch.mul(args[0], mul)
        abs_1 = torch.abs(mul_1)
        mul_2 = torch.mul(mul_1, mul)
        return (abs_1, mul_2)

model_0 = Model0()
output_names_0 = ['v5_0', 'v4_0']

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v3_0 = p0
        self.v5_0 = p1

    def forward(self, *args):
        v3_0 = self.v3_0
        v5_0 = self.v5_0
        mul = torch.mul(v3_0, args[0])
        mul_1 = torch.mul(v5_0, mul)
        abs_1 = torch.abs(mul_1)
        return (abs_1, )

model_1 = Model1()
output_names_1 = ['v5_0',]

data = np.random.normal(10, 0.1, size=(53, 33)).astype(np.int8)
input_data_0 = [data]

optmodel_0 = torch.compile(model_0, fullgraph=True, backend='hidet', mode=None)
model_out_0 = optmodel_0(*[torch.from_numpy(v).to(DEVICE) for v in input_data_0])
model_out_0 = [v.to(DEVICE).detach() for v in model_out_0] if isinstance(model_out_0, tuple) else [model_out_0.to(DEVICE).detach()]
model_out_0 = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_0]
print(model_out_0)
output_0 = dict(zip(output_names_0, model_out_0))
print(output_0)

input_data_1 = [data]

optmodel_1 = torch.compile(model_1, fullgraph=True, backend='hidet', mode=None)
model_out_1 = optmodel_1(*[torch.from_numpy(v).to(DEVICE) for v in input_data_1])
model_out_1 = [v.to(DEVICE).detach() for v in model_out_1] if isinstance(model_out_1, tuple) else [model_out_1.to(DEVICE).detach()]
model_out_1 = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_1]
output_1 = dict(zip(output_names_1, model_out_1))
output_name_dict = {'v5_0': 'v5_0'}

print('=========================')
try:
    for tensor_name_0, tensor_name_1 in output_name_dict.items():
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1], rtol=1, err_msg=f'at {tensor_name_0}, {tensor_name_1}')
    print("hidet does not trigger assertion")
except AssertionError as e:
    print("hidet triggers assertion")
    print(e)
print('=========================')

model_out_0 = model_0(*[torch.from_numpy(v).to(DEVICE) for v in input_data_0])
model_out_0 = [v.to(DEVICE).detach() for v in model_out_0] if isinstance(model_out_0, tuple) else [model_out_0.to(DEVICE).detach()]
model_out_0 = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_0]
output_0 = dict(zip(output_names_0, model_out_0))

model_out_1 = model_1(*[torch.from_numpy(v).to(DEVICE) for v in input_data_1])
model_out_1 = [v.to(DEVICE).detach() for v in model_out_1] if isinstance(model_out_1, tuple) else [model_out_1.to(DEVICE).detach()]
model_out_1 = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_1]
output_1 = dict(zip(output_names_1, model_out_1))

print('=========================')
try:
    for tensor_name_0, tensor_name_1 in output_name_dict.items():
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1], rtol=1, err_msg=f'at {tensor_name_0}, {tensor_name_1}')
    print("torch_eager does not trigger assertion")
except AssertionError as e:
    print("torch_eager triggers assertion")
    print(e)
print('=========================')