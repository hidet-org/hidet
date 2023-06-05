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
# %%
from hidet.utils.model_translator import AstInterpreter, transpiled_str, vis_interpreter

# %%
import torch
import numpy as np

# transpile a simple function
def orig_func(a, b: torch.Tensor, c: float, dim: int = 1):
    """
        doc str
    """
    x = torch.matmul(a, b)
    y = np.matmul(a.numpy(), b.numpy())
    return torch.nn.functional.softmax(x.view([1, -1]), dim) + c

interpreter = AstInterpreter()
args = [torch.rand([4, 5]), torch.rand([5, 5]), 1.0]
orig = orig_func(*args)
# res is the result of evaluating the function
res = interpreter(orig_func, args)
assert torch.allclose(res, orig)

# shows the state the interpreter
vis_interpreter(interpreter)
print("final result:\n")
print(transpiled_str(interpreter))

# %%
# conditionals with multiple branches may need to have all branches covered
def conditional(a: torch.Tensor, b: torch.Tensor, c):
    if ((a - b).abs() < c).any():
        d = a + b
    elif c < 3:
        d = a - b.pow(2)
    else:
        d = None
    return d

interpreter = AstInterpreter()
# each call activates a branch, updating the type of the nodes
interpreter(conditional, [torch.rand([4, 4]), torch.rand([4, 4]), 1])
vis_interpreter(interpreter)
interpreter(conditional, [torch.rand([4, 4]), torch.rand([4, 4]) * 10000, 1])
vis_interpreter(interpreter)
interpreter(conditional, [torch.rand([4, 4]), torch.rand([4, 4]) * 10000, 5])
vis_interpreter(interpreter)

print(transpiled_str(interpreter))

# %%
def forloop(a: torch.Tensor, c, l):
    for i in range(l):
        a = a.pow(2) + c
        if i > 10:
            break
    else:
        print("hi")
    return a
 
interpreter = AstInterpreter()
res = interpreter(forloop, [torch.rand([4, 4]), torch.rand([4, 4]), 13])
vis_interpreter(interpreter)
print(transpiled_str(interpreter))

# %%
# calling a function that is not in the torch name space will trigger a recursive trace
#   asking for user input whether to trace or not
def mysoftmax(a, dim):
    return torch.softmax(a, dim)

def raw_function(a, b, c, d):
    d = torch.tensor([1, 2]).to(c.device)
    x = a + mysoftmax(b, dim = 1)
    a, b = (1, 2)
    a, b = (torch.tensor([1, 2]).float(), c)
    a, b = (torch.rand(4, 3).to(a.device), c)
    return d

interpreter = AstInterpreter()
res = interpreter(raw_function, [torch.rand([2, 2]), torch.rand([2, 2]), torch.rand([1, 2]), torch.rand([1, 1])])
vis_interpreter(interpreter)
print(transpiled_str(interpreter))

# %%
# tracing classes works a bit differently
class TestClass(torch.nn.Module):
    h = 1
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(dim, dim, kernel_size=[3, 3], padding=1)
        self.conv2 = torch.nn.Conv2d(dim, dim, kernel_size=[3, 3], padding=1)
        self.num = self.test(dim)
        self.num2 = self.test1(1)
    
    def forward(self, x):
        y1 = self.conv(x)
        y2 = self.test2
        return self.conv2(x) + y1 + y2

    @staticmethod
    def test1(x):
        return x + 2
    
    def test(self, x):
        return x + 1
    
    @property
    def test2(self):
        return self.num + 1

# visualize(TestClass, [3])
intp = AstInterpreter()
# we call the interpreter on the class, which is equivalent to tracing the class's constructor, if applicable
#   h is the result of evaluating the constructor
h = intp(TestClass, [3])
# we then trace the forward method
intp(h.forward, [torch.rand([1, 3, 5, 5])])

vis_interpreter(intp)
print(transpiled_str(intp))

# %%
# inheritance would trigger more recursive traces
class TestClass2(TestClass):
    def __init__(self, dim) -> None:
        super().__init__(dim)
        self.conv3 = torch.nn.Conv2d(dim, dim, kernel_size=[3, 3], padding=1)
        self.num1 = self.test(3)
        self.num4 = self.test2
    
    def forward(self, x):
        y = super().forward(x)
        y = self.conv3(y)
        return y

intp = AstInterpreter()
h = intp(TestClass2, [3])
intp(h.forward, [torch.rand([1, 3, 5, 5])])
vis_interpreter(intp)
print(transpiled_str(intp))
