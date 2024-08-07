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
from typing import List, Optional, Callable, Union
from hidet.ir.func import Function
from hidet.ir.module import IRModule

from .instruments import PassInstrument


class PassContext:
    stack: List['PassContext'] = []

    def __init__(self, instruments: Optional[List[PassInstrument]] = None, verbose: bool = False):
        self.instruments: List[PassInstrument] = instruments if instruments else []
        self.verbose: bool = verbose

    @classmethod
    def current(cls):
        return cls.stack[-1]

    def __enter__(self):
        self.stack.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not (len(self.stack) > 0 and self.stack[-1] is self):
            print(self.stack, self.stack[-1] is self)
        assert len(self.stack) > 0 and self.stack[-1] is self
        self.stack.pop()


PassContext.stack.append(PassContext())


class Pass:
    def __init__(self, name=None):
        self.name = name if name else self.__class__.__name__

    def __call__(self, ir_module: IRModule) -> IRModule:
        ctx = PassContext.current()
        for instrument in ctx.instruments:
            instrument.before_pass(self.name, ir_module)
        ir_module = self.process_module(ir_module)
        for instrument in ctx.instruments:
            instrument.after_pass(self.name, ir_module)
        return ir_module

    @staticmethod
    def apply_transforms(
        node: Union[IRModule, Function],
        transforms: List[Callable[[Union[IRModule, Function]], Union[IRModule, Function]]],
        repeat_limit=1,
    ):
        while True:
            prev_node = node
            for rewriter in transforms:
                node = rewriter(node)
            # with open(./outs/apply_transform_{}.txt.format(repeat_limit), w) as f:
            #     f.write(str(node))
            if prev_node is node:
                break
            repeat_limit -= 1
            if repeat_limit == 0:
                break
            if repeat_limit < -100:
                raise RuntimeError("Exceeded repeat hard limit 100")
        return node

    def process_module(self, ir_module: IRModule) -> IRModule:
        new_funcs = {}
        for name, func in ir_module.functions.items():
            new_funcs[name] = self.process_func(func)
        if all(new_funcs[name] is ir_module.functions[name] for name in new_funcs):
            return ir_module
        else:
            return ir_module.copy().reset_funcs(new_funcs, ir_module.global_vars)

    def process_func(self, func: Function) -> Function:
        return func


class SequencePass(Pass):
    def __init__(self, passes: List[Pass], name=None):
        super().__init__(name)
        self.passes = passes

    def process_module(self, ir_module: IRModule) -> IRModule:
        for p in self.passes:
            ir_module = p(ir_module)
        return ir_module


class FunctionPass(Pass):
    def process_func(self, func: Function) -> Function:
        raise NotImplementedError()


class RepeatFunctionPass(FunctionPass):
    def __init__(self, passes: List[FunctionPass], repeat_limit=10, name=None):
        super().__init__(name)
        assert all(isinstance(p, FunctionPass) for p in passes)
        self.passes = passes
        self.repeat_limit = repeat_limit

    def process_func(self, func: Function) -> Function:
        for i in range(self.repeat_limit):
            orig_func = func
            for p in self.passes:
                func = p.process_func(func)
            if orig_func is func:
                return func
        print(f"Exceeded: {i} {self.name} on {func.name}")
        return func
