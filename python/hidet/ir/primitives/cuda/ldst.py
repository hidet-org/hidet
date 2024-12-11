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
# pylint: disable=cell-var-from-loop
from typing import Optional

from hidet.ir.type import PointerType, TensorPointerType, data_type
from hidet.ir.expr import Expr
from hidet.ir.func import Function
from hidet.ir.tools import infer_type
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.ir.type import DataType
from hidet.utils import initialize


def resolve_load_inst_name(dtype: str, space: str, sync: Optional[str], scope: str) -> str:
    dtype = data_type(dtype)
    nbytes = dtype.nbytes
    nbits = nbytes * 8
    if sync:
        if space == 'generic':
            inst_name = f'ld.{sync}.{scope}.b{nbits}'
        else:
            inst_name = f'ld.{sync}.{scope}.{space}.b{nbits}'
    else:
        if space == 'generic':
            inst_name = f'ld.b{nbits}'
        else:
            inst_name = f'ld.{space}.b{nbits}'
    return inst_name


def resolve_store_inst_name(dtype: str, space: str, sync: Optional[str], scope: str) -> str:
    dtype = data_type(dtype)
    nbytes = dtype.nbytes
    nbits = nbytes * 8
    if sync:
        if space == 'generic':
            inst_name = f'st.{sync}.{scope}.b{nbits}'
        else:
            inst_name = f'st.{sync}.{scope}.{space}.b{nbits}'
    else:
        if space == 'generic':
            inst_name = f'st.b{nbits}'
        else:
            inst_name = f'st.{space}.b{nbits}'
    return inst_name


@initialize()
def register_functions():
    from hidet.lang import attrs, script, asm  # pylint: disable=import-outside-toplevel

    registered = set()
    for dtype in ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int32', 'int64', 'float16', 'float32']:
        for space in ['generic', 'global']:
            for sync in ['acquire']:
                for scope in ['gpu']:
                    inst_name = resolve_load_inst_name(dtype, space, sync, scope)
                    func_name = 'cuda_' + inst_name.replace('.', '_') + f'_{dtype}'
                    if func_name in registered:
                        continue
                    registered.add(func_name)

                    @script
                    def cuda_load(addr: ~data_type(dtype)) -> data_type(dtype):
                        attrs.func_kind = 'cuda_internal'
                        attrs.func_name = func_name
                        template = inst_name + ' %0, [%1];'
                        ret: data_type(dtype) = 0  # define a variable used to store the loaded data
                        asm(template, outputs=[ret], inputs=[addr], is_volatile=True)
                        return ret

                    assert isinstance(cuda_load, Function)
                    register_primitive_function(name=cuda_load.name, func_or_type=cuda_load)

    for dtype in ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int32', 'int64', 'float16', 'float32']:
        for space in ['generic', 'global']:
            for sync in ['release']:
                for scope in ['gpu']:
                    inst_name = resolve_store_inst_name(dtype, space, sync, scope)
                    func_name = 'cuda_' + inst_name.replace('.', '_') + f'_{dtype}'
                    if func_name in registered:
                        continue
                    registered.add(func_name)

                    @script
                    def cuda_store(addr: ~data_type(dtype), value: data_type(dtype)):
                        attrs.func_kind = 'cuda_internal'
                        attrs.func_name = func_name
                        template = inst_name + ' [%0], %1;'
                        asm(template, inputs=[addr, value], is_volatile=True)

                    assert isinstance(cuda_store, Function)
                    register_primitive_function(name=cuda_store.name, func_or_type=cuda_store)


def resolve_pointed_dtype(addr: Expr) -> str:
    ptr_type = infer_type(addr)
    if not isinstance(ptr_type, (PointerType, TensorPointerType)):
        raise ValueError("Expect a pointer type, got {}".format(ptr_type))
    if isinstance(ptr_type, PointerType):
        dtype = ptr_type.base_type
    else:
        dtype = ptr_type.tensor_type.dtype
    if not isinstance(dtype, DataType):
        raise ValueError("Expect a pointer to a scalar type, got {}".format(ptr_type))
    return dtype.name


def load(addr: Expr, space: str = 'generic', sync: Optional[str] = None, scope: Optional[str] = None):
    """
    Load data from memory.

    Parameters
    ----------
    addr: Expr
        The address of the data, in a type of pointer.

    space: str
        The memory space of the address. Candidates: 'generic', 'global', 'shared', 'local'

    sync: Optional[str]
        The synchronization behavior. Candidates: None, 'acquire', and 'relaxed'.

    scope: Optional[str]
        The scope of the synchronization. Candidates: None, 'cta', 'gpu', 'sys'.

    Returns
    -------
    ret: Expr
        The loaded data.
    """
    dtype = resolve_pointed_dtype(addr)
    func_name = 'cuda_' + resolve_load_inst_name(dtype, space, sync, scope).replace('.', '_') + f'_{dtype}'
    return call_primitive_func(func_name, [addr])


def store(addr: Expr, value: Expr, space: str = 'generic', sync: Optional[str] = 'release', scope: str = 'gpu'):
    """
    Store data to memory.

    Parameters
    ----------
    addr: Expr
        The address to store the data.

    value: Expr
        The value to store.

    space: str
        The memory space of the address. Candidates: 'generic', 'global', 'shared', 'local'

    sync: Optional[str]
        The synchronization behavior. Candidates: 'release', and 'relaxed'.

    scope: str
        The scope of the synchronization. Candidates: 'cta', 'gpu', 'sys'.
    """
    dtype = resolve_pointed_dtype(addr)
    func_name = 'cuda_' + resolve_store_inst_name(dtype, space, sync, scope).replace('.', '_') + f'_{dtype}'
    return call_primitive_func(func_name, [addr, value])


@initialize()
def register_ldg():
    from hidet.ir.dtypes import u32, i32, u16, boolean
    from hidet.ir.expr import Var, cast, deref
    from hidet.ir.stmt import AsmStmt
    from hidet.ir.builders import FunctionBuilder
    from hidet.ir.type import void_p
    from hidet.lang import attrs, script, asm  # pylint: disable=import-outside-toplevel

    for cache_operation in ["always", "last_use"]:
        cache_policy = ".lu" if cache_operation == "last_use" else ""
        func_name = f"cuda_ldg2{cache_policy}".replace(".", "_")

        @script
        def cuda_load(reg: void_p, addr: void_p, pred: boolean):
            attrs.func_kind = 'cuda_internal'
            attrs.func_name = func_name
            template = '{ .reg .pred p; setp.ne.b32 p, %2, 0; mov.b16 %0, 0; @p ld.global.u16 %0, [%1]; }'
            asm(template, outputs=[deref(cast(reg, ~u16))], inputs=[addr, cast(pred, i32)], is_volatile=True)

        assert isinstance(cuda_load, Function)
        register_primitive_function(name=cuda_load.name, func_or_type=cuda_load)

    for load_bytes in [4, 8, 16, 32]:
        for cache_operation in ["always", "last_use"]:
            cache_policy = ".lu" if cache_operation == "last_use" else ""
            func_name = f"cuda_ldg{load_bytes}{cache_policy}".replace(".", "_")
            with FunctionBuilder(func_name, kind="cuda_internal") as fb:
                insts = []
                insts.append(".reg .pred p;")
                nr_outputs = load_bytes // u32.nbytes
                nr_outputs_per_inst = min(nr_outputs, 4)
                nr_insts = nr_outputs // nr_outputs_per_inst
                reg_vars = [Var(f"reg{i}", void_p) for i in range(nr_outputs)]
                addr_var = Var("addr", void_p)
                pred_var = Var("pred", boolean)
                fb.extend_params(reg_vars + [addr_var, pred_var])
                insts.append(f"setp.ne.b32 p, %{1 + nr_outputs}, 0;")
                vector_size = f".v{nr_outputs_per_inst}" if nr_outputs_per_inst > 1 else ""
                ld_inst = f"ld.global{vector_size}{cache_policy}.u32"
                mov = [f"mov.b32 %{i}, %{i + nr_outputs + 2};" for i in range(nr_outputs)]
                insts += mov
                cur = 0
                extra_addr_var = []
                for i in range(nr_insts):
                    dst = [f"%{j}" for j in range(cur, cur + nr_outputs_per_inst)]
                    dst = "{" + ", ".join(dst) + "}"
                    src = nr_outputs if cur == 0 else 2 * nr_outputs + 2 + i - 1
                    insts.append(f"@p {ld_inst} {dst}, [%{src}];")
                    cur += nr_outputs_per_inst
                    if i > 0:
                        extra_addr_var.append(("l", cast(addr_var, ~u32) + 4 * i))

                doc = "{" + "".join(insts) + "}"
                body = AsmStmt(
                    doc,
                    outputs=[("=r", deref(cast(var, PointerType(u32)))) for var in reg_vars],
                    inputs=[("l", addr_var), ("r", cast(pred_var, i32))]
                    + [("r", deref(cast(var, PointerType(u32)))) for var in reg_vars]
                    + extra_addr_var,
                    is_volatile=True,
                )
                fb.set_body(body)
            register_primitive_function(name=func_name, func_or_type=fb.get())


@initialize()
def register_stg():
    from hidet.ir.dtypes import u32, i32, u16, boolean
    from hidet.ir.expr import Var, cast, deref
    from hidet.ir.stmt import AsmStmt
    from hidet.ir.builders import FunctionBuilder
    from hidet.ir.type import void_p
    from hidet.lang import attrs, script, asm  # pylint: disable=import-outside-toplevel

    @script
    def cuda_store(reg: void_p, addr: void_p, pred: boolean):
        attrs.func_kind = 'cuda_internal'
        attrs.func_name = 'cuda_stg2'
        template = '{ .reg .pred p; setp.ne.b32 p, %1, 0; @p st.global.u16 [%0], %2; }'
        asm(template, outputs=[], inputs=[addr, cast(pred, i32), deref(cast(reg, ~u16))], is_volatile=True)

    assert isinstance(cuda_store, Function)
    register_primitive_function(name=cuda_store.name, func_or_type=cuda_store)

    for store_bytes in [4, 8, 16, 32, 64]:
        func_name = f"cuda_stg{store_bytes}"
        with FunctionBuilder(func_name, kind="cuda_internal") as fb:
            insts = []
            insts.append(".reg .pred p;")
            nr_inputs = store_bytes // u32.nbytes
            nr_inputs_per_inst = min(nr_inputs, 4)
            nr_insts = nr_inputs // nr_inputs_per_inst
            reg_vars = [Var(f"reg{i}", void_p) for i in range(nr_inputs)]
            addr_var = Var("addr", void_p)
            pred_var = Var("pred", boolean)
            fb.extend_params(reg_vars + [addr_var, pred_var])
            insts.append("setp.ne.b32 p, %1, 0;")
            vector_size = f".v{nr_inputs_per_inst}" if nr_inputs_per_inst > 1 else ""
            st_inst = f"st.global{vector_size}.u32"
            cur = 2
            extra_addr_var = []
            for i in range(nr_insts):
                src = [f"%{j}" for j in range(cur, cur + nr_inputs_per_inst)]
                src = "{" + ", ".join(src) + "}"
                dst = 0 if i == 0 else nr_inputs + 2 + i - 1
                insts.append(f"@p {st_inst} [%{dst}], {src};")
                cur += nr_inputs_per_inst
                if i > 0:
                    extra_addr_var.append(("l", cast(addr_var, ~u32) + 4 * i))

            doc = "{" + "".join(insts) + "}"
            body = AsmStmt(
                doc,
                outputs=[],
                inputs=[("l", addr_var), ("r", cast(pred_var, i32))]
                + [("r", deref(cast(var, PointerType(u32)))) for var in reg_vars]
                + extra_addr_var,
                is_volatile=True,
            )
            fb.set_body(body)
        register_primitive_function(name=func_name, func_or_type=fb.get())


def ldg256(reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, gmem_addr, pred_guard):
    return call_primitive_func("cuda_ldg32", [reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, gmem_addr, pred_guard])


def ldg128(reg0, reg1, reg2, reg3, gmem_addr, pred_guard):
    return call_primitive_func("cuda_ldg16", [reg0, reg1, reg2, reg3, gmem_addr, pred_guard])


def ldg64(reg0, reg1, gmem_addr, pred_guard):
    return call_primitive_func("cuda_ldg8", [reg0, reg1, gmem_addr, pred_guard])


def ldg32(reg0, gmem_addr, pred_guard):
    return call_primitive_func("cuda_ldg4", [reg0, gmem_addr, pred_guard])


def ldg16(reg0, gmem_addr, pred_guard):
    return call_primitive_func("cuda_ldg2", [reg0, gmem_addr, pred_guard])


def ldg256_lu(reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, gmem_addr, pred_guard):
    return call_primitive_func("cuda_ldg32_lu", [reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, gmem_addr, pred_guard])


def ldg128_lu(reg0, reg1, reg2, reg3, gmem_addr, pred_guard):
    return call_primitive_func("cuda_ldg16_lu", [reg0, reg1, reg2, reg3, gmem_addr, pred_guard])


def ldg64_lu(reg0, reg1, gmem_addr, pred_guard):
    return call_primitive_func("cuda_ldg8_lu", [reg0, reg1, gmem_addr, pred_guard])


def ldg32_lu(reg0, gmem_addr, pred_guard):
    return call_primitive_func("cuda_ldg4_lu", [reg0, gmem_addr, pred_guard])


def ldg16_lu(reg0, gmem_addr, pred_guard):
    return call_primitive_func("cuda_ldg2_lu", [reg0, gmem_addr, pred_guard])


def stg512(
    reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9, reg10, reg11, reg12, reg13, reg15, gmem_addr, pred_guard
):
    return call_primitive_func(
        "cuda_stg64",
        [
            reg0,
            reg1,
            reg2,
            reg3,
            reg4,
            reg5,
            reg6,
            reg7,
            reg8,
            reg9,
            reg10,
            reg11,
            reg12,
            reg13,
            reg15,
            gmem_addr,
            pred_guard,
        ],
    )


def stg256(reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, gmem_addr, pred_guard):
    return call_primitive_func("cuda_stg32", [reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, gmem_addr, pred_guard])


def stg128(reg0, reg1, reg2, reg3, gmem_addr, pred_guard):
    return call_primitive_func("cuda_stg16", [reg0, reg1, reg2, reg3, gmem_addr, pred_guard])


def stg64(reg0, reg1, gmem_addr, pred_guard):
    return call_primitive_func("cuda_stg8", [reg0, reg1, gmem_addr, pred_guard])


def stg32(reg0, gmem_addr, pred_guard):
    return call_primitive_func("cuda_stg4", [reg0, gmem_addr, pred_guard])


def stg16(reg0, gmem_addr, pred_guard):
    return call_primitive_func("cuda_stg2", [reg0, gmem_addr, pred_guard])


@initialize()
def register_lds():
    from hidet.ir.dtypes import u8, u16, u32
    from hidet.ir.expr import Var, cast, deref
    from hidet.ir.stmt import AsmStmt
    from hidet.ir.builders import FunctionBuilder
    from hidet.ir.type import void_p
    from hidet.lang import attrs, script, asm  # pylint: disable=import-outside-toplevel

    @script
    def cuda_lds16(reg: void_p, addr: void_p):
        attrs.func_kind = "cuda_internal"
        attrs.func_name = "cuda_lds16"
        template = '{ .reg.u64 u64addr; cvta.to.shared.u64 u64addr, %1; ld.shared.u16 %0, [u64addr]; }'
        asm(template, outputs=[deref(cast(reg, ~u16))], inputs=[addr], is_volatile=True)

    assert isinstance(cuda_lds16, Function)
    register_primitive_function(name=cuda_lds16.name, func_or_type=cuda_lds16)

    load_bytes = 1
    func_name = f"cuda_lds{load_bytes*8}"

    @script
    def cuda_lds8(v0: void_p, addr: void_p):
        attrs.func_kind = 'cuda_internal'
        attrs.func_name = func_name
        v0_u8 = cast(v0, ~u8)
        v0_u8[0] = deref(cast(addr, ~u8))

    register_primitive_function(name=cuda_lds8.name, func_or_type=cuda_lds8)

    for load_bytes in [4, 8, 16]:
        func_name = f"cuda_lds{load_bytes*8}"
        with FunctionBuilder(func_name, kind="cuda_internal") as fb:
            insts = []
            insts.append(".reg.u64 u64addr;")
            nr_outputs = load_bytes // u32.nbytes if load_bytes >= u32.nbytes else 1
            assert nr_outputs <= 4, "Exceeds maximum outputs"
            reg_vars = [Var(f"reg{i}", void_p) for i in range(nr_outputs)]
            addr_var = Var("addr", void_p)
            fb.extend_params(reg_vars + [addr_var])
            insts.append(f"cvta.to.shared.u64 u64addr, %{nr_outputs};")
            vector_size = f".v{nr_outputs}" if nr_outputs > 1 else ""
            dtype = u16 if load_bytes == 2 else u32
            ld_inst = f"ld.shared{vector_size}.{dtype.short_name}"
            dst = [f"%{j}" for j in range(nr_outputs)]
            dst = "{" + ", ".join(dst) + "}"
            insts.append(f"{ld_inst} {dst}, [u64addr];")
            indicator = "h" if load_bytes == 2 else "r"

            doc = "{" + "".join(insts) + "}"
            body = AsmStmt(
                doc,
                outputs=[(f"={indicator}", deref(cast(var, PointerType(dtype)))) for var in reg_vars],
                inputs=[("l", addr_var)],
                is_volatile=True,
            )
            fb.set_body(body)
        register_primitive_function(name=func_name, func_or_type=fb.get())


@initialize()
def register_sts():
    from hidet.ir.dtypes import u8, u16, u32
    from hidet.ir.expr import Var, cast, deref
    from hidet.ir.stmt import AsmStmt
    from hidet.ir.builders import FunctionBuilder
    from hidet.ir.type import void_p
    from hidet.lang import attrs, script, asm  # pylint: disable=import-outside-toplevel

    @script
    def cuda_store(reg: void_p, addr: void_p):
        attrs.func_kind = "cuda_internal"
        attrs.func_name = "cuda_sts16"
        template = '{ .reg.u64 u64addr; cvta.to.shared.u64 u64addr, %1; st.shared.u16 [u64addr], %0; }'
        asm(template, outputs=[], inputs=[deref(cast(reg, ~u16)), addr], is_volatile=True)

    assert isinstance(cuda_store, Function)
    register_primitive_function(name=cuda_store.name, func_or_type=cuda_store)

    for store_bytes in [1, 4, 8, 16]:
        func_name = f"cuda_sts{store_bytes*8}"
        with FunctionBuilder(func_name, kind="cuda_internal") as fb:
            insts = []
            insts.append(".reg.u64 u64addr;")
            nr_inputs = store_bytes // u32.nbytes if store_bytes >= u32.nbytes else 1
            assert nr_inputs <= 4, "Exceeds maximum inputs"
            reg_vars = [Var(f"reg{i}", void_p) for i in range(nr_inputs)]
            addr_var = Var("addr", void_p)
            fb.extend_params(reg_vars + [addr_var])
            insts.append("cvta.to.shared.u64 u64addr, %0;")
            vector_size = f".v{nr_inputs}" if nr_inputs > 1 else ""
            dtype = u8 if store_bytes == 1 else u16 if store_bytes == 2 else u32
            st_inst = f"st.shared{vector_size}.{dtype.short_name}"
            src = [f"%{j + 1}" for j in range(nr_inputs)]
            src = "{" + ", ".join(src) + "}"
            insts.append(f"{st_inst} [u64addr], {src};")
            indicator = "b" if store_bytes == 1 else "h" if store_bytes == 2 else "r"

            doc = "{" + "".join(insts) + "}"
            body = AsmStmt(
                doc,
                outputs=[],
                inputs=[("l", addr_var)] + [(indicator, deref(cast(var, PointerType(dtype)))) for var in reg_vars],
                is_volatile=True,
            )
            fb.set_body(body)
        register_primitive_function(name=func_name, func_or_type=fb.get())


def lds128(reg0, reg1, reg2, reg3, smem_addr):
    return call_primitive_func("cuda_lds128", [reg0, reg1, reg2, reg3, smem_addr])


def lds64(reg0, reg1, smem_addr):
    return call_primitive_func("cuda_lds64", [reg0, reg1, smem_addr])


def lds32(reg0, smem_addr):
    return call_primitive_func("cuda_lds32", [reg0, smem_addr])


def lds16(reg0, smem_addr):
    return call_primitive_func("cuda_lds16", [reg0, smem_addr])


def lds8(reg0, smem_addr):
    return call_primitive_func("cuda_lds8", [reg0, smem_addr])


def sts128(reg0, reg1, reg2, reg3, smem_addr):
    return call_primitive_func("cuda_sts128", [reg0, reg1, reg2, reg3, smem_addr])


def sts64(reg0, reg1, smem_addr):
    return call_primitive_func("cuda_sts64", [reg0, reg1, smem_addr])


def sts32(reg0, smem_addr):
    return call_primitive_func("cuda_sts32", [reg0, smem_addr])


def sts16(reg0, smem_addr):
    return call_primitive_func("cuda_sts16", [reg0, smem_addr])


def sts8(reg0, smem_addr):
    return call_primitive_func("cuda_sts8", [reg0, smem_addr])
