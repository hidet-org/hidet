from typing import List, Type, Optional
import contextlib
from hidet.transforms import Pass
from hidet.ir.type import TensorType
from hidet.ir.expr import Expr, Var, TensorElement, Address
from hidet.ir.stmt import Stmt, SeqStmt, BufferStoreStmt, AssignStmt, EvaluateStmt
from hidet.ir.func import Function
from hidet.ir.functors import StmtRewriter, equal, same_list
from hidet.ir.primitives import lds128, sts128


class Vectorizer:
    @property
    def num_stmts(self) -> int:
        raise NotImplementedError()

    @property
    def stmt_cls(self) -> Type[Stmt]:
        raise NotImplementedError()

    def vectorize(self, seq: List[Stmt]) -> Optional[Stmt]:
        raise NotImplementedError()

    @staticmethod
    def is_greater_one(lhs: Expr, rhs: Expr) -> bool:
        return equal(lhs + 1, rhs)

    @staticmethod
    def is_contiguous(seq: List[Expr]) -> bool:
        for i in range(len(seq) - 1):
            if not Vectorizer.is_greater_one(seq[i], seq[i + 1]):
                return False
        return True


class CudaLds128Vectorizer(Vectorizer):
    @property
    def num_stmts(self) -> int:
        return 4

    @property
    def stmt_cls(self) -> Type[Stmt]:
        return BufferStoreStmt

    def vectorize(self, seq: List[BufferStoreStmt]) -> Optional[Stmt]:
        with contextlib.suppress(AssertionError):
            assert len(seq) == 4
            assert all(isinstance(s, BufferStoreStmt) for s in seq)
            assert all(isinstance(s.value, TensorElement) for s in seq)
            dst_vars: List[Var] = [s.buf for s in seq]
            src_vars: List[Var] = [s.value.base for s in seq]
            assert all(isinstance(v.type, TensorType) and v.type.scope.name == 'shared' for v in src_vars)
            assert all(isinstance(v.type, TensorType) and v.type.scope.name == 'register' for v in dst_vars)
            assert all(len(s.value.indices) == 1 for s in seq)
            shared_indices = [s.value.indices[0] for s in seq]
            assert self.is_contiguous(shared_indices)
            regs = [TensorElement(s.buf, s.indices) for s in seq]
            smem_addr = Address(seq[0].value)
            return EvaluateStmt(lds128(regs[0], regs[1], regs[2], regs[3], smem_addr))
        return None


class CudaSts128Vectorizer(Vectorizer):
    @property
    def num_stmts(self) -> int:
        return 4

    @property
    def stmt_cls(self) -> Type[Stmt]:
        return BufferStoreStmt

    def vectorize(self, seq: List[BufferStoreStmt]) -> Optional[Stmt]:
        with contextlib.suppress(AssertionError):
            assert len(seq) == 4
            assert all(isinstance(s, BufferStoreStmt) for s in seq)
            assert all(isinstance(s.value, TensorElement) for s in seq)
            dst_vars: List[Var] = [s.buf for s in seq]
            src_vars: List[Var] = [s.value.base for s in seq]
            assert all(isinstance(v.type, TensorType) and v.type.scope.name == 'register' for v in src_vars)
            assert all(isinstance(v.type, TensorType) and v.type.scope.name == 'shared' for v in dst_vars)
            assert all(len(s.indices) == 1 for s in seq)
            shared_indices = [s.indices[0] for s in seq]
            assert self.is_contiguous(shared_indices)
            regs = [s.value for s in seq]
            smem_addr = Address(TensorElement(seq[0].buf, seq[0].indices))
            return EvaluateStmt(sts128(regs[0], regs[1], regs[2], regs[3], smem_addr))
        return None


class StmtVectorizer(StmtRewriter):
    def __init__(self):
        super().__init__()
        self.vectorizers: List[Vectorizer] = [
            CudaLds128Vectorizer(),
            CudaSts128Vectorizer()
        ]

    def visit_SeqStmt(self, stmt: SeqStmt):
        seq: List[Stmt] = [self(s) for s in stmt.seq]
        new_seq = []
        n = len(seq)
        i = 0
        while i < n:
            success = False
            for vectorizer in self.vectorizers:
                cls = vectorizer.stmt_cls
                m = vectorizer.num_stmts
                if i + m - 1 >= n:
                    continue
                if not all(isinstance(s, cls) for s in seq[i: i+m]):
                    continue
                new_stmt = vectorizer.vectorize(seq[i: i+m])
                if new_stmt:
                    new_seq.append(new_stmt)
                    i += m
                    success = True
                    break
            if not success:
                new_seq.append(seq[i])
                i += 1
        if same_list(new_seq, stmt.seq):
            return stmt
        else:
            return SeqStmt(new_seq)


class VectorizeLoadStorePass(Pass):
    def process_func(self, func: Function) -> Function:
        vectorizer = StmtVectorizer()
        body = vectorizer(func.body)
        if body is func.body:
            return func
        else:
            return Function(func.name, func.params, body, func.ret_type,
                            local_const_vars=func.local_const_vars, extern_vars=func.extern_vars, attrs=func.attrs)


def vectorize_load_store_pass():
    return VectorizeLoadStorePass()
