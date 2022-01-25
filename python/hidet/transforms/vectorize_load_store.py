from typing import List, Type, Optional
from hidet.transforms import Pass
from hidet.ir.stmt import Stmt, SeqStmt, BufferStoreStmt, AssignStmt
from hidet.ir.func import Function
from hidet.ir.functors import StmtRewriter


class Vectorizer:
    @property
    def num_stmts(self) -> int:
        raise NotImplementedError()

    @property
    def stmt_cls(self) -> Type[Stmt]:
        raise NotImplementedError()

    def vectorize(self, seq: List[Stmt]) -> Optional[Stmt]:
        raise NotImplementedError()


class CudaLds128Vectorizer(Vectorizer):
    @property
    def num_stmts(self) -> int:
        return 4

    @property
    def stmt_cls(self) -> Type[Stmt]:
        return AssignStmt

    def vectorize(self, seq: List[AssignStmt]) -> Optional[Stmt]:
        pass


class StmtVectorizer(StmtRewriter):
    def __init__(self):
        super().__init__()
        self.vectorizers: List[Vectorizer] = [
            CudaLds128Vectorizer()
        ]

    def visit_SeqStmt(self, stmt: SeqStmt):
        seq: List[Stmt] = [self(s) for s in stmt.seq]
        pass


class VectorizeLoadStore(Pass):
    def __init__(self):
        super().__init__('vectorize_load_store')

    def process_func(self, func: Function) -> Function:
        pass


def vectorize_load_store_pass():
    return VectorizeLoadStore()
