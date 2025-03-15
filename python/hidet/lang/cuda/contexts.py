from typing import Sequence
from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, logical_and
from hidet.ir.stmt import Stmt
from hidet.ir.builders import StmtBuilder
from hidet.lang.constructs.context import HidetContext


class WarpGroupContext(HidetContext):
    """
    A context manager for CUDA warp specialization, which allows different warps (groups of 32 threads)
    to perform different roles in producer-consumer patterns.

    Example usage:
    ```python
    with warp_groups([0, 1]) as tid
        # Code for warp groups 0 and 1 (warps 0-7)
        body1(tid)

    with warp_groups([2, 3]) as tid
        # Code for warp groups 2 and 3 (warps 8-15)
        body2(tid)
    ```

    And Hidet will make it semantically equivalent to the following code:
    ```python
    if threadIdx.x // 128 in [0, 1]:
        tid = threadIdx.x % 256
        body1(tid)

    if threadIdx.x // 128 in [2, 3]:
        tid = threadIdx.x % 256
        body2(tid)
    ```

    The warp groups must be contiguous and non-overlapping, and the warp group IDs must be unique.
    """

    def __init__(self, group_ids: Sequence[int]):
        """
        Initialize a warp specialization context.

        Args:
            group_ids: List of warp group IDs (each group = 4 warps = 128 threads) assigned to this role
        """
        assert len(set(group_ids)) == len(group_ids), "Duplicate warp group IDs are not allowed"
        assert len(set(group_ids)) == max(group_ids) - min(group_ids) + 1, "Warp group IDs must be contiguous"

        from hidet.lang.cuda import threadIdx

        min_warpgroup_id: int = min(group_ids)
        max_warpgroup_id: int = max(group_ids)
        num_warpgroups: int = max_warpgroup_id - min_warpgroup_id + 1

        self.condition: Expr = logical_and(
            min_warpgroup_id * 128 <= threadIdx.x, threadIdx.x < (max_warpgroup_id + 1) * 128
        )
        self.tid_value = (threadIdx.x - int32(min_warpgroup_id * 128)) % int32(num_warpgroups * 128)

    def bind_value(self) -> Expr:
        return self.tid_value

    def post_process(self, stmt: Stmt) -> Stmt:
        sb = StmtBuilder()

        with sb.if_then(self.condition):
            sb.append(stmt)

        return sb.finish()


def warp_groups(group_ids: Sequence[int]) -> WarpGroupContext:
    return WarpGroupContext(group_ids)
