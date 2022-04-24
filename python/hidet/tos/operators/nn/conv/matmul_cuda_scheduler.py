from hidet.ir.type import scalar_type
from hidet.ir.expr import Var, scalar_var
from hidet.ir.stmt import AssignStmt, BufferStoreStmt
from hidet.ir.func import IRModule
from hidet.ir.ntask import Task, Scheduler
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.task import Grid
from hidet.ir.layout import TaskLayout
from hidet.ir.primitives import thread_idx, block_idx


class MatmulScheduler(Scheduler):
    def schedule(self, task: Task) -> IRModule:
        assert len(task.prologues) == 0 and len(task.epilogues) == 0
        a_shape = task.inputs[0].const_shape()
        b_shape = task.inputs[1].const_shape()
        m_size, n_size, k_size = a_shape[0], b_shape[1], a_shape[1]
        block_dim = 512
        grid_dim = (m_size * n_size + block_dim - 1) // block_dim
        layout = TaskLayout.row_major(task_shape=[m_size, n_size])
        with FunctionBuilder(name=task.name + '.grid', worker=Grid(grid_dim=grid_dim, block_dim=block_dim)) as fb:
            # params
            params = [Var(param.name, param.data_type) for param in task.parameters]
            a, b, c = params
            fb.extend_params(params)

            # local vars
            acc = scalar_var('acc', 'float32')
            fb.extend_local_vars([acc])

            sb = StmtBuilder()
            sb += AssignStmt(acc, 0.0)
            wid = block_idx() * block_dim + thread_idx()
            with sb.if_then(wid < (m_size * n_size)):
                i, j = layout(wid)[0]
                with sb.for_loop('k', k_size) as k:
                    sb += AssignStmt(acc, acc + a[i, k] * b[k, j])
                sb += BufferStoreStmt(c, [i, j], acc)
            fb.set_body(sb.finish())
        func = fb.get()
        return IRModule(funcs={func.name: func}, task=task)

