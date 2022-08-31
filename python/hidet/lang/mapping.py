from typing import List
from hidet.ir.mapping import TaskMapping
from hidet.ir.mapping import row_repeat as repeat
from hidet.ir.mapping import row_spatial as spatial
from hidet.ir.mapping import auto_map


def chain(*task_mappings) -> TaskMapping:
    assert len(task_mappings) > 0
    composed = task_mappings[0]
    for mapping in task_mappings[1:]:
        composed = composed * mapping
    return composed

