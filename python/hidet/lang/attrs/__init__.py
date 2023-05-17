from typing import Optional
from hidet.ir.expr import Var
from hidet.lang.attrs import cuda


"""
general attributes
"""
# The label of the scope
label: Optional[str] = None

"""
function attributes
"""
# The name of the function. The default hidet function name is the name of wrapped python function.
# Please set this attribute if we want to have a different name
func_name: Optional[str] = None

# The kind of this function. Candidates: 'cuda_kernel', 'cuda_device', 'host_kernel', 'packed_func'
func_kind: Optional[str] = None


# If the func_kind == packed_func, then this attribute should be set to the var to function to be packed.
packed_func: Optional[Var] = None
