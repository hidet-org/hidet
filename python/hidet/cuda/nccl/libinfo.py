import os

def _get_nccl_dirs():
    import site
    return [os.path.join(root, 'nvidia', 'nccl') for root in site.getsitepackages()]

def get_nccl_include_dirs():
    return [os.path.join(root, 'include') for root in _get_nccl_dirs()]

def get_nccl_library_search_dirs():
    return [os.path.join(root, 'lib') for root in _get_nccl_dirs()]