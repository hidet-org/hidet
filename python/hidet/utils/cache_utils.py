import os
import shutil
import hidet.option


def hidet_cache_dir(category='./') -> str:
    root = hidet.option.get_cache_dir()
    ret = os.path.abspath(os.path.join(root, category))
    os.makedirs(ret, exist_ok=True)
    return ret


def hidet_cache_file(*items: str) -> str:
    root_dir = hidet_cache_dir('./')
    ret_path = os.path.abspath(os.path.join(root_dir, *items))
    os.makedirs(os.path.dirname(ret_path), exist_ok=True)
    return ret_path


def hidet_clear_op_cache():
    op_cache = hidet_cache_dir('ops')
    print('Clearing operator cache: {}'.format(op_cache))
    shutil.rmtree(op_cache, ignore_errors=True)
