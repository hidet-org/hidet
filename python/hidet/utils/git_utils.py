import os
import os.path
import functools
import datetime
import logging
import shutil


logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def get_repo_sha(short=False):
    """
    Get the current commit (i.e., HEAD) sha hash.

    Parameters
    ----------
    short: bool, default False
        Whether to get a short version of hash.

    Returns
    -------
    ret: str
        The commit sha hash.
    """
    import git

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    if short:
        return sha[:7]
    else:
        return sha


def get_repo_commit_date(strftime='%Y-%m-%d') -> str:
    """
    Get the commit date time of current commit (i.e., HEAD).

    Parameters
    ----------
    strftime: str, default '%Y-%m-%d'
        The format of the date time. The default format will return a date time like '2023-03-24'.
        Others: %H: hour, %M: minutes

    Returns
    -------
    ret: str
        The commit date time in given format.
    """
    import git

    repo = git.Repo(search_parent_directories=True)
    commit = repo.head
    committed_date = commit.commit.committed_date
    dt = datetime.datetime.fromtimestamp(committed_date)
    return str(dt.strftime(strftime))


@functools.lru_cache(maxsize=1)
def repo_root() -> str:
    """
    Get the root directory of current git repository.

    Returns
    -------
    ret: str
        The root directory.
    """
    hidet_cache = os.path.expanduser('~/.cache/hidet')
    os.makedirs(hidet_cache, exist_ok=True)
    try:
        import git
    except ImportError:
        return hidet_cache
    else:
        try:
            repo = git.Repo(search_parent_directories=True)
            return repo.working_dir
        except git.InvalidGitRepositoryError:
            return hidet_cache


_hidet_cache_root_dir = os.path.join(repo_root(), '.hidet_cache')
os.makedirs(_hidet_cache_root_dir, exist_ok=True)


class CacheDir:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.old_cache_dir = None
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if not os.path.isdir(cache_dir):
            raise ValueError('{} is not a directory.'.format(repr(cache_dir)))

    def __enter__(self):
        self.old_cache_dir = hidet_cache_dir()
        hidet_set_cache_root(self.cache_dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        hidet_set_cache_root(self.old_cache_dir)
        self.old_cache_dir = None


def hidet_set_cache_root(root_dir: str):
    global _hidet_cache_root_dir
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if not os.path.isdir(root_dir):
        raise ValueError('Expect {} to be a directory.'.format(root_dir))
    _hidet_cache_root_dir = root_dir
    logger.info('Hidet cache root dir: {}'.format(root_dir))


def hidet_cache_dir(category='./') -> str:
    root = _hidet_cache_root_dir
    if category == './':
        ret = root
    else:
        ret = os.path.join(root, category)
    os.makedirs(ret, exist_ok=True)
    return ret


def hidet_cache_file(*items: str) -> str:
    root_dir = hidet_cache_dir('./')
    ret_path = os.path.join(root_dir, *items)
    os.makedirs(os.path.dirname(ret_path), exist_ok=True)
    return ret_path


def hidet_clear_op_cache():
    op_cache = hidet_cache_dir('ops')
    print('Clearing operator cache in {}'.format(op_cache))
    shutil.rmtree(op_cache, ignore_errors=True)
