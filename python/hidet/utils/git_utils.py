from typing import List
import os
import git
import functools
import datetime
import logging


logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def get_repo_sha(short=False):
    """
    Get the current commit (i.e., HEAD) sha hash.

    Parameters
    ----------
    short: bool, default False
        Whether get a short version of hash.

    Returns
    -------
    ret: str
        The commit sha hash.
    """
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
    repo = git.Repo(search_parent_directories=True)
    return repo.working_dir


_hidet_cache_root_dir = os.path.join(repo_root(), '.hidet_cache')
os.makedirs(_hidet_cache_root_dir, exist_ok=True)


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


if __name__ == '__main__':
    print(repo_root())

