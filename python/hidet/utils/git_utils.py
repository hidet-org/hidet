import os
import git
import functools
import datetime


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


def get_repo_commit_date(strftime='%Y-%m-%d-%H-%M') -> str:
    """
    Get the commit date time of current commit (i.e., HEAD).

    Parameters
    ----------
    strftime: str, default '%Y-%m-%d-%H-%M'
        The format of the date time. The default format will return a date time like '2023-03-24-13-46'.

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


def hidet_cache_dir(category='root') -> str:
    root = os.path.join(repo_root(), '.hidet_cache')
    if category == 'root':
        ret = root
    else:
        ret = os.path.join(root, category)
    os.makedirs(ret, exist_ok=True)
    return ret


if __name__ == '__main__':
    print(repo_root())

