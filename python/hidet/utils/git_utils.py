import datetime
import logging
import git


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


def in_git_repo():
    """
    Check whether the source code is in a git repository.

    Returns
    -------
    ret: bool
        True if the source code is in a git repository. Otherwise, False.
    """
    try:
        git.Repo(path=__file__, search_parent_directories=True)
        return True
    except git.exc.InvalidGitRepositoryError:
        return False


def git_repo_root():
    """
    Get the root directory of the git repository that contains the current source code.

    Returns
    -------
    ret: str
        The root directory of the git repository.
    """
    repo = git.Repo(path=__file__, search_parent_directories=True)
    return repo.working_dir
