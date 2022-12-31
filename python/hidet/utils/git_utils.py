# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime
import logging


logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def in_git_repo():
    """
    Check whether git program has been installed and the source code is in a git repository.

    This function must be called before any other git-related functions.

    Returns
    -------
    ret: bool
        True if the source code is in a git repository. Otherwise, False.
    """
    try:
        import git
    except ImportError:
        return False

    try:
        git.Repo(path=__file__, search_parent_directories=True)
        return True
    except git.exc.InvalidGitRepositoryError:
        return False


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


def git_repo_root():
    """
    Get the root directory of the git repository that contains the current source code.

    Returns
    -------
    ret: str
        The root directory of the git repository.
    """
    import git

    repo = git.Repo(path=__file__, search_parent_directories=True)
    return repo.working_dir
