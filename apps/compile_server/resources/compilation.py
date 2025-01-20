from typing import Dict, Any, Tuple
import time
import re
import os
import traceback
import threading
import subprocess
import zipfile
import logging
import pickle
import git
from flask import request
from flask_jwt_extended import jwt_required
from flask_restful import Resource
from hashlib import sha256
from filelock import FileLock

from .compile_worker import CompilationWorkers

lock = threading.Lock()
logger = logging.Logger(__name__)

pid = os.getpid()
JOBS_DIR = os.path.join(os.getcwd(), 'jobs')
REPOS_DIR = os.path.join(os.getcwd(), 'repos')
COMMITS_DIR = os.path.join(os.getcwd(), 'commits')
RESULTS_DIR = os.path.join(os.getcwd(), 'results')

compilation_workers = CompilationWorkers(max_workers=5)


def should_update(repo_timestamp) -> bool:
    if os.path.exists(repo_timestamp):
        with open(repo_timestamp, 'r') as f:
            timestamp = f.read()
        return time.time() - float(timestamp) > 3 * 60  # 3 minutes
    else:
        return True


def clone_github_repo(owner: str, repo: str, version: str) -> str:
    repo_dir = os.path.join(REPOS_DIR, "{}_{}".format(owner, repo))
    repo_timestamp = os.path.join(REPOS_DIR, "{}_{}_timestamp".format(owner, repo))
    os.makedirs(repo_dir, exist_ok=True)
    with FileLock(os.path.join(REPOS_DIR, '{}_{}.lock'.format(owner, repo))):
        if not os.path.exists(os.path.join(repo_dir, '.git')):
            repo = git.Repo.clone_from(
                url="https://github.com/{}/{}.git".format(owner, repo),
                to_path=repo_dir
            )
        else:
            repo = git.Repo(repo_dir)

        # `version` is either a branch name, or 'pull/{n}' if coming from a pull request
        if should_update(repo_timestamp):
            branches = repo.git.branch("--all").split()
            # If local branch already exists, delete it as we prepare to do a new fresh checkout
            # This is because the local branch might be divergent with remote, so we just discard it
            # The exception is the main branch, since it should never diverge
            if version in branches and version != 'main':
                repo.git.checkout('main')
                repo.git.branch('-D', version)
            if 'pull/' in version:
                # Equivalent to `git fetch origin pull/{n}/head:pull/{n}`. Checks out PR#n into branch 'pull/{n}'
                repo.remotes.origin.fetch(version + '/head:' + version)
                repo.git.checkout(version)
                repo.remotes.origin.pull(version + '/head')
            else:
                # Not a PR, just a regular branch
                repo.remotes.origin.fetch(version)
                repo.git.checkout(version)
                repo.remotes.origin.pull(version)
            with open(repo_timestamp, 'w') as f:
                f.write(str(time.time()))
        else:
            repo.git.checkout(version)
        commit_id = repo.head.commit.hexsha

        commit_dir = os.path.join(COMMITS_DIR, commit_id)
        if os.path.exists(commit_dir):
            return commit_id
        with FileLock(os.path.join(COMMITS_DIR, commit_id + '.lock')):
            repo.git.archive(commit_id, format='zip', output=os.path.join(COMMITS_DIR, f'{commit_id}.zip'))
            with zipfile.ZipFile(os.path.join(COMMITS_DIR, f'{commit_id}.zip'), 'r') as zip_ref:
                os.makedirs(commit_dir, exist_ok=True)
                zip_ref.extractall(commit_dir)
            # build the hidet
            os.makedirs(os.path.join(commit_dir, 'build'), exist_ok=True)
            commands = [
                ("cmake ..", "./build"),
                ("make -j1", "./build")
            ]
            for command, cwd in commands:
                subprocess.run(
                    command.split(),
                    cwd=os.path.join(commit_dir, cwd),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )

        return commit_id


def parse_repo_url(url: str) -> Tuple[str, str]:
    patterns = [
        r"https://github.com/([^/]+)/([^/]+)(?:\.git)?",
        r"git://github.com/([^/]+)/([^/]+)\.git",
        r"([^/]+)/([^/]+)"
    ]
    for pattern in patterns:
        match = re.match(pattern, url)
        if match:
            return match.group(1), match.group(2)
    raise ValueError(f'Invalid repository url: {repr(url)}')


class CompilationResource(Resource):
    @jwt_required()  # Requires JWT authentication for this endpoint
    def post(self):
        try:
            # Retrieve the ir modules
            job_data: bytes = request.data

            raw_job: Dict[str, Any] = pickle.loads(job_data)

            # download the repository if needed
            hidet_repo_url = raw_job['hidet_repo_url']
            hidet_repo_version = raw_job['hidet_repo_version']
            owner, repo = parse_repo_url(hidet_repo_url)
            commit_id: str = clone_github_repo(owner, repo, hidet_repo_version)

            workload: bytes = raw_job['workload']

            job = {
                'commit_id': commit_id,
                'workload': workload
            }

            job_id: str = sha256(commit_id.encode() + workload).hexdigest()
            job_path = os.path.join(JOBS_DIR, job_id + '.pickle')
            job_response_path = os.path.join(JOBS_DIR, job_id + '.response')

            print('[{}] Received a job: {}'.format(pid, job_id[:16]))

            # check if the job is already done
            if os.path.exists(job_response_path):
                print('[{}] Job {} has already done before, respond directly'.format(pid, job_id[:16]))
                with open(job_response_path, 'rb') as f:
                    return pickle.load(f)

            # write the job to the disk
            job_lock = os.path.join(JOBS_DIR, job_id + '.lock')
            with FileLock(job_lock):
                if not os.path.exists(job_path):
                    with open(job_path, 'wb') as f:
                        pickle.dump(job, f)

            version_path = os.path.join(COMMITS_DIR, commit_id)
            with lock:  # Only one thread can access the following code at the same time
                print('[{}] Start compiling: {}'.format(pid, job_id[:16]), flush=True)
                start_time = time.time()
                compilation_workers.submit_job(job_id, version_path)
                compilation_workers.wait_all_jobs_finished()
                end_time = time.time()

            # respond to the client
            response_path = os.path.join(JOBS_DIR, job_id + '.response')
            if not os.path.exists(response_path):
                raise RuntimeError('Can not find the response file')
            else:
                print(f'[{pid}] Finish compiling: {job_id[:16]} in {end_time - start_time:.2f}s', flush=True)
                with open(response_path, 'rb') as f:
                    response: Tuple[Dict, int] = pickle.load(f)
                    return response
        except Exception as e:
            msg = traceback.format_exc()
            print('[{}] Failed to compile:\n{}'.format(pid, msg))
            return {'message': '[Remote] {}'.format(msg)}, 500
