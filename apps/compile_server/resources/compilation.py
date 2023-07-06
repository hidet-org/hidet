from typing import Dict, Any, List, Tuple
import time
import re
import sys
import os
import traceback
import threading
import requests
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

lock = threading.Lock()
logger = logging.Logger(__name__)

pid = os.getpid()
jobs_dir = os.path.join(os.getcwd(), 'jobs')
repos_dir = os.path.join(os.getcwd(), 'repos')
commits_dir = os.path.join(os.getcwd(), 'commits')
results_dir = os.path.join(os.getcwd(), 'results')

compile_script = os.path.join(os.path.dirname(__file__), 'compile_worker.py')


def _get_github_repo_commit_id(owner: str, repo: str, version: str) -> str:
    try:
        # try branch first
        response = requests.get(
            url="https://api.github.com/repos/{}/{}/branches/{}".format(owner, repo, version),
        )
        if response.status_code == 200:
            return response.json()['commit']['sha']
        # try tag
        response = requests.get(
            url="https://api.github.com/repos/{}/{}/git/ref/tags/{}".format(owner, repo, version),
        )
        if response.status_code == 200:
            return response.json()['object']['sha']
        else:
            print(response.json())

    except requests.exceptions.ConnectionError:
        raise RuntimeError('Can not connect to compiler server') from None

    raise RuntimeError(
        'Failed to get commit id for {}/{}:{}: {}'.format(owner, repo, version, response.json()['message'])
    )

def get_github_repo_commit_id(owner: str, repo: str, version: str) -> str:
    records = getattr(get_github_repo_commit_id, 'records', {})

    key = (owner, repo, version)
    if key in records and (time.time() - records[key][0]) < 10: # cache for 10 seconds
        return records[key][1]
    else:
        commit_id = _get_github_repo_commit_id(owner, repo, version)
        records[key] = (time.time(), commit_id)
        get_github_repo_commit_id.records = records
        return commit_id


def clone_github_repo(owner: str, repo: str, version: str) -> str:
    commit_id: str = get_github_repo_commit_id(owner, repo, version)
    commit_dir = os.path.join(commits_dir, commit_id)
    if os.path.exists(commit_dir):
        return commit_id

    repo_dir = os.path.join(repos_dir, "{}_{}".format(owner, repo))
    os.makedirs(repo_dir, exist_ok=True)
    with FileLock(os.path.join(repos_dir, '{}_{}.lock'.format(owner, repo))):
        if not os.path.exists(os.path.join(repo_dir, '.git')):
            repo = git.Repo.clone_from(
                url="https://github.com/{}/{}.git".format(owner, repo),
                to_path=repo_dir
            )
        else:
            repo = git.Repo(repo_dir)

        repo.remotes.origin.fetch()
        repo.git.fetch('--all')
        repo.git.checkout(version)
        repo.git.pull('origin', version)
        commit_id = repo.head.commit.hexsha
        commit_dir = os.path.join(commits_dir, commit_id)

        if os.path.exists(commit_dir):
            return commit_id
        with FileLock(os.path.join(commits_dir, commit_id + '.lock')):
            repo.git.archive(commit_id, format='zip', output=os.path.join(commits_dir, f'{commit_id}.zip'))
            with zipfile.ZipFile(os.path.join(commits_dir, f'{commit_id}.zip'), 'r') as zip_ref:
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
            job_path = os.path.join(jobs_dir, job_id + '.pickle')
            job_response_path = os.path.join(jobs_dir, job_id + '.response')

            print('[{}] Received a job: {}'.format(pid, job_id[:16]))

            # check if the job is already done
            if os.path.exists(job_response_path):
                print('[{}] Job {} has already done before, respond directly'.format(pid, job_id[:16]))
                with open(job_response_path, 'rb') as f:
                    return pickle.load(f)

            # write the job to the disk
            job_lock = os.path.join(jobs_dir, job_id + '.lock')
            with FileLock(job_lock):
                if not os.path.exists(job_path):
                    with open(job_path, 'wb') as f:
                        pickle.dump(job, f)

            with lock:  # Only one thread can access the following code at the same time
                print('[{}] Start compiling {}'.format(pid, job_id[:16]))
                ret = subprocess.run([sys.executable, compile_script, '--job_id', job_id])

            # respond to the client
            response_path = os.path.join(jobs_dir, job_id + '.response')
            if not os.path.exists(response_path):
                raise RuntimeError('Can not find the response file:\n{}{}'.format(ret.stderr, ret.stdout))
            else:
                print('[{}] Finish compiling {}'.format(pid, job_id[:16]))
                with open(response_path, 'rb') as f:
                    response: Tuple[Dict, int] = pickle.load(f)
                    return response
        except Exception as e:
            msg = traceback.format_exc()
            print('[{}] Failed to compile:\n{}'.format(pid, msg))
            return {'message': '[Remote] {}'.format(msg)}, 500
