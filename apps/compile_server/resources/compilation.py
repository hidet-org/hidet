from typing import Dict, Any, Tuple, Sequence, Union
import time
import re
import sys
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

lock = threading.Lock()
logger = logging.Logger(__name__)

pid = os.getpid()
jobs_dir = os.path.join(os.getcwd(), 'jobs')
repos_dir = os.path.join(os.getcwd(), 'repos')
commits_dir = os.path.join(os.getcwd(), 'commits')
results_dir = os.path.join(os.getcwd(), 'results')


def save_response(response, response_file: str):
    with open(response_file, 'wb') as f:
        pickle.dump(response, f) 


def compile_job(job_id: str):
    try:
        job_file = os.path.join(jobs_dir, job_id + '.pickle')
        if not os.path.exists(job_file):
            # job not found
            return 1

        job_lock = os.path.join(jobs_dir, job_id + '.lock')
        with FileLock(job_lock):
            response_file = os.path.join(jobs_dir, job_id + '.response')
            if os.path.exists(response_file):
                # job already compiled
                return 0

            # unpacking the job
            with open(job_file, 'rb') as f:
                job: Dict[str, Any] = pickle.load(f)

            # import the hidet from the commit
            commit_id: str = job['commit_id']
            commit_dir = os.path.join(commits_dir, commit_id)
            sys.path.insert(0, os.path.join(commit_dir, 'python'))
            import hidet  # import the hidet from the commit

            # load the workload
            workload: Dict[str, Any] = pickle.loads(job['workload'])
            ir_module: Union[hidet.ir.IRModule, Sequence[hidet.ir.IRModule]] = workload['ir_module']
            target: str = workload['target']
            output_kind: str = workload['output_kind']

            # perform the compilation
            module_string = str(ir_module)
            key = module_string + target + output_kind + commit_id
            hash_digest: str = sha256(key.encode()).hexdigest()
            zip_file_path: str = os.path.join(results_dir, hash_digest + '.zip')
            if not os.path.exists(zip_file_path):
                output_dir: str = os.path.join(results_dir, hash_digest)
                with FileLock(os.path.join(results_dir, f'{hash_digest}.lock')):
                    if not os.path.exists(os.path.join(output_dir, 'lib.so')):
                        hidet.drivers.build_ir_module(
                            ir_module,
                            output_dir=output_dir,
                            target=target,
                            output_kind=output_kind
                        )
                    with zipfile.ZipFile(zip_file_path, 'w') as f:
                        for root, dirs, files in os.walk(output_dir):
                            for file in files:
                                f.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_dir))
    except Exception:
        response = {
            'message': '[Remote] ' + traceback.format_exc()
        }, 400
        save_response(response, response_file)
        return 0

    response = {
        'download_filename': hash_digest + '.zip',
    }, 200
    save_response(response, response_file)
    return 0


def should_update(repo_timestamp) -> bool:
    if os.path.exists(repo_timestamp):
        with open(repo_timestamp, 'r') as f:
            timestamp = f.read()
        return time.time() - float(timestamp) > 3 * 60  # 3 minutes
    else:
        return True


def clone_github_repo(owner: str, repo: str, version: str) -> str:
    repo_dir = os.path.join(repos_dir, "{}_{}".format(owner, repo))
    repo_timestamp = os.path.join(repos_dir, "{}_{}_timestamp".format(owner, repo))
    os.makedirs(repo_dir, exist_ok=True)
    with FileLock(os.path.join(repos_dir, '{}_{}.lock'.format(owner, repo))):
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

            print('[{}] Received a job: {}'.format(pid, job_id[:16]), flush=True)

            # check if the job is already done
            if os.path.exists(job_response_path):
                print('[{}] Job {} has already done before, respond directly'.format(pid, job_id[:16]), flush=True)
                with open(job_response_path, 'rb') as f:
                    return pickle.load(f)

            # write the job to the disk
            job_lock = os.path.join(jobs_dir, job_id + '.lock')
            with FileLock(job_lock):
                if not os.path.exists(job_path):
                    with open(job_path, 'wb') as f:
                        pickle.dump(job, f)

            with lock:  # Only one thread can access the following code at the same time
                print('[{}] Start compiling: {}'.format(pid, job_id[:16]), flush=True)
                compile_job(job_id)

            # respond to the client
            response_path = os.path.join(jobs_dir, job_id + '.response')
            if not os.path.exists(response_path):
                raise RuntimeError('Can not find the response file')
            else:
                print('[{}] Finish compiling: {}'.format(pid, job_id[:16]), flush=True)
                with open(response_path, 'rb') as f:
                    response: Tuple[Dict, int] = pickle.load(f)
                    return response
        except Exception as e:
            msg = traceback.format_exc()
            print('[{}] Failed to compile:\n{}'.format(pid, msg), flush=True)
            return {'message': '[Remote] {}'.format(msg)}, 500
