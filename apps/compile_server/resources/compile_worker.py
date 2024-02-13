from typing import Dict, Any, List, Tuple, Sequence, Union
import os
import traceback
import argparse
import sys
import re
import subprocess
import zipfile
import logging
import pickle
import git
from hashlib import sha256
from filelock import FileLock

logger = logging.Logger(__name__)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', type=str, required=True)
    args = parser.parse_args()
    exit(compile_job(args.job_id))


if __name__ == '__main__':
    main()
