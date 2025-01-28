from typing import Dict, Any, Sequence, Union
import os
import traceback
import sys
import zipfile
import logging
import pickle
from hashlib import sha256
from filelock import FileLock
import multiprocessing
import gc

logger = logging.Logger(__name__)

JOBS_DIR = os.path.join(os.getcwd(), 'jobs')
REPOS_DIR = os.path.join(os.getcwd(), 'repos')
COMMITS_DIR = os.path.join(os.getcwd(), 'commits')
RUSULTS_DIR = os.path.join(os.getcwd(), 'results')


def save_response(response, response_file: str):
    with open(response_file, 'wb') as f:
        pickle.dump(response, f)


def compile_job(job_id: str):
    try:
        job_file = os.path.join(JOBS_DIR, job_id + '.pickle')
        if not os.path.exists(job_file):
            # job not found
            return 1

        job_lock = os.path.join(JOBS_DIR, job_id + '.lock')
        with FileLock(job_lock):
            response_file = os.path.join(JOBS_DIR, job_id + '.response')
            if os.path.exists(response_file):
                # job already compiled
                return 0

            # unpacking the job
            with open(job_file, 'rb') as f:
                job: Dict[str, Any] = pickle.load(f)

            # import the hidet from the commit
            commit_id: str = job['commit_id']
            import hidet
             # load the workload
            workload: Dict[str, Any] = pickle.loads(job['workload'])
            ir_module: Union[hidet.ir.IRModule, Sequence[hidet.ir.IRModule]] = workload['ir_module']
            target: str = workload['target']
            output_kind: str = workload['output_kind']

            # perform the compilation
            module_string = str(ir_module)
            key = module_string + target + output_kind + commit_id
            hash_digest: str = sha256(key.encode()).hexdigest()
            zip_file_path: str = os.path.join(RUSULTS_DIR, hash_digest + '.zip')
            if not os.path.exists(zip_file_path):
                output_dir: str = os.path.join(RUSULTS_DIR, hash_digest)
                with FileLock(os.path.join(RUSULTS_DIR, f'{hash_digest}.lock')):
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


# Worker process function to handle compilation jobs using a specific version of the 'hidet' module.
def worker_process(version, job_queue, result_queue, parent_pid):
    sys.path.insert(0, os.path.join(version, 'python'))  # Ensure the version path is first in sys.path
    print(f"[{parent_pid}] Worker loaded hidet version from {version}", flush=True)

    while True:
        job = job_queue.get()
        if job == "STOP":
            print(f"[{parent_pid}] Shutting down worker for version: {version}", flush=True)
            break

        # Compile
        job_id = job
        print(f"[{parent_pid}] Worker processing job {job_id[:16]} with hidet version {version}", flush=True)
        compile_job(job_id)
        result_queue.put((job_id, 'DONE'))
        gc.collect()   # Collect garbage to free memory


class CompilationWorkers:
    """
    A class to manage a pool of compilation workers.
    It is needed to avoid the overhead of loading the hidet module for every job.
    Every worker processes a compilation with a fixed version of hidet (fixed commit hash).
    One worker per version.
    Only one worker is compiling at the same time. No concurrent compilation. 
    Concurrency compilation is processed on upper level.
    """
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.workers = {} # {version_path: (worker_process, job_queue)}
        self.result_queue = multiprocessing.Queue()

    def _get_or_create_worker(self, version_path):
        # If a worker for the version exists, return it
        if version_path in self.workers:
            return self.workers[version_path]

        # If the worker pool is full, remove the oldest worker
        if len(self.workers) >= self.max_workers:
            _, (worker, job_queue) = self.workers.popitem()
            job_queue.put("STOP")  # Send shutdown signal to the removing worker
            worker.join()  # Wait for it to exit

        # Create a new worker for the version
        job_queue = multiprocessing.Queue()
        worker = multiprocessing.Process(target=worker_process, 
                                         args=(version_path, job_queue, self.result_queue, os.getpid())
                                        )
        worker.start()
        self.workers[version_path] = (worker, job_queue)
        return self.workers[version_path]

    
    def run_and_wait_job(self, job_id, version_path):
        # Run the job and wait until it is finished
        _, job_queue = self._get_or_create_worker(version_path)
        job_queue.put(job_id)
        self.result_queue.get()   # multiprocessing.Queue.get() waits until a new item is available

    def shutdown(self):
        for _, (worker, job_queue) in self.workers.items():
            job_queue.put("STOP")
            worker.join()
        print(f"[{os.getpgid}] All compilation workers are shuted down.", flush=True)
