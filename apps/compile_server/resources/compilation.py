from typing import Dict, Any, List
import os
import traceback
import threading
import zipfile
import logging
import pickle
from flask import request
from flask_jwt_extended import jwt_required
from flask_restful import Resource
import hidet
from hidet.ir.module import IRModule
from hashlib import sha256
import filelock

lock = threading.Lock()
logger = logging.Logger(__name__)

class CompilationResource(Resource):
    @jwt_required()  # Requires JWT authentication for this endpoint
    def post(self):
        # Retrieve the ir modules
        with lock:
            logger.info('Retrieved a job...')
            try:
                workload: Dict[str, Any] = pickle.loads(request.data)
                ir_module: IRModule = workload['ir_module']
                target: str = workload['target']
                output_kind: str = workload['output_kind']
            except Exception:
                return {
                    'message': '[Remote] ' + traceback.format_exc()
                }, 400

            # Perform the compilation
            module_string = str(ir_module)
            key = module_string + target + output_kind
            hash_digest: str = sha256(key.encode()).hexdigest()
            zip_file_path: str = hidet.utils.cache_file('compilefarm', hash_digest + '.zip')
            logger.info('Compiling the job [{}] {}...'.format(target, module_string[:32]))
            try:
                if not os.path.exists(zip_file_path):
                    output_dir: str = hidet.utils.cache_dir('compilefarm', hash_digest)
                    with filelock.FileLock(os.path.join(output_dir, 'lock')):
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
                logger.warning("Failed to compile a job: [{}] {}".format(target, module_string[:32]))
                return {
                    'message': '[Remote] ' + traceback.format_exc()
                }, 500

            # generate a download url
            logger.info('Successfully complete job: [{}] {}!'.format(target, module_string[:32]))
            return {
                'download_filename': hash_digest + '.zip',
            }, 200
