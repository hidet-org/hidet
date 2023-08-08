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
import zipfile
import shutil
import tempfile
import os
import pickle
import requests

import hidet.utils.net_utils
from hidet.ir.module import IRModule
from .core import api_url, access_token


def remote_build(ir_module: IRModule, output_dir: str, *, target: str, output_kind: str = '.so'):
    # upload the IRModule
    if 'cuda' in target and 'arch' not in target:
        arch = hidet.option.cuda.get_arch()
        target = '{} --arch={}'.format(target, arch)
    job_data = pickle.dumps(
        {
            'workload': pickle.dumps({'ir_module': ir_module, 'target': target, 'output_kind': output_kind}),
            'hidet_repo_url': hidet.option.get_option('compile_server.repo_url'),
            'hidet_repo_version': hidet.option.get_option('compile_server.repo_version'),
        }
    )
    response = requests.post(api_url('compile'), data=job_data, headers={'Authorization': f'Bearer {access_token()}'})
    if response.status_code != 200:
        msg = response.json()['message']
        raise RuntimeError('Failed to remotely compile an IRModule: \n{}'.format(msg))

    # download the compiled module
    with tempfile.TemporaryDirectory() as tmp_dir:
        filename = response.json()['download_filename']
        download_url = api_url(f'download/{filename}')
        save_path = os.path.join(tmp_dir, 'download.zip')
        hidet.utils.net_utils.download_url_to_file(
            download_url, save_path, progress=False, headers={'Authorization': f'Bearer {access_token()}'}
        )

        # extract the downloaded zip file to the output directory
        extract_dir = os.path.join(tmp_dir, 'extract')
        with zipfile.ZipFile(save_path) as f:
            f.extractall(extract_dir)

        # copy the extracted files to the output directory
        shutil.copytree(extract_dir, output_dir, dirs_exist_ok=True)
