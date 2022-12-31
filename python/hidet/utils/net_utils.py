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
from typing import Optional
import os
import sys
import shutil
import tempfile
import urllib.parse
import urllib.request
from tqdm import tqdm

import hidet


def download(url: str, file_name: Optional[str] = None, progress: bool = True) -> str:
    if file_name is None:
        parts = urllib.parse.urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.join(hidet.utils.hidet_cache_dir(), file_name)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, cached_file, progress=progress)
    return cached_file


def download_url_to_file(url, dst, progress=True):
    # modified based on PyTorch
    file_size = None
    req = urllib.request.Request(url, headers={"User-Agent": ""})
    with urllib.request.urlopen(req) as u:
        meta = u.info()
        if hasattr(meta, 'getheaders'):
            content_length = meta.getheaders("Content-Length")
        else:
            content_length = meta.get_all("Content-Length")
        if content_length is not None and len(content_length) > 0:
            file_size = int(content_length[0])

        dst = os.path.expanduser(dst)
        dst_dir = os.path.dirname(dst)
        os.makedirs(dst_dir, exist_ok=True)
        f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)  # pylint: disable=consider-using-with

        try:
            with tqdm(total=file_size, disable=not progress, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                while True:
                    buffer = u.read(8192)
                    if len(buffer) == 0:
                        break
                    f.write(buffer)
                    pbar.update(len(buffer))

            f.close()
            shutil.move(f.name, dst)
        finally:
            f.close()
            if os.path.exists(f.name):
                os.remove(f.name)
