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
import shutil
from pathlib import Path


def _copy_tree_ignore_existing(src, dst):
    src_path = Path(src)
    dst_path = Path(dst)

    # Ensure the destination directory exists
    dst_path.mkdir(parents=True, exist_ok=True)

    for item in src_path.iterdir():
        dst_item = dst_path / item.name
        if item.is_dir():
            # Recursively copy subdirectories only if they contain new content
            _copy_tree_ignore_existing(item, dst_item)
        else:
            # Copy files only if they don't already exist
            if not dst_item.exists():
                shutil.copy2(item, dst_item)


def copy_tree_ignore_existing(src_list, dst_list):
    assert len(src_list) == len(dst_list)
    for src, dst in zip(src_list, dst_list):
        _copy_tree_ignore_existing(src, dst)
