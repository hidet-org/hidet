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
import os
from filelock import FileLock


class FolderLock(FileLock):
    """
    A context manager for file-based locking using flock.
    Ensures that only one process can acquire the lock at a time.

    Parameters
    ----------
    lock_file_path : str
        Path to the lock file.
    """

    def __init__(self, lock_dir):
        self.lock_file_path = os.path.join(lock_dir, ".lock")
        super().__init__(self.lock_file_path)
