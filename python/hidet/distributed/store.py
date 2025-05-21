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

from typing import List, Optional, Dict
from datetime import timedelta, datetime
import time
import struct
import os
import atexit
import signal
import filelock


class Store:
    def set(self, key: str, value: bytes) -> None:
        raise NotImplementedError()

    def get(self, key: str) -> bytes:
        raise NotImplementedError()

    def add(self, key: str, amount: int) -> int:
        raise NotImplementedError()

    def compare_set(self, key: str, expected: bytes, desired: bytes) -> bytes:
        raise NotImplementedError()

    def wait(self, keys: List[str], timeout: Optional[timedelta] = None) -> None:
        raise NotImplementedError()

    def num_keys(self) -> int:
        raise NotImplementedError()

    def delete_key(self, key: str) -> bool:
        raise NotImplementedError()

    def set_timeout(self, timeout: timedelta):
        raise NotImplementedError()


class FileStore(Store):
    """
    A shared KV-store based on the local filesystem.

    It will create a binary file (specified by the filename argument) and a locking file.
    Each time an new entry (key, value) is requested to be inserted, it will be inserted to
    the end of the file. Only the newest is effective among all entries with the same key.
    So when scanning the file from beginning, we can get the up-to-date status of the KV-store.

    All keys requested by public methods will be given a prefix '+' (REGULAR_PREFIX) to be
    distinguished from some internal keys used by the store itself. For example, we have an
    internal entry 'cnt' to maintain how many clients are using this store currently.

    Keys will be converted from Python strings to bytes automatically, while values won't since
    values can be arbitary bytes arrays that might not be decodable. So please do the conversion
    manually if required.

    We use a 4-byte integer to record the length of each (encoded) key and value. So do not insert
    more than 2^31 - 1 bytes for each entry.

    Deletion of an entry is done by adding a new entry with a suffix '-' (DELETE_PREFIX). It will
    overwrite the insertion of the given entry when we scanning the file.
    """

    REGULAR_PREFIX = '+'
    DELETE_PREFIX = '-'

    def __init__(self, filename: str, world_size: Optional[int] = -1):
        self._filename: str = filename
        self._lock_filename: str = filename + '.lock'
        self._world_size: int = world_size

        self._lock: filelock.FileLock = filelock.FileLock(self._lock_filename)
        self._cache: Dict[str, bytes] = {}
        self._timeout: timedelta = None

        num_peers = self._add('cnt', 1)
        if 0 <= world_size < num_peers:
            raise RuntimeError("Warning: more peers than world size.")

        # We cannot operate files in __del__, and we don't want to call close explicitly
        # So we register a atexit function doing cleanup when python interpreter exits
        @atexit.register
        def cleanup():
            with self._lock:
                if os.path.exists(self._filename):
                    rest = self._add('cnt', -1)
                    if rest == 0:
                        os.remove(self._filename)

        old_int = signal.getsignal(signal.SIGINT)
        old_term = signal.getsignal(signal.SIGTERM)

        def signal_handler(signum, frame):
            cleanup()
            if signum == signal.SIGTERM and callable(old_term):
                old_term(signum, frame)
            if signum == signal.SIGINT and callable(old_int):
                old_int(signum, frame)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        # Cleanup the file when the process is killed
        # But if the process is killed by SIGKILL, we cannot do cleanup
        # In this case we have to remove the file manually
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _write(self, f, content):
        f.write(struct.pack('i', len(content)))
        f.write(content)

    def _read(self, f):
        len_str = f.read(4)
        if len_str == b'':
            return None
        l = struct.unpack('i', len_str)[0]
        return f.read(l)

    def _file_size(self, f):
        origin_pos = f.tell()
        f.seek(0, 2)  # 2 means the file's end
        size = f.tell()
        f.seek(origin_pos, 0)
        return size

    def _update(self, f):
        self._cache = {}
        f.seek(0)
        while True:
            k = self._read(f)
            if k is None:
                return
            v = self._read(f)
            k = k.decode()
            if k.startswith(self.DELETE_PREFIX):
                k = k[len(self.DELETE_PREFIX) :]
                del self._cache[k]
            else:
                self._cache[k] = v

    def _add(self, key: str, amount: int) -> int:
        with self._lock:
            with open(self._filename, "ab+") as f:
                self._update(f)
            value = int(self._cache.get(key, '0')) + amount
            with open(self._filename, "ab+") as f:
                self._write(f, bytes(key, encoding='utf-8'))
                self._write(f, bytes(str(value), encoding='utf-8'))
        return value

    def _check(self, keys: List[str]):
        with self._lock:
            with open(self._filename, "ab+") as f:
                self._update(f)
        return all((key in self._cache for key in keys))

    def _set(self, key: str, value: bytes):
        with self._lock:
            with open(self._filename, "ab+") as f:
                self._write(f, bytes(key, encoding='utf-8'))
                self._write(f, value)

    def set(self, key: str, value: bytes) -> None:
        self._set(self.REGULAR_PREFIX + key, value)

    def get(self, key: str) -> bytes:
        last_file_size = None
        key = self.REGULAR_PREFIX + key
        start_t = datetime.now()
        while True:
            self._lock.acquire()
            with open(self._filename, "ab+") as f:
                file_size = self._file_size(f)
                if key not in self._cache and file_size == last_file_size:
                    # No new entries
                    self._lock.release()
                    if self._timeout is not None and datetime.now() - start_t > self._timeout:
                        raise TimeoutError()
                    time.sleep(0.01)
                    continue
                last_file_size = file_size
                self._update(f)
                self._lock.release()
                value = self._cache.get(key)
                if value is not None:
                    return value

    def add(self, key: str, amount: int) -> int:
        return self._add(self.REGULAR_PREFIX + key, amount)

    def compare_set(self, key: str, expected: bytes, desired: bytes) -> bytes:
        key = self.REGULAR_PREFIX + key
        with self._lock:
            with open(self._filename, "ab+") as f:
                self._update(f)
                has_key = key in self._cache
                if (not has_key and expected == b'') or (has_key and self._cache[key] == expected):
                    f.seek(0, 2)
                    self._write(f, bytes(key, encoding='utf-8'))
                    self._write(f, desired)
                    return desired
                elif not has_key:
                    return expected
        return self._cache[key]

    def wait(self, keys: List[str], timeout: Optional[timedelta] = None) -> None:
        timeout = self._timeout if self._timeout is not None else timeout
        start_t = datetime.now()
        keys = [self.REGULAR_PREFIX + key for key in keys]
        while not self._check(keys):
            if timeout is not None and datetime.now() - start_t > timeout:
                raise TimeoutError()
            time.sleep(0.01)

    def num_keys(self):
        with self._lock():
            with open(self._filename, "rb") as f:
                self._update(f)
        return len(self._cache)

    def delete_key(self, key: str):
        self._set(self.DELETE_PREFIX + self.REGULAR_PREFIX + key, b'')

    def set_timeout(self, timeout: timedelta):
        self._timeout = timeout
