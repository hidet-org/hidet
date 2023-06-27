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

from typing import List, Optional
from datetime import timedelta, datetime
import time
import struct
import os
import atexit
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
    REGULAR_PREFIX = '+'
    DELETE_PREFIX = '-'

    def __init__(self, filename: str, world_size: Optional[int] = -1):
        self._filename = filename
        self._lock_filename = filename + '.lock'
        self._world_size = world_size

        self._lock = filelock.FileLock(self._lock_filename)
        self._cache = {}
        self._timeout = None

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
            k = str(k, encoding='raw_unicode_escape')
            if k.startswith(self.DELETE_PREFIX):
                k = k[len(self.DELETE_PREFIX) :]
                del self._cache[k]
            else:
                self._cache[k] = v

    def _add(self, key: str, amount: int) -> int:
        with self._lock:
            with open(self._filename, "ab+") as f:
                f.seek(0)
                self._update(f)
            value = int(self._cache.get(key, '0')) + amount
            with open(self._filename, "ab+") as f:
                self._write(f, bytes(key, encoding='raw_unicode_escape'))
                self._write(f, bytes(str(value), encoding='raw_unicode_escape'))
        return value

    def _check(self, keys: List[str]):
        with self._lock:
            with open(self._filename, "ab+") as f:
                f.seek(0)
                self._update(f)
        return all((key in self._cache for key in keys))

    def _set(self, key: str, value: bytes):
        with self._lock:
            with open(self._filename, "ab+") as f:
                self._write(f, bytes(key, encoding='raw_unicode_escape'))
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
                f.seek(0)
                file_size = self._file_size(f)
                if key not in self._cache and file_size == last_file_size:
                    # No new entries
                    last_file_size = file_size
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
                f.seek(0)
                self._update(f)
                has_key = key in self._cache
                print(has_key, self._cache[key])
                if (not has_key and expected == b'') or (has_key and self._cache[key] == expected):
                    f.seek(0, 2)
                    self._write(f, bytes(key, encoding='raw_unicode_escape'))
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


if __name__ == '__main__':
    store = FileStore('tmp')
    store.set_timeout(timedelta(seconds=30))
    ret = store.add('baga', 2)
    store.set('yarou', b'haha')
    store.wait(['baga', 'yarou'])
    print(ret)
    ret = store.add('baga', 5)
    print(ret)
    print(store.get('baga'))
