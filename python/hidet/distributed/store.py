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
import socket
import threading
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


class TCPStore(Store):
    """
    A TCP-based distributed key-value store implementation.
    """

    # client message types
    VALIDATE = 0
    SET = 1
    GET = 2
    ADD = 3
    CHECK = 4
    # server response types
    READY = 0
    NOT_READY = 1

    def __init__(
        self,
        host_name: str,
        port: int,
        world_size: Optional[int] = None,
        is_server: bool = False,
        timeout: timedelta = timedelta(seconds=20),
    ):
        super().__init__()
        self.host = host_name
        self.port = port
        self.world_size = world_size
        self.is_server = is_server
        self.timeout = timeout

        self._lock = threading.Lock()
        self._store: Dict[str, bytes] = {}
        self._server_socket = None
        self._client_socket = None

        self._shutdown = threading.Event()
        self._server_thread = None
        self._client_threads = []
        self._client_threads_lock = threading.Lock()

        if is_server:
            self._start_server()
        else:
            self._connect_to_server()

        num_peers = self.add('worker_count', 1)
        if self.world_size is not None and num_peers > self.world_size:
            raise RuntimeError("Warning: more peers than world size.")

        old_int = signal.getsignal(signal.SIGINT)
        old_term = signal.getsignal(signal.SIGTERM)

        def signal_handler(signum, frame):
            self.shutdown()
            if signum == signal.SIGTERM and callable(old_term):
                old_term(signum, frame)
            if signum == signal.SIGINT and callable(old_int):
                old_int(signum, frame)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _send_message(self, msg_type: int, data: bytes = b''):
        header = struct.pack('!B', msg_type)
        self._client_socket.sendall(header + data)

    def _receive_message(self) -> bytes:
        try:
            return self._client_socket.recv(4096)
        except socket.timeout:
            return b''

    def _validate_connection(self):
        self._send_message(self.VALIDATE, struct.pack('!I', 0x3C85F7CE))
        response = self._receive_message()
        if not response or response != b'\x01':
            raise RuntimeError("Failed to validate connection")

    def _start_server(self):
        # create a server socket
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(5)
        self.port = self._server_socket.getsockname()[1]
        # Start server thread
        self._server_thread = threading.Thread(target=self._server_loop)
        self._server_thread.daemon = True
        self._server_thread.start()

    def _connect_to_server(self):
        for _ in range(5):
            try:
                self._client_socket = socket.create_connection((self.host, self.port), timeout=2)
                break
            except ConnectionRefusedError:
                time.sleep(0.5)
        else:
            raise RuntimeError("Could not connect to server after retries")
        self._validate_connection()

    def check(self, keys: List[str]) -> bool:
        if self.is_server:
            with self._lock:
                return all(key in self._store for key in keys)
        else:
            self._send_message(
                self.CHECK,
                struct.pack('!I', len(keys)) + b''.join(struct.pack('!I', len(key)) + key.encode() for key in keys),
            )
            response = self._receive_message()
            return struct.unpack('!B', response)[0] == self.READY

    def wait(self, keys: List[str], timeout: Optional[timedelta] = None) -> None:
        timeout = timeout or self.timeout
        end_time = datetime.now() + timeout

        while datetime.now() < end_time:
            if all(self.check([key]) for key in keys):
                return
            time.sleep(0.01)
        raise TimeoutError(f"Timeout waiting for keys: {keys}")

    def set(self, key: str, value: bytes) -> None:
        if self.is_server:
            with self._lock:
                self._store[key] = value
        else:
            self._send_message(self.SET, struct.pack('!I', len(key)) + key.encode() + value)
            self._receive_message()

    def get(self, key: str, wait: bool = True) -> bytes:
        if self.is_server:
            with self._lock:
                return self._store.get(key, b'')
        else:
            if wait:
                self.wait([key])
            self._send_message(self.GET, struct.pack('!I', len(key)) + key.encode())
            response_len = struct.unpack('!I', self._client_socket.recv(4))[0]
            return self._client_socket.recv(response_len)

    def add(self, key: str, amount: int) -> int:
        if self.is_server:
            with self._lock:
                current = int(self._store.get(key, b'0'))
                new_value = current + amount
                self._store[key] = str(new_value).encode()
                return new_value
        else:
            self._send_message(self.ADD, struct.pack('!I', len(key)) + key.encode() + struct.pack('!q', amount))
            return struct.unpack('!q', self._receive_message())[0]

    def set_timeout(self, timeout: timedelta):
        self.timeout = timeout

    def _server_loop(self):
        while not self._shutdown.is_set():
            try:
                self._server_socket.settimeout(1.0)
                client, _ = self._server_socket.accept()

                client_thread = threading.Thread(target=self._handle_client, args=(client,))
                client_thread.daemon = True
                with self._client_threads_lock:
                    self._client_threads.append(client_thread)
                client_thread.start()

                with self._client_threads_lock:
                    self._client_threads = [t for t in self._client_threads if t.is_alive()]
            except socket.timeout:
                continue
            except Exception as e:  # pylint: disable=broad-except
                if not self._shutdown.is_set():
                    raise RuntimeError(f"Error in TCP server loop: {e} pid: {os.getpid()}") from e
                break

    def _handle_client(self, client: socket.socket):
        try:
            while not self._shutdown.is_set():
                client.settimeout(1)
                try:
                    msg_type_bytes = client.recv(1)
                    if not msg_type_bytes:
                        break
                    msg_type = struct.unpack('!B', msg_type_bytes)[0]
                    if msg_type == self.VALIDATE:
                        magic_number = struct.unpack('!I', client.recv(4))[0]
                        if magic_number == 0x3C85F7CE:
                            client.sendall(b'\x01')
                        else:
                            client.sendall(b'\x00')
                    elif msg_type == self.SET:
                        key_len = struct.unpack('!I', client.recv(4))[0]
                        key = client.recv(key_len).decode()
                        value = client.recv(4096)
                        self.set(key, value)
                        client.sendall(b'\x01')
                    elif msg_type == self.GET:
                        key_len = struct.unpack('!I', client.recv(4))[0]
                        key = client.recv(key_len).decode()
                        value = self.get(key)
                        client.sendall(struct.pack('!I', len(value)))
                        client.sendall(value)
                    elif msg_type == self.ADD:
                        key_len = struct.unpack('!I', client.recv(4))[0]
                        key = client.recv(key_len).decode()
                        amount = struct.unpack('!q', client.recv(8))[0]
                        result = self.add(key, amount)
                        client.sendall(struct.pack('!q', result))
                    elif msg_type == self.CHECK:
                        key_count = struct.unpack('!I', client.recv(4))[0]
                        keys = []
                        for _ in range(key_count):
                            key_len = struct.unpack('!I', client.recv(4))[0]
                            key = client.recv(key_len).decode()
                            keys.append(key)
                        client.sendall(struct.pack('!B', self.READY if self.check(keys) else self.NOT_READY))
                except socket.timeout:
                    continue
                except Exception as e:  # pylint: disable=broad-except
                    if not self._shutdown.is_set():
                        raise RuntimeError(
                            f"Error handling client request: {e} pid: {os.getpid()} msg_type: {msg_type}"
                        ) from e
                    break
        finally:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            client.close()

    def shutdown(self):
        self._shutdown.set()
        if self.is_server:
            with self._client_threads_lock:
                for thread in self._client_threads:
                    if thread.is_alive():
                        thread.join(timeout=1.0)
            if self._server_thread:
                self._server_thread.join()
            if self._server_socket:
                try:
                    self._server_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                finally:
                    self._server_socket.close()
                    self._server_socket = None
        else:
            if self._client_socket:
                try:
                    self._client_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                finally:
                    self._client_socket.close()
                    self._client_socket = None

    def __del__(self):
        self.shutdown()
