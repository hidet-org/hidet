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
import pytest
import multiprocessing
from multiprocessing import Process, Queue
import os
import time
from datetime import timedelta
import random

from hidet.distributed import FileStore

TMP_PATH = './tmp'


def test_filestore_get_hold():
    if os.path.exists(TMP_PATH):
        os.remove(TMP_PATH)

    def subproc():
        store = FileStore(TMP_PATH)
        store.get('non-existing-key')

    p = Process(target=subproc)
    p.start()
    store = FileStore(TMP_PATH)
    store.set('key', b'value')
    time.sleep(1)
    assert p.is_alive()
    p.terminate()


def test_filestore_set_get():
    if os.path.exists(TMP_PATH):
        os.remove(TMP_PATH)

    def subproc(q):
        store = FileStore(TMP_PATH)
        store.set_timeout(timedelta(seconds=10))
        b = store.get('key')
        q.put(b)

    store = FileStore(TMP_PATH)
    store.set('key', b'u98guj89ks')
    new_value = b'32894728934798'
    store.set('key', new_value)
    q = Queue()
    p = Process(target=subproc, args=(q,))
    p.start()
    ret = q.get()
    assert ret == new_value
    p.join()


def test_filestore_add():
    if os.path.exists(TMP_PATH):
        os.remove(TMP_PATH)

    def subproc():
        store = FileStore(TMP_PATH)
        store.add('cnt', 1)
        store.add('cnt', 2)

    store = FileStore(TMP_PATH)
    store.add('cnt', 1)
    p = Process(target=subproc)
    p.start()
    p.join()
    ret = store.add('cnt', 2)
    assert ret == 6


def test_filestore_del():
    if os.path.exists(TMP_PATH):
        os.remove(TMP_PATH)

    def subproc():
        store = FileStore(TMP_PATH)
        store.get('key')

    p = Process(target=subproc)
    p.start()
    store = FileStore(TMP_PATH)
    store.set('key', b'value')
    store.delete_key('key')
    time.sleep(1)
    assert p.is_alive()
    p.terminate()


def test_filestore_wait():
    if os.path.exists(TMP_PATH):
        os.remove(TMP_PATH)

    def subproc():
        store = FileStore(TMP_PATH)
        store.wait(['key'], timeout=timedelta(seconds=10))

    p = Process(target=subproc)
    p.start()
    store = FileStore(TMP_PATH)
    time.sleep(1)
    assert p.is_alive()
    store.set('key', b'test')
    p.join()
    assert not p.is_alive()


def test_filestore_compare_set():
    if os.path.exists(TMP_PATH):
        os.remove(TMP_PATH)

    def subproc():
        store = FileStore(TMP_PATH)
        store.compare_set("key", b"first", b"second")

    store = FileStore(TMP_PATH)
    store.set("key", b"random")
    p = Process(target=subproc)
    p.start()
    p.join()
    assert store.get("key") == b"random"
    store.set("key", b"first")
    store.compare_set("key", b"first", b"second")
    p = Process(target=subproc)
    p.start()
    p.join()
    assert store.get("key") == b"second"
