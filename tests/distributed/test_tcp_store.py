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
import random
from datetime import timedelta
from hidet.distributed import TCPStore


def client_process(port, unique_id, expected_value, result_queue):
    try:
        store = TCPStore('localhost', port)
        store.wait(['unique_id'])
        value = store.get('unique_id')
        result_queue.put((unique_id, value == expected_value))
    except Exception as e:
        result_queue.put((unique_id, False, str(e)))


@pytest.fixture
def server_port():
    return random.randint(29500, 30000)


@pytest.fixture
def server_store(server_port):
    store = TCPStore('localhost', server_port, is_server=True)
    yield store
    store.shutdown()


def test_distributed_key_value(server_port, server_store):
    num_clients = 7
    expected_value = str(random.randint(1000, 9999)).encode()
    result_queue = multiprocessing.Queue()
    server_store.set('unique_id', expected_value)

    processes = []
    for i in range(num_clients):
        p = multiprocessing.Process(target=client_process, args=(server_port, i, expected_value, result_queue))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    assert len(results) == num_clients, "Not all clients reported results"

    for client_id, success in results:
        assert success, f"Client {client_id} failed to get correct value"


def test_store_timeout(server_port, server_store):
    """Test that store operations timeout properly"""
    store = TCPStore('localhost', server_port)
    store.set_timeout(timedelta(seconds=1))

    with pytest.raises(TimeoutError):
        store.get('non_existent_key')


def test_store_add(server_port, server_store):
    """Test atomic add operation"""
    store = TCPStore('localhost', server_port)

    assert store.add('counter', 5) == 5
    assert store.add('counter', 3) == 8

    assert store.add('counter', -2) == 6


def test_store_check(server_port, server_store):
    """Test check operation"""
    store = TCPStore('localhost', server_port)

    store.set('key1', b'value1')
    store.set('key2', b'value2')

    assert store.check(['key1', 'key2'])

    assert not store.check(['non_existent'])

    assert not store.check(['key1', 'non_existent'])
