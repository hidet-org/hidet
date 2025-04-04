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
import logging
from typing import Optional, Tuple, Dict
from hashlib import sha256
from hidet.graph.flow_graph import FlowGraph
from hidet.option import get_cache_dir
from hidet.utils.counters import counters
from hidet.runtime.compiled_graph import CompiledGraph, load_compiled_graph, save_compiled_graph
from hidet.utils.cache_utils import clear_cache_dir

logger = logging.getLogger(__name__)

FLOW_GRAPH_CACHE_DIR_NAME = "flowgraph"
COMPILED_GRAPH_CACHE_DIR_NAME = "graphs"
COMPILED_GRAPH_SAVE_FILE_NAME = "compiled_graph.hidet"


def compute_flow_graph_hash(fg: FlowGraph, kwargs) -> str:
    """
    Generate a unique hash of the FlowGraph for caching.
    """
    kwargs_str = ''.join(f'{k}={v}' for k, v in sorted(kwargs.items()))
    flow_graph_detail = str(fg) + kwargs_str
    return sha256(flow_graph_detail.encode()).hexdigest()[:32]


class CompiledGraphInMemoryCache:
    def __init__(self):
        self.cached: Dict[str, CompiledGraph] = {}

    def contains(self, key: str) -> bool:
        return key in self.cached

    def get(self, key: str) -> Tuple[str, Optional[CompiledGraph]]:
        return self.cached.get(key) if key in self.cached else None

    def add(self, key: str, compiled_graph: CompiledGraph):
        self.cached[key] = compiled_graph

    def clear(self):
        self.cached.clear()


compiled_graph_in_memory_cache = CompiledGraphInMemoryCache()


def get_compiled_graph_path_with_hash(key: str) -> str:
    return os.path.join(get_cache_dir(), COMPILED_GRAPH_CACHE_DIR_NAME, key)


def flow_graph_get_cache_dir() -> str:
    """
    Get the toplevel temporary directory for storing compiled models.
    """
    return os.path.join(get_cache_dir(), FLOW_GRAPH_CACHE_DIR_NAME)


def flow_graph_get_cache_dir_for_key(key: str) -> str:
    """
    Return the disk location for a given cache key.
    """
    return os.path.join(flow_graph_get_cache_dir(), key)


def flow_graph_get_cache_filename_for_key(key: str) -> str:
    """
    Return the disk location for a given cache key.
    """
    return os.path.join(flow_graph_get_cache_dir_for_key(key), COMPILED_GRAPH_SAVE_FILE_NAME)


def flow_graph_get_hash_key_path(key: str, compiled_graph_hash: str) -> str:
    """
    Return the disk location for a given cache key.
    """
    return os.path.join(flow_graph_get_cache_dir_for_key(key), compiled_graph_hash)


def flow_graph_lookup_graph(key: str) -> Optional[CompiledGraph]:
    """
    Look up a CompiledGraph from the file system cache.
    """
    cache_dir_for_key = flow_graph_get_cache_dir_for_key(key)
    if not os.path.isdir(cache_dir_for_key):
        return None
    cached_files = os.listdir(cache_dir_for_key)
    # there should be exactly one file if this key directory exists
    # one FlowGraph maps to one CompiledGraph
    assert len(cached_files) == 1, f"{cache_dir_for_key} files found in cache dir!"
    cache_file_name = cached_files[0]
    if cache_file_name != COMPILED_GRAPH_SAVE_FILE_NAME:
        # the file name is the hash key, we load compiled graph from 'graph' directory
        try:
            compiled_graph_path = get_compiled_graph_path_with_hash(cache_file_name)
            graph = load_compiled_graph(compiled_graph_path)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("flow graph unable to read from graph cache {}, exception {}".format(compiled_graph_path, e))
            return None
        return graph

    # if the file name is 'compiled_graph.hidet', we load saved compiled graph from 'flowgraph' directory
    file_path = flow_graph_get_cache_filename_for_key(key)
    if not os.path.isfile(file_path):
        return None
    try:
        graph = load_compiled_graph(file_path)
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("flow graph unable to read from cache {}, exception {}".format(file_path, e))
        return None

    return graph


def flow_graph_cache_save(key: str, compiled_graph: CompiledGraph) -> None:
    """
    Store a serialized CompiledGraph on disk.
    """
    # if not present in in-memory cache, add it
    if not compiled_graph_in_memory_cache.contains(key):
        compiled_graph_in_memory_cache.add(key, compiled_graph)
    compiled_graph_hash = compiled_graph.meta.graph_hash
    compiled_graph_dir = get_compiled_graph_path_with_hash(compiled_graph_hash)
    if os.path.exists(compiled_graph_dir):
        logger.info(f"compiled graph already exists for hash: {compiled_graph_hash}")
        # save an empty file since we only need the hash key of compiled_graph
        os.makedirs(flow_graph_get_cache_dir_for_key(key), exist_ok=True)
        with open(flow_graph_get_hash_key_path(key, compiled_graph_hash), 'w'):
            pass
    else:
        file_path = flow_graph_get_cache_filename_for_key(key)
        try:
            save_compiled_graph(compiled_graph, file_path)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("flow graph unable to write to cache {}, exception {}".format(file_path, e))


def flow_graph_cache_load(fg: FlowGraph, kwargs) -> Tuple[CompiledGraph, str]:
    """
    Compute hash key and load a FlowGraph from the cache if it exists.
    """
    compiled_graph = None
    key = compute_flow_graph_hash(fg, kwargs)
    # check in-memory cache first
    compiled_graph = compiled_graph_in_memory_cache.get(key)
    if compiled_graph is not None:
        logger.info(f"flow graph cache hit for key: {key}")
        counters["flow_graph_cache"]["hit"] += 1
        return compiled_graph, key
    compiled_graph = flow_graph_lookup_graph(key)

    if compiled_graph is None:
        logger.info(f"flow graph cache miss for key: {key}")
        counters["flow_graph_cache"]["miss"] += 1
    else:
        compiled_graph_in_memory_cache.add(key, compiled_graph)
        logger.info(f"flow graph cache hit for key: {key}")
        counters["flow_graph_cache"]["hit"] += 1
    return compiled_graph, key


def flow_graph_cache_clear():
    """
    Clear out the on-disk cache.
    """
    try:
        clear_cache_dir(FLOW_GRAPH_CACHE_DIR_NAME)
    except FileNotFoundError:
        pass
