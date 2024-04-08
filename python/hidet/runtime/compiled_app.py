from typing import Dict, List, Optional, Union
import json
import dataclasses
import os
import zipfile
import tempfile
import hashlib
from collections import defaultdict
from dataclasses import asdict
import numpy as np

import hidet.utils
from hidet.runtime.compiled_module import CompiledModule
from hidet.runtime.compiled_graph import CompiledGraph, save_compiled_graph, load_compiled_graph, GraphExecution


Tensor = 'hidet.graph.tensor.Tensor'  # used in type hint


@dataclasses.dataclass
class AppMetaData:
    name: str
    hidet_version: str
    graphs: List[str]
    app_hash: str


class CompiledApp:
    def __init__(
        self,
        meta: AppMetaData,
        graphs: Dict[str, CompiledGraph],
        modules: Dict[str, CompiledModule],
        tensors: Dict[str, Tensor],
        attributes: Dict[str, Union[bool, int, float, str]],
    ):
        self.meta: AppMetaData = meta
        self.graphs: Dict[str, CompiledGraph] = graphs
        self.tensors: Dict[str, Tensor] = tensors
        self.attributes: Dict[str, Union[bool, int, float, str]] = attributes


def create_compiled_app(
    graphs: Dict[str, CompiledGraph],
    modules: Dict[str, CompiledModule],
    tensors: Dict[str, Tensor],
    attributes: Dict[str, Union[bool, int, float, str]],
    name: Optional[str] = None,
) -> CompiledApp:
    """
    Create a compiled app from a dict of compiled graphs.

    Parameters
    ----------
    graphs: Dict[str, CompiledGraph]
        The compiled graphs used in the app.

    modules: Dict[str, CompiledModule]
        The compiled modules used in the app.

    tensors: Dict[str, Tensor]
        The tensors used in the app.

    attributes: Dict[str, Union[bool, int, float, str]]
        The attributes of the app.

    name: Optional[str]
        The name of the app. If None, the name will be set to 'app'.

    Returns
    -------
    ret: CompiledApp
        The compiled app.
    """
    if name is None:
        name = 'app'

    hash_obj = hashlib.sha256()
    hash_obj.update(name.encode())
    for graph_name, graph in graphs.items():
        hash_obj.update(graph_name.encode())
        hash_obj.update(graph.meta.graph_hash.encode())
    app_hash: str = hash_obj.hexdigest()[:16]

    meta = AppMetaData(name=name, hidet_version=hidet.__version__, graphs=list(graphs.keys()), app_hash=app_hash)
    return CompiledApp(meta=meta, graphs=graphs, modules=modules, tensors=tensors, attributes=attributes)


def save_compiled_app(app: CompiledApp, path: str):
    """
    Save a compiled app to a file.

    Parameters
    ----------
    app: CompiledApp
        The compiled app to save.

    path: str
        The path to save the compiled app.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # save the meta data
        with open(os.path.join(tmp_dir, 'meta.json'), 'w') as f:
            meta_bytes = json.dumps(asdict(app.meta), indent=4)
            f.write(meta_bytes)

        # save the kernel-only graphs to files
        for name, graph in app.graphs.items():
            graph_path = os.path.join(tmp_dir, '{}.hidet'.format(name))
            save_compiled_graph(graph, file=graph_path, save_dispatch_table=False, save_weights=False)
            with zipfile.ZipFile(graph_path, 'r') as zip_file:
                graph_dir = os.path.join(tmp_dir, 'graphs', name)
                os.makedirs(graph_dir)
                zip_file.extractall(path=graph_dir)
            os.remove(graph_path)

        # save the weights
        weights: List[np.ndarray] = []
        weight_hash_map: Dict[str, int] = {}  # the hash of the weight -> the index of the weight in the weights list
        for name, graph in app.graphs.items():
            with open(os.path.join(tmp_dir, 'graphs', '{}-weights-index.txt'.format(name)), 'w') as weight_index_file:
                for weight in graph.weights:
                    weight_ndarray = weight.cpu().numpy()
                    hash_obj = hashlib.sha256()
                    hash_obj.update(weight_ndarray.tobytes())
                    hash_obj.update(weight.signature().encode())
                    weight_hash: str = hash_obj.hexdigest()
                    if weight_hash not in weight_hash_map:
                        weight_hash_map[weight_hash] = len(weights)
                        weights.append(weight_ndarray)
                    weight_index = weight_hash_map[weight_hash]
                    weight_index_file.write('{}\n'.format(weight_index))

        np.savez(os.path.join(tmp_dir, 'weights.npz'), *weights)

        # save the contents of the current dir to a zip file
        with zipfile.ZipFile(path, 'w') as zip_file:
            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    zip_file.write(os.path.join(root, file), arcname=os.path.relpath(os.path.join(root, file), tmp_dir))


def load_compiled_app(path: str) -> CompiledApp:
    """
    Load a compiled app from a file.

    Parameters
    ----------
    path: str
        The path to the compiled app file.

    Returns
    -------
    ret: CompiledApp
        The loaded compiled app.
    """
    from hidet import Tensor  # pylint: disable=redefined-outer-name
    from hidet.utils.dataclass import from_dict

    with zipfile.ZipFile(path, 'r') as zip_file:
        # load the meta data
        with zip_file.open('meta.json', 'r') as f:
            meta_bytes = f.read()
            meta: AppMetaData = from_dict(AppMetaData, json.loads(meta_bytes))

        # extract the app if needed
        app_dir = hidet.utils.cache_file('apps', meta.app_hash)
        meta_path = os.path.join(app_dir, 'meta.json')
        if not os.path.exists(meta_path):
            # we only extract the app when it is not in our cache dir.
            # we used 'meta.json' as the indicator whether the app is there or not.
            # if the app is not there, we extract everything but the weights in the app to the cache dir
            files_to_extract = [name for name in zip_file.namelist() if name != 'weights.npz']
            zip_file.extractall(app_dir, files_to_extract)

        # load the compiled graphs
        graphs: Dict[str, CompiledGraph] = {}
        for graph_name in meta.graphs:
            graphs[graph_name] = load_compiled_graph(os.path.join(app_dir, 'graphs', graph_name))

        # load the weights from the app file
        device2weights: Dict[str, Dict[int, Tensor]] = defaultdict(dict)
        with zip_file.open('weights.npz', 'r') as npz:
            weights: List[np.ndarray] = list(np.load(npz).values())
        for graph_name in meta.graphs:
            graph: CompiledGraph = graphs[graph_name]
            weight_index_file = os.path.join(app_dir, 'graphs', '{}-weights-index.txt'.format(graph_name))
            graph_weights = []
            with open(weight_index_file, 'r') as f:
                weight_indices = [int(line.strip()) for line in f.readlines()]
                for idx, weight_index in enumerate(weight_indices):
                    execution: GraphExecution = graph.graph_execution
                    device: str = execution.tensor_device[execution.weights_index[idx]]
                    if weight_index not in device2weights[device]:
                        device2weights[device][weight_index] = hidet.asarray(weights[weight_index], device=device)
                    graph_weights.append(device2weights[device][weight_index])
            graphs[graph_name].set_weights(graph_weights)

        return CompiledApp(meta=meta, graphs=graphs, modules={}, tensors={}, attributes={})
