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
import json
from typing import Dict, Tuple, List, Any, Union, Optional
from datetime import datetime
from filelock import FileLock

from tabulate import tabulate
import numpy as np

import hidet
from hidet.ffi import runtime_api
from hidet.ffi.array import Array
from hidet.ffi.utils import ctypes_func_pointer
from hidet.ir.type import void_p
from hidet.runtime.compiled_module import CompiledFunction
from hidet import option


class DispatchTable:
    """
    Base class for dispatch tables, managing candidate selection based on performance.
    """

    def __init__(self, candidates: List[CompiledFunction], task_dir: str, symbols: List[str], name: str):
        """
        Initializes the dispatch table.

        Parameters
        ----------
        candidates : List[CompiledFunction]
            The compiled function candidates.
        task_dir : str
            Directory to save or load dispatch data.
        symbols : List[str]
            Names of runtime symbols.
        name : str
            A label for this dispatch table (used in benchmarking reports).
        """
        self.candidates: List[CompiledFunction] = candidates
        self.task_dir: str = task_dir
        self.symbols: List[str] = symbols
        self.name: str = name

    def pick_best_candidate(self, inputs: List['Tensor'], outputs: List['Tensor']) -> int:
        """
        Must be overridden to pick a best candidate index for the given inputs (and optional outputs).

        Parameters
        ----------
        inputs : List[Any]
            Input tensors or data.
        outputs : List[Any], optional
            Output tensors or data, if applicable.

        Returns
        -------
        int
            Index of the best candidate implementation.
        """
        raise NotImplementedError("Subclasses must implement pick_best_candidate(...).")

    def _record_candidate_selection(self, key: Tuple[int, ...], latencies: List[float], report_path='reports'):
        """
        Records performance details of candidates for a specific key, which helps in analyzing
        or logging latency data.

        Parameters
        ----------
        key : Tuple[int, ...]
            Key representing runtime symbol values.
        latencies : List[float]
            Latency measurements (ms) for each candidate.
        report_path : str
            Subdirectory name for storing the report file.
        """
        if not self.task_dir:
            return
        candidates_json_path = os.path.join(self.task_dir, 'candidates.json')
        if not os.path.exists(candidates_json_path):
            return

        report_dir = os.path.join(self.task_dir, report_path)
        os.makedirs(report_dir, exist_ok=True)

        name_parts = [f"{sym_name}_{sym_val}" for sym_val, sym_name in zip(key, self.symbols)]
        filename = "_".join(name_parts) + ".txt"
        out_path = os.path.join(report_dir, filename)

        if os.path.exists(out_path):
            return  # avoid overwriting if it already exists

        with open(candidates_json_path, 'r') as f:
            candidates_json = json.load(f)
            headers = candidates_json['headers']
            candidate_lines = candidates_json['candidates']

        headers.extend(['latency', 'rank'])
        sorted_indices = sorted(range(len(latencies)), key=lambda i: latencies[i])

        for i, line in enumerate(candidate_lines):
            line.extend([f'{latencies[i]:.3f} ms', sorted_indices.index(i)])
        candidate_lines.sort(key=lambda l: l[-1])

        with open(out_path, 'w') as rf:
            rf.write(tabulate(candidate_lines, headers=headers, tablefmt='plain'))


class IntervalsDispachTable(DispatchTable):
    """
    Handles a single dynamic dimension over [1..∞). It splits the domain into intervals,
    each assigned a single best candidate. Intervals and candidates are decided by sampling latencies.
    """

    def __init__(
        self,
        candidates: List[CompiledFunction],
        input_shapes: List[List[Union[str, int]]],
        output_shapes: List[List[Union[str, int]]],
        task_dir: str,
        symbols: List[str],
        name: str,
    ):
        """
        Creates a dynamic dispatch table with interval splitting.

        Parameters
        ----------
        candidates : List[CompiledFunction]
            Candidate implementations.
        input_shapes : List[List[Union[str, int]]]
            Shapes for each input tensor (only one dimension may be dynamic).
        output_shapes : List[List[Union[str, int]]]
            Shapes for output tensors (optionally dynamic).
        task_dir : str
            Directory for storing/loading dispatch data.
        symbols : List[str]
            Runtime symbol names controlling dispatch.
        name : str
            Label for the dispatch table (used in logging or reporting).
        """
        super().__init__(candidates, task_dir, symbols, name)
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes or []
        self.intervals: List[Dict[str, Any]] = []

        self.dynamic_input_dim = self._find_dynamic_input_dim()
        if self.dynamic_input_dim is None:
            raise ValueError("No dynamic dimension found in 'input_shapes'.")

        split_points = option.internal.dispatch_table.get_split_points()

        if split_points is None:
            raise NotImplementedError("Split Points must be set as an option in order to use IntervalsDispachTable")

        with FileLock(os.path.join(self.task_dir, 'dispatch_table.txt.t_lock')):
            self._init_intervals(split_points)

    def pick_best_candidate(self, inputs: List['Tensor'], outputs: List['Tensor']) -> int:
        """
        Determines the best candidate by looking up the dynamic dimension size
        in intervals or in a cached direct mapping if available.
        """
        (i, j) = self.dynamic_input_dim[0]
        dynamic_size = inputs[i].shape[j]

        for entry in self.intervals:
            rmin, rmax = entry["range"]
            if rmin <= dynamic_size <= rmax:
                return entry["best_candidate"]

        raise RuntimeError(f"No interval covers size {dynamic_size}.")

    # -------------------------------------------------------------------------
    # Interval-based splitting and initialization
    # -------------------------------------------------------------------------
    def _init_intervals(self, split_points: List[int]):
        """
        Loads existing intervals or, if none are saved, splits the ranges
        defined by split_points to build intervals.
        """
        self._load()
        if self.intervals:
            return

        assert len(split_points) > 1
        assert split_points[0] == 1

        for i in range(len(split_points) - 1):
            self.intervals.extend(self._add_intervals(split_points[i], split_points[i + 1]))
            split_points[i + 1] += 1

        last_interval = self.intervals[-1]
        best_cand_for_max = last_interval["best_candidate"]
        self.intervals.append({"range": (split_points[-1], float('inf')), "best_candidate": best_cand_for_max})

        self._save()

    def _add_intervals(self, start: int, end: int) -> List[Dict[str, Any]]:
        from hidet.utils.benchmark.bench import find_best_candidate

        if option.internal.dispatch_table.get_candidate_selection_method() == 'find_best_candidate':
            input_tensors, output_tensors = self._fake_inputs(end)
            best_idx, _ = find_best_candidate(self.candidates, self.name, *input_tensors, *output_tensors)
            return [{"range": (start, end), "best_candidate": best_idx}]
        else:
            raise NotImplementedError('Only find_best_candidate method is supported')

    def _find_dynamic_input_dim(self) -> Optional[Tuple[int, int]]:
        """
        Identifies which input dimension is dynamic str
        Raises NotImplementedError if more than one dynamic dimension is found.
        """
        found = []
        found_symbols = set()
        for i, shape in enumerate(self.input_shapes):
            for j, dim in enumerate(shape):
                if isinstance(dim, str):
                    found.append((i, j))
                    found_symbols.add(dim)
        if len(found_symbols) != len(self.symbols):
            assert (
                False
            ), f"Expected {len(self.symbols)} symbols in input shapes {self.input_shapes}, found {len(found_symbols)}"
        if len(found_symbols) > 1:
            raise NotImplementedError(f"Only one dynamic dimension is supported, found: {found_symbols}")
        return found if found else None

    def _fake_inputs(self, shape_val: int) -> List[Any]:
        """
        Creates input and output tensors for the given shape_val in the dynamic dimension
        (wherever self.dim_name occurs).
        """
        input_shapes = [list(shape) for shape in self.input_shapes]
        input_tensors = []
        output_tensors = []
        for (i, j) in self.dynamic_input_dim:
            runtime_api.set_symbol_value(input_shapes[i][j], shape_val)
            input_shapes[i][j] = shape_val
        for in_shape in input_shapes:
            input_tensors.append(hidet.randn(in_shape, device='cuda'))

        for out_shape in self.output_shapes:
            final_shape = [shape_val if isinstance(dim, str) else dim for dim in out_shape]
            output_tensors.append(hidet.empty(final_shape, device='cuda'))

        return input_tensors, output_tensors

    # -------------------------------------------------------------------------
    # Approximation loss
    # -------------------------------------------------------------------------
    def measure_approximation_loss(self, test_shapes: List[int]) -> Dict[str, Any]:
        """
        Evaluates how much performance is lost by using the intervals/caches
        vs. always picking the actual best candidate for each shape.
        """
        import csv

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        losses = {}
        total_loss_time = 0.0
        total_actual_time = 0.0

        for shape_val in test_shapes:
            input_tensors, output_tensors = self._fake_inputs(shape_val)
            num_candidates = len(self.candidates)
            actual_latencies = np.zeros(num_candidates, dtype=np.float32)
            for c_idx in range(num_candidates):
                actual_latencies[c_idx] = self._benchmark_candidate(
                    self.candidates[c_idx], [*input_tensors, *output_tensors], warmup=10, repeat=100
                )

            # Optionally record all candidate latencies
            self._record_candidate_selection([shape_val], actual_latencies, report_path=timestamp_str)

            actual_best_idx = int(np.argmin(actual_latencies))
            actual_best_lat = float(actual_latencies[actual_best_idx])
            approx_idx = self.pick_best_candidate(input_tensors, output_tensors)
            approx_lat = float(actual_latencies[approx_idx])

            time_loss = approx_lat - actual_best_lat
            pct_loss = (time_loss / actual_best_lat) * 100 if actual_best_lat > 0 else 0

            losses[shape_val] = {
                "actual_best_idx": actual_best_idx,
                "actual_best_latency": actual_best_lat,
                "approx_idx": approx_idx,
                "approx_latency": approx_lat,
                "time_loss_ms": time_loss,
                "percentage_loss": pct_loss,
            }

            total_loss_time += time_loss
            total_actual_time += actual_best_lat

        total_shapes = len(test_shapes)
        avg_time_loss = total_loss_time / total_shapes if total_shapes else 0
        avg_time_loss_pct = (total_loss_time / total_actual_time * 100) if total_actual_time else 0

        max_loss = max(losses.values(), key=lambda x: x["time_loss_ms"]) if losses else {}
        min_loss = min(losses.values(), key=lambda x: x["time_loss_ms"]) if losses else {}

        summary = {
            "total_shapes_tested": total_shapes,
            "avg_time_loss_ms": avg_time_loss,
            "avg_percentage_loss": avg_time_loss_pct,
            "max_time_loss_ms": max_loss.get("time_loss_ms", 0.0),
            "min_time_loss_ms": min_loss.get("time_loss_ms", 0.0),
            "max_loss_shape": max(losses, key=lambda x: losses[x]["time_loss_ms"]) if losses else None,
            "min_loss_shape": min(losses, key=lambda x: losses[x]["time_loss_ms"]) if losses else None,
            "loss_details": losses,
        }

        report_dir = os.path.join(self.task_dir, 'reports')
        os.makedirs(report_dir, exist_ok=True)
        txt_path = os.path.join(report_dir, f'approximation_loss_{timestamp_str}.txt')
        with open(txt_path, 'w') as f:
            f.write("[APPROXIMATION LOSS REPORT]\n\n")
            f.write(f"Total Shapes Tested: {total_shapes}\n")
            f.write(f"Average Time Loss (ms): {avg_time_loss:.4f}\n")
            f.write(f"Average Percentage Loss: {avg_time_loss_pct:.2f}%\n")
            f.write(
                f"Maximum Time Loss (ms): {summary['max_time_loss_ms']:.4f} " f"(Shape: {summary['max_loss_shape']})\n"
            )
            f.write(
                f"Minimum Time Loss (ms): {summary['min_time_loss_ms']:.4f} "
                f"(Shape: {summary['min_loss_shape']})\n\n"
            )
            f.write("Details per shape:\n")
            for s_val, det in losses.items():
                f.write(f"Shape {s_val}: {det}\n")

        csv_path = os.path.join(report_dir, f'approximation_loss_{timestamp_str}.csv')
        fieldnames = [
            "shape_val",
            "actual_best_idx",
            "actual_best_latency",
            "approx_idx",
            "approx_latency",
            "time_loss_ms",
            "percentage_loss",
        ]
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for s_val, det in losses.items():
                row = {
                    "shape_val": s_val,
                    "actual_best_idx": det["actual_best_idx"],
                    "actual_best_latency": det["actual_best_latency"],
                    "approx_idx": det["approx_idx"],
                    "approx_latency": det["approx_latency"],
                    "time_loss_ms": det["time_loss_ms"],
                    "percentage_loss": det["percentage_loss"],
                }
                writer.writerow(row)

        return summary

    def _benchmark_candidate(
        self, candidate: CompiledFunction, shape_inputs: List[Any], warmup: int = 10, repeat: int = 100
    ) -> float:
        """
        Benchmarks a single candidate with given inputs and returns the median latency in ms.
        """
        from hidet.utils.benchmark.bench import benchmark_func

        return benchmark_func(lambda: candidate(*shape_inputs), warmup=warmup, number=1, repeat=repeat)

    # -------------------------------------------------------------------------
    # Persistence: save/load table to file
    # -------------------------------------------------------------------------
    def _save(self):
        """
        Saves intervals and the best-candidate cache to a JSON file.
        """
        data = {"intervals": self.intervals}
        path = os.path.join(self.task_dir, 'dispatch_table.txt')
        with FileLock(path + '.lock'), open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def _load(self):
        """
        Loads intervals and cache from the JSON file if available.
        """
        path = os.path.join(self.task_dir, 'dispatch_table.txt')
        if not os.path.exists(path):
            return
        with FileLock(path + '.lock'), open(path, 'r') as f:
            data = json.load(f)
        self.intervals = data["intervals"]


class PointsDispachTable(DispatchTable):
    """
    Static dispatch table with a key-value lookup from symbol values to a best candidate.
    """

    def __init__(self, candidates: List[CompiledFunction], task_dir: str, symbols: List[str], name: str):
        """
        Initializes a static dispatch table.

        Parameters
        ----------
        candidates : List[CompiledFunction]
            The candidate functions.
        task_dir : str
            Directory to store logs/reports.
        symbols : List[str]
            Symbol names for dispatch decisions.
        name : str
            A label for the task (used in benchmarking reports).
        """
        super().__init__(candidates, task_dir, symbols, name)
        self.dispatch_table: Dict[Tuple[int, ...], int] = {}
        self._load()

    def pick_best_candidate(self, inputs: List['Tensor'], outputs: List['Tensor']) -> int:
        """
        Returns a candidate index for the current symbol values, or benchmarks to find the best if unknown.
        """
        from hidet.utils.benchmark.bench import find_best_candidate

        key = self._get_symbol_values()
        if key in self.dispatch_table:
            return self.dispatch_table[key]
        with FileLock(os.path.join(self.task_dir, 'dispatch_table.txt.t_lock')):
            if len(self.candidates) > 1:
                best_idx, latencies = find_best_candidate(
                    self.candidates, self.name, *(inputs if inputs else []), *(outputs if outputs else [])
                )
                self.dispatch_table[key] = best_idx
                self._record_candidate_selection(key, latencies)
            else:
                self.dispatch_table[key] = 0
            self._append_dispatch_table_entry(key, self.dispatch_table[key])
        return self.dispatch_table[key]

    def _get_symbol_values(self) -> Tuple[int, ...]:
        """
        Returns current runtime symbol values as a tuple.
        """
        return tuple(runtime_api.get_symbol_value(sym) for sym in self.symbols)

    def _load(self):
        """
        Loads dispatch entries from file if available.
        """
        if not self.task_dir:
            return
        path = os.path.join(self.task_dir, 'dispatch_table.txt')
        if not os.path.exists(path):
            return
        with FileLock(path + '.lock'), open(path, 'r') as f:
            lines = f.readlines()
            if not lines:
                return
            for i, line in enumerate(lines):
                if i == 0 and any(sym in line for sym in self.symbols):
                    continue
                items = line.strip().split()
                if not items:
                    continue
                key_tuple = tuple(int(x) for x in items[:-1])
                cand_idx = int(items[-1])
                self.dispatch_table[key_tuple] = cand_idx

    def _append_dispatch_table_entry(self, key: Tuple[int, ...], cand_idx: int):
        """
        Appends a new entry to the dispatch table file.
        """
        if not self.task_dir:
            return
        path = os.path.join(self.task_dir, 'dispatch_table.txt')
        with FileLock(path + '.lock'):
            if not os.path.exists(path):
                with open(path, 'w') as fw:
                    fw.write(" ".join(self.symbols) + "\n")
            with open(path, 'a') as fw:
                line = " ".join(map(str, key)) + f" {cand_idx}\n"
                fw.write(line)


class GraphIntervalDispatchTable:
    """
    A dispatch table for a compiled graph that uses per-integer lookup.

    Instead of storing intervals, we now fill a list entry for each integer
    so that self.dispatch_table[symbol_value] works directly.
    """

    def __init__(self, graph: 'CompiledGraph'):
        """
        Initialize the GraphIntervalDispatchTable.

        Parameters
        ----------
        graph : CompiledGraph
            The compiled graph whose kernels are to be dispatched using an
            integer-based lookup.
        """
        from hidet.runtime.compiled_graph import CompiledGraph

        self.dispatch_table: List[Dict[str, Any]] = []
        self.dispatch_table_path = graph.dispatch_table_path
        self.compiled_graph: CompiledGraph = graph

        # Load any existing dispatch table from disk (reconstructing kernel arrays).
        self.load()

        # If dispatch table is empty, build it.
        if len(self.dispatch_table) == 0:
            with FileLock(self.dispatch_table_path + '.t_lock'):
                self.populate_dispatch_table()

    def __getitem__(self, symbol_val):
        """
        Retrieve the prebuilt array of kernel pointers based on an exact integer index.

        Parameters
        ----------
        symbol_val : tuple
            A tuple of symbol values (currently supporting only one symbol).

        Returns
        -------
        Array
            An Array of function pointers, each pointer being the best candidate
            for the corresponding compiled task in the graph.
        """
        assert len(symbol_val) == 1
        idx = symbol_val[0]
        if idx < len(self.dispatch_table):
            return self.dispatch_table[idx]["kernel_array"]
        else:
            return self.dispatch_table[-1]["kernel_array"]

    def __contains__(self, _):
        """
        Always return True to allow the runtime to query
        without KeyErrors. If out of range, __getitem__
        simply uses the last entry anyway.
        """
        return True

    def populate_dispatch_table(self):
        """
        Build (or rebuild) the dispatch table entries for each integer within
        the dynamic domain. We still rely on 'split points' to do one benchmark
        per interval, but we replicate the results for every integer in that interval.
        """
        from hidet.graph.tensor import Tensor

        graph = self.compiled_graph
        exe = graph.graph_execution
        split_points = option.internal.dispatch_table.get_split_points()

        assert split_points[0] == 1, (
            "We expect split_points to start at 1 (or some positive integer) so that "
            "we can fill dispatch_table from index=1 upward."
        )

        max_split = split_points[-1]
        self.dispatch_table = [{} for _ in range(max_split)]

        def create_inputs(symbol_val: int):
            input_shapes = [list(tensor.shape) for tensor in graph.meta.inputs]
            input_tensors = []
            for shape in input_shapes:
                for i, dim in enumerate(shape):
                    if isinstance(dim, str):
                        runtime_api.set_symbol_value(dim, symbol_val)
                        shape[i] = symbol_val
            for in_shape in input_shapes:
                input_tensors.append(hidet.randn(in_shape, device='cuda'))
            return input_tensors

        for interval_num, _ in enumerate(split_points[:-1]):
            interval_beg = split_points[interval_num]
            interval_end = split_points[interval_num + 1]

            symbol_val = interval_end

            inputs = create_inputs(symbol_val)

            index2tensor: Dict[int, Tensor] = {}
            for idx_inp, inp_tensor in zip(exe.inputs_index, inputs):
                index2tensor[idx_inp] = inp_tensor
            for idx_w, w_tensor in zip(exe.weights_index, graph.weights):
                index2tensor[idx_w] = w_tensor

            best_candidates = [-1 for _ in range(len(graph.compiled_tasks))]

            for inst in exe.instructions:
                node_inputs = [index2tensor[i] for i in inst.inputs]
                node_kernel = graph.compiled_tasks[inst.task_idx]
                node_outputs = node_kernel.create_outputs(node_inputs)

                if len(node_kernel.meta_data.symbols) > 1:
                    raise NotImplementedError(
                        "Currently only supports populating the dispatch table with at most one symbol"
                    )

                for out_idx, output_index in enumerate(inst.outputs):
                    index2tensor[output_index] = node_outputs[out_idx]

                best_candidates[inst.task_idx] = node_kernel.pick_best_candidate(node_inputs, node_outputs)

                for idx in inst.free:
                    if idx in index2tensor:
                        del index2tensor[idx]

            kernel_array = Array(void_p, len(graph.compiled_tasks))
            for task_idx, bc in enumerate(best_candidates):
                kernel_array[task_idx] = ctypes_func_pointer(graph.compiled_tasks[task_idx].candidates[bc].ctypes_func)

            for val in range(interval_beg, min(interval_end, len(self.dispatch_table))):
                self.dispatch_table[val] = {"best_candidates": best_candidates, "kernel_array": kernel_array}

            split_points[i + 1] += 1

        assert any(self.dispatch_table)

        for i in reversed(range(len(self.dispatch_table))):
            if self.dispatch_table[i]:
                last_entry = self.dispatch_table[i]
                break

        self.dispatch_table.append(last_entry)

        self.save()

    def save(self):
        """
        Saves dispatch_table (best_candidates for each index) to a JSON file.
        We do NOT store the kernel arrays (function pointers).
        """
        to_serialize = []
        for entry in self.dispatch_table:
            assert entry
            to_serialize.append(entry["best_candidates"])

        data = {"dispatch_table": to_serialize}
        path = self.dispatch_table_path
        with FileLock(path + '.lock'), open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def load(self):
        """
        Loads dispatch_table from the JSON file if available, and recreates
        the array of kernel pointers for each entry.
        """
        path = self.dispatch_table_path
        if not os.path.exists(path):
            return

        with FileLock(path + '.lock'), open(path, 'r') as f:
            data = json.load(f)

        loaded_candidates_list = data["dispatch_table"]
        self.dispatch_table = []

        for best_candidates in loaded_candidates_list:
            if best_candidates is None:
                self.dispatch_table.append({})
            else:
                kernel_array = Array(void_p, len(self.compiled_graph.compiled_tasks))
                for task_idx, bc in enumerate(best_candidates):
                    kernel_array[task_idx] = ctypes_func_pointer(
                        self.compiled_graph.compiled_tasks[task_idx].candidates[bc].ctypes_func
                    )
                self.dispatch_table.append({"best_candidates": best_candidates, "kernel_array": kernel_array})


class GraphPointsDispatchTable:
    """
    A dispatch table for a compiled graph that uses a points-based key-value approach.

    Instead of intervals, this class explicitly stores the best candidate for each
    distinct combination of symbol values that occur at runtime. When the graph
    executes and a new symbol combination appears, it is benchmarked, stored, and
    reused on subsequent invocations.
    """

    def __init__(self, graph: 'CompiledGraph'):
        """
        Initialize the GraphPointsDispatchTable.

        Parameters
        ----------
        graph : CompiledGraph
            The compiled graph whose kernels are to be dispatched by exact symbol-value lookup.
        """
        from hidet.runtime.compiled_graph import CompiledGraph

        self.dispatch_table: Dict[Tuple[int, ...], Array] = {}
        self.dispatch_table_path = graph.dispatch_table_path
        self.compiled_graph: CompiledGraph = graph

        self.load()

    def __getitem__(self, symbol_dims):
        """
        Retrieve the array of best-candidate kernel pointers for a particular tuple of symbol values.

        Parameters
        ----------
        symbol_dims : Tuple[int, ...]
            The dynamic symbol values that define a unique graph shape.

        Returns
        -------
        Array
            An array of function pointers, where each pointer is the best candidate
            for its associated compiled task.
        """
        return self.dispatch_table[symbol_dims]

    def __contains__(self, symbol_dims):
        """
        Check whether the dispatch table has a best-candidate record for the given symbol values.

        Parameters
        ----------
        symbol_dims : Tuple[int, ...]
            The dynamic symbol values in question.

        Returns
        -------
        bool
            True if an entry exists; False otherwise.
        """
        return symbol_dims in self.dispatch_table

    def update_symbol_table(self, symbol_dims: Tuple[int, ...], best_candidates: List[int]):
        """
        Store a new set of best candidates for a given symbol combination and append it to the dispatch file.

        Parameters
        ----------
        symbol_dims : Tuple[int, ...]
            The dynamic symbol values for which best_candidates applies.
        best_candidates : List[int]
            Indices of the best schedule (candidate) for each compiled task in the graph.
        """
        kernel_array = Array(void_p, len(self.compiled_graph.compiled_tasks))
        for task_idx, best_candidate in enumerate(best_candidates):
            kernel_array[task_idx] = ctypes_func_pointer(
                self.compiled_graph.compiled_tasks[task_idx].candidates[best_candidate].ctypes_func
            )
        self.dispatch_table[symbol_dims] = kernel_array

        with FileLock(self.dispatch_table_path + '.lock'):
            if not os.path.exists(self.dispatch_table_path):
                with open(self.dispatch_table_path, 'w') as f:
                    f.write(' '.join(n for n, _ in self.compiled_graph.dynamic_dims) + '\n')

            # Read the last line to see if it matches current line
            append_line = ' '.join(str(x) for x in symbol_dims) + ' ' + ' '.join(str(x) for x in best_candidates) + '\n'
            with open(self.dispatch_table_path, 'r') as f:
                lines = f.readlines()
                if lines[-1] == append_line:
                    return

            with open(self.dispatch_table_path, 'a') as f:
                f.write(append_line)

    def load(self):
        """
        Load existing dispatch records from disk if the table file exists,
        and build the in-memory dictionary of best candidates.
        """
        graph = self.compiled_graph
        if os.path.exists(self.dispatch_table_path):
            with FileLock(self.dispatch_table_path + '.lock'), open(self.dispatch_table_path, 'r') as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    if idx == 0:
                        continue  # skip the header line
                    items = line.split()
                    if len(items) == 0:
                        continue  # skip empty lines
                    if len(items) != len(graph.dynamic_dims) + len(graph.compiled_tasks):
                        raise RuntimeError('Invalid dispatch table')
                    items = [int(item) for item in items]
                    symbol_dims = items[: len(graph.dynamic_dims)]
                    schedule_indices = items[len(graph.dynamic_dims) :]
                    kernel_array = Array(void_p, len(graph.compiled_tasks))
                    for task_idx, (compiled_task, sch_idx) in enumerate(zip(graph.compiled_tasks, schedule_indices)):
                        if not 0 <= sch_idx < len(compiled_task.candidates):
                            raise RuntimeError(
                                'Invalid schedule index {} for compiled task at {}'.format(
                                    sch_idx, compiled_task.task_dir
                                )
                            )
                        kernel_array[task_idx] = ctypes_func_pointer(compiled_task.candidates[sch_idx].ctypes_func)
                    self.dispatch_table[tuple(symbol_dims)] = kernel_array
