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

from tabulate import tabulate
import numpy as np

import hidet
from hidet.ffi import runtime_api
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

    def pick_best_candidate(self, inputs: List[Any], outputs: Optional[List[Any]] = None) -> int:
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

        dynamic_input_dim = self._find_dynamic_input_dim()
        if dynamic_input_dim is None:
            raise ValueError("No dynamic dimension found in 'input_shapes'.")
        self.in_idx, self.dim_idx = dynamic_input_dim  # pylint: disable=unpacking-non-sequence

        split_points = option.internal.dispatch_table.get_split_points()

        if split_points is None:
            raise NotImplementedError("Split Points must be set as an option in order to use IntervalsDispachTable")

        self._init_intervals(split_points)

    def pick_best_candidate(self, inputs: List[Any], outputs: Optional[List[Any]] = None) -> int:
        """
        Determines the best candidate by looking up the dynamic dimension size
        in intervals or in a cached direct mapping if available.
        """
        dynamic_size = inputs[self.in_idx].shape[self.dim_idx]

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
        self.intervals.append({"range": (split_points[-1] + 1, float('inf')), "best_candidate": best_cand_for_max})

        self._save()

    def _add_intervals(self, start: int, end: int) -> List[Dict[str, Any]]:
        from hidet.utils.benchmark.bench import find_best_candidate

        if option.internal.dispatch_table.get_candidate_selection_method() == 'find_best_candidate':
            shape_inputs = self._fake_inputs(end)
            best_idx, _ = find_best_candidate(self.candidates, self.name, *shape_inputs)
            return [{"range": (start, end), "best_candidate": best_idx}]
        else:
            raise NotImplementedError('Only find_best_candidate method is supported')

    def _find_dynamic_input_dim(self) -> Optional[Tuple[int, int]]:
        """
        Identifies which input dimension is dynamic (str or None).
        Raises NotImplementedError if more than one dynamic dimension is found.
        """
        found = []
        for i, shape in enumerate(self.input_shapes):
            for j, dim in enumerate(shape):
                if isinstance(dim, (str, type(None))):
                    found.append((i, j))
        if len(found) > 1:
            raise NotImplementedError(f"Only one dynamic dimension is supported, found: {found}")
        return found[0] if found else None

    def _fake_inputs(self, shape_val: int) -> List[Any]:
        """
        Creates input and output tensors for the given shape_val in the dynamic dimension.
        Sets symbol values if the dimension is a string.
        """
        all_tensors = []
        for i, shape in enumerate(self.input_shapes):
            final_shape = []
            for j, dim in enumerate(shape):
                if i == self.in_idx and j == self.dim_idx:
                    final_shape.append(shape_val)
                    if isinstance(dim, str):
                        runtime_api.set_symbol_value(dim, shape_val)
                else:
                    final_shape.append(dim)
            all_tensors.append(hidet.randn(final_shape, device='cuda'))
        for out_shape in self.output_shapes:
            final_shape = [shape_val if isinstance(dim, (str, type(None))) else dim for dim in out_shape]
            all_tensors.append(hidet.empty(final_shape, device='cuda'))
        return all_tensors

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
            shape_inputs = self._fake_inputs(shape_val)
            num_candidates = len(self.candidates)
            actual_latencies = np.zeros(num_candidates, dtype=np.float32)
            for c_idx in range(num_candidates):
                actual_latencies[c_idx] = self._benchmark_candidate(
                    self.candidates[c_idx], shape_inputs, warmup=10, repeat=100
                )

            # Optionally record all candidate latencies
            self._record_candidate_selection([shape_val], actual_latencies, report_path=timestamp_str)

            actual_best_idx = int(np.argmin(actual_latencies))
            actual_best_lat = float(actual_latencies[actual_best_idx])
            approx_idx = self.pick_best_candidate(shape_inputs)
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
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def _load(self):
        """
        Loads intervals and cache from the JSON file if available.
        """
        path = os.path.join(self.task_dir, 'dispatch_table.txt')
        if not os.path.exists(path):
            return
        with open(path, 'r') as f:
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

    def pick_best_candidate(self, inputs: List[Any], outputs: Optional[List[Any]] = None) -> int:
        """
        Returns a candidate index for the current symbol values, or benchmarks to find the best if unknown.
        """
        from hidet.utils.benchmark.bench import find_best_candidate

        key = self._get_symbol_values()
        if key in self.dispatch_table:
            return self.dispatch_table[key]
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
        with open(path, 'r') as f:
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
        if not os.path.exists(path):
            with open(path, 'w') as fw:
                fw.write(" ".join(self.symbols) + "\n")
        with open(path, 'a') as fw:
            line = " ".join(map(str, key)) + f" {cand_idx}\n"
            fw.write(line)
