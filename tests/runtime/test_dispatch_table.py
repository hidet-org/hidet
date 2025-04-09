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
import pytest
import tempfile
from unittest.mock import MagicMock, patch

from hidet.runtime.utils.dispatch_table import DispatchTable, IntervalsDispachTable, PointsDispachTable
from hidet.ffi.runtime_api import RuntimeAPI


class MockCompiledFunction:
    """Minimal stand-in for a CompiledFunction."""

    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        return f"Result from {self.name}"


@pytest.fixture
def mock_intervals_table(tmp_path, monkeypatch):
    """
    Provides an IntervalsDispachTable for testing internal methods.
    We patch get_split_points to return a default range so the constructor won't raise.
    """
    # Patch get_split_points to return something like [1, 10].
    def mock_get_split_points():
        return [1, 10]

    monkeypatch.setattr("hidet.option.internal.dispatch_table.get_split_points", mock_get_split_points)

    candidates = [MockCompiledFunction("cand0"), MockCompiledFunction("cand1")]
    input_shapes = [["s0", 32]]
    output_shapes = [["s0", 32]]
    table = IntervalsDispachTable(
        candidates=candidates,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        task_dir=str(tmp_path),
        symbols=["s0"],
        name="test_internal_methods",
    )
    return table


@pytest.fixture
def fresh_task_dir():
    """Yields a temporary directory path for each test requiring a unique task_dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_dispatch_table_base_pick_best_candidate():
    dt = DispatchTable(candidates=[], task_dir="", symbols=[], name="test_dt")
    with pytest.raises(NotImplementedError):
        dt.pick_best_candidate(inputs=[], outputs=None)


def test_intervals_init_no_dynamic_dimension():
    candidates = [MockCompiledFunction("cand0"), MockCompiledFunction("cand1")]
    input_shapes = [[32, 128]]
    output_shapes = [[32, 128]]
    with pytest.raises(ValueError, match="No dynamic dimension found"):
        IntervalsDispachTable(
            candidates=candidates,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            task_dir="",
            symbols=[],
            name="test_intervals_no_dyn",
        )


def test_intervals_init_multiple_dynamic_dimension():
    candidates = [MockCompiledFunction("cand0")]
    input_shapes = [["s0", "s1"]]
    output_shapes = [["s0", "s1"]]
    with pytest.raises(NotImplementedError, match="Only one dynamic dimension is supported"):
        IntervalsDispachTable(
            candidates=candidates,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            task_dir="",
            symbols=["s0", "s1"],
            name="test_intervals_multi_dyn",
        )


@pytest.mark.parametrize("split_points", [[1, 4, 8], [1, 10]])
def test_intervals_init_okay_with_split_points(split_points, monkeypatch, fresh_task_dir):
    def mock_get_split_points():
        return split_points

    monkeypatch.setattr("hidet.option.internal.dispatch_table.get_split_points", mock_get_split_points)
    monkeypatch.setattr(
        "hidet.option.internal.dispatch_table.get_candidate_selection_method", lambda: 'find_best_candidate'
    )

    candidates = [MockCompiledFunction("cand0"), MockCompiledFunction("cand1")]
    input_shapes = [["s0", 128]]
    output_shapes = [["s0", 128]]

    with patch("hidet.utils.benchmark.bench.find_best_candidate") as mock_find_best:
        mock_find_best.return_value = (0, [1.0, 2.0])
        table = IntervalsDispachTable(
            candidates=candidates,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            task_dir=fresh_task_dir,
            symbols=["s0"],
            name="test_intervals_ok",
        )
        assert table.dynamic_input_dim == [(0, 0)]
        assert len(table.intervals) > 0
        assert table.intervals[-1]["range"][1] == float('inf')
        for interval in table.intervals:
            assert interval["best_candidate"] == 0


def test_intervals_pick_best_candidate(monkeypatch, fresh_task_dir):
    def mock_get_split_points():
        return [1, 4, 10]

    monkeypatch.setattr("hidet.option.internal.dispatch_table.get_split_points", mock_get_split_points)
    monkeypatch.setattr(
        "hidet.option.internal.dispatch_table.get_candidate_selection_method", lambda: 'find_best_candidate'
    )

    def mock_find_best_candidate(cands, name, *inputs):
        shape_val = inputs[0].shape[0] if inputs else 1
        return (0, [10.0, 20.0]) if shape_val <= 4 else (1, [20.0, 10.0])

    monkeypatch.setattr("hidet.utils.benchmark.bench.find_best_candidate", mock_find_best_candidate)

    mock_randn = MagicMock(side_effect=lambda shape, device: MagicMock(shape=shape))
    monkeypatch.setattr("hidet.randn", mock_randn)
    mock_empty = MagicMock(side_effect=lambda shape, device: MagicMock(shape=shape))
    monkeypatch.setattr("hidet.empty", mock_empty)

    candidates = [MockCompiledFunction("cand0"), MockCompiledFunction("cand1")]
    input_shapes = [["s0", 128]]
    output_shapes = [["s0", 128]]

    table = IntervalsDispachTable(
        candidates=candidates,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        task_dir=fresh_task_dir,
        symbols=["s0"],
        name="test_intervals_pick",
    )

    shape_inputs = [MagicMock(shape=[3, 128])]
    shape_outputs = [MagicMock(shape=[3, 128])]
    assert table.pick_best_candidate(shape_inputs, shape_outputs) == 0

    shape_inputs = [MagicMock(shape=[10, 128])]
    shape_outputs = [MagicMock(shape=[10, 128])]
    assert table.pick_best_candidate(shape_inputs, shape_outputs) == 1


def test_points_dispatch_table_pick_best_candidate(fresh_task_dir):
    with patch("hidet.utils.benchmark.bench.find_best_candidate") as mock_find_best:
        mock_find_best.return_value = (0, [5.0, 10.0])
        candidates = [MockCompiledFunction("cand0"), MockCompiledFunction("cand1")]
        table = PointsDispachTable(candidates=candidates, task_dir=fresh_task_dir, symbols=["s0"], name="test_points")
        with patch.object(RuntimeAPI, 'get_symbol_value', return_value=42):
            idx = table.pick_best_candidate([], [])
            assert idx == 0
            assert table.dispatch_table[(42,)] == 0

            mock_find_best.reset_mock()
            idx2 = table.pick_best_candidate([], [])
            assert idx2 == 0
            mock_find_best.assert_not_called()


def test_points_dispatch_table_load_save(fresh_task_dir):
    candidates = [MockCompiledFunction("cand0")]
    dt_path = os.path.join(fresh_task_dir, 'dispatch_table.txt')
    with open(dt_path, 'w') as fw:
        fw.write("s0\n")
        fw.write("10 0\n")

    table = PointsDispachTable(candidates=candidates, task_dir=fresh_task_dir, symbols=["s0"], name="test_points_load")
    assert table.dispatch_table == {(10,): 0}

    table.dispatch_table[(20,)] = 0
    table._append_dispatch_table_entry((20,), 0)

    with open(dt_path, 'r') as fr:
        lines = fr.read().splitlines()
        assert len(lines) == 3
        assert lines[0] == "s0"
        assert lines[1] == "10 0"
        assert lines[2] == "20 0"


def test_intervals_init_symbols_mismatch():
    with patch("hidet.option.internal.dispatch_table.get_split_points", return_value=[1, 10]):
        with patch(
            "hidet.option.internal.dispatch_table.get_candidate_selection_method", return_value='find_best_candidate'
        ):
            candidates = [MockCompiledFunction("cand0")]
            input_shapes = [["(s0 * 2)", 128], ['s0']]
            output_shapes = [["s0", 128]]
            symbols = ["s0"]

            with pytest.raises(AssertionError, match=r"Expected 1 symbols.*found 2"):
                IntervalsDispachTable(
                    candidates=candidates,
                    input_shapes=input_shapes,
                    output_shapes=output_shapes,
                    task_dir="",
                    symbols=symbols,
                    name="test_mismatch_symbols",
                )


def test_find_dynamic_dimension_name_complex(mock_intervals_table):
    pytest.skip("Not implemented yet")
    nested_shapes = [["((s0 + 4) * 2)", 128], [64, "s0"]]
    mock_intervals_table.input_shapes = nested_shapes
    assert mock_intervals_table._find_dynamic_dimension_name() == "s0"

    multiple_symbol_shapes = [["(s0 + 4)", "(s1 + 5)"]]
    mock_intervals_table.input_shapes = multiple_symbol_shapes
    with pytest.raises(NotImplementedError, match="Only one dynamic dimension"):
        mock_intervals_table._find_dynamic_dimension_name()

    mock_intervals_table.input_shapes = [[32, 64]]
    assert mock_intervals_table._find_dynamic_dimension_name() is None


@pytest.mark.parametrize(
    "shape_str, s0_val, expected",
    [
        ("(s0 + 10)", 5, 15),
        ("(s0 * 2)", 6, 12),
        ("((s0 + 4) * 3)", 2, 18),
        ("(((s0 * 2) + 10) * 2)", 5, 40),
        ("((s0 * 2) < 10) ? 100 : 200", 3, 100),
        ("((s0 * 2) < 10) ? 100 : 200", 6, 200),
        ("(s0 < 5) ? (s0 + 100) : (s0 + 200)", 4, 104),
        ("(s0 < 5) ? (s0 + 100) : (s0 + 200)", 5, 205),
    ],
)
def test_evaluate_with_symbols_complex(mock_intervals_table, shape_str, s0_val, expected):
    pytest.skip("Not implemented yet")
    result = mock_intervals_table.evaluate_with_symbols(shape_str, {"s0": s0_val})
    assert result == expected

    list_shape = [shape_str, 256]
    res_list = mock_intervals_table.evaluate_with_symbols(list_shape, {"s0": s0_val})
    assert res_list[0] == expected
    assert res_list[1] == 256
