"""Tests for parameter sweep functionality."""

import pytest

from moshpit.pipeline.sweep import (
    ParameterSweep,
    SweepParameter,
    generate_output_path,
    get_nested_value,
    parse_vary_spec,
    set_nested_value,
)


class TestParseVarySpec:
    def test_range_two_values(self):
        param = parse_vary_spec("pframe.dup=2:8")
        assert param.path == "pframe.dup"
        assert len(param.values) > 0
        assert param.values[0] == 2
        assert param.values[-1] <= 8

    def test_range_with_step(self):
        param = parse_vary_spec("strength=0.1:1.0:0.3")
        assert param.path == "strength"
        assert 0.1 in param.values
        assert len(param.values) == 4  # 0.1, 0.4, 0.7, 1.0

    def test_explicit_values(self):
        param = parse_vary_spec("levels=2,4,8,16")
        assert param.path == "levels"
        assert param.values == [2, 4, 8, 16]

    def test_single_value(self):
        param = parse_vary_spec("factor=2.5")
        assert param.path == "factor"
        assert param.values == [2.5]


class TestNestedValues:
    def test_set_simple(self):
        obj = {"a": 1}
        set_nested_value(obj, "a", 2)
        assert obj["a"] == 2

    def test_set_nested(self):
        obj = {"a": {"b": 1}}
        set_nested_value(obj, "a.b", 2)
        assert obj["a"]["b"] == 2

    def test_set_array_index(self):
        obj = {"items": [{"x": 1}, {"x": 2}]}
        set_nested_value(obj, "items[1].x", 99)
        assert obj["items"][1]["x"] == 99

    def test_get_simple(self):
        obj = {"a": 1}
        assert get_nested_value(obj, "a") == 1

    def test_get_nested(self):
        obj = {"a": {"b": {"c": 42}}}
        assert get_nested_value(obj, "a.b.c") == 42

    def test_get_array_index(self):
        obj = {"items": [10, 20, 30]}
        assert get_nested_value(obj, "items[1]") == 20


class TestParameterSweep:
    def test_grid_sweep(self):
        base = {"value": 0}
        params = [SweepParameter("value", [1, 2, 3])]
        sweep = ParameterSweep(base, params, mode="grid")

        configs = list(sweep)
        assert len(configs) == 3
        assert configs[0][1]["value"] == 1
        assert configs[1][1]["value"] == 2
        assert configs[2][1]["value"] == 3

    def test_grid_sweep_multiple_params(self):
        base = {"a": 0, "b": 0}
        params = [
            SweepParameter("a", [1, 2]),
            SweepParameter("b", [10, 20]),
        ]
        sweep = ParameterSweep(base, params, mode="grid")

        configs = list(sweep)
        assert len(configs) == 4  # 2 * 2

    def test_count_limit(self):
        base = {"value": 0}
        params = [SweepParameter("value", [1, 2, 3, 4, 5])]
        sweep = ParameterSweep(base, params, count=3, mode="grid")

        configs = list(sweep)
        assert len(configs) == 3

    def test_random_sweep_reproducible(self):
        base = {"value": 0}
        params = [SweepParameter("value", [1, 2, 3, 4, 5])]

        sweep1 = ParameterSweep(base, params, count=5, mode="random", seed=42)
        sweep2 = ParameterSweep(base, params, count=5, mode="random", seed=42)

        configs1 = [c for _, c in sweep1]
        configs2 = [c for _, c in sweep2]

        assert configs1 == configs2


class TestGenerateOutputPath:
    def test_directory(self, tmp_path):
        result = generate_output_path(tmp_path, 5)
        assert result.parent == tmp_path
        assert "005" in result.name
        assert result.suffix == ".mp4"

    def test_file_path(self, tmp_path):
        base = tmp_path / "output.mp4"
        result = generate_output_path(base, 3)
        assert result.name == "output_003.mp4"
