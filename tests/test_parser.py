"""Tests for the pipeline DSL parser."""

import pytest

from moshpit.pipeline.parser import (
    parse_effect,
    parse_params,
    parse_pipeline,
    parse_value,
    pipeline_to_config,
)


class TestParseValue:
    def test_integer(self):
        assert parse_value("42") == 42
        assert parse_value("-10") == -10

    def test_float(self):
        assert parse_value("3.14") == 3.14
        assert parse_value("-0.5") == -0.5

    def test_boolean(self):
        assert parse_value("true") is True
        assert parse_value("false") is False
        assert parse_value("True") is True
        assert parse_value("False") is False

    def test_string(self):
        assert parse_value("hello") == "hello"
        assert parse_value('"quoted"') == "quoted"
        assert parse_value("'single'") == "single"


class TestParseParams:
    def test_empty(self):
        params, start, end = parse_params("")
        assert params == {}
        assert start is None
        assert end is None

    def test_range(self):
        params, start, end = parse_params("30:90")
        assert params == {}
        assert start == 30
        assert end == 90

    def test_partial_range(self):
        params, start, end = parse_params("30:")
        assert start == 30
        assert end is None

        params, start, end = parse_params(":90")
        assert start is None
        assert end == 90

    def test_key_value(self):
        params, _, _ = parse_params("dup=5")
        assert params == {"dup": 5}

    def test_multiple_params(self):
        params, _, _ = parse_params("dup=5,start=10")
        assert params == {"dup": 5, "start": 10}

    def test_transform_shorthand(self):
        params, _, _ = parse_params("jitter,strength=0.5")
        assert params == {"transform": "jitter", "strength": 0.5}


class TestParseEffect:
    def test_simple(self):
        effect = parse_effect("iframe")
        assert effect.name == "iframe"
        assert effect.params == {}

    def test_with_range(self):
        effect = parse_effect("iframe[30:90]")
        assert effect.name == "iframe"
        assert effect.range_start == 30
        assert effect.range_end == 90

    def test_with_params(self):
        effect = parse_effect("pframe[dup=5]")
        assert effect.name == "pframe"
        assert effect.params == {"dup": 5}

    def test_vector_transform(self):
        effect = parse_effect("vector[jitter,strength=0.5]")
        assert effect.name == "vector"
        assert effect.params == {"transform": "jitter", "strength": 0.5}


class TestParsePipeline:
    def test_single_effect(self):
        effects = parse_pipeline("iframe[30:90]")
        assert len(effects) == 1
        assert effects[0].name == "iframe"

    def test_chained_effects(self):
        effects = parse_pipeline("iframe[30:90] -> pframe[dup=5]")
        assert len(effects) == 2
        assert effects[0].name == "iframe"
        assert effects[1].name == "pframe"

    def test_complex_chain(self):
        effects = parse_pipeline(
            "iframe[10:40] -> vector[jitter,strength=0.5] -> pframe[dup=3]"
        )
        assert len(effects) == 3
        assert effects[0].name == "iframe"
        assert effects[1].name == "vector"
        assert effects[2].name == "pframe"


class TestPipelineToConfig:
    def test_iframe(self):
        effects = parse_pipeline("iframe[30:90]")
        configs = pipeline_to_config(effects)
        assert configs[0]["effect"] == "iframe"
        assert configs[0]["start"] == 30
        assert configs[0]["end"] == 90

    def test_pframe(self):
        effects = parse_pipeline("pframe[dup=5]")
        configs = pipeline_to_config(effects)
        assert configs[0]["effect"] == "pframe"
        assert configs[0]["dup"] == 5

    def test_vector(self):
        effects = parse_pipeline("vector[jitter,strength=0.5]")
        configs = pipeline_to_config(effects)
        assert configs[0]["effect"] == "vector"
        assert configs[0]["transform"] == "jitter"
        assert configs[0]["params"]["strength"] == 0.5
