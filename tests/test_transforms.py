"""Tests for vector transforms."""

import numpy as np
import pytest

from moshpit.transforms.builtin import (
    amplify,
    horizontal_kill,
    invert,
    jitter,
    quantize,
    register_builtin_transforms,
    sine_wave,
    swap_axes,
    threshold,
    vertical_kill,
)
from moshpit.transforms.registry import get_transform, list_transforms, transform_exists


# Ensure transforms are registered
register_builtin_transforms()


class TestRegistry:
    def test_transform_exists(self):
        assert transform_exists("jitter")
        assert transform_exists("amplify")
        assert not transform_exists("nonexistent")

    def test_get_transform(self):
        fn = get_transform("jitter")
        assert callable(fn)

    def test_get_unknown_transform(self):
        with pytest.raises(KeyError):
            get_transform("nonexistent")

    def test_list_transforms(self):
        transforms = list_transforms()
        assert len(transforms) > 0
        # Check it returns tuples of (name, description)
        for name, desc in transforms:
            assert isinstance(name, str)
            assert isinstance(desc, str)


class TestBasicTransforms:
    @pytest.fixture
    def sample_frame(self):
        """Create a sample motion vector frame."""
        # Shape: (height, width, 2) for x/y components
        return np.random.randn(10, 10, 2).astype(np.float32)

    def test_horizontal_kill(self, sample_frame):
        result = horizontal_kill(sample_frame, 0, {})
        assert np.all(result[..., 0] == 0)  # X component should be zero
        assert not np.all(result[..., 1] == 0)  # Y component preserved

    def test_vertical_kill(self, sample_frame):
        result = vertical_kill(sample_frame, 0, {})
        assert not np.all(result[..., 0] == 0)  # X component preserved
        assert np.all(result[..., 1] == 0)  # Y component should be zero

    def test_amplify(self, sample_frame):
        factor = 2.0
        result = amplify(sample_frame, 0, {"factor": factor})
        np.testing.assert_allclose(result, sample_frame * factor)

    def test_invert_both(self, sample_frame):
        result = invert(sample_frame, 0, {"axis": "both"})
        np.testing.assert_allclose(result, -sample_frame)

    def test_invert_x(self, sample_frame):
        result = invert(sample_frame, 0, {"axis": "x"})
        assert np.all(result[..., 0] == -sample_frame[..., 0])
        assert np.all(result[..., 1] == sample_frame[..., 1])

    def test_swap_axes(self, sample_frame):
        result = swap_axes(sample_frame, 0, {})
        np.testing.assert_allclose(result[..., 0], sample_frame[..., 1])
        np.testing.assert_allclose(result[..., 1], sample_frame[..., 0])


class TestNoiseTransforms:
    @pytest.fixture
    def sample_frame(self):
        return np.zeros((10, 10, 2), dtype=np.float32)

    def test_jitter_adds_noise(self, sample_frame):
        result = jitter(sample_frame, 0, {"strength": 1.0})
        assert not np.allclose(result, sample_frame)

    def test_jitter_reproducible(self, sample_frame):
        result1 = jitter(sample_frame, 0, {"strength": 1.0, "seed": 42})
        result2 = jitter(sample_frame, 0, {"strength": 1.0, "seed": 42})
        np.testing.assert_allclose(result1, result2)


class TestQuantizationTransforms:
    @pytest.fixture
    def sample_frame(self):
        return np.linspace(-10, 10, 100).reshape(10, 10).astype(np.float32)

    def test_quantize(self, sample_frame):
        result = quantize(sample_frame, 0, {"levels": 4})
        unique_values = np.unique(result)
        # With 4 levels, we get at most 5 unique values (4 intervals + boundary)
        assert len(unique_values) <= 5
        # Values should be fewer than original
        assert len(unique_values) < len(np.unique(sample_frame))

    def test_threshold(self):
        frame = np.array([[-2, -1, 0, 1, 2]], dtype=np.float32)
        result = threshold(frame, 0, {"cutoff": 0.5, "high": 1.0, "low": 0.0})
        # Values above cutoff get high * sign, below get low
        # abs(-2)=2 > 0.5 -> -1, abs(-1)=1 > 0.5 -> -1, abs(0)=0 < 0.5 -> 0
        expected = np.array([[-1, -1, 0, 1, 1]], dtype=np.float32)
        np.testing.assert_allclose(result, expected)


class TestTrigTransforms:
    @pytest.fixture
    def sample_frame(self):
        return np.zeros((10, 10, 2), dtype=np.float32)

    def test_sine_wave_adds_displacement(self, sample_frame):
        result = sine_wave(sample_frame, 0, {"freq": 0.1, "amp": 5.0, "axis": "x"})
        assert not np.allclose(result, sample_frame)
        # X component should be modified
        assert not np.allclose(result[..., 0], sample_frame[..., 0])
