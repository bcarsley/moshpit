"""Parameter sweep generation for creating variations."""

import copy
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass
class SweepParameter:
    """A parameter to vary in a sweep."""

    path: str  # e.g., "pipeline[1].kernel" or "pframe.dup"
    values: list[Any]  # Explicit values to use


def parse_vary_spec(spec: str) -> SweepParameter:
    """Parse a --vary specification string.

    Formats:
        - "param=start:end" - Linear range
        - "param=start:end:step" - Linear range with step
        - "param=v1,v2,v3" - Explicit values

    Args:
        spec: Vary specification string

    Returns:
        SweepParameter object
    """
    if "=" not in spec:
        raise ValueError(f"Invalid vary spec (missing '='): {spec}")

    path, value_spec = spec.split("=", 1)
    path = path.strip()
    value_spec = value_spec.strip()

    values: list[Any] = []

    if ":" in value_spec:
        # Range specification
        parts = value_spec.split(":")
        if len(parts) == 2:
            start, end = float(parts[0]), float(parts[1])
            step = (end - start) / 10  # Default 10 steps
        elif len(parts) == 3:
            start, end, step = float(parts[0]), float(parts[1]), float(parts[2])
        else:
            raise ValueError(f"Invalid range spec: {value_spec}")

        # Generate values
        current = start
        while current <= end:
            values.append(current)
            current += step

        # Convert to int if all values are whole numbers
        if all(v == int(v) for v in values):
            values = [int(v) for v in values]

    elif "," in value_spec:
        # Explicit values
        for v in value_spec.split(","):
            v = v.strip()
            # Try to parse as number
            try:
                if "." in v:
                    values.append(float(v))
                else:
                    values.append(int(v))
            except ValueError:
                values.append(v)  # Keep as string
    else:
        # Single value
        try:
            if "." in value_spec:
                values.append(float(value_spec))
            else:
                values.append(int(value_spec))
        except ValueError:
            values.append(value_spec)

    return SweepParameter(path=path, values=values)


def set_nested_value(obj: dict, path: str, value: Any) -> None:
    """Set a value in a nested dictionary using dot notation.

    Args:
        obj: Dictionary to modify
        path: Path like "pipeline[1].kernel" or "pframe.dup"
        value: Value to set
    """
    parts = re.split(r"\.|\[|\]", path)
    parts = [p for p in parts if p]  # Remove empty strings

    current = obj
    for i, part in enumerate(parts[:-1]):
        # Check if this is an array index
        try:
            idx = int(part)
            current = current[idx]
        except (ValueError, TypeError):
            if part not in current:
                current[part] = {}
            current = current[part]

    # Set the final value
    final_key = parts[-1]
    try:
        idx = int(final_key)
        current[idx] = value
    except (ValueError, TypeError):
        current[final_key] = value


def get_nested_value(obj: dict, path: str) -> Any:
    """Get a value from a nested dictionary using dot notation.

    Args:
        obj: Dictionary to read from
        path: Path like "pipeline[1].kernel"

    Returns:
        Value at path
    """
    parts = re.split(r"\.|\[|\]", path)
    parts = [p for p in parts if p]

    current = obj
    for part in parts:
        try:
            idx = int(part)
            current = current[idx]
        except (ValueError, TypeError):
            current = current[part]

    return current


class ParameterSweep:
    """Generates parameter variations for sweep runs."""

    def __init__(
        self,
        base_config: dict[str, Any],
        parameters: list[SweepParameter],
        count: int | None = None,
        mode: str = "grid",  # "grid" or "random"
        seed: int | None = None,
    ):
        """Initialize a parameter sweep.

        Args:
            base_config: Base configuration to vary
            parameters: Parameters to sweep over
            count: Maximum number of variations (for random mode)
            mode: "grid" for exhaustive, "random" for random sampling
            seed: Random seed for reproducibility
        """
        self.base_config = base_config
        self.parameters = parameters
        self.count = count
        self.mode = mode
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[int, dict[str, Any]]]:
        """Iterate over parameter combinations.

        Yields:
            Tuple of (index, config) for each variation
        """
        if self.mode == "grid":
            yield from self._grid_sweep()
        else:
            yield from self._random_sweep()

    def _grid_sweep(self) -> Iterator[tuple[int, dict[str, Any]]]:
        """Generate all combinations (grid sweep)."""
        if not self.parameters:
            yield 0, copy.deepcopy(self.base_config)
            return

        # Get all value lists
        value_lists = [p.values for p in self.parameters]

        # Generate combinations
        for i, combo in enumerate(itertools.product(*value_lists)):
            if self.count is not None and i >= self.count:
                break

            config = copy.deepcopy(self.base_config)
            for param, value in zip(self.parameters, combo):
                set_nested_value(config, param.path, value)

            yield i, config

    def _random_sweep(self) -> Iterator[tuple[int, dict[str, Any]]]:
        """Generate random combinations."""
        rng = np.random.default_rng(self.seed)
        count = self.count or 10

        for i in range(count):
            config = copy.deepcopy(self.base_config)

            for param in self.parameters:
                value = rng.choice(param.values)
                set_nested_value(config, param.path, value)

            yield i, config

    @property
    def total_combinations(self) -> int:
        """Get total number of possible combinations (for grid mode)."""
        if not self.parameters:
            return 1

        total = 1
        for param in self.parameters:
            total *= len(param.values)

        if self.count is not None:
            return min(total, self.count)
        return total


def generate_sweep_configs(
    base_config: dict[str, Any],
    vary_specs: list[str],
    count: int | None = None,
    mode: str = "grid",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate sweep configurations from vary specifications.

    Args:
        base_config: Base configuration
        vary_specs: List of --vary specification strings
        count: Maximum variations
        mode: "grid" or "random"
        seed: Random seed

    Returns:
        List of configuration dictionaries
    """
    parameters = [parse_vary_spec(spec) for spec in vary_specs]
    sweep = ParameterSweep(
        base_config,
        parameters,
        count=count,
        mode=mode,
        seed=seed,
    )
    return [config for _, config in sweep]


def generate_output_path(
    base_path: Path | str,
    index: int,
    params: dict[str, Any] | None = None,
) -> Path:
    """Generate an output path for a sweep variation.

    Args:
        base_path: Base output path or directory
        index: Variation index
        params: Optional parameter values to include in filename

    Returns:
        Output path for this variation
    """
    base_path = Path(base_path)

    if base_path.is_dir() or not base_path.suffix:
        # Directory - generate filename
        output_dir = base_path
        output_dir.mkdir(parents=True, exist_ok=True)

        if params:
            # Include param values in filename
            param_str = "_".join(f"{k}{v}" for k, v in params.items())
            filename = f"variation_{index:03d}_{param_str}.mp4"
        else:
            filename = f"variation_{index:03d}.mp4"

        return output_dir / filename
    else:
        # File - insert index before extension
        stem = base_path.stem
        suffix = base_path.suffix
        return base_path.parent / f"{stem}_{index:03d}{suffix}"
