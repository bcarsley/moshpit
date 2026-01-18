"""Pipeline DSL parser.

Parses the pipeline syntax:
    effect[params] -> effect[params] -> ...

Examples:
    iframe[30:90] -> pframe[dup=5]
    vector[transform=sine,freq=0.1,axis=x] -> iframe[50:100]
    transfer[source=clouds.mp4] -> vector[jitter,strength=0.3]
"""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedEffect:
    """A parsed effect from the DSL."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    range_start: int | None = None
    range_end: int | None = None


def parse_value(value: str) -> Any:
    """Parse a parameter value to appropriate type.

    Args:
        value: String value to parse

    Returns:
        Parsed value (int, float, bool, or str)
    """
    value = value.strip()

    # Boolean
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # String (remove quotes if present)
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    return value


def parse_params(params_str: str) -> tuple[dict[str, Any], int | None, int | None]:
    """Parse the parameters portion of an effect.

    Args:
        params_str: String inside brackets, e.g., "30:90" or "dup=5,start=10"

    Returns:
        Tuple of (params_dict, range_start, range_end)
    """
    params: dict[str, Any] = {}
    range_start: int | None = None
    range_end: int | None = None

    if not params_str:
        return params, range_start, range_end

    # Check for range syntax first (e.g., "30:90")
    range_match = re.match(r"^(\d+)?:(\d+)?$", params_str.strip())
    if range_match:
        if range_match.group(1):
            range_start = int(range_match.group(1))
        if range_match.group(2):
            range_end = int(range_match.group(2))
        return params, range_start, range_end

    # Parse comma-separated params
    # Handle nested brackets and quoted strings
    parts = []
    current = ""
    bracket_depth = 0
    in_quotes = False
    quote_char = None

    for char in params_str:
        if char in ('"', "'") and not in_quotes:
            in_quotes = True
            quote_char = char
            current += char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
            current += char
        elif char == "[":
            bracket_depth += 1
            current += char
        elif char == "]":
            bracket_depth -= 1
            current += char
        elif char == "," and bracket_depth == 0 and not in_quotes:
            parts.append(current.strip())
            current = ""
        else:
            current += char

    if current.strip():
        parts.append(current.strip())

    # Parse each part
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            params[key.strip()] = parse_value(value)
        else:
            # Positional parameter - could be transform name or flag
            # First positional is typically the transform name for vector effects
            if "transform" not in params and part:
                params["transform"] = part
            else:
                # Treat as boolean flag
                params[part] = True

    return params, range_start, range_end


def parse_effect(effect_str: str) -> ParsedEffect:
    """Parse a single effect string.

    Args:
        effect_str: Effect string like "iframe[30:90]" or "vector[jitter,strength=0.5]"

    Returns:
        ParsedEffect object
    """
    effect_str = effect_str.strip()

    # Match effect name and optional params
    match = re.match(r"^(\w+)(?:\[(.*)\])?$", effect_str)
    if not match:
        raise ValueError(f"Invalid effect syntax: {effect_str}")

    name = match.group(1)
    params_str = match.group(2) or ""

    params, range_start, range_end = parse_params(params_str)

    return ParsedEffect(
        name=name,
        params=params,
        range_start=range_start,
        range_end=range_end,
    )


def parse_pipeline(pipeline_str: str) -> list[ParsedEffect]:
    """Parse a complete pipeline string.

    Args:
        pipeline_str: Pipeline string like "iframe[30:90] -> vector[jitter] -> pframe[dup=3]"

    Returns:
        List of ParsedEffect objects
    """
    # Split on arrow (with flexible whitespace)
    parts = re.split(r"\s*->\s*", pipeline_str.strip())

    effects = []
    for part in parts:
        if part:
            effects.append(parse_effect(part))

    return effects


def pipeline_to_config(effects: list[ParsedEffect]) -> list[dict[str, Any]]:
    """Convert parsed effects to configuration dictionaries.

    Args:
        effects: List of ParsedEffect objects

    Returns:
        List of configuration dictionaries suitable for RecipeConfig
    """
    configs = []

    for effect in effects:
        config: dict[str, Any] = {"effect": effect.name}

        # Add range parameters for iframe/pframe effects
        if effect.name == "iframe":
            if effect.range_start is not None:
                config["start"] = effect.range_start
            if effect.range_end is not None:
                config["end"] = effect.range_end
            # Add any other params
            for key, value in effect.params.items():
                if key not in ("start", "end"):
                    config[key] = value

        elif effect.name == "pframe":
            if effect.range_start is not None:
                config["start"] = effect.range_start
            # Map common params
            for key, value in effect.params.items():
                config[key] = value

        elif effect.name == "vector":
            # Transform name should be in params
            config["transform"] = effect.params.get("transform", "jitter")
            # All other params go into params dict
            config["params"] = {k: v for k, v in effect.params.items() if k != "transform"}

        elif effect.name == "transfer":
            config["source"] = effect.params.get("source", "")
            if "blend" in effect.params:
                config["blend"] = effect.params["blend"]

        else:
            # Unknown effect, pass through
            config.update(effect.params)

        configs.append(config)

    return configs


def format_pipeline(effects: list[ParsedEffect]) -> str:
    """Format parsed effects back to pipeline string.

    Args:
        effects: List of ParsedEffect objects

    Returns:
        Pipeline string
    """
    parts = []

    for effect in effects:
        params_parts = []

        # Add range if present
        if effect.range_start is not None or effect.range_end is not None:
            start = effect.range_start if effect.range_start is not None else ""
            end = effect.range_end if effect.range_end is not None else ""
            params_parts.append(f"{start}:{end}")
        else:
            # Add named params
            for key, value in effect.params.items():
                if isinstance(value, bool):
                    if value:
                        params_parts.append(key)
                elif isinstance(value, str) and " " in value:
                    params_parts.append(f'{key}="{value}"')
                else:
                    params_parts.append(f"{key}={value}")

        if params_parts:
            parts.append(f"{effect.name}[{','.join(params_parts)}]")
        else:
            parts.append(effect.name)

    return " -> ".join(parts)
