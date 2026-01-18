"""Configuration loading and validation."""

from pathlib import Path
from typing import Any

import yaml

from moshpit.config.schema import RecipeConfig, parse_pipeline_step


def load_recipe(path: Path | str) -> RecipeConfig:
    """Load a recipe from a YAML file.

    Args:
        path: Path to YAML recipe file

    Returns:
        Validated RecipeConfig
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Recipe file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid recipe format in {path}")

    # Parse pipeline steps
    if "pipeline" in data:
        data["pipeline"] = [parse_pipeline_step(step) for step in data["pipeline"]]

    return RecipeConfig(**data)


def save_recipe(recipe: RecipeConfig, path: Path | str) -> None:
    """Save a recipe to a YAML file.

    Args:
        recipe: Recipe configuration to save
        path: Output path for YAML file
    """
    path = Path(path)

    # Convert to dict for YAML serialization
    data = recipe.model_dump(exclude_none=True)

    # Convert pipeline steps to dicts
    if "pipeline" in data:
        data["pipeline"] = [
            step.model_dump(exclude_none=True) if hasattr(step, "model_dump") else step
            for step in recipe.pipeline
        ]

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_recipe_by_name(name: str, recipes_dir: Path | str | None = None) -> RecipeConfig:
    """Load a recipe by name from the recipes directory.

    Args:
        name: Recipe name (without .yaml extension)
        recipes_dir: Optional custom recipes directory

    Returns:
        Validated RecipeConfig
    """
    if recipes_dir is None:
        # Default to recipes/ in current directory or package
        recipes_dir = Path.cwd() / "recipes"

    recipes_dir = Path(recipes_dir)

    # Try with and without .yaml extension
    yaml_path = recipes_dir / f"{name}.yaml"
    if yaml_path.exists():
        return load_recipe(yaml_path)

    yml_path = recipes_dir / f"{name}.yml"
    if yml_path.exists():
        return load_recipe(yml_path)

    # Check if name is already a path
    if Path(name).exists():
        return load_recipe(name)

    raise FileNotFoundError(
        f"Recipe '{name}' not found in {recipes_dir}. "
        f"Available: {list_available_recipes(recipes_dir)}"
    )


def list_available_recipes(recipes_dir: Path | str | None = None) -> list[str]:
    """List available recipes in the recipes directory.

    Args:
        recipes_dir: Optional custom recipes directory

    Returns:
        List of recipe names (without extension)
    """
    if recipes_dir is None:
        recipes_dir = Path.cwd() / "recipes"

    recipes_dir = Path(recipes_dir)
    if not recipes_dir.exists():
        return []

    recipes = []
    for path in recipes_dir.iterdir():
        if path.suffix in (".yaml", ".yml"):
            recipes.append(path.stem)

    return sorted(recipes)


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Override configuration (takes precedence)

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result
