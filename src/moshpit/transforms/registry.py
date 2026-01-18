"""Transform registry for managing vector transforms."""

from typing import Callable

import numpy as np

# Type alias for transform functions
# Transform signature: (frame: np.ndarray, frame_index: int, context: dict) -> np.ndarray
TransformFunc = Callable[[np.ndarray, int, dict], np.ndarray]


class TransformRegistry:
    """Registry for vector transform functions."""

    def __init__(self):
        self._transforms: dict[str, TransformFunc] = {}
        self._descriptions: dict[str, str] = {}

    def register(
        self,
        name: str,
        func: TransformFunc,
        description: str = "",
    ) -> None:
        """Register a transform function.

        Args:
            name: Name to register the transform under
            func: Transform function
            description: Human-readable description
        """
        self._transforms[name] = func
        self._descriptions[name] = description

    def get(self, name: str) -> TransformFunc:
        """Get a transform function by name.

        Args:
            name: Transform name

        Returns:
            Transform function

        Raises:
            KeyError: If transform not found
        """
        if name not in self._transforms:
            available = ", ".join(sorted(self._transforms.keys()))
            raise KeyError(f"Unknown transform '{name}'. Available: {available}")
        return self._transforms[name]

    def list(self) -> list[tuple[str, str]]:
        """List all registered transforms.

        Returns:
            List of (name, description) tuples
        """
        return [
            (name, self._descriptions.get(name, "")) for name in sorted(self._transforms.keys())
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._transforms


# Global registry instance
_registry = TransformRegistry()


def register_transform(
    name: str,
    description: str = "",
) -> Callable[[TransformFunc], TransformFunc]:
    """Decorator to register a transform function.

    Args:
        name: Name to register under
        description: Human-readable description

    Returns:
        Decorator function
    """

    def decorator(func: TransformFunc) -> TransformFunc:
        _registry.register(name, func, description)
        return func

    return decorator


def get_transform(name: str) -> TransformFunc:
    """Get a transform function by name.

    Args:
        name: Transform name

    Returns:
        Transform function
    """
    return _registry.get(name)


def list_transforms() -> list[tuple[str, str]]:
    """List all registered transforms.

    Returns:
        List of (name, description) tuples
    """
    return _registry.list()


def transform_exists(name: str) -> bool:
    """Check if a transform is registered.

    Args:
        name: Transform name

    Returns:
        True if transform exists
    """
    return name in _registry
