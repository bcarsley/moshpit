"""Inline expression evaluator for vector transforms.

Allows users to write simple Python expressions for quick transforms
without creating full transform functions.
"""

from typing import Any

import numpy as np

ALLOWED_NUMPY_FUNCS = {
    # Math functions
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round,
    "clip": np.clip,
    "sign": np.sign,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "power": np.power,
    "mod": np.mod,
    # Array functions
    "zeros_like": np.zeros_like,
    "ones_like": np.ones_like,
    "where": np.where,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "mean": np.mean,
    "std": np.std,
    "sum": np.sum,
    # Random
    "random": np.random.random,
    "randn": np.random.randn,
    # Constants
    "pi": np.pi,
    "e": np.e,
}


def create_expression_transform(expression: str):
    """Create a transform function from an expression string.

    The expression has access to:
    - frame: The motion vector array (numpy)
    - i: Frame index
    - ctx: Context dictionary
    - All functions in ALLOWED_NUMPY_FUNCS

    The expression should modify 'frame' in place or assign to it.

    Examples:
        "frame *= 2"
        "frame[:,:,0] = 0"
        "frame += randn(*frame.shape) * 0.5"
        "frame = where(abs(frame) > 1, frame, 0)"

    Args:
        expression: Python expression string

    Returns:
        Transform function
    """

    def transform(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
        # Make a copy to work with
        local_frame = frame.copy()

        # Build namespace
        namespace: dict[str, Any] = {
            "frame": local_frame,
            "i": i,
            "ctx": ctx,
            "np": np,
            **ALLOWED_NUMPY_FUNCS,
        }

        # Add frame shape info for convenience
        if frame.ndim >= 2:
            namespace["h"], namespace["w"] = frame.shape[:2]
        if frame.ndim >= 3:
            namespace["c"] = frame.shape[2]

        try:
            # Execute expression
            exec(expression, {"__builtins__": {}}, namespace)
            return namespace["frame"]
        except Exception as e:
            raise ValueError(f"Expression evaluation failed: {e}") from e

    return transform


def validate_expression(expression: str) -> tuple[bool, str]:
    """Validate an expression without executing it.

    Args:
        expression: Expression to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Try to compile the expression
        compile(expression, "<expression>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, str(e)


def parse_expression_params(expr_string: str) -> tuple[str, dict]:
    """Parse an expression string that may include parameters.

    Format: "expression; param1=value1; param2=value2"

    Args:
        expr_string: Expression with optional parameters

    Returns:
        Tuple of (expression, params_dict)
    """
    parts = [p.strip() for p in expr_string.split(";")]
    expression = parts[0]
    params = {}

    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Try to parse as number
            try:
                if "." in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                params[key] = value

    return expression, params
