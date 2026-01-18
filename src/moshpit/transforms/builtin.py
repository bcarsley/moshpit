"""Built-in vector transforms for datamoshing."""

import numpy as np

from moshpit.transforms.registry import register_transform


def register_builtin_transforms() -> None:
    """Register all built-in transforms.

    This function is called during initialization to populate
    the transform registry with standard transforms.
    """
    pass  # Transforms are registered via decorators below


# ============================================================================
# Basic Transforms
# ============================================================================


@register_transform("horizontal-kill", "Zero out horizontal motion")
def horizontal_kill(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Kill all horizontal motion."""
    frame = frame.copy()
    if frame.ndim >= 3:
        frame[..., 0] = 0
    return frame


@register_transform("vertical-kill", "Zero out vertical motion")
def vertical_kill(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Kill all vertical motion."""
    frame = frame.copy()
    if frame.ndim >= 3:
        frame[..., 1] = 0
    return frame


@register_transform("amplify", "Scale motion vectors by factor")
def amplify(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Amplify motion by a factor."""
    factor = ctx.get("factor", 2.0)
    return (frame * factor).astype(frame.dtype)


@register_transform("invert", "Flip motion direction")
def invert(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Invert motion direction."""
    axis = ctx.get("axis", "both")
    frame = frame.copy()
    if frame.ndim >= 3:
        if axis in ("x", "both"):
            frame[..., 0] = -frame[..., 0]
        if axis in ("y", "both"):
            frame[..., 1] = -frame[..., 1]
    else:
        frame = -frame
    return frame


@register_transform("swap-axes", "Swap horizontal and vertical motion")
def swap_axes(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Swap X and Y motion components."""
    frame = frame.copy()
    if frame.ndim >= 3:
        frame[..., 0], frame[..., 1] = frame[..., 1].copy(), frame[..., 0].copy()
    return frame


# ============================================================================
# Noise / Stochastic Transforms
# ============================================================================


@register_transform("jitter", "Add random noise to motion")
def jitter(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Add random jitter to motion vectors."""
    strength = ctx.get("strength", 0.5)
    seed = ctx.get("seed")

    if seed is not None:
        rng = np.random.default_rng(seed + i)  # Different per frame but reproducible
    else:
        rng = np.random.default_rng()

    noise = rng.standard_normal(frame.shape) * strength
    return (frame + noise).astype(frame.dtype)


@register_transform("uniform", "Add uniform random noise")
def uniform_noise(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Add uniform random noise."""
    strength = ctx.get("strength", 1.0)
    seed = ctx.get("seed")

    if seed is not None:
        rng = np.random.default_rng(seed + i)
    else:
        rng = np.random.default_rng()

    noise = (rng.random(frame.shape) - 0.5) * 2 * strength
    return (frame + noise).astype(frame.dtype)


@register_transform("gaussian", "Add Gaussian noise")
def gaussian_noise(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Add Gaussian noise to motion vectors."""
    sigma = ctx.get("sigma", 1.0)
    seed = ctx.get("seed")

    if seed is not None:
        rng = np.random.default_rng(seed + i)
    else:
        rng = np.random.default_rng()

    noise = rng.normal(0, sigma, frame.shape)
    return (frame + noise).astype(frame.dtype)


@register_transform("perlin", "Apply Perlin-like coherent noise")
def perlin_noise(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Apply coherent noise (simplified Perlin-like)."""
    strength = ctx.get("strength", 2.0)
    seed = ctx.get("seed", 0)
    scale = ctx.get("scale", 8)  # Grid divisor (higher = smoother noise)

    if frame.ndim < 2:
        return frame

    h, w = frame.shape[:2]
    rng = np.random.default_rng(seed)

    # Generate smooth noise using interpolation
    # Low-res noise grid (scale controls smoothness)
    grid_h, grid_w = max(2, h // scale), max(2, w // scale)
    grid = rng.random((grid_h, grid_w, 2)) * 2 - 1

    # Interpolate to full size
    from scipy.ndimage import zoom

    try:
        noise = zoom(grid, (h / grid_h, w / grid_w, 1), order=1)
    except ImportError:
        # Fallback to simple noise if scipy not available
        noise = rng.random((h, w, 2)) * 2 - 1

    # Ensure proper shape
    if frame.ndim == 3 and frame.shape[2] >= 2:
        noise_shaped = np.zeros_like(frame, dtype=float)
        noise_shaped[..., :2] = noise[..., :2] * strength
        return (frame + noise_shaped).astype(frame.dtype)

    return frame


# ============================================================================
# Trigonometric Transforms
# ============================================================================


@register_transform("sine-wave", "Apply sinusoidal displacement")
def sine_wave(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Apply sine wave displacement."""
    freq = ctx.get("freq", 0.1)
    amp = ctx.get("amp", 5.0)
    axis = ctx.get("axis", "x")
    phase = ctx.get("phase", 0.0)

    if frame.ndim < 2:
        return frame

    h, w = frame.shape[:2]
    frame = frame.copy().astype(float)

    if axis == "x":
        for y in range(h):
            displacement = amp * np.sin(2 * np.pi * freq * y + phase)
            if frame.ndim >= 3:
                frame[y, :, 0] += displacement
            else:
                frame[y, :] += displacement
    else:  # y axis
        for x in range(w):
            displacement = amp * np.sin(2 * np.pi * freq * x + phase)
            if frame.ndim >= 3:
                frame[:, x, 1] += displacement
            else:
                frame[:, x] += displacement

    return frame.astype(np.int16) if frame.dtype == np.float64 else frame


@register_transform("cosine-wave", "Apply cosine wave displacement")
def cosine_wave(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Apply cosine wave displacement (phase-shifted sine)."""
    ctx = dict(ctx)
    ctx["phase"] = ctx.get("phase", 0.0) + np.pi / 2
    return sine_wave(frame, i, ctx)


@register_transform("spiral", "Apply radial spiral pattern")
def spiral(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Apply spiral/radial pattern."""
    freq = ctx.get("freq", 0.02)
    amp = ctx.get("amp", 3.0)

    if frame.ndim < 2:
        return frame

    h, w = frame.shape[:2]
    cy, cx = h / 2, w / 2
    frame = frame.copy().astype(float)

    for y in range(h):
        for x in range(w):
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            displacement = amp * np.sin(r * freq * 2 * np.pi)
            if frame.ndim >= 3 and frame.shape[2] >= 2:
                frame[y, x, 0] += displacement
                frame[y, x, 1] += displacement

    return frame.astype(np.int16) if frame.dtype == np.float64 else frame


# ============================================================================
# Quantization / Discrete Transforms
# ============================================================================


@register_transform("quantize", "Posterize motion to discrete levels")
def quantize(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Quantize motion vectors to discrete levels."""
    levels = ctx.get("levels", 4)

    if levels < 2:
        return frame

    # Find range of values
    vmin, vmax = frame.min(), frame.max()
    if vmax == vmin:
        return frame

    # Quantize
    step = (vmax - vmin) / levels
    quantized = np.round((frame - vmin) / step) * step + vmin
    return quantized.astype(frame.dtype)


@register_transform("threshold", "Binary threshold motion")
def threshold(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Apply binary threshold to motion vectors."""
    cutoff = ctx.get("cutoff", 0.5)
    high = ctx.get("high", 1.0)
    low = ctx.get("low", 0.0)

    magnitude = np.abs(frame)
    result = np.where(magnitude > cutoff, high * np.sign(frame), low)
    return result.astype(frame.dtype)


@register_transform("step", "Apply staircase function")
def step_function(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Apply staircase function to motion."""
    step_size = ctx.get("step_size", 2.0)
    return (np.floor(frame / step_size) * step_size).astype(frame.dtype)


@register_transform("modulo", "Wrap values at boundary")
def modulo(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Wrap motion values at boundary (creates discontinuities)."""
    boundary = ctx.get("boundary", 10.0)
    return (frame % boundary).astype(frame.dtype)


# ============================================================================
# Spatial / Gradient Transforms
# ============================================================================


@register_transform("radial-gradient", "Effect varies from center outward")
def radial_gradient(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Apply effect with strength varying from center."""
    inner_strength = ctx.get("inner_strength", 0.0)
    outer_strength = ctx.get("outer_strength", 1.0)

    if frame.ndim < 2:
        return frame

    h, w = frame.shape[:2]
    cy, cx = h / 2, w / 2
    max_r = np.sqrt(cx**2 + cy**2)

    result = frame.copy().astype(float)
    for y in range(h):
        for x in range(w):
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            t = r / max_r  # 0 at center, 1 at corners
            strength = inner_strength + t * (outer_strength - inner_strength)
            result[y, x] *= strength

    return result.astype(frame.dtype)


@register_transform("linear-gradient", "Effect varies along axis")
def linear_gradient(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Apply effect with strength varying along axis."""
    start_strength = ctx.get("start_strength", 0.0)
    end_strength = ctx.get("end_strength", 1.0)
    axis = ctx.get("axis", "y")

    if frame.ndim < 2:
        return frame

    h, w = frame.shape[:2]
    result = frame.copy().astype(float)

    if axis == "y":
        for y in range(h):
            t = y / (h - 1) if h > 1 else 0
            strength = start_strength + t * (end_strength - start_strength)
            result[y] *= strength
    else:  # x axis
        for x in range(w):
            t = x / (w - 1) if w > 1 else 0
            strength = start_strength + t * (end_strength - start_strength)
            result[:, x] *= strength

    return result.astype(frame.dtype)


@register_transform("vignette", "Stronger effect at edges")
def vignette(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Apply vignette effect (stronger at edges)."""
    # Invert radial gradient
    ctx = dict(ctx)
    ctx["inner_strength"] = ctx.get("inner_strength", 0.0)
    ctx["outer_strength"] = ctx.get("outer_strength", 2.0)
    return radial_gradient(frame, i, ctx)


@register_transform("region-mask", "Apply transform to specific region")
def region_mask(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Apply transform only to a specific region."""
    x = ctx.get("x", 0)
    y = ctx.get("y", 0)
    w = ctx.get("w", frame.shape[1] if frame.ndim >= 2 else 1)
    h_param = ctx.get("h", frame.shape[0] if frame.ndim >= 1 else 1)
    inner_transform = ctx.get("transform", "amplify")
    inner_params = ctx.get("transform_params", {})

    if frame.ndim < 2:
        return frame

    from moshpit.transforms.registry import get_transform

    result = frame.copy()
    region = frame[y : y + h_param, x : x + w].copy()

    # Apply inner transform to region
    inner_ctx = dict(ctx)
    inner_ctx.update(inner_params)
    transform_fn = get_transform(inner_transform)
    transformed_region = transform_fn(region, i, inner_ctx)

    result[y : y + h_param, x : x + w] = transformed_region
    return result


# ============================================================================
# Blur / Smoothing Transforms
# ============================================================================


@register_transform("blur", "Gaussian blur motion vectors spatially")
def blur(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Apply Gaussian blur to motion vectors."""
    kernel = ctx.get("kernel", 5)

    if frame.ndim < 2:
        return frame

    try:
        from scipy.ndimage import gaussian_filter

        sigma = kernel / 3.0
        if frame.ndim == 3:
            result = np.zeros_like(frame, dtype=float)
            for c in range(frame.shape[2]):
                result[..., c] = gaussian_filter(frame[..., c].astype(float), sigma)
            return result.astype(frame.dtype)
        else:
            return gaussian_filter(frame.astype(float), sigma).astype(frame.dtype)
    except ImportError:
        # Fallback: simple box blur
        return frame  # Skip blur if scipy not available


# ============================================================================
# Temporal Transforms
# ============================================================================


# These are stateful and require special handling
_temporal_state: dict = {}


@register_transform("feedback", "Blend with previous frame's vectors")
def feedback(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Blend current vectors with previous frame's vectors."""
    blend = ctx.get("blend", 0.5)
    state_key = ctx.get("_state_key", "default")

    key = f"feedback_{state_key}"
    if key not in _temporal_state or i == 0:
        _temporal_state[key] = frame.copy()
        return frame

    prev = _temporal_state[key]
    if prev.shape != frame.shape:
        _temporal_state[key] = frame.copy()
        return frame

    result = ((1 - blend) * frame + blend * prev).astype(frame.dtype)
    _temporal_state[key] = result.copy()
    return result


@register_transform("accumulate", "Sum vectors over time (trails)")
def accumulate(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Accumulate vectors over time."""
    decay = ctx.get("decay", 0.9)  # How much of previous to keep
    state_key = ctx.get("_state_key", "default")

    key = f"accumulate_{state_key}"
    if key not in _temporal_state or i == 0:
        _temporal_state[key] = frame.copy().astype(float)
        return frame

    acc = _temporal_state[key]
    if acc.shape != frame.shape:
        _temporal_state[key] = frame.copy().astype(float)
        return frame

    acc = acc * decay + frame
    _temporal_state[key] = acc.copy()
    return acc.astype(frame.dtype)


@register_transform("diff", "Use difference between consecutive frames")
def diff(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Output difference between consecutive frames."""
    state_key = ctx.get("_state_key", "default")

    key = f"diff_{state_key}"
    if key not in _temporal_state or i == 0:
        _temporal_state[key] = frame.copy()
        return np.zeros_like(frame)

    prev = _temporal_state[key]
    _temporal_state[key] = frame.copy()

    if prev.shape != frame.shape:
        return np.zeros_like(frame)

    return (frame - prev).astype(frame.dtype)


@register_transform("echo", "Delayed copy of vectors overlaid")
def echo(frame: np.ndarray, i: int, ctx: dict) -> np.ndarray:
    """Overlay delayed copy of vectors."""
    delay = ctx.get("delay", 5)  # Frames of delay
    strength = ctx.get("strength", 0.5)
    state_key = ctx.get("_state_key", "default")

    key = f"echo_{state_key}"
    if key not in _temporal_state:
        _temporal_state[key] = []

    history = _temporal_state[key]
    history.append(frame.copy())

    # Keep only needed history
    max_history = delay + 1
    if len(history) > max_history:
        history.pop(0)

    if len(history) <= delay:
        return frame

    delayed = history[0]
    if delayed.shape != frame.shape:
        return frame

    return (frame + strength * delayed).astype(frame.dtype)


def clear_temporal_state() -> None:
    """Clear all temporal transform state.

    Call this between processing different videos to reset
    stateful transforms like feedback and accumulate.
    """
    global _temporal_state
    _temporal_state = {}
