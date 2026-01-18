"""Pydantic schemas for YAML configuration."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class OutputConfig(BaseModel):
    """Output video settings."""

    format: str = "mp4"
    codec: str = "libx264"
    crf: int = Field(default=18, ge=0, le=51)
    resolution: str | None = None  # e.g., "1080p", "720p", "480p", "original"


class IframeEffect(BaseModel):
    """I-frame removal effect configuration."""

    effect: Literal["iframe"] = "iframe"
    start: int | None = None
    end: int | None = None
    keep_first: bool = True


class PframeEffect(BaseModel):
    """P-frame duplication effect configuration."""

    effect: Literal["pframe"] = "pframe"
    dup: int = Field(default=3, ge=1, le=10)
    start: int | None = None


class VectorEffect(BaseModel):
    """Vector transform effect configuration."""

    effect: Literal["vector"] = "vector"
    transform: str
    params: dict[str, Any] = Field(default_factory=dict)


class TransferEffect(BaseModel):
    """Motion transfer effect configuration."""

    effect: Literal["transfer"] = "transfer"
    source: str  # Path to source video
    blend: float = Field(default=1.0, ge=0.0, le=1.0)


# Union of all effect types
PipelineStep = IframeEffect | PframeEffect | VectorEffect | TransferEffect


class SweepParameter(BaseModel):
    """Parameter to vary in a sweep."""

    path: str  # e.g., "pipeline[1].kernel"
    range: tuple[float, float] | list[float]  # [min, max] or list of values
    step: float | None = None  # Step size for range


class SweepConfig(BaseModel):
    """Sweep configuration for generating variations."""

    variations: int = Field(default=10, ge=1, le=100)
    parameters: list[SweepParameter] = Field(default_factory=list)


class RecipeConfig(BaseModel):
    """Complete recipe configuration."""

    name: str
    description: str = ""

    input: dict[str, str] = Field(default_factory=lambda: {"format": "auto"})
    output: OutputConfig = Field(default_factory=OutputConfig)

    pipeline: list[PipelineStep] = Field(default_factory=list)
    sweep: SweepConfig | None = None

    model_config = ConfigDict(extra="allow")  # Allow additional fields for extensibility


def parse_pipeline_step(data: dict[str, Any]) -> PipelineStep:
    """Parse a dictionary into the appropriate effect type.

    Args:
        data: Dictionary with effect configuration

    Returns:
        Appropriate effect model instance
    """
    effect_type = data.get("effect", "")

    if effect_type == "iframe":
        return IframeEffect(**data)
    elif effect_type == "pframe":
        return PframeEffect(**data)
    elif effect_type == "vector":
        return VectorEffect(**data)
    elif effect_type == "transfer":
        return TransferEffect(**data)
    else:
        raise ValueError(f"Unknown effect type: {effect_type}")


def validate_pipeline(pipeline: list[dict]) -> list[PipelineStep]:
    """Validate and parse a pipeline configuration.

    Args:
        pipeline: List of effect dictionaries

    Returns:
        List of validated PipelineStep objects
    """
    return [parse_pipeline_step(step) for step in pipeline]
