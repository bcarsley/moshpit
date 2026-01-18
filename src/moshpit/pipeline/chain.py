"""Effect chaining for pipeline execution."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from moshpit.config.schema import (
    PipelineStep,
)
from moshpit.core.iframe import remove_iframes
from moshpit.core.pframe import duplicate_pframes
from moshpit.core.transfer import transfer_motion
from moshpit.core.vectors import apply_vector_transform
from moshpit.io.video import PreviewSettings
from moshpit.pipeline.parser import ParsedEffect, pipeline_to_config


@dataclass
class ChainContext:
    """Context passed through the effect chain."""

    input_path: Path
    output_path: Path
    preview: PreviewSettings | None = None
    seed: int | None = None
    verbose: bool = False
    dry_run: bool = False
    temp_dir: Path | None = None


class EffectChain:
    """Manages a chain of effects to apply to a video."""

    def __init__(self, effects: list[PipelineStep] | list[ParsedEffect] | list[dict]):
        """Initialize the effect chain.

        Args:
            effects: List of effects (PipelineStep, ParsedEffect, or dict)
        """
        self.effects = self._normalize_effects(effects)

    def _normalize_effects(
        self, effects: list[PipelineStep] | list[ParsedEffect] | list[dict]
    ) -> list[dict[str, Any]]:
        """Normalize effects to dictionaries.

        Args:
            effects: Effects in various formats

        Returns:
            List of effect dictionaries
        """
        normalized = []
        for effect in effects:
            if isinstance(effect, dict):
                normalized.append(effect)
            elif isinstance(effect, ParsedEffect):
                # Convert ParsedEffect to dict
                normalized.extend(pipeline_to_config([effect]))
            elif hasattr(effect, "model_dump"):
                # Pydantic model
                normalized.append(effect.model_dump())
            else:
                raise ValueError(f"Unknown effect type: {type(effect)}")
        return normalized

    def execute(
        self,
        input_path: Path | str,
        output_path: Path | str,
        preview: PreviewSettings | None = None,
        seed: int | None = None,
        verbose: bool = False,
        dry_run: bool = False,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> Path:
        """Execute the effect chain.

        Args:
            input_path: Path to input video
            output_path: Path for output video
            preview: Preview settings for fast rendering
            seed: Random seed for reproducibility
            verbose: Enable verbose logging
            dry_run: Show what would happen without executing
            progress_callback: Optional callback(current, total, message)

        Returns:
            Path to the output video
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not self.effects:
            # No effects, just copy
            import shutil

            shutil.copy(input_path, output_path)
            return output_path

        if dry_run:
            self._print_dry_run()
            return output_path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            current_input = input_path
            total_effects = len(self.effects)

            for i, effect in enumerate(self.effects):
                effect_name = effect.get("effect", "unknown")

                if progress_callback:
                    progress_callback(i + 1, total_effects, f"Applying {effect_name}")

                # Determine output path for this step
                if i == total_effects - 1:
                    # Last effect, write to final output
                    step_output = output_path
                else:
                    # Intermediate step, write to temp file
                    step_output = tmpdir / f"step_{i}.mp4"

                # Apply the effect
                current_input = self._apply_effect(
                    effect,
                    current_input,
                    step_output,
                    preview=preview,
                    seed=seed,
                    verbose=verbose,
                )

        return output_path

    def _apply_effect(
        self,
        effect: dict[str, Any],
        input_path: Path,
        output_path: Path,
        preview: PreviewSettings | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ) -> Path:
        """Apply a single effect.

        Args:
            effect: Effect configuration dictionary
            input_path: Input video path
            output_path: Output video path
            preview: Preview settings
            seed: Random seed
            verbose: Verbose logging

        Returns:
            Path to output video
        """
        effect_type = effect.get("effect", "")

        if effect_type == "iframe":
            return remove_iframes(
                input_path,
                output_path,
                start_frame=effect.get("start"),
                end_frame=effect.get("end"),
                keep_first=effect.get("keep_first", True),
                preview=preview,
            )

        elif effect_type == "pframe":
            return duplicate_pframes(
                input_path,
                output_path,
                dup_count=effect.get("dup", 3),
                start_frame=effect.get("start"),
                preview=preview,
            )

        elif effect_type == "vector":
            transform = effect.get("transform", "jitter")
            params = effect.get("params", {})

            # Add seed if provided
            if seed is not None:
                params["seed"] = seed

            return apply_vector_transform(
                input_path,
                output_path,
                transform=transform,
                preview=preview,
                **params,
            )

        elif effect_type == "transfer":
            source = effect.get("source", "")
            if not source:
                raise ValueError("Transfer effect requires 'source' parameter")

            return transfer_motion(
                source_path=source,
                target_path=input_path,
                output_path=output_path,
                blend=effect.get("blend", 1.0),
                preview=preview,
            )

        else:
            raise ValueError(f"Unknown effect type: {effect_type}")

    def _print_dry_run(self) -> None:
        """Print what would happen in dry-run mode."""
        print("Would execute the following effects:")
        for i, effect in enumerate(self.effects):
            effect_type = effect.get("effect", "unknown")
            params = {k: v for k, v in effect.items() if k != "effect"}
            print(f"  {i + 1}. {effect_type}: {params}")


def execute_pipeline(
    input_path: Path | str,
    output_path: Path | str,
    pipeline_str: str,
    preview: PreviewSettings | None = None,
    seed: int | None = None,
    verbose: bool = False,
    dry_run: bool = False,
) -> Path:
    """Execute a pipeline from a DSL string.

    Args:
        input_path: Path to input video
        output_path: Path for output video
        pipeline_str: Pipeline DSL string
        preview: Preview settings
        seed: Random seed
        verbose: Verbose logging
        dry_run: Dry run mode

    Returns:
        Path to output video
    """
    from moshpit.pipeline.parser import parse_pipeline

    effects = parse_pipeline(pipeline_str)
    chain = EffectChain(effects)
    return chain.execute(
        input_path,
        output_path,
        preview=preview,
        seed=seed,
        verbose=verbose,
        dry_run=dry_run,
    )
