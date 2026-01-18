"""Moshpit CLI - Creative video datamoshing tool."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from moshpit import __version__
from moshpit.io.video import PreviewSettings

app = typer.Typer(
    name="moshpit",
    help="Creative video datamoshing CLI tool.",
    no_args_is_help=True,
)

console = Console()


# Common options
def version_callback(value: bool):
    if value:
        console.print(f"moshpit version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
):
    """Moshpit - A CLI tool for creative video datamoshing."""
    pass


# ============================================================================
# I-frame removal command
# ============================================================================


@app.command()
def iframe(
    input_path: Annotated[Path, typer.Argument(help="Input video file")],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output video file")] = None,
    start: Annotated[Optional[int], typer.Option("--start", help="Start frame")] = None,
    end: Annotated[Optional[int], typer.Option("--end", help="End frame")] = None,
    keep_first: Annotated[
        bool, typer.Option("--keep-first/--no-keep-first", help="Keep first I-frame")
    ] = True,
    preview: Annotated[bool, typer.Option("--preview", help="Fast preview mode (480p)")] = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Verbose output")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would happen")] = False,
    info: Annotated[bool, typer.Option("--info", help="Show I-frame info only")] = False,
):
    """Remove I-frames from a video to create datamosh effect.

    The classic datamoshing technique that creates "pixel bleeding" where
    motion from one scene bleeds into imagery from a previous scene.
    """
    if info:
        from moshpit.core.iframe import get_iframe_info

        info_data = get_iframe_info(input_path)
        console.print(f"[bold]I-frame Info for {input_path.name}[/bold]")
        console.print(f"  Total frames: {info_data['total_frames']}")
        console.print(
            f"  I-frames: {info_data['iframe_count']} ({info_data['iframe_percentage']:.1f}%)"
        )
        console.print(f"  P-frames: {info_data['pframe_count']}")
        console.print(
            f"  I-frame positions: {info_data['iframe_positions'][:20]}"
            + f"{'...' if len(info_data['iframe_positions']) > 20 else ''}"
        )
        return

    if output is None:
        output = input_path.parent / f"{input_path.stem}_moshed{input_path.suffix}"

    if dry_run:
        console.print(f"Would remove I-frames from {input_path}")
        console.print(f"  Start frame: {start or 0}")
        console.print(f"  End frame: {end or 'end'}")
        console.print(f"  Keep first: {keep_first}")
        console.print(f"  Output: {output}")
        return

    preview_settings = PreviewSettings() if preview else None

    from moshpit.core.iframe import remove_iframes

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Removing I-frames...", total=None)
        remove_iframes(
            input_path,
            output,
            start_frame=start,
            end_frame=end,
            keep_first=keep_first,
            preview=preview_settings,
        )

    console.print(f"[green]Created:[/green] {output}")


# ============================================================================
# P-frame duplication command
# ============================================================================


@app.command()
def pframe(
    input_path: Annotated[Path, typer.Argument(help="Input video file")],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output video file")] = None,
    dup: Annotated[int, typer.Option("--dup", help="Number of duplications (1-10)")] = 3,
    start: Annotated[Optional[int], typer.Option("--start", help="Start frame")] = None,
    preview: Annotated[bool, typer.Option("--preview", help="Fast preview mode (480p)")] = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Verbose output")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would happen")] = False,
):
    """Duplicate P-frames to create bloom/smear effect.

    Amplifies motion by repeating P-frames, causing motion deltas to
    accumulate dramatically. Note: This increases video duration.
    """
    if output is None:
        output = input_path.parent / f"{input_path.stem}_pframe{input_path.suffix}"

    if dry_run:
        console.print(f"Would duplicate P-frames in {input_path}")
        console.print(f"  Duplication count: {dup}")
        console.print(f"  Start frame: {start or 0}")
        console.print(f"  Output: {output}")
        return

    preview_settings = PreviewSettings() if preview else None

    from moshpit.core.pframe import duplicate_pframes

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Duplicating P-frames...", total=None)
        duplicate_pframes(
            input_path,
            output,
            dup_count=dup,
            start_frame=start,
            preview=preview_settings,
        )

    console.print(f"[green]Created:[/green] {output}")


# ============================================================================
# Vector manipulation command
# ============================================================================


@app.command()
def vector(
    input_path: Annotated[Path, typer.Argument(help="Input video file")],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output video file")] = None,
    transform: Annotated[str, typer.Option("--transform", "-t", help="Transform name")] = "jitter",
    expr: Annotated[Optional[str], typer.Option("--expr", help="Inline expression")] = None,
    script: Annotated[Optional[Path], typer.Option("--script", help="Custom Python script")] = None,
    preview: Annotated[bool, typer.Option("--preview", help="Fast preview mode (480p)")] = False,
    seed: Annotated[Optional[int], typer.Option("--seed", help="Random seed")] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Verbose output")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would happen")] = False,
    list_transforms: Annotated[
        bool, typer.Option("--list", help="List available transforms")
    ] = False,
    # Transform parameters
    strength: Annotated[
        Optional[float], typer.Option("--strength", help="Transform strength")
    ] = None,
    factor: Annotated[
        Optional[float], typer.Option("--factor", help="Amplification factor")
    ] = None,
    freq: Annotated[
        Optional[float], typer.Option("--freq", help="Frequency for wave transforms")
    ] = None,
    amp: Annotated[
        Optional[float], typer.Option("--amp", help="Amplitude for wave transforms")
    ] = None,
    axis: Annotated[Optional[str], typer.Option("--axis", help="Axis (x, y, both)")] = None,
    levels: Annotated[Optional[int], typer.Option("--levels", help="Quantization levels")] = None,
    kernel: Annotated[Optional[int], typer.Option("--kernel", help="Blur kernel size")] = None,
):
    """Apply motion vector transforms for advanced datamoshing.

    Requires FFglitch tools (ffedit, ffgac) to be installed.
    Use --list to see available transforms.
    """
    # Ensure builtin transforms are registered
    from moshpit.transforms.builtin import register_builtin_transforms

    register_builtin_transforms()

    if list_transforms:
        from moshpit.transforms.registry import list_transforms as get_transforms

        table = Table(title="Available Vector Transforms")
        table.add_column("Name", style="cyan")
        table.add_column("Description")

        for name, desc in get_transforms():
            table.add_row(name, desc)

        console.print(table)
        return

    if output is None:
        output = input_path.parent / f"{input_path.stem}_vector{input_path.suffix}"

    if dry_run:
        console.print(f"Would apply vector transform to {input_path}")
        if expr:
            console.print(f"  Expression: {expr}")
        elif script:
            console.print(f"  Script: {script}")
        else:
            console.print(f"  Transform: {transform}")
        console.print(f"  Output: {output}")
        return

    preview_settings = PreviewSettings() if preview else None

    # Build transform kwargs
    kwargs = {}
    if strength is not None:
        kwargs["strength"] = strength
    if factor is not None:
        kwargs["factor"] = factor
    if freq is not None:
        kwargs["freq"] = freq
    if amp is not None:
        kwargs["amp"] = amp
    if axis is not None:
        kwargs["axis"] = axis
    if levels is not None:
        kwargs["levels"] = levels
    if kernel is not None:
        kwargs["kernel"] = kernel
    if seed is not None:
        kwargs["seed"] = seed

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Applying {transform}...", total=None)

        if expr:
            from moshpit.core.vectors import apply_expression

            apply_expression(input_path, output, expr, preview=preview_settings)
        elif script:
            from moshpit.core.vectors import apply_script

            apply_script(input_path, output, script, preview=preview_settings)
        else:
            from moshpit.core.vectors import apply_vector_transform

            apply_vector_transform(
                input_path,
                output,
                transform=transform,
                preview=preview_settings,
                **kwargs,
            )

    console.print(f"[green]Created:[/green] {output}")


# ============================================================================
# Transfer command
# ============================================================================


@app.command()
def transfer(
    source: Annotated[Path, typer.Option("--source", "-s", help="Source video (motion from)")],
    target: Annotated[Path, typer.Option("--target", "-t", help="Target video (apply motion to)")],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output video file")] = None,
    blend: Annotated[
        float, typer.Option("--blend", help="Blend factor (0=target, 1=source)")
    ] = 1.0,
    preview: Annotated[bool, typer.Option("--preview", help="Fast preview mode (480p)")] = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Verbose output")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would happen")] = False,
):
    """Transfer motion vectors from one video to another.

    Takes motion characteristics from source video and applies them to
    target video, making target imagery move like the source.
    """
    if output is None:
        output = target.parent / f"{target.stem}_transfer{target.suffix}"

    if dry_run:
        console.print(f"Would transfer motion from {source} to {target}")
        console.print(f"  Blend: {blend}")
        console.print(f"  Output: {output}")
        return

    preview_settings = PreviewSettings() if preview else None

    from moshpit.core.transfer import transfer_motion

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Transferring motion...", total=None)
        transfer_motion(
            source,
            target,
            output,
            blend=blend,
            preview=preview_settings,
        )

    console.print(f"[green]Created:[/green] {output}")


# ============================================================================
# Pipeline command
# ============================================================================


@app.command()
def pipe(
    input_path: Annotated[Path, typer.Argument(help="Input video file")],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output video file")] = None,
    chain: Annotated[str, typer.Option("--chain", "-c", help="Pipeline chain DSL")] = "",
    preview: Annotated[bool, typer.Option("--preview", help="Fast preview mode (480p)")] = False,
    seed: Annotated[Optional[int], typer.Option("--seed", help="Random seed")] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Verbose output")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would happen")] = False,
):
    """Execute a pipeline of chained effects.

    Use DSL syntax to chain effects:
        iframe[30:90] -> vector[jitter,strength=0.5] -> pframe[dup=3]
    """
    # Ensure builtin transforms are registered
    from moshpit.transforms.builtin import register_builtin_transforms

    register_builtin_transforms()

    if not chain:
        console.print("[red]Error:[/red] --chain is required")
        raise typer.Exit(1)

    if output is None:
        output = input_path.parent / f"{input_path.stem}_pipe{input_path.suffix}"

    if dry_run:
        from moshpit.pipeline.parser import parse_pipeline

        effects = parse_pipeline(chain)
        console.print(f"Would execute pipeline on {input_path}")
        for i, effect in enumerate(effects):
            console.print(f"  {i + 1}. {effect.name}: {effect.params}")
        console.print(f"  Output: {output}")
        return

    preview_settings = PreviewSettings() if preview else None

    from moshpit.pipeline.chain import execute_pipeline

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        _task = progress.add_task("Running pipeline...", total=None)

        execute_pipeline(
            input_path,
            output,
            chain,
            preview=preview_settings,
            seed=seed,
            verbose=verbose,
        )

    console.print(f"[green]Created:[/green] {output}")


# ============================================================================
# Run recipe command
# ============================================================================


@app.command()
def run(
    recipe_path: Annotated[Path, typer.Argument(help="Recipe YAML file or name")],
    input_path: Annotated[Path, typer.Option("--input", "-i", help="Input video file")] = None,
    output: Annotated[Path, typer.Option("-o", "--output", help="Output video file")] = None,
    preview: Annotated[bool, typer.Option("--preview", help="Fast preview mode (480p)")] = False,
    seed: Annotated[Optional[int], typer.Option("--seed", help="Random seed")] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Verbose output")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would happen")] = False,
):
    """Run a saved recipe from a YAML file.

    Recipes can be loaded by name from the recipes/ directory,
    or by path to any YAML file.
    """
    # Ensure builtin transforms are registered
    from moshpit.transforms.builtin import register_builtin_transforms

    register_builtin_transforms()

    from moshpit.config.loader import load_recipe, load_recipe_by_name

    # Try to load recipe
    try:
        if recipe_path.exists():
            recipe = load_recipe(recipe_path)
        else:
            recipe = load_recipe_by_name(str(recipe_path))
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if input_path is None:
        console.print("[red]Error:[/red] --input is required")
        raise typer.Exit(1)

    if output is None:
        output = input_path.parent / f"{input_path.stem}_{recipe.name}{input_path.suffix}"

    if dry_run:
        console.print(f"[bold]Recipe:[/bold] {recipe.name}")
        if recipe.description:
            console.print(f"  {recipe.description}")
        console.print("[bold]Pipeline:[/bold]")
        for i, step in enumerate(recipe.pipeline):
            step_dict = step.model_dump() if hasattr(step, "model_dump") else step
            console.print(f"  {i + 1}. {step_dict}")
        console.print(f"[bold]Output:[/bold] {output}")
        return

    preview_settings = PreviewSettings() if preview else None

    from moshpit.pipeline.chain import EffectChain

    chain = EffectChain(recipe.pipeline)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        _task = progress.add_task(f"Running {recipe.name}...", total=None)

        chain.execute(
            input_path,
            output,
            preview=preview_settings,
            seed=seed,
            verbose=verbose,
        )

    console.print(f"[green]Created:[/green] {output}")


# ============================================================================
# Sweep command
# ============================================================================


@app.command()
def sweep(
    input_path: Annotated[Path, typer.Argument(help="Input video file")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o", help="Output directory")] = None,
    vary: Annotated[
        list[str], typer.Option("--vary", help="Parameter to vary (e.g., 'pframe.dup=2:8')")
    ] = [],
    count: Annotated[Optional[int], typer.Option("--count", help="Maximum variations")] = None,
    mode: Annotated[str, typer.Option("--mode", help="Sweep mode: grid or random")] = "grid",
    preview: Annotated[bool, typer.Option("--preview", help="Fast preview mode (480p)")] = False,
    seed: Annotated[Optional[int], typer.Option("--seed", help="Random seed")] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Verbose output")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would happen")] = False,
    recipe: Annotated[Optional[Path], typer.Option("--recipe", help="Base recipe to vary")] = None,
    chain: Annotated[
        Optional[str], typer.Option("--chain", "-c", help="Base pipeline chain")
    ] = None,
):
    """Generate multiple variations with parameter sweeps.

    Vary parameters across a range to explore the effect space:
        moshpit sweep input.mp4 --vary "pframe.dup=2:8" --vary "vector.strength=0.1:1.0:0.2"
    """
    # Ensure builtin transforms are registered
    from moshpit.transforms.builtin import register_builtin_transforms

    register_builtin_transforms()

    if output_dir is None:
        output_dir = input_path.parent / "sweep"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not vary:
        console.print("[red]Error:[/red] At least one --vary parameter is required")
        raise typer.Exit(1)

    from moshpit.pipeline.sweep import ParameterSweep, generate_output_path, parse_vary_spec

    # Build base config
    base_config: dict = {"pipeline": []}

    if recipe:
        from moshpit.config.loader import load_recipe

        recipe_obj = load_recipe(recipe)
        base_config["pipeline"] = [
            s.model_dump() if hasattr(s, "model_dump") else s for s in recipe_obj.pipeline
        ]
    elif chain:
        from moshpit.pipeline.parser import parse_pipeline, pipeline_to_config

        effects = parse_pipeline(chain)
        base_config["pipeline"] = pipeline_to_config(effects)
    else:
        # Default: simple pframe effect
        base_config["pipeline"] = [{"effect": "pframe", "dup": 3}]

    # Parse vary specs
    parameters = [parse_vary_spec(spec) for spec in vary]

    sweep_gen = ParameterSweep(
        base_config,
        parameters,
        count=count,
        mode=mode,
        seed=seed,
    )

    total = sweep_gen.total_combinations
    if count:
        total = min(total, count)

    if dry_run:
        console.print(f"Would generate {total} variations")
        console.print("Parameters:")
        for param in parameters:
            console.print(f"  {param.path}: {param.values}")
        console.print(f"Output directory: {output_dir}")
        return

    preview_settings = PreviewSettings() if preview else None

    from moshpit.pipeline.chain import EffectChain

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Generating {total} variations...", total=total)

        for i, config in sweep_gen:
            output_path = generate_output_path(output_dir, i)

            chain_obj = EffectChain(config["pipeline"])
            chain_obj.execute(
                input_path,
                output_path,
                preview=preview_settings,
                seed=seed,
                verbose=verbose,
            )

            progress.update(task, advance=1, description=f"Generated {output_path.name}")

    console.print(f"[green]Created {total} variations in:[/green] {output_dir}")


# ============================================================================
# List recipes command
# ============================================================================


@app.command("recipes")
def list_recipes():
    """List available recipes in the recipes directory."""
    from moshpit.config.loader import list_available_recipes

    recipes = list_available_recipes()

    if not recipes:
        console.print("No recipes found in recipes/ directory")
        return

    table = Table(title="Available Recipes")
    table.add_column("Name", style="cyan")

    for recipe_name in recipes:
        table.add_row(recipe_name)

    console.print(table)


if __name__ == "__main__":
    app()
