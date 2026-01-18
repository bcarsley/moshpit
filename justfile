# Moshpit justfile - command runner for datamoshing CLI

# Default preview flags for development
default_flags := "--preview"

# Quick test with a sample video
test-iframe:
    uv run moshpit iframe samples/test.mp4 -o out/iframe.mp4 --start 10 --end 40 {{default_flags}}

test-pframe:
    uv run moshpit pframe samples/test.mp4 -o out/pframe.mp4 --dup 5 {{default_flags}}

test-vector transform="jitter":
    uv run moshpit vector samples/test.mp4 -o out/vector.mp4 --transform {{transform}} {{default_flags}}

test-pipe:
    uv run moshpit pipe samples/test.mp4 -o out/pipe.mp4 \
        --chain "iframe[10:40] -> vector[jitter] -> pframe[dup=3]" {{default_flags}}

# Run a saved recipe
recipe name:
    uv run moshpit run recipes/{{name}}.yaml --input samples/test.mp4 --output out/{{name}}.mp4 {{default_flags}}

# Generate variations
sweep:
    uv run moshpit sweep samples/test.mp4 --output-dir out/sweep \
        --vary "pframe.dup=2:8" --count 10 {{default_flags}}

# Install dependencies
install:
    uv sync

# Development mode - show help
dev:
    uv run moshpit --help

# Format code
fmt:
    uv run ruff format src/

# Lint code
lint:
    uv run ruff check src/

# Run tests
test:
    uv run pytest tests/ -v

# Full render (no preview)
render input output *ARGS:
    uv run moshpit pipe {{input}} -o {{output}} {{ARGS}}

# Show available transforms
transforms:
    uv run moshpit vector --list samples/test.mp4

# Show I-frame info for a video
info video:
    uv run moshpit iframe {{video}} --info

# List available recipes
recipes:
    uv run moshpit recipes

# Create output directory
setup:
    mkdir -p out samples

# Clean output files
clean:
    rm -rf out/*

# Check if ffglitch is installed
check-ffglitch:
    @which ffedit && echo "ffedit: OK" || echo "ffedit: NOT FOUND"
    @which ffgac && echo "ffgac: OK" || echo "ffgac: NOT FOUND"

# Check if ffmpeg is installed
check-ffmpeg:
    @which ffmpeg && echo "ffmpeg: OK" || echo "ffmpeg: NOT FOUND"
    @which ffprobe && echo "ffprobe: OK" || echo "ffprobe: NOT FOUND"

# Check all dependencies
check: check-ffmpeg check-ffglitch
    @echo "Dependency check complete"

# Download and install ffglitch (macOS Apple Silicon)
install-ffglitch:
    curl -LO https://ffglitch.org/pub/bin/macos-aarch64/ffglitch-0.10.2-macos-aarch64.zip
    unzip -o ffglitch-0.10.2-macos-aarch64.zip
    sudo cp ffglitch-0.10.2-macos-aarch64/ffedit ffglitch-0.10.2-macos-aarch64/ffgac ffglitch-0.10.2-macos-aarch64/fflive /usr/local/bin/
    rm -rf ffglitch-0.10.2-macos-aarch64.zip ffglitch-0.10.2-macos-aarch64/
    @echo "FFglitch installed successfully"
