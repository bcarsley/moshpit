# Moshpit

A Python CLI tool for creative video datamoshing with maximum tunability.

## Installation

### Prerequisites

**Required:**
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [just](https://github.com/casey/just) (command runner)
- `ffmpeg` and `ffprobe`

**For vector manipulation (optional):**
- FFglitch tools (`ffedit`, `ffgac`) from [ffglitch.org](https://ffglitch.org/)

### Install FFglitch (macOS Apple Silicon)

```bash
# Download and install
just install-ffglitch

# Or manually:
curl -LO https://ffglitch.org/pub/bin/macos-aarch64/ffglitch-0.10.2-macos-aarch64.zip
unzip ffglitch-0.10.2-macos-aarch64.zip
sudo cp ffglitch-0.10.2-macos-aarch64/bin/* /usr/local/bin/
```

### Install moshpit

```bash
# Clone and install
git clone <repo> && cd video_augmentation
just install

# Verify
just dev
```

## Quick Start

```bash
# Basic I-frame removal (classic datamosh)
moshpit iframe input.mp4 -o out.mp4 --start 30 --end 90

# P-frame duplication (bloom effect)
moshpit pframe input.mp4 -o out.mp4 --dup 5

# Vector manipulation (requires FFglitch)
moshpit vector input.mp4 -o out.mp4 --transform jitter --strength 0.5

# Chain effects with pipeline DSL
moshpit pipe input.mp4 -o out.mp4 \
  --chain "iframe[30:90] -> vector[jitter,strength=0.5] -> pframe[dup=3]"

# Use preview mode for fast iteration
moshpit iframe input.mp4 -o out.mp4 --preview
```

## Commands

### `moshpit iframe`

Remove I-frames to create the classic "pixel bleeding" datamosh effect.

```bash
moshpit iframe input.mp4 -o out.mp4 --start 30 --end 90
moshpit iframe input.mp4 --info  # Show I-frame positions
```

### `moshpit pframe`

Duplicate P-frames to amplify motion and create bloom/smear effects.

```bash
moshpit pframe input.mp4 -o out.mp4 --dup 5 --start 40
```

### `moshpit vector`

Apply motion vector transforms for advanced effects. Requires FFglitch.

```bash
# Use a built-in transform
moshpit vector input.mp4 -o out.mp4 --transform jitter --strength 0.5

# List available transforms
moshpit vector --list input.mp4

# Use inline expression
moshpit vector input.mp4 -o out.mp4 --expr "frame[:,:,0] *= 2"

# Use custom Python script
moshpit vector input.mp4 -o out.mp4 --script my_transform.py
```

### `moshpit transfer`

Transfer motion from one video to another.

```bash
moshpit transfer --source motion.mp4 --target static.mp4 -o out.mp4
```

### `moshpit pipe`

Chain multiple effects with the pipeline DSL.

```bash
moshpit pipe input.mp4 -o out.mp4 \
  --chain "iframe[10:40] -> vector[sine-wave,freq=0.1] -> pframe[dup=3]"
```

### `moshpit run`

Run a saved YAML recipe.

```bash
moshpit run recipes/dreamy-melt.yaml --input video.mp4 --output result.mp4
```

### `moshpit sweep`

Generate variations with parameter sweeps.

```bash
moshpit sweep input.mp4 --output-dir ./variations \
  --vary "pframe.dup=2:8" --vary "vector.strength=0.1:1.0:0.2" --count 10
```

## Pipeline DSL

Chain effects with the `->` operator:

```
effect[params] -> effect[params] -> ...
```

**Syntax:**
- `iframe[start:end]` - Frame range
- `pframe[dup=5]` - Named parameters
- `vector[jitter,strength=0.5]` - Transform with params

**Examples:**
```
iframe[30:90]
iframe[30:90] -> pframe[dup=5]
vector[transform=sine-wave,freq=0.1,axis=x] -> iframe[50:100]
```

## Vector Transforms

| Transform | Description | Parameters |
|-----------|-------------|------------|
| `jitter` | Random noise | `strength`, `seed` |
| `amplify` | Scale motion | `factor` |
| `horizontal-kill` | Zero X motion | - |
| `vertical-kill` | Zero Y motion | - |
| `invert` | Flip direction | `axis` |
| `swap-axes` | Swap X/Y | - |
| `sine-wave` | Sinusoidal | `freq`, `amp`, `axis` |
| `quantize` | Posterize | `levels` |
| `threshold` | Binary | `cutoff` |
| `blur` | Gaussian blur | `kernel` |
| `feedback` | Temporal blend | `blend` |
| `accumulate` | Motion trails | `decay` |

## YAML Recipes

```yaml
name: dreamy-melt
description: Soft melting transitions

pipeline:
  - effect: iframe
    start: 30
    end: 90

  - effect: vector
    transform: blur
    params:
      kernel: 5

  - effect: pframe
    dup: 3
```

## Global Flags

| Flag | Description |
|------|-------------|
| `--preview` | Fast render: 480p + CRF 28 |
| `--seed <int>` | RNG seed for reproducibility |
| `-v, --verbose` | Detailed logging |
| `--dry-run` | Show what would happen |

## Development

```bash
# Install dev dependencies
just install

# Run tests
just test

# Format code
just fmt

# Lint
just lint

# Check dependencies
just check
```

## License

MIT
