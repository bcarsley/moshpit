# Moshpit Recipes

This directory contains pre-made datamoshing recipes that can be run with:

```bash
moshpit run recipes/recipe-name.yaml --input video.mp4 --output result.mp4
```

## Available Recipes

### dreamy-melt
Soft melting transitions with gentle motion blur. Creates a dreamy, ethereal effect.
- I-frame removal for pixel bleeding
- Motion vector blur for smoothing
- P-frame duplication for trails

### chaos-jitter
High-energy random motion chaos. Creates chaotic, glitchy movement.
- Aggressive I-frame removal
- High-strength random jitter
- Motion amplification

### wave-distort
Wavy, liquid-like motion distortion. Creates a fluid, underwater effect.
- Horizontal sine wave displacement
- Vertical cosine wave for organic movement

### bloom-trails
Motion trails with accumulated bloom effect. Creates ghosting and persistence.
- Motion vector accumulation over time
- P-frame duplication for trail intensity

## Creating Your Own Recipes

Copy one of the existing recipes and modify it. The YAML schema supports:

```yaml
name: my-recipe
description: What this recipe does

pipeline:
  - effect: iframe
    start: 30          # Start frame
    end: 90            # End frame
    keep_first: true   # Keep first I-frame

  - effect: pframe
    dup: 3             # Duplication count (1-10)
    start: 40          # Start frame

  - effect: vector
    transform: jitter  # Transform name
    params:
      strength: 0.5    # Transform-specific parameters

  - effect: transfer
    source: other.mp4  # Source video for motion
    blend: 1.0         # Blend factor (0-1)
```

See `moshpit vector --list` for all available transforms.
