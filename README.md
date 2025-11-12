
## Features

- Smooth morphing transitions using Procrustes alignment and easing curves
- Organic vibration effects with Perlin noise (not random jitter)
- 31 different easing function types (bounce, elastic, smooth, etc.)
- Per-letter configuration for fine-tuned animations
- Batch processing for generating multiple letters
- Comprehensive logging and error handling
- SOLID architecture for maintainability
- OWASP-compliant security validation
- Optimized GIF output with compression

## Requirements

- Python 3.13 or higher
- OpenCV 4.12+
- NumPy 2.2+
- Pillow 12.0+
- SciPy 1.16+
- perlin-noise 1.12+
- easing-functions 1.0.4+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Vibing-Letters.git
cd Vibing-Letters
```

2. Install dependencies:
```bash
pip install opencv-python numpy pillow scipy perlin-noise easing-functions
```

3. Verify installation:
```bash
python -m pytest tests/ -v
```

## Quick Start

### Generate a Single Letter

Generate an animated letter from base and target images:

```bash
python generate_letter.py _Static_O.png input/A.png output/A_animated.gif
```

With a specific letter configuration:

```bash
python generate_letter.py _Static_O.png input/A.png output/A.gif --letter A --preset bouncy
```

### Build a Text String

Create a text animation from pre-generated letter GIFs:

```bash
python build_string.py "HELLO" output/hello.gif
```

### Batch Generate All Letters

Process all images in the input directory:

```bash
python batch_generate.py --preset smooth --input-dir input --output-dir output
```

## Usage

### generate_letter.py

Generate individual letter animations.

**Basic Usage:**
```bash
python generate_letter.py <base_image> <target_image> [output_path]
```

**Options:**
- `--background PATH` - Background image path (default: clean_background.png)
- `--letter LETTER` - Letter name for custom config (e.g., A)
- `--preset {default,bouncy,smooth,energetic}` - Animation preset
- `--log-level {DEBUG,INFO,WARNING,ERROR}` - Logging verbosity
- `--output-dir PATH` - Output directory (default: output/)

**Examples:**
```bash
# Generate with default settings
python generate_letter.py _Static_O.png input/A.png

# Use bouncy preset
python generate_letter.py _Static_O.png input/B.png --preset bouncy

# Custom output path
python generate_letter.py _Static_O.png input/C.png my_animations/C.gif

# Use letter-specific config
python generate_letter.py _Static_O.png input/S.png --letter S
```

### build_string.py

Combine pre-generated letter GIFs into text strings.

**Basic Usage:**
```bash
python build_string.py "TEXT" [output_path]
```

**Options:**
- `--input-dir PATH` - Directory with letter GIFs (default: input/)
- `--output-dir PATH` - Output directory (default: output/)
- `--suffix STR` - Letter GIF filename suffix (default: _animated)
- `--log-level {DEBUG,INFO,WARNING,ERROR}` - Logging verbosity

**Examples:**
```bash
# Create text animation
python build_string.py "HELLO"

# Custom output path
python build_string.py "WORLD" output/world.gif

# Different input directory
python build_string.py "TEST" --input-dir generated/
```

### batch_generate.py

Batch process multiple images.

**Basic Usage:**
```bash
python batch_generate.py [options]
```

**Options:**
- `--base-image PATH` - Base shape image (default: _Static_O.png)
- `--background PATH` - Background image (default: clean_background.png)
- `--input-dir PATH` - Input directory (default: input/)
- `--output-dir PATH` - Output directory (default: output/)
- `--preset {default,bouncy,smooth,energetic}` - Animation preset
- `--file-pattern STR` - File pattern to match (default: *.png)
- `--log-level {DEBUG,INFO,WARNING,ERROR}` - Logging verbosity

**Examples:**
```bash
# Generate all PNG images with default settings
python batch_generate.py

# Use energetic preset
python batch_generate.py --preset energetic

# Custom directories
python batch_generate.py --input-dir my_letters --output-dir my_output

# Process only JPG files
python batch_generate.py --file-pattern "*.jpg"
```

## Configuration

### Animation Presets

Four built-in presets control the animation style:

**default** - Balanced settings for general use
- Easing: ease_in_out_cubic (smooth)
- Vibration: moderate (noise_scale=0.3, frequency=2.5)
- Overshoot: 110% (bouncy morph)

**bouncy** - Playful, elastic motion
- Easing: ease_out_bounce (bouncing ball)
- Vibration: high (noise_scale=0.4, frequency=3.5)
- Overshoot: 120% (very bouncy)

**smooth** - Gentle, flowing motion
- Easing: ease_in_out_sine (very smooth)
- Vibration: low (noise_scale=0.2, frequency=1.5)
- Overshoot: 105% (minimal bounce)

**energetic** - Fast, intense motion
- Easing: ease_out_elastic (spring-like)
- Vibration: very high (noise_scale=0.5, frequency=4.5)
- Overshoot: 130% (extreme bounce)

### Per-Letter Configuration

Fine-tune individual letter animations by modifying [src/config/letter_config.py](src/config/letter_config.py).

Example configuration:

```python
EXAMPLE_LETTER_CONFIGS = {
    'A': {
        'easing_type': 'ease_out_bounce',
        'noise_scale': 0.3,              # Vibration amplitude
        'vibration_frequency': 2.5,      # Vibration speed
        'overshoot_values': [0.0, 1.15, 1.0],  # Morph sequence
    },
    'S': {
        'easing_type': 'ease_out_elastic',
        'noise_scale': 0.4,
        'vibration_frequency': 3.2,
        'overshoot_values': [0.0, 1.2, 1.0],
    },
}
```

### Configuration Parameters

**Morphing:**
- `easing_type` - Easing function (see list below)
- `overshoot_values` - List of interpolation values for morph sequence
- `use_procrustes` - Enable Procrustes alignment (default: True)
- `procrustes_scaling` - Allow scaling in alignment (default: True)

**Vibration:**
- `vibration_cycles` - Number of vibration cycles (default: 3)
- `noise_octaves` - Perlin noise detail level (default: 4)
- `noise_persistence` - Noise amplitude falloff (default: 0.5)
- `noise_scale` - Vibration amplitude in pixels (default: 0.3)
- `vibration_frequency` - Vibration speed multiplier (default: 2.5)

**Timing:**
- `frame_duration_ms` - Duration per frame in milliseconds (default: 20)
- `blank_pause_duration_ms` - Pause at end in milliseconds (default: 60)
- `static_start_frames` - Static frames at start (default: 3)
- `static_end_frames` - Static frames at end (default: 2)

**Contours:**
- `n_points` - Number of contour points for morphing (default: 120)

**Rendering:**
- `image_size` - Output image dimensions (default: (512, 512))
- `line_thickness` - Line thickness in pixels (default: 2)
- `line_color` - RGB tuple for lines (default: (0, 0, 0))
- `background_color` - RGB tuple for background (default: (255, 255, 255))

### Available Easing Functions

**Linear:**
- linear - Constant speed

**Quadratic:**
- ease_in_quad - Slow start
- ease_out_quad - Fast start
- ease_in_out_quad - Slow start and end

**Cubic:**
- ease_in_cubic - Slow start
- ease_out_cubic - Fast start
- ease_in_out_cubic - Smooth start and end (recommended)

**Quartic:**
- ease_in_quart - Very slow start
- ease_out_quart - Very fast start
- ease_in_out_quart - Very slow start and end

**Quintic:**
- ease_in_quint - Extremely slow start
- ease_out_quint - Extremely fast start
- ease_in_out_quint - Extremely slow start and end

**Sine:**
- ease_in_sine - Gentle slow start
- ease_out_sine - Gentle fast start
- ease_in_out_sine - Very smooth (recommended)

**Circular:**
- ease_in_circ - Circular acceleration
- ease_out_circ - Circular deceleration
- ease_in_out_circ - Circular both

**Exponential:**
- ease_in_expo - Exponential acceleration
- ease_out_expo - Exponential deceleration
- ease_in_out_expo - Exponential both

**Elastic:**
- ease_in_elastic - Elastic at start
- ease_out_elastic - Springy at end (recommended for energetic)
- ease_in_out_elastic - Elastic both

**Back:**
- ease_in_back - Pull back before start
- ease_out_back - Overshoot at end
- ease_in_out_back - Pull back and overshoot

**Bounce:**
- ease_in_bounce - Bounce at start
- ease_out_bounce - Bounce at end (recommended for bouncy)
- ease_in_out_bounce - Bounce both

## Project Structure

```
Vibing-Letters/
├── src/
│   ├── config/
│   │   ├── morph_config.py         # Global configuration
│   │   └── letter_config.py        # Per-letter configuration
│   ├── morphing/
│   │   ├── contour_extractor.py    # Extract/resample contours
│   │   ├── procrustes_aligner.py   # Shape alignment
│   │   ├── perlin_vibrator.py      # Vibration effects
│   │   ├── easing_curve.py         # Easing functions
│   │   ├── morph_engine.py         # Orchestration
│   │   ├── frame_generator.py      # Render frames
│   │   └── gif_builder.py          # Create GIFs
│   └── utils/
│       ├── logger.py               # Logging system
│       └── validators.py           # Input validation
├── tests/
│   ├── test_validators.py
│   ├── test_contour_extractor.py
│   └── test_easing_curve.py
├── generate_letter.py              # Single letter generator
├── build_string.py                 # Text string builder
├── batch_generate.py               # Batch processor
├── PLANS.md                        # Development plans
├── README.md                       # This file
└── requirements.txt                # (optional) Dependencies

Legacy files (not used):
├── vibing_letter_generator.py      # Old implementation
├── string_builder.py               # Old implementation
└── collapse_O.py                   # Experimental
```

## Architecture

The project follows SOLID principles:

**Single Responsibility Principle:**
- Each class has one clear purpose
- ContourExtractor: Extract and resample contours
- ProcrustesAligner: Align shapes
- PerlinVibrator: Generate vibration
- EasingCurve: Apply easing functions
- MorphEngine: Orchestrate morphing pipeline
- FrameGenerator: Render frames
- GifBuilder: Create GIF files

**Open-Closed Principle:**
- Easy to add new easing functions
- Can swap alignment strategies
- Extensible configuration system

**Dependency Inversion:**
- Components depend on abstractions
- Dependency injection for all major classes

## Security

The codebase follows OWASP Top 10 security practices:

**Path Traversal Prevention:**
- All file paths are validated and sanitized
- Base directory restrictions prevent access outside allowed directories

**Input Validation:**
- All user inputs are validated before use
- File formats verified using magic bytes (not just extensions)
- Image size limits enforced (max 4096x4096, max 50MB)

**Filename Sanitization:**
- Special characters removed from filenames
- Path separators stripped
- Reserved Windows names handled

**Resource Limits:**
- Maximum file sizes enforced
- Memory usage controlled
- No unbounded operations

**Error Handling:**
- Comprehensive exception handling
- Clear error messages without exposing internals
- Proper resource cleanup

## Development

### Running Tests

Run all tests:
```bash
python -m pytest tests/ -v
```

Run specific test file:
```bash
python -m pytest tests/test_validators.py -v
```

Run with coverage:
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

### Adding a New Easing Function

1. Easing functions are provided by the `easing-functions` library
2. To add a custom easing, use `create_custom_easing()`:

```python
from src.morphing.easing_curve import create_custom_easing

def my_easing(t):
    return t ** 3  # Cubic easing

custom_ease = create_custom_easing(my_easing)
```

### Adding Per-Letter Configuration

Edit [src/config/letter_config.py](src/config/letter_config.py):

```python
EXAMPLE_LETTER_CONFIGS = {
    'X': {
        'easing_type': 'ease_out_bounce',
        'noise_scale': 0.35,
        'vibration_frequency': 2.8,
        'overshoot_values': [0.0, 1.15, 1.0],
    },
}
```

Then use it:
```bash
python generate_letter.py _Static_O.png input/X.png --letter X
```

## Troubleshooting

### "No contours found in image"

The image may be all white or have no clear shapes. Ensure:
- Image has a dark shape on light background
- Image is not corrupted
- Image format is supported (PNG, JPG, BMP)

### "Path is outside allowed directory"

Security validation prevents path traversal. Ensure:
- File paths are within the project directory
- No `../` in paths
- Absolute paths point to valid locations

### Large GIF file sizes

Reduce file size by:
1. Decreasing `n_points` in configuration (fewer points = smaller file)
2. Increasing `frame_duration_ms` (fewer frames per second)
3. Reducing `vibration_cycles`
4. Using the `--optimize` flag (enabled by default)

### Slow performance

Improve performance by:
1. Reducing `n_points` (default: 120)
2. Reducing `vibration_cycles` (default: 3)
3. Using fewer `noise_octaves` (default: 4)
4. Batch processing instead of individual letters

### Import errors

Ensure src/ is in Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

## Performance

Typical performance on modern hardware:

- Single letter generation: 1-2 seconds
- Batch generation (26 letters): 30-60 seconds
- Text string (5 letters): <1 second
- GIF file size: 1-3 MB per letter (compressed)

## Contributing

Contributions are welcome. Please:

1. Follow existing code style
2. Add unit tests for new features
3. Update documentation
4. Ensure SOLID principles are maintained
5. Run all tests before submitting

## License

This project is open source. Please check the LICENSE file for details.


## Technical Details

### Morphing Algorithm

The morphing process uses:

1. **Contour Extraction** - OpenCV findContours with binary thresholding
2. **Resampling** - Arc length-based resampling to fixed point count
3. **Procrustes Alignment** - Optimal rotation, translation, and scaling
4. **Easing Interpolation** - Non-linear interpolation with easing curves
5. **Perlin Noise Vibration** - Smooth, organic vibration patterns
6. **Frame Rendering** - OpenCV polylines on background image
7. **GIF Optimization** - Pillow with optimize=True and color quantization

### Why Perlin Noise?

Perlin noise creates smooth, continuous variations that look more organic than random jitter. Each point on the contour vibrates independently but smoothly over time, creating a realistic "string vibration" effect.

### Why Procrustes Alignment?

Procrustes analysis finds the optimal alignment between shapes, minimizing distortion during morphing. Without it, shapes may rotate or scale awkwardly during transition.


