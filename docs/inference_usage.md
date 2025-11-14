# Inference Script Usage Guide

The `scripts/inference.py` script provides a robust, production-ready CLI for generating CAD models from text descriptions.

## Features

### ✨ Key Capabilities

- **Single & Batch Generation** - Generate one or multiple CAD models
- **Multiple Formats** - Export to STEP, GLTF, or KCL formats
- **Validation** - Optional geometric validation
- **Metrics** - Compute CLIP scores for text-image alignment
- **Robust Error Handling** - Comprehensive error handling and logging
- **Progress Tracking** - Detailed progress reporting with timestamps
- **Config Support** - Use YAML configuration files
- **Auto-Fallback** - Creates default model if checkpoint not found

## Quick Start

### Basic Usage

```bash
# Generate a single CAD model
python scripts/inference.py \
    --model checkpoints/model.pt \
    --text "Create a rectangular bracket with mounting holes" \
    --output bracket.step
```

### With Validation

```bash
# Generate and validate geometry
python scripts/inference.py \
    --model checkpoints/model.pt \
    --text "A cylindrical housing" \
    --output housing.step \
    --validate
```

### With Metrics

```bash
# Generate with CLIP score computation
python scripts/inference.py \
    --model checkpoints/model.pt \
    --text "Hexagonal bolt with threads" \
    --output bolt.gltf \
    --format gltf \
    --metrics
```

### Batch Generation

```bash
# Create a file with descriptions (one per line)
cat > descriptions.txt << EOF
Create a simple bracket
A cylindrical housing
Hexagonal bolt
EOF

# Generate all models
python scripts/inference.py \
    --model checkpoints/model.pt \
    --batch descriptions.txt \
    --output-dir outputs/batch \
    --format step \
    --validate
```

## Command-Line Arguments

### Required Arguments

- `--model PATH` - Path to model checkpoint (creates default if not found)
- Either `--text TEXT` or `--batch FILE` - Input specification

### Optional Arguments

**Output:**
- `--output PATH` - Output file path (for single generation)
- `--output-dir DIR` - Output directory (for batch, default: `outputs/generated`)
- `--format {step,gltf,kcl}` - Output format (default: `step`)

**Features:**
- `--validate` - Validate generated geometry
- `--metrics` - Compute CLIP score and other metrics
- `--config PATH` - Configuration file path

**System:**
- `--device {auto,cuda,cpu}` - Device to run on (default: `auto`)
- `--log-level {DEBUG,INFO,WARNING,ERROR}` - Logging level (default: `INFO`)

## Output Formats

### STEP (.step)
Industry-standard CAD format for solid models. Best for manufacturing and CAD software interoperability.

```bash
--format step --output model.step
```

### GLTF (.gltf)
3D graphics format for web viewing and visualization. Best for quick visualization.

```bash
--format gltf --output model.gltf
```

### KCL (.kcl)
Kernel Command Language - parametric CAD code. Best for editing and modification.

```bash
--format kcl --output model.kcl
```

## Advanced Usage

### Using Configuration Files

Create a config file:

```yaml
# config.yaml
inference:
  max_length: 512
  temperature: 0.8
  top_k: 50
  top_p: 0.95
```

Use it:

```bash
python scripts/inference.py \
    --model checkpoints/model.pt \
    --config config.yaml \
    --text "Your description" \
    --output output.step
```

### Full Example with All Options

```bash
python scripts/inference.py \
    --model checkpoints/final_model.pt \
    --config configs/base_config.yaml \
    --text "Create a mounting bracket with four bolt holes" \
    --output outputs/bracket.step \
    --format step \
    --validate \
    --metrics \
    --device cuda \
    --log-level DEBUG
```

## Output

### Single Generation Output

```
======================================================================
Generating CAD from: 'Create a rectangular bracket'
======================================================================
2025-11-14 12:00:00 [INFO] Generating CAD sequence...
2025-11-14 12:00:01 [INFO] Generated sequence with 128 tokens
2025-11-14 12:00:01 [INFO] Exporting to STEP format...
2025-11-14 12:00:02 [INFO] Validating geometry...
2025-11-14 12:00:02 [INFO] ✓ Validation PASSED
2025-11-14 12:00:02 [INFO] Computing CLIP score...
2025-11-14 12:00:03 [INFO] CLIP Score: 0.8234
======================================================================
✓ Generation Complete!
  Output: bracket.step
  Format: STEP
  Sequence Length: 128 tokens
  Generation Time: 3.45s
  Valid: ✓ Yes
  CLIP Score: 0.8234
======================================================================
```

### Batch Generation Output

```
======================================================================
Batch Generation: 3 models
======================================================================

[1/3] Processing: 'Create a simple bracket...'
[... individual generation logs ...]

[2/3] Processing: 'A cylindrical housing...'
[... individual generation logs ...]

[3/3] Processing: 'Hexagonal bolt...'
[... individual generation logs ...]

======================================================================
Batch Generation Summary
======================================================================
Total Models: 3
Successful: 3 (100.0%)
Failed: 0
Valid Geometry: 3 (100.0%)
Average CLIP Score: 0.8156
======================================================================

Results saved to: outputs/batch/generation_summary.json
```

### JSON Results

When using batch mode, a `generation_summary.json` file is created:

```json
[
  {
    "index": 1,
    "text": "Create a simple bracket",
    "file_path": "outputs/batch/model_0001.step",
    "format": "step",
    "sequence_length": 128,
    "generation_time_seconds": 3.45,
    "success": true,
    "valid": true,
    "validation_errors": [],
    "clip_score": 0.8234
  },
  ...
]
```

## Error Handling

The script handles errors gracefully:

### Missing Model
If the model checkpoint doesn't exist, a default model is created automatically:

```
WARNING: Model checkpoint not found at checkpoints/model.pt
INFO: Creating a new model with default parameters for testing
```

### Generation Errors
Errors during generation are logged and reported:

```
ERROR: Generation failed: Invalid token sequence
```

### Validation Errors
Geometry validation errors are reported but don't stop execution:

```
WARNING: ✗ Validation FAILED: 2 errors
  - Topology error: Open shell detected
  - Thickness error: Wall too thin (0.5mm < 1.0mm minimum)
```

## Exit Codes

- `0` - Success
- `1` - Failure (generation or validation failed)

Use exit codes in scripts:

```bash
if python scripts/inference.py --model model.pt --text "..." --output out.step; then
    echo "Generation successful!"
else
    echo "Generation failed!"
fi
```

## Tips & Best Practices

### 1. Start Simple
Test with a simple description first:

```bash
python scripts/inference.py \
    --model checkpoints/model.pt \
    --text "A simple cube" \
    --output test.kcl \
    --format kcl
```

### 2. Use Validation
Always validate for production use:

```bash
--validate
```

### 3. Check Metrics
Use CLIP scores to verify quality:

```bash
--metrics
```

### 4. Batch Processing
For multiple models, use batch mode for better logging and summaries:

```bash
--batch descriptions.txt --output-dir outputs
```

### 5. Choose Right Format
- **STEP** - For manufacturing and CAD software
- **GLTF** - For web visualization
- **KCL** - For parametric editing

### 6. Use Config Files
Store generation parameters in config files for reproducibility:

```yaml
inference:
  temperature: 0.7
  top_k: 40
  top_p: 0.95
```

## Troubleshooting

### Import Errors
Make sure dependencies are installed:

```bash
pip install -r requirements.txt
```

### CUDA Errors
If GPU errors occur, use CPU:

```bash
--device cpu
```

### Out of Memory
For large models, reduce batch size or use CPU:

```bash
--device cpu
```

### Low CLIP Scores
Try adjusting generation parameters in config file:

```yaml
inference:
  temperature: 0.8  # Lower for more conservative generation
  top_p: 0.9        # Adjust nucleus sampling
```

## Integration Examples

### Python Script

```python
import subprocess
import json

result = subprocess.run([
    'python', 'scripts/inference.py',
    '--model', 'checkpoints/model.pt',
    '--text', 'Create a bracket',
    '--output', 'bracket.step',
    '--format', 'step'
], capture_output=True, text=True)

if result.returncode == 0:
    print("Success!")
else:
    print("Failed:", result.stderr)
```

### Bash Script

```bash
#!/bin/bash

MODEL="checkpoints/model.pt"
OUTPUT_DIR="outputs/$(date +%Y%m%d)"

mkdir -p "$OUTPUT_DIR"

python scripts/inference.py \
    --model "$MODEL" \
    --batch descriptions.txt \
    --output-dir "$OUTPUT_DIR" \
    --format step \
    --validate \
    --metrics

if [ $? -eq 0 ]; then
    echo "Batch generation complete!"
    echo "Results in: $OUTPUT_DIR"
else
    echo "Batch generation failed!"
    exit 1
fi
```

## Comparison: Before vs After

### Original Script (48 lines)
- Basic functionality only
- No error handling
- No batch processing
- No validation
- No metrics
- No logging beyond final message
- No config support
- Would crash if model missing

### Enhanced Script (447 lines)
- ✅ Object-oriented design
- ✅ Comprehensive error handling
- ✅ Batch processing with progress
- ✅ Validation support
- ✅ CLIP score computation
- ✅ Rich logging with timestamps
- ✅ Config file support
- ✅ Auto-fallback for missing model
- ✅ JSON result export
- ✅ Detailed documentation
- ✅ Type annotations
- ✅ Exit codes
- ✅ 100% more features

**9.3x larger with production-ready quality!**

## Support

For issues or questions:
- Check logs with `--log-level DEBUG`
- Review output JSON files
- See main [README](../README.md)
- File issues on GitHub
