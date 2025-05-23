<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Text-to-CAD AI Model Development Instructions

## Project Context
This is a state-of-the-art AI model that converts natural language descriptions into parametric CAD files. The project implements a multi-modal transformer architecture with geometric validation and visual feedback.

## Key Technical Components
- **Multi-modal Architecture**: BERT text encoder + 24-layer transformer decoder + CLIP visual feedback
- **Geometric Validation**: KCL code generation with feature-by-feature validation
- **Training Pipeline**: Sequential pre-training + visual feedback fine-tuning with PPO
- **CAD Processing**: Parametric sequence generation with 8-bit quantization
- **Manufacturing Validation**: Wall thickness, geometric constraints, and STEP export

## Coding Guidelines

### Model Architecture
- Use PyTorch for all deep learning components
- Implement adaptive projection layers for multi-modal alignment
- Follow transformer best practices for attention mechanisms
- Include geometric validation in forward passes

### CAD Processing
- Use parametric sequences with quantized parameters
- Implement proper error handling for geometric operations
- Support multiple CAD formats (STEP, GLTF, KCL)
- Validate manufacturing constraints (thickness, tolerances)

### Data Pipeline
- Process DeepCAD dataset with geometric hash deduplication
- Generate multi-level text annotations (visual + parametric)
- Implement proper train/validation/test splits (80/15/5)
- Handle sequence tokenization with special tokens

### Training
- Implement multi-stage training (pre-training + fine-tuning)
- Use cross-entropy loss for sequence prediction
- Include CLIP-based reward functions for visual feedback
- Apply gradient clipping and learning rate scheduling

### Validation
- Implement geometric validation with KCL execution
- Check manufacturing constraints (wall thickness, etc.)
- Generate CLIP scores for visual-text alignment
- Measure Chamfer distance for geometric accuracy

### Code Style
- Use type hints for all function signatures
- Include comprehensive docstrings with mathematical notation
- Follow PEP 8 style guidelines
- Add logging for training and inference stages

## Mathematical Notation
- Use LaTeX-style notation in docstrings: L_seq = -∑log p(c_t|c_<t, T)
- Document loss functions, reward calculations, and geometric operations
- Include parameter ranges and quantization details

## Performance Targets
- Inference: <2s for 50-operation sequences
- Accuracy: <0.87mm Chamfer Distance, >0.82 CLIP Score
- Model size: ~350M parameters with FP16 quantization

## Dependencies Priority
1. PyTorch ecosystem (torch, transformers, datasets)
2. CAD libraries (opencascade, cadquery, freecad)
3. Geometric processing (open3d, trimesh, vtk)
4. Multi-modal models (clip, llava)
5. Deployment tools (fastapi, tensorrt, triton)
