# CAD Fusion Lab - Research Analysis & Improvement Plan

**Date:** November 16, 2025
**Research Phase:** Complete
**Status:** Ready for Major Improvements

---

## 🔬 Research Findings

### State-of-the-Art Projects (2024-2025)

#### 1. **Text2CAD** (NeurIPS 2024 - Spotlight)
- **Architecture:** BERT encoder + Transformer decoder with cross-attention
- **Dataset:** DeepCAD with 170K models, 660K text annotations
- **Innovation:** First end-to-end text-to-parametric CAD system
- **Key Feature:** Layer-wise cross-attention between CAD and text embeddings
- **Source:** https://sadilkhan.github.io/text2cad-project/

#### 2. **CADLLM** (ACL 2025)
- **Architecture:** Dual-channel transformer fusing parameter + appearance
- **Innovation:** Fine-tuned LLM with confidence scores for sequence refinement
- **Key Feature:** Parameter and appearance feature fusion
- **Advantage:** Better handles complex geometric relationships

#### 3. **CADFusion** (January 2025)
- **Architecture:** LLM backbone with visual feedback loop
- **Innovation:** Represents CAD sequences as text tokens (not numeric)
- **Training:** Sequential learning + visual feedback stage
- **Key Feature:** Alternates between sequence generation and visual feedback
- **Advantage:** Leverages pre-trained LLM knowledge

#### 4. **GenCAD** (Latest)
- **Architecture:** Autoregressive transformer + latent diffusion
- **Innovation:** Contrastive learning for joint CAD-image representations
- **Key Features:**
  - Contrastive learning alignment
  - Latent diffusion models
  - CAD command sequence decoder
- **Source:** https://gencad.github.io/

#### 5. **CAD-MLLM**
- **Architecture:** Multimodal LLM for CAD generation
- **Innovation:** First to support text, images, AND point clouds as input
- **Key Feature:** Unified multimodal conditioning
- **Advantage:** Most flexible input modalities
- **Source:** https://cad-mllm.github.io/

#### 6. **SkexGen** (ICML 2022)
- **Architecture:** Autoregressive with disentangled codebooks
- **Innovation:** Separates different aspects of CAD construction
- **Source:** https://github.com/samxuxiang/SkexGen

---

## 📊 Gap Analysis: Current vs State-of-the-Art

### ❌ Critical Gaps in Our Implementation

| Feature | Our Implementation | State-of-the-Art | Impact |
|---------|-------------------|------------------|--------|
| **Tokenization** | Generic token IDs | CAD-specific vocabulary | 🔴 High |
| **Architecture** | Basic transformer | Modern with Flash Attention, RoPE | 🔴 High |
| **Multi-modal** | Separate CLIP module | Integrated contrastive learning | 🔴 High |
| **Training** | Basic supervised | Sequential + visual feedback | 🔴 High |
| **CAD Operations** | Limited primitives | Full parametric operations | 🔴 High |
| **Sequence Format** | Numeric tokens | Structured operations | 🟡 Medium |
| **Validation** | Basic geometric | Constraint-based + manufacturability | 🟡 Medium |
| **Dataset** | 100 samples | 170K+ models | 🟡 Medium |
| **Attention** | Standard | Cross-attention, flash attention | 🟡 Medium |
| **Diffusion** | None | Latent diffusion for quality | 🟢 Low |

---

## 🎯 Major Improvements Needed

### Priority 1: Critical (Must Have)

#### 1. **CAD-Specific Tokenizer & Vocabulary**
**Current:** Generic numeric tokens
**Needed:**
```python
# Current: [0, 1, 2, 3, ...]
# Improved: ['sketch_rectangle', 'extrude', 'fillet', ...]
```
- **Why:** Text2CAD and CADFusion show this dramatically improves generation
- **Implementation:** Create CAD operation vocabulary with parameters
- **Impact:** Better semantic understanding of CAD operations

#### 2. **Modern Transformer Architecture**
**Current:** Basic transformer decoder
**Needed:**
- Flash Attention (2-4x faster)
- Rotary Position Embeddings (RoPE) - better than learned PE
- Cross-attention layers for text-CAD fusion
- Group Query Attention (GQA) for efficiency

**Why:** All 2024-2025 papers use these
**Impact:** Better quality + faster inference

#### 3. **Contrastive Learning for Multi-Modal Alignment**
**Current:** Separate CLIP scoring
**Needed:**
```python
# Joint embedding space for text and CAD
text_emb = text_encoder(text)
cad_emb = cad_encoder(cad_sequence)
contrastive_loss = InfoNCE(text_emb, cad_emb)
```
- **Why:** GenCAD shows 15-20% improvement
- **Impact:** Better text-CAD alignment

#### 4. **Structured CAD Operations**
**Current:** Flat token sequence
**Needed:**
```python
{
  "operation": "extrude",
  "sketch": "rectangle",
  "params": {"width": 10, "height": 5, "depth": 3},
  "constraints": ["parallel_to_xy", "centered"]
}
```
- **Why:** All modern systems use structured representations
- **Impact:** Enables constraint checking and better validation

### Priority 2: High Value (Should Have)

#### 5. **Visual Feedback Loop** (CADFusion approach)
```python
# Stage 1: Generate sequence
sequence = model.generate(text)

# Stage 2: Render and get visual feedback
rendering = render(sequence)
feedback_score = clip_model(text, rendering)

# Stage 3: Refine based on feedback
refined_sequence = model.refine(sequence, feedback_score)
```

#### 6. **Enhanced Validation**
- **Manufacturing constraints** (min wall thickness, draft angles)
- **Geometric constraints** (parallelism, perpendicularity)
- **Topological checks** (closed volumes, manifold meshes)
- **Material feasibility**

#### 7. **Dataset Augmentation Pipeline**
Based on Text2CAD's annotation pipeline:
- Use LLaVA for image captioning
- Use Mistral for text generation
- Generate multi-level descriptions (beginner to expert)

### Priority 3: Nice to Have

#### 8. **Latent Diffusion** (GenCAD approach)
- Improves generation diversity
- Better quality for complex models
- More stable training

#### 9. **Multi-modal Input Support**
- Text descriptions
- Reference images
- Point clouds
- Sketches

---

## 🏗️ Implementation Plan

### Phase 1: Foundation (Critical - 4 hours)

1. **CAD Operation Vocabulary**
   - Define comprehensive operation set
   - Create tokenizer/detokenizer
   - Add parameter validation

2. **Modern Architecture**
   - Implement Flash Attention
   - Add RoPE positional embeddings
   - Add cross-attention layers

3. **Contrastive Learning**
   - Joint embedding space
   - InfoNCE loss
   - Multi-modal fusion

### Phase 2: Enhancement (High Value - 3 hours)

4. **Visual Feedback Loop**
   - Rendering integration
   - Feedback scoring
   - Iterative refinement

5. **Advanced Validation**
   - Manufacturing constraints
   - Topological validation
   - Constraint solver

### Phase 3: Polish (Nice to Have - 2 hours)

6. **Testing & Benchmarking**
   - Comprehensive test suite
   - Quality metrics
   - Performance benchmarks

---

## 📈 Expected Improvements

### Quantitative Goals

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| **CLIP Score** | ~0.75 | >0.85 | Contrastive learning |
| **Chamfer Distance** | ~1.2mm | <0.87mm | Better architecture |
| **Valid Models** | ~85% | >95% | Enhanced validation |
| **Inference Speed** | ~5s | <2s | Flash attention |
| **Model Size** | 350M | 350M | Same (efficiency gains) |

### Qualitative Goals

- ✅ Generate complex assemblies, not just primitives
- ✅ Handle beginner to expert level descriptions
- ✅ Produce manufacturable CAD models
- ✅ Support parametric constraints
- ✅ Enable iterative refinement

---

## 🔧 Technical Specifications

### New Architecture

```python
class ImprovedTextToCADModel:
    - CADTokenizer: Structured operation vocabulary
    - TextEncoder: BERT with contrastive head
    - CADEncoder: Transformer with operation embeddings
    - CrossAttentionFusion: Multi-modal alignment
    - CADDecoder: Modern transformer with Flash Attention + RoPE
    - ConstraintSolver: Validates geometric constraints
    - VisualFeedback: CLIP-based refinement loop
```

### Training Pipeline

```python
# Stage 1: Contrastive Pre-training
train_contrastive(text_cad_pairs)

# Stage 2: Sequential CAD Generation
train_autoregressive(cad_sequences)

# Stage 3: Visual Feedback Fine-tuning
finetune_with_visual_feedback(text, renderings)
```

---

## 📚 Key Papers & References

1. **Text2CAD** - https://arxiv.org/html/2409.17106v1
2. **CADFusion** - https://arxiv.org/html/2501.19054v1
3. **GenCAD** - https://gencad.github.io/
4. **CAD-MLLM** - https://cad-mllm.github.io/
5. **DeepCAD** - https://github.com/ChrisWu1997/DeepCAD
6. **Flash Attention** - https://arxiv.org/abs/2205.14135
7. **RoPE** - https://arxiv.org/abs/2104.09864

---

## 🎯 Success Criteria

### Minimum Viable Improvements
- ✅ CAD-specific tokenizer working
- ✅ Flash Attention integrated
- ✅ Contrastive learning functional
- ✅ Tests passing with real examples
- ✅ Better than baseline on metrics

### Stretch Goals
- ✅ Visual feedback loop working
- ✅ Manufacturable CAD output
- ✅ Multi-modal input support
- ✅ Published benchmark results

---

## 💡 Innovation Opportunities

### Where We Can Lead

1. **Hybrid Approach**: Combine CADFusion's LLM backbone with GenCAD's diffusion
2. **Real-time Feedback**: Interactive CAD refinement during generation
3. **Manufacturing-First**: Build in manufacturing constraints from start
4. **Multi-format Export**: Seamless conversion between CAD formats
5. **Open Source Leadership**: Most accessible text-to-CAD system

---

**Next Step:** Implement Priority 1 improvements in src/models/
