# GroundDerm: Training-Free Spatially-Verified Concept-Based Explanations from Vision–Language Models for Dermoscopic Skin Lesion Diagnosis

## Overview

GroundDerm is a fully training-free pipeline for dermoscopic melanoma diagnosis that closes the spatial reliability gap in existing vision-language pipelines. Prior training-free methods score clinical concepts (e.g., "blue-whitish veil") over the full image field — a high score can be driven entirely by an out-of-lesion artefact, costing up to 20 percentage points in balanced accuracy. GroundDerm verifies each concept's spatial origin at inference time using text-conditioned GradCAM cross-referenced against an automatic lesion mask, producing a bounded reliability score r_k ∈ [0,1] that vanishes when spatial fidelity or semantic confidence is low. Verified concepts are stratified into confidence tiers and passed to a frozen MedLLaMA2-7B for constrained single-token decoding.

**Key result:** 87.34% ± 14.14% balanced accuracy on PH2 (5-fold CV, n=200), +2.29 pp above the prior training-free state of the art, while jointly providing per-concept spatial heatmaps and reliability scores absent from all competing methods.

**No training. No weight updates. No pixel-level annotations. No backbone changes.**

## Pipeline

![GroundDerm Framework](imgs/framework.png)

GroundDerm extends the standard concept-bottleneck formulation x → c → y with an interposed spatial verification stage:

```
x ──Stage 1──▶ s ──Stages 2–3──▶ s* ──Stages 4–6──▶ ŷ
```

where **s** ∈ [0,1]^N is the raw concept score vector and **s\*** is the spatially verified subset passed to the LLM. The lesion mask M is generated in parallel and feeds exclusively into Stage 3, sharing no parameters with the diagnostic pipeline.

```
Input dermoscopic image
    │
    ├──► [Lesion segmentation] DeepLabV3+ (ResNet-101) → mask M
    │
    ▼
Stage 1: VLM-Based Concept Extraction
    BiomedCLIP (frozen) × 8 concepts × T=3 templates
    → softmax concept scores s_k ∈ [0,1]            (~0.15s)
    │
Stage 2: Text-Conditioned Spatial Grounding
    GradCAM on frozen BiomedCLIP, L=4 transformer layers
    → per-concept heatmaps H_k ∈ [0,1]^{H×W}        (~1.20s)
    │
Stage 3: Spatial Reliability Verification          ← core contribution
    r_k = coverage(H_k, M) × σ(s_k) ∈ [0,1]
    → 3 tiers: Reliable (r_k ≥ ρ) / Uncertain / Unreliable (<ρ/2)
    → Unreliable concepts filtered from prompt        (<0.01s)
    │
Stage 4: Few-Shot Retrieval (PICES, K=1)
    BiomedCLIP embedding similarity → retrieve K training cases (~0.05s)
    │
Stage 5: Graduated Confidence Prompt Construction
    [PRESENT|HIGH CONF] / [ABSENT|HIGH CONF]    for r_k ≥ ρ
    [PRESENT|MODERATE CONF] / [ABSENT|MODERATE CONF]  for ρ/2 ≤ r_k < ρ
    │
Stage 6: LLM-Based Diagnostic Inference
    MedLLaMA2-7B (frozen), τ=0, single-token {A=Nevus, B=Melanoma}
    → ŷ                                              (~0.70s)
    │
    Total: ~2.1s per image
```

## Clinical Concept Vocabulary

Eight dermoscopic concepts grounded in the 7-Point Checklist and ABCD rule:

| k | Concept | Clinical Significance | Source |
|---|---------|----------------------|--------|
| 1 | Atypical Pigment Network | Irregular melanocytic proliferation | 7-Point |
| 2 | Typical Pigment Network | Regular honeycomb pattern (benign) | ABCD |
| 3 | Blue-Whitish Veil | Dermal melanosis; melanoma indicator | 7-Point |
| 4 | Irregular Dots & Globules | Asymmetric melanocytic nests | 7-Point |
| 5 | Regular Dots & Globules | Symmetric benign structures | ABCD |
| 6 | Irregular Streaks | Radial growth-phase indicator | 7-Point |
| 7 | Regression Structures | White scar-like / blue-grey peppering | 7-Point |
| 8 | Irregular Pigmentation | Asymmetric blotch distribution | ABCD |

## Stage Details

**Stage 1 — VLM-Based Concept Extraction**
BiomedCLIP (ViT-B/16 + PubMedBERT, pretrained on PMC-15M) is used frozen throughout. Input is a 224×224 crop of the lesion bounding box expanded by 15%. Each concept is scored via T=3 syntactically diverse prompt templates, mean-pooled and softmax-normalised against contrastive references. Domain pretraining matters: BiomedCLIP achieves 24.2% mean per-concept F1 vs 14.3% for general-domain CLIP.

**Stage 2 — Text-Conditioned Spatial Grounding**
GradCAM gradients are computed exactly via automatic differentiation (no parameter updates) through the frozen encoder. Heatmaps are aggregated across the last L=4 transformer layers to capture both low-level texture and high-level semantics, then bilinearly upsampled and min-max normalised to H_k ∈ [0,1]^{H×W}. Operates on the full uncropped image so heatmaps share the mask's coordinate frame.

**Stage 3 — Spatial Reliability Verification (core contribution)**
The reliability score multiplicatively couples spatial fidelity with semantic confidence:

$$r_k = \underbrace{\frac{\sum_{i,j} H_k(i,j) \cdot M(i,j)}{\sum_{i,j} H_k(i,j) + \epsilon}}_{\text{coverage}(H_k, M)} \times \underbrace{\sigma(s_k)}_{\text{calibrated confidence}}$$

Formal properties: (1) Boundedness — r_k ∈ [0,1]; (2) Monotonicity — r_k → 0 when heatmap energy lies outside the mask OR σ(s_k) → 0; (3) Mutual requirement — r_k approaches its maximum only when both spatial fidelity and semantic confidence are simultaneously high. No competing method produces this continuous score.

**Stage 4 — Few-Shot Retrieval**
PICES strategy (K=1): retrieve the training case whose BiomedCLIP embedding is most cosine-similar to the query. Using the same encoder for retrieval and concept scoring ensures embedding-space similarity corresponds to concept-feature similarity. K=1 is optimal: K=2 trades 1.31 pp mean BAcc for 40% reduction in standard deviation.

**Stage 5 — Graduated Confidence Prompt Construction**
Unreliable concepts (r_k < ρ/2) are hard-filtered. A minimum-concept guarantee ensures ≥3 concepts always reach the LLM (triggered in 29% of images). Surviving concepts receive graduated confidence tags; presence threshold τ_pres=0.2 (deliberately lower than the Stage 1 binarisation threshold τ_abs=0.5 to preserve borderline signals).

**Stage 6 — LLM-Based Diagnostic Inference**
MedLLaMA2-7B (LLaMA-2 fine-tuned on PMC-LLaMA), temperature τ=0, vocabulary logit mask restricting generation to {A, B}. Constrained decoding prevents intermediate reasoning from overriding prompt evidence — chain-of-thought is catastrophically harmful at 7B scale (−18.69% BAcc, sensitivity collapses to 37.3%).

## Results

### Main Results on PH2 (5-Fold CV, ρ=0.35, K=1)

| | BAcc (%) | Sensitivity (%) | Specificity (%) | F1 (%) |
|---|----------|----------------|----------------|--------|
| Fold 0 | 62.54 | 44.44 | 80.65 | 42.11 |
| Fold 1 | 97.06 | 100.00 | 94.12 | 85.71 |
| Fold 2 | 98.21 | 100.00 | 96.43 | 96.00 |
| Fold 3 | 80.39 | 66.67 | 94.12 | 66.67 |
| Fold 4 | 98.48 | 100.00 | 96.97 | 93.33 |
| **Mean ± Std** | **87.34 ± 14.14** | **82.22 ± 22.88** | **92.46 ± 6.02** | **76.76 ± 20.14** |

GroundDerm surpasses the prior training-free SOTA (Patrício et al. Two-Step, 85.05%) by +2.29 pp while additionally providing per-concept heatmaps and reliability scores absent from that method.

### Per-Concept Analysis

| Concept | F1 (%) | Mel Rate | Nev Rate | Δs | Reliable | Uncertain | Unreliable |
|---------|--------|----------|----------|----|----------|-----------|------------|
| Regression Structures | 52.29 | 0.700 | 0.338 | +0.182 | 89 | 67 | 44 |
| Blue-Whitish Veil | 47.88 | **0.800** | 0.150 | **+0.501** | 55 | 32 | 113 |
| Irreg. Dots & Glob. | 40.93 | 0.275 | 0.062 | +0.088 | 77 | 57 | 66 |
| Irregular Streaks | 33.75 | 0.125 | 0.019 | +0.034 | 46 | 38 | 116 |
| Regular Streaks | 17.11 | 0.625 | 0.588 | −0.011 | 41 | 25 | 134 |
| Reg. Dots & Glob. | 1.67 | 0.000 | 0.000 | −0.022 | 65 | 33 | 102 |
| Typical Pigment Network | 0.00 | 0.000 | 0.000 | −0.026 | 134 | 46 | 20 |
| Atypical Pigment Network | 0.00 | 0.000 | 0.000 | +0.005 | 142 | 42 | 16 |
| **Average / Total** | **24.20** | | | | **649** | **340** | **611** |

38.2% of the 1,600 concept evaluations are classified as unreliable and suppressed from the prompt. Blue-whitish veil (Δs=+0.501) is the strongest discriminator.

## Ablation Study

### Component Contribution (PH2, 5-fold CV)

| Configuration | BAcc (%) | Std | Δ |
|---------------|----------|-----|---|
| **Full GroundDerm** | **87.34** | 14.14 | — |
| w/o spatial grounding | 87.99 | 12.19 | +0.65 |
| w/o reliability filtering | 85.54 | 13.87 | −1.80 |
| w/o graduated confidence | 80.69 | 8.26 | −6.65 |
| w/o few-shot retrieval (K=0) | 80.36 | 8.36 | −6.98 |

Graduated confidence prompting (+6.65%) and few-shot retrieval (+6.98%) are the dominant performance levers. Spatial grounding's primary value is the clinical infrastructure it provides (per-concept heatmaps, auditable reasoning) rather than direct accuracy gain on this fixed test set.

### Prompting Strategy Comparison (PH2, 5-fold CV)

| Strategy | BAcc (%) | Sensitivity (%) | Specificity (%) | Δ |
|----------|----------|----------------|----------------|---|
| **Direct (graduated conf.)** | **87.34 ± 14.14** | 82.22 | 92.46 | — |
| Binary (present/absent) | 80.69 ± 8.26 | 77.78 | 83.61 | −6.65 |
| Chain-of-Thought | 68.65 ± 9.48 | 37.30 | 100.00 | −18.69 |
| CoT Self-Consistency | 66.74 ± 8.86 | 34.13 | 99.35 | −20.60 |

Chain-of-thought prompting is **catastrophically harmful** at 7B medical-LLM scale: sensitivity collapses to 37.3% while specificity reaches 100%, revealing a systematic over-conservative corpus-prior bias. Constrained single-token decoding is the correct strategy for sub-10B medical LLMs.

### Hyperparameter Grid: ρ × K

| ρ | K=0 | K=1 | K=2 |
|---|-----|-----|-----|
| 0.20 | — | 87.96 ± 14.21 | 84.28 ± 8.31 |
| 0.25 | — | 87.04 ± 14.30 | — |
| 0.30 | — | 87.63 ± 14.01 | 84.92 ± 10.50 |
| 0.35 | 80.36 ± 8.36 | **87.34 ± 14.14** | **86.03 ± 8.43** |

BAcc varies <1 pp across all ρ values at K=1. K dominates performance: K=0→K=1 gives +6.98 pp regardless of ρ. K=2 at ρ=0.35 is recommended for deployment (best accuracy–variance trade-off: 86.03% ± 8.43%).

## Inference Time Breakdown

| Stage | Time (s) | Operation |
|-------|----------|-----------|
| 1: VLM scoring | ~0.15 | 8 forward passes, BiomedCLIP |
| 2: GradCAM maps | ~1.20 | 8 backward passes (4 layers each) |
| 3: Reliability | <0.01 | Matrix multiply, CPU |
| 4: Retrieval | ~0.05 | Cosine sim., pre-cached embeddings |
| 5: Prompt build | <0.01 | String formatting, CPU |
| 6: LLM decode | ~0.70 | 1-token decode, MedLLaMA2-7B |
| **Total** | **~2.11** | RTX 3090 (24 GB) |

Full 5-fold CV over 200 images completes in under 15 minutes.

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- BiomedCLIP (ViT-B/16 + PubMedBERT)
- MedLLaMA2-7B (via LM Studio, local inference)
- DeepLabV3+ (ResNet-101, for lesion segmentation)
- NVIDIA GPU with ≥24 GB VRAM (RTX 3090)

```bash
pip install torch torchvision transformers open-clip-torch opencv-python numpy scipy matplotlib
```

## Dataset

PH2: 200 dermoscopic images with binary diagnostic labels (80 melanomas, 120 nevi), ground-truth segmentation masks, and expert annotations for 8 dermoscopic concepts.

| Dataset | Total | Train (per fold) | Test (per fold) | Folds |
|---------|-------|-----------------|----------------|-------|
| PH2 | 200 | 160 | 40 | 5 (stratified) |

Ground-truth masks are used in ablation experiments to isolate mask quality; predicted DeepLabV3+ masks are used in fully automated inference. Sensitivity to mask quality is analysed in the paper.

## Inference

```bash
python run_pipeline.py \
  --image path/to/dermoscopic_image.jpg \
  --threshold 0.35 \
  --k_shot 1
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{alharith2025groundderm,
  title   = {{GroundDerm}: Training-Free Spatially-Verified Concept-Based Explanations
             from Vision--Language Models for Dermoscopic Skin Lesion Diagnosis},
  author  = {Alharith, Razan and Qasim, Kaleem Ullah and Zhang, Jiashu and Al-Huda, Zaid},
  year    = {2025},
  url     = {https://github.com/razanharith/GroundDerm}
}
```

## Contact

For questions about this research, contact Razan Alharith at razanalharith@my.swjtu.edu.cn.
