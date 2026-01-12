
# Task 3: Custom Vision–Language Model (VLM) for Industrial PCB Inspection

## Overview

This repository documents the **design of a custom Vision–Language Model (VLM)** for explainable PCB defect inspection in semiconductor manufacturing. The proposed system enables inspectors to ask **natural-language questions** about PCB images and receive **structured, visually grounded responses** describing defect type, location, confidence, and severity.

The design prioritizes **industrial constraints**:

* Sub-2-second inference
* Offline / on-prem deployment
* High localization accuracy
* Strong hallucination control
* Scalable GPU inference for multiple users


---

## Problem Statement

Traditional Automated Optical Inspection (AOI) systems detect defects but lack interpretability. Generic Vision–Language Models, while expressive, are not reliable for industrial use due to hallucinations, poor localization, and deployment constraints.

**Goal:**
Design a VLM that produces **trustworthy, grounded explanations** for PCB defects using only:

* 50,000 PCB images
* Bounding box annotations
* No pre-existing QA pairs

---

## Task-3 Objectives

This README addresses all Task-3 requirements:

* Custom VLM design
* Image → explanation pipeline
* Architecture explanation
* Training strategy (without QA data)
* Inference optimization (<2s)
* Hallucination mitigation
* Cloud GPU inference & scaling
* MLOps considerations

---

## Model Selection

### Selected Base Model: **Qwen-VL (7B)**

**Why Qwen-VL?**

* Native support for **visual grounding**
* Strong spatial reasoning via cross-modal attention
* Balanced model size for real-time inference
* Fully open weights (Apache-2.0 license)
* Supports LoRA, quantization, pruning, and distillation

### Alternatives Considered

| Model    | Limitation                         |
| -------- | ---------------------------------- |
| GPT-4V   | API-only, no offline control       |
| LLaVA    | Weak spatial grounding             |
| BLIP-2   | Limited reasoning capacity         |
| Kosmos-2 | Insufficient localization accuracy |

Qwen-VL provides the best trade-off between **accuracy, control, and deployability**.

---

## System Architecture

### High-Level Pipeline

```
PCB Image
   ↓
Vision Encoder (multi-scale)
   ↓
Cross-Modal Fusion
   ↓
Grounded Language Decoder
   ↓
Structured Defect Explanation
```

### Core Components

#### Vision Encoder

* Dual-resolution processing (global layout + fine defects)
* Edge- and texture-aware feature extraction
* Feature pyramid for scale robustness

#### Cross-Modal Fusion

* Multi-layer cross-attention
* Language queries attend to spatial visual tokens
* Spatial positional encoding for region-aware reasoning

#### Language Decoder

* Structured output (not free-form captions)
* Explicit confidence and severity reporting
* Constrained generation to prevent malformed responses

---

## Example Output (Conceptual)

**User Query:**
“How many defects are present and where are they located?”

**Model Response:**

* Total defects: 3
* Defect 1: Open circuit, top-left region, high confidence
* Defect 2: Short circuit, center-right region, high severity
* Defect 3: Mousebite, bottom-left region, low severity

All statements are **explicitly grounded in image regions**.

---

## Training Strategy

### Stage 1: Synthetic QA Generation

* Convert bounding boxes into synthetic QA pairs:

  * Counting
  * Localization
  * Classification
* Include defect-free images as negative samples

### Stage 2: Vision Encoder Adaptation

* Fine-tune visual backbone for PCB textures
* Emphasize edge-level and micro-defect patterns

### Stage 3: Cross-Modal Fine-Tuning

* Freeze most of the backbone
* Fine-tune fusion layers and decoder using LoRA

### Stage 4: Calibration

* Temperature scaling on validation set
* Threshold tuning for abstention behavior

---

## Hallucination Mitigation

Hallucination is treated as a **reliability problem**, not a language problem.

### Key Techniques

* Explicit grounding enforcement
* Negative (defect-free) training samples
* Confidence calibration (ECE monitoring)
* Abstention when visual evidence is weak
* Post-generation visual verification

**Target hallucination rate:** ≤ 5%

---

## Inference Optimization (<2s)

### Model-Level

* INT8 quantization
* LoRA fine-tuning
* Structured pruning
* Knowledge distillation

### Deployment-Level

* TensorRT compilation
* Static micro-batching
* GPU memory pre-allocation

**Target latency:** ~1.5–1.8 seconds per image

---

### Scaling Strategy

* Stateless inference containers
* Kubernetes GPU autoscaling
* Horizontal scaling based on GPU utilization
* Micro-batching to maximize throughput

### Expected Throughput

* ~12–15 images/sec per GPU
* Linear scaling with additional GPUs
* Stable latency under concurrent load

---

## MLOps Considerations

* Model versioning and registry
* Reproducible training pipelines
* Latency and confidence drift monitoring
* Human-in-the-loop feedback for low-confidence cases
* Safe rollback on performance degradation

---

## Evaluation Metrics

| Metric                 | Target |
| ---------------------- | ------ |
| Localization (mAP@0.5) | ≥ 85%  |
| Hallucination Rate     | ≤ 5%   |
| Inference Latency      | ≤ 2s   |
| Calibration (ECE)      | ≤ 0.10 |

---

## Conclusion

This Task-3 design demonstrates how a Vision–Language Model can be **adapted, optimized, and deployed** for safety-critical industrial inspection. The emphasis on grounding, calibration, and deployment realism reflects **modern applied computer vision research**, bridging the gap between academic VLMs and factory-ready systems.

