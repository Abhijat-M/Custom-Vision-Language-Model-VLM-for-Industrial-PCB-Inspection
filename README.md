# Custom Vision-Language Model for Industrial PCB Inspection

Production-grade PCB defect detection system combining **Faster R-CNN** (ResNet-50 FPN) for precise defect localization with **Qwen2-VL-2B** (LoRA fine-tuned) for natural language inspection queries.

## Architecture

```
PCB Image ──► Faster R-CNN (ResNet-50 FPN) ──► Bounding Boxes + Classes + Severity
                                                         │
                                                         ▼
              Qwen2-VL-2B (LoRA) ◄── Grounded Prompt + Image ──► Natural Language Answer
                                                         │
                                                         ▼
                                              Confidence Calibration
                                          (hallucination detection)
```

## Problem Statement

Traditional Automated Optical Inspection (AOI) systems detect defects but lack interpretability. Generic Vision-Language Models, while expressive, are not reliable for industrial use due to hallucinations, poor localization, and deployment constraints.

**Goal:** Build a VLM that produces **trustworthy, grounded explanations** for PCB defects with:
- Sub-2-second detection, ~3-4s for full VLM query
- Offline / on-prem deployment
- High localization accuracy
- Strong hallucination control

## Defect Classes

| Class | Description |
|-------|-------------|
| `open` | Open circuit — broken trace |
| `short` | Short circuit — unintended connection |
| `mousebite` | Irregular edge nibbling |
| `spur` | Unwanted copper protrusion |
| `copper` | Excess copper residue |
| `pin-hole` | Small hole in copper layer |

## Results

| Component | Metric | Value |
|-----------|--------|-------|
| Faster R-CNN | Val Loss | 0.3388 |
| Faster R-CNN | Inference | ~50ms/image (GPU) |
| Faster R-CNN | Detection Confidence | 0.98–1.00 |
| VLM (LoRA) | Train Loss | 0.181 |
| VLM (LoRA) | Eval Loss | 0.211 |
| VLM (LoRA) | Confidence Calibration | 0.947 |
| VLM (LoRA) | Query Response Time | ~3–4s |

## Quick Start

### 1. Install

```bash
git clone https://github.com/Abhijat-M/Custom-Vision-Language-Model-VLM-for-Industrial-PCB-Inspection.git
cd Custom-Vision-Language-Model-VLM-for-Industrial-PCB-Inspection
pip install -r requirements.txt
```

### 2. Train Detector

```bash
python -m src.train --data-dir /path/to/pcb/data --epochs 50
```

### 3. Fine-tune VLM

```bash
# Generate QA pairs from annotations
python -m src.vlm.generate_qa --data-dir /path/to/pcb/data --output data/qa_pairs.json

# Fine-tune Qwen2-VL with LoRA
python -m src.vlm.finetune --qa-json data/qa_pairs.json --data-dir /path/to/pcb/data --epochs 3
```

### 4. Run API Server

```bash
python scripts/run_api.py
# Server starts at http://localhost:8500
```

### 5. Query the API

```bash
# Fast detection only (~50ms)
curl -X POST http://localhost:8500/detect \
  -F "file=@pcb_image.jpg"

# Full VLM inspection (~3-4s)
curl -X POST http://localhost:8500/inspect \
  -F "file=@pcb_image.jpg" \
  -F "question=What defects are present?"

# Health check
curl http://localhost:8500/health
```

### 6. Docker

```bash
docker-compose up --build
```

## API Endpoints

| Endpoint | Method | Description | Latency |
|----------|--------|-------------|---------|
| `/detect` | POST | Faster R-CNN detection only | ~50ms |
| `/inspect` | POST | Full VLM-powered inspection | ~3-4s |
| `/health` | GET | Liveness + model readiness | <1ms |
| `/metrics` | GET | Request counts, latency stats | <1ms |
| `/classes` | GET | Supported defect classes | <1ms |

## Project Structure

```
pcb-vlm-inspection/
├── src/
│   ├── config.py           # Central configuration
│   ├── dataset.py          # PCB dataset with augmentation
│   ├── model.py            # ResNet-50 FPN Faster R-CNN
│   ├── train.py            # Training with early stopping
│   ├── detect.py           # Detection + visualization
│   ├── evaluate.py         # mAP evaluation
│   ├── export_onnx.py      # ONNX + TorchScript export
│   ├── api.py              # FastAPI production server
│   └── vlm/
│       ├── generate_qa.py  # Synthetic QA generation
│       ├── dataset.py      # VLM training dataset
│       ├── finetune.py     # LoRA fine-tuning
│       └── inference.py    # NL query interface
├── tests/
│   └── test_api.py         # Unit + integration tests
├── scripts/
│   ├── run_api.py          # API launcher
│   └── test_api_client.py  # Smoke test client
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/ci.yml
```

## System Design

### Model Selection: Qwen2-VL-2B-Instruct

| Model | Limitation |
|-------|------------|
| GPT-4V | API-only, no offline control |
| LLaVA | Weak spatial grounding |
| BLIP-2 | Limited reasoning capacity |
| Kosmos-2 | Insufficient localization accuracy |
| **Qwen2-VL-2B** | **Best trade-off: accuracy, size, deployability** |

### Training Strategy

1. **Synthetic QA Generation** — Convert bounding box annotations into QA pairs (counting, localization, classification, severity)
2. **Detector Training** — Fine-tune Faster R-CNN (ResNet-50 FPN) on all 6 defect classes with augmentation
3. **VLM LoRA Fine-tuning** — Fine-tune Qwen2-VL-2B with LoRA (rank=16, α=32) on generated QA pairs
4. **Calibration** — Confidence calibration via detector-VLM consistency checking

### Hallucination Mitigation

- **Grounded prompts** — VLM receives detector results as context, must answer consistently
- **Confidence calibration** — Cross-checks VLM answer against detector findings
- **Negative samples** — Defect-free images included in training
- **Penalty scoring** — Hallucinated classes penalize confidence by 0.2 per class

### Inference Optimization

- **4-bit quantization** (bitsandbytes NF4) — 2B model fits in ~2GB VRAM
- **LoRA adapters** — 74MB vs full fine-tune
- **ONNX export** — TensorRT-ready for edge deployment
- **Two-tier API** — Fast `/detect` (50ms) for volume, `/inspect` (3-4s) for detailed queries

## Export for Deployment

```bash
# Export to both ONNX and TorchScript
python -m src.export_onnx --checkpoint checkpoints/best_model.pth --format both

# ONNX only (for TensorRT)
python -m src.export_onnx --checkpoint checkpoints/best_model.pth --format onnx
```

## Evaluation Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Detection Confidence | ≥ 0.90 | **0.98–1.00** |
| Hallucination Rate | ≤ 5% | **<3%** (calibration score 0.947) |
| Detection Latency | ≤ 100ms | **~50ms** |
| VLM Query Latency | ≤ 4s | **~3-4s** |

## Tech Stack

- **Detection**: PyTorch, torchvision (Faster R-CNN + ResNet-50 FPN)
- **VLM**: Qwen2-VL-2B, Hugging Face Transformers, PEFT (LoRA), bitsandbytes (4-bit)
- **API**: FastAPI, Uvicorn
- **Deployment**: Docker, ONNX, TorchScript
- **CI/CD**: GitHub Actions

## License

MIT
