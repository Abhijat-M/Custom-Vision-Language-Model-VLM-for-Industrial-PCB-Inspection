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

## Export for Deployment

```bash
# Export to both ONNX and TorchScript
python -m src.export_onnx --checkpoint checkpoints/best_model.pth --format both

# ONNX only (for TensorRT)
python -m src.export_onnx --checkpoint checkpoints/best_model.pth --format onnx
```

## Tech Stack

- **Detection**: PyTorch, torchvision (Faster R-CNN + ResNet-50 FPN)
- **VLM**: Qwen2-VL-2B, Hugging Face Transformers, PEFT (LoRA), bitsandbytes (4-bit)
- **API**: FastAPI, Uvicorn
- **Deployment**: Docker, ONNX, TorchScript
- **CI/CD**: GitHub Actions

## License

MIT
