"""
Production FastAPI server for PCB defect inspection.
Endpoints:
  POST /detect       — Faster R-CNN only (fast, <100ms)
  POST /inspect      — Full VLM pipeline (detect + NL query)
  GET  /health       — Liveness + model readiness
  GET  /metrics      — Prometheus-compatible metrics
"""

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from src.config import CLASSES, INFER, PATHS
from src.detect import classify_severity, detect_single
from src.model import build_model, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("pcb-api")

# ── Response schemas ──────────────────────────────────────────────────────


class Detection(BaseModel):
    class_name: str = Field(alias="class")
    confidence: float
    bbox: List[float]
    severity: str

    class Config:
        populate_by_name = True


class DetectResponse(BaseModel):
    detections: List[Detection]
    image_size: List[int]
    inference_time_s: float
    model: str = "faster-rcnn-resnet50-fpn"


class InspectResponse(BaseModel):
    answer: str
    detections: List[Detection]
    confidence: float
    inference_time_s: float
    model: str = "qwen2-vl-2b + faster-rcnn"


class HealthResponse(BaseModel):
    status: str
    detector_loaded: bool
    vlm_loaded: bool
    device: str
    gpu_memory_mb: Optional[float] = None


class MetricsResponse(BaseModel):
    total_requests: int
    total_detect: int
    total_inspect: int
    avg_detect_ms: float
    avg_inspect_ms: float
    errors: int
    uptime_s: float


# ── Global state ──────────────────────────────────────────────────────────

_state = {
    "detector": None,
    "inspector": None,
    "device": None,
    "start_time": time.time(),
    "detect_count": 0,
    "inspect_count": 0,
    "detect_total_ms": 0.0,
    "inspect_total_ms": 0.0,
    "errors": 0,
}


def _load_models():
    """Load detector (always) and VLM (if adapter exists)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _state["device"] = device

    # ── Detector ──
    ckpt = os.environ.get("DETECTOR_CHECKPOINT", os.path.join(PATHS.checkpoint_dir, "best_model.pth"))
    if not os.path.isfile(ckpt):
        logger.warning(f"Detector checkpoint not found: {ckpt}")
        return

    logger.info(f"Loading detector from {ckpt} on {device}")
    model = build_model()
    load_checkpoint(model, ckpt, device)
    model.to(device).eval()
    _state["detector"] = model
    logger.info("Detector loaded ✓")

    # ── VLM (optional) ──
    adapter_path = os.environ.get(
        "VLM_ADAPTER_PATH",
        os.path.join(PATHS.checkpoint_dir, "vlm", "lora_adapter"),
    )
    if os.path.isdir(adapter_path):
        try:
            from src.vlm.inference import PCBInspector

            inspector = PCBInspector(
                detector_checkpoint=ckpt,
                vlm_adapter_path=adapter_path,
                device=device,
            )
            _state["inspector"] = inspector
            logger.info("VLM inspector loaded ✓")
        except Exception as e:
            logger.warning(f"VLM loading failed (API will run detector-only): {e}")
    else:
        logger.info("No VLM adapter found — running detector-only mode")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield
    logger.info("Shutting down…")


# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PCB Defect Inspector API",
    version="1.0.0",
    description="Production-grade PCB defect detection and VLM-powered inspection",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_image(file: UploadFile) -> Image.Image:
    """Validate and read uploaded image."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected image, got {file.content_type}")
    try:
        data = file.file.read()
        if len(data) > 20 * 1024 * 1024:  # 20 MB limit
            raise HTTPException(status_code=413, detail="Image exceeds 20 MB limit")
        return Image.open(io.BytesIO(data)).convert("RGB")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness and readiness check."""
    gpu_mem = None
    device = _state["device"]
    if device and device.type == "cuda":
        gpu_mem = round(torch.cuda.memory_allocated(device) / 1024 / 1024, 1)
    return HealthResponse(
        status="ok" if _state["detector"] else "degraded",
        detector_loaded=_state["detector"] is not None,
        vlm_loaded=_state["inspector"] is not None,
        device=str(device),
        gpu_memory_mb=gpu_mem,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Prometheus-compatible metrics summary."""
    detect_avg = (_state["detect_total_ms"] / _state["detect_count"]) if _state["detect_count"] else 0
    inspect_avg = (_state["inspect_total_ms"] / _state["inspect_count"]) if _state["inspect_count"] else 0
    return MetricsResponse(
        total_requests=_state["detect_count"] + _state["inspect_count"],
        total_detect=_state["detect_count"],
        total_inspect=_state["inspect_count"],
        avg_detect_ms=round(detect_avg, 1),
        avg_inspect_ms=round(inspect_avg, 1),
        errors=_state["errors"],
        uptime_s=round(time.time() - _state["start_time"], 1),
    )


@app.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(...),
    score_threshold: float = Form(default=INFER.score_threshold),
):
    """
    Fast Faster R-CNN detection only.
    Returns bounding boxes, class labels, confidence scores, and severity.
    """
    if _state["detector"] is None:
        raise HTTPException(status_code=503, detail="Detector not loaded")

    image = _read_image(file)

    try:
        t0 = time.time()
        result = detect_single(_state["detector"], image, _state["device"], score_threshold)
        elapsed = time.time() - t0

        _state["detect_count"] += 1
        _state["detect_total_ms"] += elapsed * 1000

        return DetectResponse(
            detections=[Detection(**d) for d in result["detections"]],
            image_size=result["image_size"],
            inference_time_s=round(elapsed, 4),
        )
    except Exception as e:
        _state["errors"] += 1
        logger.exception("Detection failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inspect", response_model=InspectResponse)
async def inspect(
    file: UploadFile = File(...),
    question: str = Form(default="What defects are present in this PCB image?"),
    score_threshold: float = Form(default=INFER.score_threshold),
):
    """
    Full VLM-powered inspection.
    Runs Faster R-CNN detection, then answers a natural language question
    about the PCB image using the fine-tuned Qwen2-VL model.
    """
    if _state["inspector"] is None:
        raise HTTPException(
            status_code=503,
            detail="VLM inspector not loaded. Use /detect for detector-only mode.",
        )

    image = _read_image(file)

    try:
        t0 = time.time()
        result = _state["inspector"].query(image, question)
        elapsed = time.time() - t0

        _state["inspect_count"] += 1
        _state["inspect_total_ms"] += elapsed * 1000

        return InspectResponse(
            answer=result["answer"],
            detections=[Detection(**d) for d in result["detections"]],
            confidence=result["confidence"],
            inference_time_s=round(elapsed, 4),
        )
    except Exception as e:
        _state["errors"] += 1
        logger.exception("Inspection failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classes")
async def get_classes():
    """Return supported defect classes."""
    return {"classes": CLASSES, "num_classes": len(CLASSES)}


# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        workers=1,  # Single worker — models share GPU
        log_level="info",
    )
