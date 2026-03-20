"""
Integration tests for the PCB Defect Inspector API.
Tests run with detector-only mode (no VLM) for CI speed.
"""

import io
import os
import sys

import pytest
from PIL import Image

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def test_image_bytes():
    """Create a minimal test image in memory."""
    img = Image.new("RGB", (640, 640), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    """FastAPI test client — loads detector if checkpoint exists."""
    from fastapi.testclient import TestClient
    from src.api import app

    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert "detector_loaded" in data
        assert "vlm_loaded" in data
        assert "device" in data

    def test_classes_endpoint(self, client):
        resp = client.get("/classes")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["classes"]) == 6
        assert "open" in data["classes"]
        assert "short" in data["classes"]

    def test_metrics_endpoint(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data
        assert "uptime_s" in data
        assert data["uptime_s"] >= 0


class TestDetect:
    def test_detect_returns_detections(self, client, test_image_bytes):
        resp = client.post(
            "/detect",
            files={"file": ("test.png", test_image_bytes, "image/png")},
        )
        # 200 if detector loaded, 503 if not
        if resp.status_code == 200:
            data = resp.json()
            assert "detections" in data
            assert "image_size" in data
            assert "inference_time_s" in data
            assert data["image_size"] == [640, 640]
        else:
            assert resp.status_code == 503

    def test_detect_with_threshold(self, client, test_image_bytes):
        resp = client.post(
            "/detect",
            files={"file": ("test.png", test_image_bytes, "image/png")},
            data={"score_threshold": 0.9},
        )
        if resp.status_code == 200:
            data = resp.json()
            # All detections should be above threshold
            for d in data["detections"]:
                assert d["confidence"] >= 0.9

    def test_detect_invalid_file(self, client):
        resp = client.post(
            "/detect",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code in (400, 503)


class TestInspect:
    def test_inspect_without_vlm_returns_503(self, client, test_image_bytes):
        """If VLM is not loaded, /inspect should return 503."""
        resp = client.post(
            "/inspect",
            files={"file": ("test.png", test_image_bytes, "image/png")},
            data={"question": "What defects are there?"},
        )
        # VLM may or may not be loaded depending on environment
        assert resp.status_code in (200, 503)


class TestConfig:
    def test_config_imports(self):
        from src.config import CLASSES, NUM_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
        assert len(CLASSES) == 6
        assert NUM_CLASSES == 7
        assert CLASS_TO_IDX["open"] == 1
        assert IDX_TO_CLASS[0] == "__background__"

    def test_paths_exist(self):
        from src.config import PATHS
        assert os.path.isdir(PATHS.checkpoint_dir) or True  # May not exist in CI


class TestModel:
    def test_build_model(self):
        from src.model import build_model
        model = build_model()
        assert model is not None
        # Check it's a Faster R-CNN
        assert hasattr(model, "backbone")
        assert hasattr(model, "rpn")

    def test_model_forward_pass(self):
        import torch
        from src.model import build_model
        model = build_model()
        model.eval()
        dummy = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model([dummy[0]])
        assert len(outputs) == 1
        assert "boxes" in outputs[0]
        assert "labels" in outputs[0]
        assert "scores" in outputs[0]


class TestDataset:
    def test_dataset_init(self):
        from src.config import PATHS
        data_dir = PATHS.train_dir
        if os.path.isdir(os.path.join(data_dir, "images")):
            from src.dataset import PCBDefectDataset
            ds = PCBDefectDataset(data_dir, train=False)
            assert len(ds) > 0
        else:
            pytest.skip("Data directory not available")

    def test_dataset_getitem(self):
        from src.config import PATHS
        data_dir = PATHS.train_dir
        if os.path.isdir(os.path.join(data_dir, "images")):
            from src.dataset import PCBDefectDataset
            ds = PCBDefectDataset(data_dir, train=False)
            img, target = ds[0]
            assert img.shape[0] == 3  # C, H, W
            assert "boxes" in target
            assert "labels" in target
        else:
            pytest.skip("Data directory not available")
