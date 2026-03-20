"""
Quick smoke-test client for the running API.
Usage: python scripts/test_api_client.py --image path/to/pcb.jpg
"""

import argparse
import json
import sys
import time

import requests


def main():
    parser = argparse.ArgumentParser(description="Test PCB Inspector API")
    parser.add_argument("--url", default="http://localhost:8500")
    parser.add_argument("--image", required=True, help="Path to PCB image")
    parser.add_argument("--question", default="What defects are present?")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    base = args.url.rstrip("/")

    # Health check
    print("=" * 60)
    print("PCB INSPECTOR API SMOKE TEST")
    print("=" * 60)

    r = requests.get(f"{base}/health")
    health = r.json()
    print(f"\n[Health]  status={health['status']}  detector={health['detector_loaded']}  "
          f"vlm={health['vlm_loaded']}  device={health['device']}")

    # Detection
    print("\n[Detect]  Sending image...")
    t0 = time.time()
    with open(args.image, "rb") as f:
        r = requests.post(
            f"{base}/detect",
            files={"file": ("image.jpg", f, "image/jpeg")},
            data={"score_threshold": args.threshold},
        )
    elapsed = time.time() - t0

    if r.status_code == 200:
        data = r.json()
        print(f"  Found {len(data['detections'])} defects in {data['inference_time_s']:.3f}s "
              f"(total RTT: {elapsed:.3f}s)")
        for d in data["detections"]:
            print(f"    - {d['class']} ({d['confidence']:.2f}) [{d['severity']}]")
    else:
        print(f"  ERROR {r.status_code}: {r.text}")

    # VLM inspection
    if health.get("vlm_loaded"):
        print(f"\n[Inspect]  Question: {args.question}")
        t0 = time.time()
        with open(args.image, "rb") as f:
            r = requests.post(
                f"{base}/inspect",
                files={"file": ("image.jpg", f, "image/jpeg")},
                data={"question": args.question, "score_threshold": args.threshold},
            )
        elapsed = time.time() - t0

        if r.status_code == 200:
            data = r.json()
            print(f"  Answer: {data['answer']}")
            print(f"  Confidence: {data['confidence']:.3f} | Time: {data['inference_time_s']:.2f}s "
                  f"(RTT: {elapsed:.2f}s)")
        else:
            print(f"  ERROR {r.status_code}: {r.text}")
    else:
        print("\n[Inspect]  Skipped (VLM not loaded)")

    # Metrics
    r = requests.get(f"{base}/metrics")
    if r.status_code == 200:
        m = r.json()
        print(f"\n[Metrics]  requests={m['total_requests']}  "
              f"avg_detect={m['avg_detect_ms']:.0f}ms  errors={m['errors']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
