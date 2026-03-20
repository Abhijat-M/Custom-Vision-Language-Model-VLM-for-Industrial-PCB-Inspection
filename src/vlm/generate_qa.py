"""
Synthetic QA pair generation from Pascal VOC annotations.
Creates training data for the VLM by generating diverse question-answer pairs
about PCB defect images using annotation ground truth.
"""

import json
import os
import random
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Dict, List, Tuple

from src.config import CLASS_TO_IDX, CLASSES, SEVERITY


# ── Question templates ─────────────────────────────────────────────────────

DETECTION_QUESTIONS = [
    "What defects are present in this PCB image?",
    "Identify all defects visible in this circuit board.",
    "List the defects found on this PCB.",
    "What quality issues can you see in this PCB?",
    "Describe the defects detected in this board image.",
    "Are there any manufacturing defects on this PCB?",
]

COUNT_QUESTIONS = [
    "How many defects are in this PCB image?",
    "Count the total number of defects on this board.",
    "How many quality issues are present?",
]

CLASS_SPECIFIC_QUESTIONS = {
    "open": [
        "Are there any open circuit defects?",
        "Can you identify broken traces on this PCB?",
        "Are there any disconnected copper traces?",
    ],
    "short": [
        "Are there any short circuit defects?",
        "Can you see any unintended connections between traces?",
        "Are there any bridging defects on the board?",
    ],
    "mousebite": [
        "Are there any mousebite defects?",
        "Can you identify any irregular edge nibbling on traces?",
        "Are there any partial trace etching defects?",
    ],
    "spur": [
        "Are there any spur defects on the traces?",
        "Can you see any unwanted copper protrusions?",
        "Are there any spurious copper extensions?",
    ],
    "copper": [
        "Are there any excess copper defects?",
        "Can you identify any copper residue issues?",
        "Are there any remaining copper spots that shouldn't be there?",
    ],
    "pin-hole": [
        "Are there any pin-hole defects in the copper?",
        "Can you see any small holes in the copper layer?",
        "Are there any voids or pinholes in the traces?",
    ],
}

LOCATION_QUESTIONS = [
    "Where are the defects located on this PCB?",
    "Describe the positions of defects on this board.",
    "In which regions of the PCB are defects found?",
]

SEVERITY_QUESTIONS = [
    "How severe are the defects on this PCB?",
    "What is the severity assessment for this board?",
    "Rate the severity of defects found.",
]

NO_DEFECT_ANSWERS = [
    "No defects are detected in this PCB image. The board appears to pass quality inspection.",
    "This PCB image shows no visible defects. The board meets quality standards.",
    "No manufacturing defects are identified in this circuit board image.",
]


def _parse_annotation(ann_path: str, img_w: int = 640, img_h: int = 640) -> List[Dict]:
    """Parse VOC XML annotation into list of defect dicts."""
    tree = ET.parse(ann_path)
    root = tree.getroot()

    size = root.find("size")
    if size is not None:
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

    defects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.lower().strip()
        if name not in CLASS_TO_IDX:
            continue

        b = obj.find("bndbox")
        xmin = int(float(b.find("xmin").text))
        ymin = int(float(b.find("ymin").text))
        xmax = int(float(b.find("xmax").text))
        ymax = int(float(b.find("ymax").text))

        if xmax <= xmin or ymax <= ymin:
            continue

        area = (xmax - xmin) * (ymax - ymin)
        ratio = area / (img_w * img_h)

        if ratio >= SEVERITY.med_threshold:
            severity = "high"
        elif ratio >= SEVERITY.low_threshold:
            severity = "medium"
        else:
            severity = "low"

        # Quadrant location
        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        if cy < 0.33:
            vert = "top"
        elif cy > 0.66:
            vert = "bottom"
        else:
            vert = "center"
        if cx < 0.33:
            horiz = "left"
        elif cx > 0.66:
            horiz = "right"
        else:
            horiz = "center"
        location = f"{vert}-{horiz}"

        defects.append({
            "class": name,
            "bbox": [xmin, ymin, xmax, ymax],
            "area": area,
            "severity": severity,
            "location": location,
        })

    return defects


def _generate_detection_answer(defects: List[Dict]) -> str:
    counts = Counter(d["class"] for d in defects)
    parts = []
    for cls in CLASSES:
        if cls in counts:
            n = counts[cls]
            parts.append(f"{n} {cls} defect{'s' if n > 1 else ''}")

    total = sum(counts.values())
    answer = f"This PCB has {total} defect{'s' if total > 1 else ''}: {', '.join(parts)}."
    return answer


def _generate_count_answer(defects: List[Dict]) -> str:
    counts = Counter(d["class"] for d in defects)
    total = len(defects)
    breakdown = ", ".join(f"{v} {k}" for k, v in counts.items())
    return f"There are {total} defects in total: {breakdown}."


def _generate_class_answer(defects: List[Dict], cls: str) -> str:
    class_defects = [d for d in defects if d["class"] == cls]
    if not class_defects:
        return f"No {cls} defects are found in this PCB image."

    n = len(class_defects)
    locations = [d["location"] for d in class_defects]
    loc_str = ", ".join(locations)
    return (
        f"Yes, there {'is' if n == 1 else 'are'} {n} {cls} defect{'s' if n > 1 else ''} "
        f"located at: {loc_str}."
    )


def _generate_location_answer(defects: List[Dict]) -> str:
    parts = []
    for d in defects:
        parts.append(f"{d['class']} at {d['location']} ({d['bbox'][0]},{d['bbox'][1]},{d['bbox'][2]},{d['bbox'][3]})")
    return "Defect locations: " + "; ".join(parts) + "."


def _generate_severity_answer(defects: List[Dict]) -> str:
    sev_counts = Counter(d["severity"] for d in defects)
    parts = []
    for sev in ["high", "medium", "low"]:
        if sev in sev_counts:
            parts.append(f"{sev_counts[sev]} {sev}-severity")
    cls_sev = []
    for d in defects:
        cls_sev.append(f"{d['class']}={d['severity']}")
    return f"Severity assessment: {', '.join(parts)}. Details: {', '.join(cls_sev)}."


def generate_qa_for_image(ann_path: str, max_pairs: int = 6) -> List[Dict]:
    """Generate diverse QA pairs for a single image annotation."""
    defects = _parse_annotation(ann_path)
    qa_pairs = []

    if not defects:
        q = random.choice(DETECTION_QUESTIONS)
        a = random.choice(NO_DEFECT_ANSWERS)
        return [{"question": q, "answer": a}]

    # 1. Detection question (always include)
    qa_pairs.append({
        "question": random.choice(DETECTION_QUESTIONS),
        "answer": _generate_detection_answer(defects),
    })

    # 2. Count question
    qa_pairs.append({
        "question": random.choice(COUNT_QUESTIONS),
        "answer": _generate_count_answer(defects),
    })

    # 3. Class-specific questions (for classes present + one absent)
    present_classes = set(d["class"] for d in defects)
    for cls in present_classes:
        if len(qa_pairs) >= max_pairs:
            break
        qa_pairs.append({
            "question": random.choice(CLASS_SPECIFIC_QUESTIONS[cls]),
            "answer": _generate_class_answer(defects, cls),
        })

    # Add one negative class question
    absent = [c for c in CLASSES if c not in present_classes]
    if absent and len(qa_pairs) < max_pairs:
        neg_cls = random.choice(absent)
        qa_pairs.append({
            "question": random.choice(CLASS_SPECIFIC_QUESTIONS[neg_cls]),
            "answer": _generate_class_answer(defects, neg_cls),
        })

    # 4. Location question
    if len(qa_pairs) < max_pairs:
        qa_pairs.append({
            "question": random.choice(LOCATION_QUESTIONS),
            "answer": _generate_location_answer(defects),
        })

    # 5. Severity question
    if len(qa_pairs) < max_pairs:
        qa_pairs.append({
            "question": random.choice(SEVERITY_QUESTIONS),
            "answer": _generate_severity_answer(defects),
        })

    return qa_pairs[:max_pairs]


def generate_dataset(data_dir: str, output_path: str, max_pairs_per_image: int = 6):
    """Generate full QA dataset from all annotations in a directory."""
    ann_dir = os.path.join(data_dir, "annotations")
    img_dir = os.path.join(data_dir, "images")

    ann_files = sorted(f for f in os.listdir(ann_dir) if f.endswith(".xml"))
    dataset = []

    for ann_file in ann_files:
        stem = os.path.splitext(ann_file)[0]

        # Find matching image
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(img_dir, stem + ext)
            if os.path.isfile(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue

        ann_path = os.path.join(ann_dir, ann_file)
        qa_pairs = generate_qa_for_image(ann_path, max_pairs=max_pairs_per_image)

        for qa in qa_pairs:
            dataset.append({
                "image": img_path,
                "question": qa["question"],
                "answer": qa["answer"],
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(dataset)} QA pairs from {len(ann_files)} images -> {output_path}")
    return dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic QA pairs")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-pairs", type=int, default=6)
    args = parser.parse_args()

    generate_dataset(args.data_dir, args.output, args.max_pairs)
