import argparse
import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Expected total object counts by counting image id.
EXPECTED_TOTAL_COUNTS: Dict[int, int] = {
    1: 10,
    2: 20,
    3: 19,
    4: 23,
    5: 17,
    6: 50,
    7: 13,
    8: 11,
    9: 8,
    10: 35,
}


@lru_cache(maxsize=4)
def _load_expected_map_from_prompt(prompt_file: str) -> Dict[int, int]:
    if not prompt_file or not os.path.exists(prompt_file):
        return {}
    expected = {}
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx = row.get("id")
            prompt = str(row.get("prompt", ""))
            if idx is None:
                continue
            m = re.search(r"\bexactly\s+(\d+)\b", prompt, flags=re.IGNORECASE)
            if m:
                expected[int(idx)] = int(m.group(1))
    return expected


def _resolve_prompt_file_path(explicit_prompt_file: Optional[str]) -> Optional[str]:
    if explicit_prompt_file and os.path.exists(explicit_prompt_file):
        return explicit_prompt_file
    candidates = [
        os.path.join("data", "prompt_data", "counting.jsonl"),
        os.path.join("prompt", "counting.jsonl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _parse_image_id(image_path: str) -> int:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    match = re.search(r"(\d+)$", stem)
    if not match:
        raise ValueError(f"Cannot parse numeric id from image name: {image_path}")
    return int(match.group(1))


def _resolve_expected_count(image_path: str, prompt_file: Optional[str] = None) -> int:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    numeric_tail = _parse_image_id(image_path)

    prompt_path = _resolve_prompt_file_path(prompt_file)
    if prompt_path:
        prompt_expected = _load_expected_map_from_prompt(prompt_path)
        if stem.isdigit() and numeric_tail in prompt_expected:
            return prompt_expected[numeric_tail]

    if stem.isdigit() and numeric_tail in EXPECTED_TOTAL_COUNTS:
        return EXPECTED_TOTAL_COUNTS[numeric_tail]
    return numeric_tail


def _load_prompt_text_by_id(prompt_file: Optional[str]) -> Dict[int, str]:
    path = _resolve_prompt_file_path(prompt_file)
    if not path:
        return {}
    records: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx = row.get("id")
            if idx is None:
                continue
            records[int(idx)] = str(row.get("prompt", ""))
    return records


def _extract_target_key(prompt_text: str) -> Optional[str]:
    p = prompt_text.lower()
    checks = [
        ("red apples", "apple"),
        ("green apples", "apple"),
        ("red apple", "apple"),
        ("green apple", "apple"),
        ("apples", "apple"),
        ("apple", "apple"),
        ("oranges", "orange"),
        ("orange", "orange"),
        ("bananas", "banana"),
        ("banana", "banana"),
        ("tomatoes", "tomato"),
        ("tomato", "tomato"),
        ("broccoli", "broccoli"),
        ("carrots", "carrot"),
        ("carrot", "carrot"),
        ("tennis balls", "sports ball"),
        ("tennis ball", "sports ball"),
        ("sports balls", "sports ball"),
        ("sports ball", "sports ball"),
    ]
    for token, key in checks:
        if token in p:
            return key
    return None


def _infer_target_key_from_image_name(image_path: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(image_path))[0].lower()
    checks = [
        ("apple", "apple"),
        ("orange", "orange"),
        ("banana", "banana"),
        ("tomato", "tomato"),
        ("broccoli", "broccoli"),
        ("carrot", "carrot"),
        ("sportsball", "sports ball"),
        ("sports_ball", "sports ball"),
        ("sports-ball", "sports ball"),
        ("tennisball", "sports ball"),
        ("tennis_ball", "sports ball"),
        ("tennis-ball", "sports ball"),
    ]
    for token, key in checks:
        if token in stem:
            return key
    return None


def _target_aliases_for_rtdetr(target_key: Optional[str]) -> List[str]:
    mapping = {
        "apple": ["apple"],
        "orange": ["orange"],
        "banana": ["banana"],
        "tomato": ["tomato"],
        "broccoli": ["broccoli"],
        "carrot": ["carrot"],
        "sports ball": ["sports ball"],
    }
    if target_key is None:
        return []
    return mapping.get(target_key, [])


@lru_cache(maxsize=2)
def _load_rtdetr(model_name: str):
    try:
        import torch
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "RT-DETR dependencies are missing. Install torch, transformers, and pillow."
        ) from exc

    processor = RTDetrImageProcessor.from_pretrained(model_name)
    model = RTDetrForObjectDetection.from_pretrained(model_name)
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    return model, processor, id2label, torch


def detect_object_count_rtdetr(
    image_path: str,
    expected_id: int,
    prompt_file: Optional[str],
    model_name: str,
    device: str,
    score_threshold: float,
) -> Tuple[int, Dict[str, object]]:
    from PIL import Image

    prompts = _load_prompt_text_by_id(prompt_file)
    prompt_text = prompts.get(expected_id, "")
    target_key = _extract_target_key(prompt_text)
    # Fallback for datasets with filenames like apple_50.png where id may not
    # exist in the prompt file.
    if target_key is None:
        target_key = _infer_target_key_from_image_name(image_path)
    aliases = _target_aliases_for_rtdetr(target_key)
    if not aliases:
        raise RuntimeError(
            f"RT-DETR unsupported target class: {target_key}. "
            f"image={os.path.basename(image_path)}, expected_id={expected_id}, "
            f"prompt_text={prompt_text!r}. "
            "Only apple/orange/banana/tomato/broccoli/carrot/sports ball are supported."
        )

    model, processor, id2label, torch = _load_rtdetr(model_name)
    image = Image.open(image_path).convert("RGB")
    target_device = device
    if device.startswith("cuda") and not torch.cuda.is_available():
        target_device = "cpu"
    model = model.to(target_device)

    inputs = processor(images=image, return_tensors="pt").to(target_device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = [image.size[::-1]]
    result = processor.post_process_object_detection(
        outputs, threshold=float(score_threshold), target_sizes=target_sizes
    )[0]

    labels = [id2label.get(int(x), str(x)) for x in result["labels"].tolist()]
    boxes = result["boxes"].tolist()
    scores = result["scores"].tolist()

    selected_idx = [i for i, label in enumerate(labels) if label in aliases]
    count = len(selected_idx)

    dbg_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for i in selected_idx:
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        cv2.rectangle(dbg_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            dbg_img,
            f"{labels[i]}:{scores[i]:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return count, {
        "total_components": len(labels),
        "valid_components": count,
        "mask_debug": np.zeros((dbg_img.shape[0], dbg_img.shape[1]), dtype=np.uint8),
        "detected_debug": dbg_img,
        "rtdetr_target_key": target_key,
        "rtdetr_aliases": aliases,
        "rtdetr_selected_labels": [labels[i] for i in selected_idx],
    }


def evaluate(
    image_path: str,
    counting_tolerance: int = 0,
    counting_debug_dir: Optional[str] = None,
    counting_prompt_file: Optional[str] = None,
    counting_rtdetr_model: str = "PekingU/rtdetr_r50vd",
    counting_rtdetr_device: str = "cpu",
    counting_rtdetr_score_threshold: float = 0.45,
):
    image_id = _parse_image_id(image_path)
    expected = _resolve_expected_count(image_path, counting_prompt_file)
    observed, debug_info = detect_object_count_rtdetr(
        image_path=image_path,
        expected_id=image_id,
        prompt_file=counting_prompt_file,
        model_name=counting_rtdetr_model,
        device=counting_rtdetr_device,
        score_threshold=float(counting_rtdetr_score_threshold),
    )
    passed = abs(observed - expected) <= int(counting_tolerance)

    if counting_debug_dir:
        os.makedirs(counting_debug_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(counting_debug_dir, f"{stem}_mask_debug.png")
        det_path = os.path.join(counting_debug_dir, f"{stem}_detected_objects.png")
        cv2.imwrite(mask_path, debug_info["mask_debug"])
        cv2.imwrite(det_path, debug_info["detected_debug"])
    else:
        mask_path = None
        det_path = None

    return {
        "passed": passed,
        "criteria": {"object_count_match": passed},
        "meta": {
            "image_id": image_id,
            "expected_count": expected,
            "observed_count": observed,
            "abs_error": abs(observed - expected),
            "tolerance": int(counting_tolerance),
            "counting_detector": "rtdetr",
            "counting_rtdetr_model": counting_rtdetr_model,
            "counting_rtdetr_device": counting_rtdetr_device,
            "counting_rtdetr_score_threshold": float(counting_rtdetr_score_threshold),
            "total_components": debug_info["total_components"],
            "valid_components": debug_info["valid_components"],
            "debug_mask_path": mask_path,
            "debug_detected_path": det_path,
            "counting_prompt_file": _resolve_prompt_file_path(counting_prompt_file),
            "detector_meta": {
                k: v
                for k, v in debug_info.items()
                if k not in {"mask_debug", "detected_debug", "total_components", "valid_components"}
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Count objects in image via RT-DETR.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--tolerance", type=int, default=0)
    parser.add_argument("--debug_dir", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--rtdetr_model", type=str, default="PekingU/rtdetr_r50vd")
    parser.add_argument("--rtdetr_device", type=str, default="cpu")
    parser.add_argument("--rtdetr_score_threshold", type=float, default=0.45)
    args = parser.parse_args()

    report = evaluate(
        image_path=args.image,
        counting_tolerance=args.tolerance,
        counting_debug_dir=args.debug_dir,
        counting_prompt_file=args.prompt_file,
        counting_rtdetr_model=args.rtdetr_model,
        counting_rtdetr_device=args.rtdetr_device,
        counting_rtdetr_score_threshold=args.rtdetr_score_threshold,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
