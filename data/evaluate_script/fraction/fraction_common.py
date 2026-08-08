"""Shared evaluators for simple fraction bar prompts."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


COLOR_RANGES = {
    "blue": [((85, 30, 35), (135, 255, 255))],
    "red": [((0, 35, 35), (12, 255, 255)), ((160, 35, 35), (180, 255, 255))],
}


def load_image(path: str) -> Optional[np.ndarray]:
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = img[:, :, :3].astype(np.float32)
        img = (bgr * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def mask_hsv(img: np.ndarray, color: str) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for low, high in COLOR_RANGES[color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(low), np.array(high)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def background_median(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    b = max(1, int(min(h, w) * 0.08))
    border = np.concatenate([
        gray[:b, :].ravel(),
        gray[h - b:, :].ravel(),
        gray[:, :b].ravel(),
        gray[:, w - b:].ravel(),
    ])
    return float(np.median(border))


def inset(img: np.ndarray, frac: float = 0.05) -> np.ndarray:
    h, w = img.shape[:2]
    p = max(1, int(round(min(h, w) * frac)))
    if h <= 2 * p + 2 or w <= 2 * p + 2:
        return img
    return img[p:h - p, p:w - p]


def candidate_bboxes(img: np.ndarray, focus_mask: np.ndarray) -> Iterable[Tuple[int, int, int, int]]:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masks = [
        cv2.dilate(focus_mask, np.ones((9, 9), np.uint8), 2),
        cv2.dilate(cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 140), np.ones((5, 5), np.uint8), 2),
    ]
    seen = set()
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 35 or bh < 10:
                continue
            aspect = bw / max(1, bh)
            area_frac = (bw * bh) / (h * w)
            if aspect < 1.8 or area_frac < 0.002 or area_frac > 0.55:
                continue
            key = (x // 3, y // 3, bw // 3, bh // 3)
            if key in seen:
                continue
            seen.add(key)
            yield (x, y, bw, bh)


def evaluate_colored_progress(image_path: str, color: str, target: float, tol: float = 0.13) -> Dict[str, object]:
    img = load_image(image_path)
    if img is None:
        return {"criteria": {"image_exists": os.path.isfile(image_path), "image_readable": False}, "passed": False}

    color_mask = mask_hsv(img, color)
    best = None
    for bbox in candidate_bboxes(img, color_mask):
        x, y, w, h = bbox
        bar = inset(img[y:y + h, x:x + w])
        if bar.size == 0:
            continue
        mask = mask_hsv(bar, color)
        density = mask.mean(axis=0) / 255.0
        cols = np.where(density > 0.08)[0]
        if len(cols) == 0:
            ratio = purity = spill = 0.0
        else:
            ratio = (cols[-1] + 1) / max(1, bar.shape[1])
            split = max(1, min(bar.shape[1] - 1, int(round(bar.shape[1] * target))))
            purity = float(np.mean(mask[:, :split] > 0))
            spill = float(np.mean(mask[:, split:] > 0))
        aspect = w / max(1, h)
        ratio_ok = abs(ratio - target) <= tol
        color_ok = purity >= 0.18
        spill_ok = spill <= 0.22 or target >= 0.75
        score = (2.0 if ratio_ok else 0.0) + purity - abs(ratio - target) + min(aspect, 8.0) / 20.0
        item = (score, bbox, ratio, purity, spill, aspect, ratio_ok, color_ok, spill_ok)
        if best is None or item[0] > best[0]:
            best = item

    if best is None:
        return {"criteria": {"bar_detected": False}, "meta": {"bbox_method": "none"}, "passed": False}

    _, bbox, ratio, purity, spill, aspect, ratio_ok, color_ok, spill_ok = best
    bg = background_median(img)
    passed = bool(bg >= 220 and aspect >= 1.8 and ratio_ok and color_ok and spill_ok)
    return {
        "criteria": {
            "image_exists": True,
            "image_readable": True,
            "background_is_light": bool(bg >= 220),
            "bar_detected": True,
            "bar_shape_ok": bool(aspect >= 1.8),
            "color_fill_present": bool(color_ok),
            "unfilled_region_ok": bool(spill_ok),
            "fill_ratio_ok": bool(ratio_ok),
        },
        "meta": {
            "background_median_intensity": round(bg, 1),
            "bbox": list(bbox),
            "bar_aspect": round(aspect, 2),
            f"{color}_fill_ratio": round(ratio, 4),
            f"{color}_purity": round(purity, 4),
            f"{color}_spill": round(spill, 4),
        },
        "passed": passed,
    }


def emit(report: Dict[str, object], image_path: str, judge: str) -> None:
    print(json.dumps({
        "topic": "proportion_mapping",
        "image": image_path,
        "passed": report["passed"],
        "criteria": report["criteria"],
        "meta": report.get("meta", {}),
        "judge": judge,
    }, indent=2))
