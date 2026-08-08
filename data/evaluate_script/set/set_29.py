#!/usr/bin/env python3
from __future__ import annotations
import argparse
import contextlib
import io
import json
import math
import os
import re
import sys
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluator_utils import print_report
from ocr_label_utils import find_labels_in_circles, point_in_circle, verify_circle_and_rect_labels, verify_circle_labels
PROMPT_ID = 29
Circle = Tuple[float, float, float]

def navy_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask (uint8 0/255) of dark navy-blue pixels."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([100, 50, 20])
    upper = np.array([130, 255, 180])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask

def black_edge_map(bgr: np.ndarray) -> np.ndarray:
    """Build an edge map emphasizing black strokes on a light background."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.Canny(binary_inv, 50, 150)

def detect_rectangle(bgr: np.ndarray, min_area_ratio: float=0.2) -> Optional[np.ndarray]:
    """Detect the universal-set rectangle and return a filled mask (uint8 255)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    h, w = bgr.shape[:2]
    if cv2.contourArea(largest) < min_area_ratio * h * w:
        return None
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.04 * peri, True)
    if len(approx) < 4 or len(approx) > 6:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask

def hough_circles(img_like: np.ndarray, dp: float, min_dist: float, param1: float, param2: float, min_radius: int, max_radius: int) -> List[Circle]:
    """Run HoughCircles and return all detected circles as float tuples."""
    circles = cv2.HoughCircles(img_like, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    if circles is None:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in circles[0]]

def dedupe_circles(circles: List[Circle], center_tol: float=25.0, radius_tol: float=25.0) -> List[Circle]:
    """Greedy de-duplication of near-identical circles."""
    out: List[Circle] = []
    for cx, cy, cr in sorted(circles, key=lambda t: t[2], reverse=True):
        if any((np.hypot(cx - sx, cy - sy) < center_tol and abs(cr - sr) < radius_tol for sx, sy, sr in out)):
            continue
        out.append((cx, cy, cr))
    return out

def circle_perimeter_support(edge_map: np.ndarray, circle: Circle, n_samples: int=360, band: int=2) -> float:
    """Score how well a circle is supported by edges near its circumference."""
    h, w = edge_map.shape[:2]
    cx, cy, r = circle
    if r <= 1:
        return 0.0
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    xs = np.round(cx + r * np.cos(angles)).astype(np.int32)
    ys = np.round(cy + r * np.sin(angles)).astype(np.int32)
    ok, total = (0, 0)
    for x, y in zip(xs, ys):
        if x < 0 or x >= w or y < 0 or (y >= h):
            continue
        x0, x1 = (max(0, x - band), min(w, x + band + 1))
        y0, y1 = (max(0, y - band), min(h, y + band + 1))
        total += 1
        ok += int(np.any(edge_map[y0:y1, x0:x1] > 0))
    return float(ok) / float(total) if total > 0 else 0.0

def circle_mask(shape_hw: Tuple[int, int], circle: Circle) -> np.ndarray:
    """Binary uint8 mask (0/255) for a filled circle."""
    h, w = shape_hw
    cx, cy, r = circle
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
    return m

def circles_overlap(c1: Circle, c2: Circle) -> bool:
    """Check whether two circles overlap (distance between centers < sum of radii)."""
    dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
    return dist < c1[2] + c2[2]

def color_ratio_in_mask(region_mask: np.ndarray, color_mask: np.ndarray) -> float:
    """Return the fraction of colored pixels within a uint8 255 region mask."""
    denom = int(cv2.countNonZero(region_mask))
    if denom == 0:
        return 0.0
    num = int(cv2.countNonZero(cv2.bitwise_and(color_mask, region_mask)))
    return float(num) / float(denom)

def verify_complement_union(image_path: str, navy_on_thresh: float=0.3, navy_off_thresh: float=0.05, support_thresh: float=0.43, spill_ratio: float=0.03) -> bool:
    """
    Evaluate a Venn diagram showing (A â\x88?B â\x88?C)' in dark navy blue
    within a rectangular universal set U.

    Criteria:
      C1: File exists and image is readable.
      C2: A rectangular universal-set border detected.
      C3: Exactly three circles detected (no more, no less).
      C4: The three circles overlap each other (pairwise).
      C5: Complement region (rect â\x88?circles) is mostly navy blue.
      C6: Navy blue does not spill into the circles.
      C7: Labels A, B, C each appear once inside a distinct circle; U appears once near the rectangle.
    """
    c1_ok = os.path.exists(image_path)
    if not c1_ok:
        _print_fail('C1_file_exists: False')
        return False
    img = cv2.imread(image_path)
    if img is None:
        _print_fail('C1_file_exists: True', 'C1_image_readable: False')
        return False
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nm = navy_mask_hsv(img)
    edges = black_edge_map(img)
    rect_mask = detect_rectangle(img)
    c2_ok = rect_mask is not None
    print(f'C1_file_exists: True')
    print(f'C1_image_readable: True')
    print(f'C2_rectangle_detected: {c2_ok}')
    if not c2_ok:
        print('Result: FAIL')
        return False
    min_dim = min(h, w)
    passes = [(1.2, 0.1 * min_dim, 80, 40, int(0.05 * min_dim), int(0.35 * min_dim)), (1.2, 0.1 * min_dim, 80, 35, int(0.3 * min_dim), int(0.8 * min_dim))]
    all_circles: List[Circle] = []
    for dp, md, p1, p2, rmin, rmax in passes:
        found = hough_circles(edges, dp, md, p1, p2, rmin, rmax)
        all_circles.extend(found)
    all_circles = dedupe_circles(all_circles, center_tol=30.0, radius_tol=30.0)
    validated = []
    for c in all_circles:
        sup = circle_perimeter_support(edges, c, n_samples=360, band=3)
        if sup >= support_thresh:
            validated.append(c)
    c3_ok = len(validated) == 3
    print(f'C3_circles_found: {len(validated)}')
    print(f'C3_exactly_three_circles: {c3_ok}')
    if not c3_ok:
        print('Result: FAIL')
        return False
    overlap_ab = circles_overlap(validated[0], validated[1])
    overlap_ac = circles_overlap(validated[0], validated[2])
    overlap_bc = circles_overlap(validated[1], validated[2])
    c4_ok = overlap_ab and overlap_ac and overlap_bc
    print(f'C4_overlap_0_1: {overlap_ab}')
    print(f'C4_overlap_0_2: {overlap_ac}')
    print(f'C4_overlap_1_2: {overlap_bc}')
    print(f'C4_all_overlapping: {c4_ok}')
    if not c4_ok:
        print('Result: FAIL')
        return False
    union_u8 = np.zeros((h, w), dtype=np.uint8)
    for c in validated:
        union_u8 = cv2.bitwise_or(union_u8, circle_mask((h, w), c))
    complement_u8 = cv2.bitwise_and(rect_mask, cv2.bitwise_not(union_u8))
    complement_ratio = color_ratio_in_mask(complement_u8, nm)
    c5_ok = complement_ratio >= navy_on_thresh
    print(f'C5_complement_navy_ratio: {complement_ratio:.4f}')
    print(f'C5_complement_is_navy: {c5_ok}')
    navy_inside = int(cv2.countNonZero(cv2.bitwise_and(nm, union_u8)))
    navy_total = int(cv2.countNonZero(cv2.bitwise_and(nm, rect_mask)))
    if navy_total > 0:
        spill = float(navy_inside) / float(navy_total)
    else:
        spill = 0.0
    c6_ok = spill <= spill_ratio
    print(f'C6_navy_spill_ratio: {spill:.4f}')
    print(f'C6_spill_ok: {c6_ok}')
    c7_ok, label_details = verify_circle_and_rect_labels(image_path, validated, ['A', 'B', 'C'], rect_label='U')
    for k, v in label_details.items():
        print(f'C7_{k}: {v}')
    success = c2_ok and c3_ok and c4_ok and c5_ok and c6_ok and c7_ok
    print(f"Result: {('PASS' if success else 'FAIL')}")
    return success

def _print_fail(*lines):
    """Print early-exit failure lines."""
    for line in lines:
        print(line)
    print('Result: FAIL')
evaluate = verify_complement_union

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen set case 29.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
