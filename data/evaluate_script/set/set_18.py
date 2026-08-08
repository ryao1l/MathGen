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
PROMPT_ID = 18
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
Circle = Tuple[float, float, float]

def red_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask (uint8 0/255) of red/reddish shaded pixels."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1, upper1 = (np.array([0, 35, 45]), np.array([15, 255, 255]))
    lower2, upper2 = (np.array([165, 35, 45]), np.array([180, 255, 255]))
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    hue_mask = cv2.bitwise_or(m1, m2)
    b, g, r = cv2.split(bgr)
    rf = r.astype(np.float32)
    gf = g.astype(np.float32)
    bf = b.astype(np.float32)
    red_dominance = ((rf > 85) & (rf > 1.18 * gf) & (rf > 1.18 * bf)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(hue_mask, red_dominance)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask

def black_edge_map(bgr: np.ndarray) -> np.ndarray:
    """Build an edge map emphasizing black strokes on white background."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.Canny(binary_inv, 50, 150)

def hough_circles(img_like: np.ndarray, dp: float, min_dist: float, param1: float, param2: float, min_radius: int, max_radius: int) -> List[Circle]:
    """Run HoughCircles and return circles as float tuples."""
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

def circle_mask(shape_hw: Tuple[int, int], circle: Circle) -> np.ndarray:
    """Binary mask for a filled circle."""
    h, w = shape_hw
    cx, cy, r = circle
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
    return m

def red_ratio_in_circle(red_mask: np.ndarray, circle: Circle) -> float:
    """Compute red pixel ratio inside a filled circle."""
    m = circle_mask(red_mask.shape[:2], circle)
    denom = int(cv2.countNonZero(m))
    if denom == 0:
        return 0.0
    num = int(cv2.countNonZero(cv2.bitwise_and(red_mask, m)))
    return float(num) / float(denom)

def red_component_circles(red_mask: np.ndarray, min_radius: float, max_radius: float) -> List[Circle]:
    """Fit circle candidates around compact red regions, which Hough often misses."""
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Circle] = []
    min_area = max(120.0, np.pi * (0.35 * min_radius) ** 2)
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if r < min_radius or r > max_radius:
            continue
        circle_area = np.pi * r * r
        fill = area / circle_area if circle_area > 0 else 0.0
        if fill < 0.35:
            continue
        out.append((float(cx), float(cy), float(r)))
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

def verify_subset_all_hough_support(image_path: str, red_thresh_for_d: float=0.25, support_thresh_for_big: float=0.6) -> bool:
    """
    Criteria:
      C1: Image file exists and is readable.
      C2: At least 3 circles detected by Hough (deduped).
      C3: A D-circle candidate exists (red_ratio >= red_thresh_for_d).
      C4: Two big-circle candidates selected (by support/radius).
      C5: D is contained in both big circles.
      C6: Red spill outside D is below tolerance.
    """
    c1_ok = os.path.exists(image_path)
    if not c1_ok:
        print('C1_file_exists: False')
        print('C2_hough_detects_>=3_circles: False')
        print('C3_d_candidate_found: False')
        print('C4_two_big_circles_selected: False')
        print('C5_d_contained_in_both: False')
        print('C6_red_spill_ok: False')
        print('Result: FAIL')
        return False
    img = cv2.imread(image_path)
    c1_readable = img is not None
    if not c1_readable:
        print('C1_file_exists: True')
        print('C1_image_readable: False')
        print('C2_hough_detects_>=3_circles: False')
        print('C3_d_candidate_found: False')
        print('C4_two_big_circles_selected: False')
        print('C5_d_contained_in_both: False')
        print('C6_red_spill_ok: False')
        print('Result: FAIL')
        return False
    h, w = img.shape[:2]
    rm = red_mask_hsv(img)
    edges = black_edge_map(img)
    min_dim = min(h, w)
    circles = hough_circles(img_like=edges, dp=1.2, min_dist=0.1 * min_dim, param1=80, param2=28, min_radius=int(0.03 * min_dim), max_radius=int(0.49 * min_dim))
    circles.extend(red_component_circles(rm, 0.025 * min_dim, 0.22 * min_dim))
    circles = dedupe_circles(circles, center_tol=25.0, radius_tol=25.0)
    c2_ok = len(circles) >= 3
    c3_ok = False
    c4_ok = False
    c5_ok = False
    c6_ok = False
    d_circle: Optional[Circle] = None
    big: List[Circle] = []
    if c2_ok:
        scored = []
        for c in circles:
            rr = red_ratio_in_circle(rm, c)
            sup = circle_perimeter_support(edges, c, n_samples=360, band=2)
            scored.append((c, rr, sup))
        d_candidates = [(c, rr, sup) for c, rr, sup in scored if rr >= red_thresh_for_d and c[2] <= 0.28 * min_dim]
        c3_ok = len(d_candidates) > 0
        if c3_ok:
            d_circle = max(d_candidates, key=lambda t: (t[1], -t[0][2]))[0]
            dx, dy, d_r = d_circle
            big_candidates = []
            for c, rr, sup in scored:
                cx, cy, cr = c
                if np.hypot(cx - dx, cy - dy) < 20.0 and abs(cr - d_r) < 20.0:
                    continue
                if cr > 1.35 * d_r and sup >= min(support_thresh_for_big, 0.15):
                    big_candidates.append((c, rr, sup))
            containment_tol = lambda cr: max(28.0, 0.09 * cr, 0.3 * d_r)
            contained_big_candidates = []
            for c, rr, sup in big_candidates:
                cx, cy, cr = c
                dist = float(np.hypot(cx - dx, cy - dy))
                if dist + d_r <= cr + containment_tol(cr):
                    contained_big_candidates.append((c, rr, sup))
            if len(contained_big_candidates) >= 2:
                pair_options = []
                for i in range(len(contained_big_candidates)):
                    for j in range(i + 1, len(contained_big_candidates)):
                        c1, _, sup1 = contained_big_candidates[i]
                        c2, _, sup2 = contained_big_candidates[j]
                        bx1, by1, br1 = c1
                        bx2, by2, br2 = c2
                        center_sep = float(np.hypot(bx1 - bx2, by1 - by2))
                        max_br = max(br1, br2)
                        min_br = min(br1, br2)
                        sep_ratio = center_sep / max_br if max_br else 0.0
                        size_ratio = min_br / max_br if max_br else 0.0
                        d_ratio = d_r / min_br if min_br else 1.0
                        if 0.45 <= sep_ratio <= 1.85 and size_ratio >= 0.72 and (d_ratio <= 0.52):
                            score = (sup1 + sup2, min_br, -abs(sep_ratio - 1.15))
                            pair_options.append((score, c1, c2))
                if pair_options:
                    _, c1, c2 = max(pair_options, key=lambda t: t[0])
                    big = [c1, c2]
            elif len(big_candidates) >= 2:
                big_candidates.sort(key=lambda t: (t[0][2], t[2]), reverse=True)
                big = [big_candidates[0][0], big_candidates[1][0]]
            else:
                fallback = []
                for c, rr, sup in scored:
                    cx, cy, cr = c
                    if np.hypot(cx - dx, cy - dy) < 20.0 and abs(cr - d_r) < 20.0:
                        continue
                    fallback.append((c, rr, sup))
                fallback.sort(key=lambda t: (t[2], t[0][2]), reverse=True)
                big = [t[0] for t in fallback[:2]] if len(fallback) >= 2 else []
            c4_ok = len(big) >= 2
            if c4_ok:
                bx1, by1, br1 = big[0]
                bx2, by2, br2 = big[1]
                center_sep = float(np.hypot(bx1 - bx2, by1 - by2))
                max_br = max(br1, br2)
                min_br = min(br1, br2)
                pair_sep_ok = 0.45 <= (center_sep / max_br if max_br else 0.0) <= 1.85
                pair_size_ok = (min_br / max_br if max_br else 0.0) >= 0.72
                d_support = circle_perimeter_support(edges, d_circle, n_samples=360, band=2)
                pair_support = 0.5 * (circle_perimeter_support(edges, big[0], n_samples=360, band=2) + circle_perimeter_support(edges, big[1], n_samples=360, band=2))
                object_evidence_ok = d_support >= 0.12 or pair_support >= 0.55
                c4_ok = pair_sep_ok and pair_size_ok and object_evidence_ok
            if c4_ok and d_circle is not None:
                d_center = np.array([dx, dy], dtype=np.float32)
                contained_flags = []
                for cx, cy, cr in big[:2]:
                    dist = float(np.linalg.norm(d_center - np.array([cx, cy], dtype=np.float32)))
                    tol = containment_tol(cr)
                    contained_flags.append(dist + d_r <= cr + tol)
                c5_ok = all(contained_flags)
                d_mask = circle_mask((h, w), (dx, dy, d_r * 1.08))
                red_outside = cv2.bitwise_and(rm, cv2.bitwise_not(d_mask))
                outside_area = int(cv2.countNonZero(red_outside))
                d_area_px = int(cv2.countNonZero(d_mask))
                spill_thresh = max(1600, int(0.6 * d_area_px))
                c6_ok = outside_area <= spill_thresh
    print(f'C1_file_exists: {c1_ok}')
    print(f'C1_image_readable: {c1_readable}')
    print(f'C2_hough_detects_>=3_circles: {c2_ok}')
    print(f'C3_d_candidate_found: {c3_ok}')
    print(f'C4_two_big_circles_selected: {c4_ok}')
    print(f'C5_d_contained_in_both: {c5_ok}')
    print(f'C6_red_spill_ok: {c6_ok}')
    success = c1_ok and c1_readable and c2_ok and c3_ok and c4_ok and c5_ok and c6_ok
    print(f"Result: {('PASS' if success else 'FAIL')}")
    return success

def evaluate(image_path: str):
    passed = verify_subset_all_hough_support(image_path)
    return {'passed': bool(passed), 'criteria': {'script_passed': bool(passed)}, 'meta': {'judge': 'set_18'}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen set case 18.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
