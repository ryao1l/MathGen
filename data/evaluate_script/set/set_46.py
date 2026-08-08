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
PROMPT_ID = 46
Circle = Tuple[float, float, float]

def red_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask (uint8 0/255) of red pixels in a BGR image."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1, upper1 = (np.array([0, 50, 50]), np.array([10, 255, 255]))
    lower2, upper2 = (np.array([170, 50, 50]), np.array([180, 255, 255]))
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)
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

def hough_circles(img_like: np.ndarray, dp: float, min_dist: float, param1: float, param2: float, min_radius: int, max_radius: int) -> List[Circle]:
    """Run HoughCircles and return all detected circles."""
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

def circle_perimeter_support(edge_map: np.ndarray, circle: Circle, n_samples: int=360, band: int=3) -> float:
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
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (int(round(circle[0])), int(round(circle[1]))), int(round(circle[2])), 255, -1)
    return m

def circles_overlap(c1: Circle, c2: Circle) -> bool:
    """Check whether two circles overlap."""
    dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
    return dist < c1[2] + c2[2]

def color_ratio_in_mask(region: np.ndarray, color: np.ndarray) -> float:
    """Fraction of colored pixels within a uint8 255 region mask."""
    denom = int(cv2.countNonZero(region))
    if denom == 0:
        return 0.0
    return float(cv2.countNonZero(cv2.bitwise_and(color, region))) / float(denom)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def classify_grid(circles: List[Circle]) -> Tuple[Circle, Circle, Circle, Circle]:
    """Classify four circles into A (TL), B (TR), C (BL), D (BR) by position."""
    sorted_y = sorted(circles, key=lambda c: c[1])
    top = sorted(sorted_y[:2], key=lambda c: c[0])
    bot = sorted(sorted_y[2:], key=lambda c: c[0])
    return (top[0], top[1], bot[0], bot[1])

def verify_four_set_intersection(image_path: str, red_on_thresh: float=0.2, red_off_thresh: float=0.05, support_thresh: float=0.4, spill_ratio: float=0.05) -> bool:
    """
    Evaluate: Four overlapping circles/ellipses A, B, C, D.
    Shade only the intersection of all four (A n B n C n D). Color: Red.

    Criteria:
      C1: File exists and image is readable.
      C2: At least four circles detected.
      C3: Labels A, B, C, D each appear once inside a distinct circle.
      C4: All six circle pairs overlap.
      C5: Four-way intersection (A n B n C n D) is red.
      C6: All non-four-way regions are unshaded.
      C7: Red does not spill outside the circles.
    """
    if not os.path.exists(image_path):
        _print_fail('C1_file_exists: False')
        return False
    img = cv2.imread(image_path)
    if img is None:
        _print_fail('C1_file_exists: True', 'C1_image_readable: False')
        return False
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('C1_file_exists: True')
    print('C1_image_readable: True')
    edges = black_edge_map(img)
    min_dim = min(h, w)
    all_c: List[Circle] = []
    for dp, md, p1, p2, rmin, rmax in [(1.2, 0.1 * min_dim, 80, 40, int(0.1 * min_dim), int(0.35 * min_dim)), (1.2, 0.1 * min_dim, 80, 35, int(0.3 * min_dim), int(0.8 * min_dim))]:
        all_c.extend(hough_circles(edges, dp, md, p1, p2, rmin, rmax))
    all_c = dedupe_circles(all_c, 30.0, 30.0)
    circles = [c for c in all_c if circle_perimeter_support(edges, c) >= support_thresh]
    c2 = len(circles) >= 4
    print(f'C2_four_circles: {c2} (found {len(circles)})')
    if not c2:
        _print_fail(f'C2_four_circles: False (found {len(circles)})')
        return False
    circles = circles[:4]
    hits = find_labels_in_circles(image_path, circles, target_letters=['A', 'B', 'C', 'D'])
    label_map = {}
    used_circles = set()
    label_ok = True
    for letter in ['A', 'B', 'C', 'D']:
        letter_hits = [(l, cx, cy, c) for l, cx, cy, c in hits if l == letter]
        if not letter_hits:
            print(f'C3_label_{letter}_ok: False (not detected)')
            label_ok = False
            continue
        letter_hits.sort(key=lambda h: h[3], reverse=True)
        _, lx, ly, conf = letter_hits[0]
        inside = [i for i, c in enumerate(circles) if point_in_circle(lx, ly, c)]
        if inside and inside[0] not in used_circles:
            used_circles.add(inside[0])
            label_map[letter] = inside[0]
            print(f'C3_label_{letter}_ok: True (conf={conf:.3f})')
        else:
            print(f'C3_label_{letter}_ok: False (not inside unique circle)')
            label_ok = False
    if not label_ok or len(label_map) < 4:
        print('C3_labels_all_ok: False')
        print('Result: FAIL')
        return False
    print('C3_labels_all_ok: True')
    a = circles[label_map['A']]
    b = circles[label_map['B']]
    c = circles[label_map['C']]
    d = circles[label_map['D']]
    names_map = {'A': a, 'B': b, 'C': c, 'D': d}
    for lbl in ['A', 'B', 'C', 'D']:
        circ = names_map[lbl]
        print(f'C3_{lbl}: cx={circ[0]:.0f} cy={circ[1]:.0f} r={circ[2]:.0f}')
    labels_list = ['A', 'B', 'C', 'D']
    c4 = True
    for i_idx in range(4):
        for j_idx in range(i_idx + 1, 4):
            l1, l2 = (labels_list[i_idx], labels_list[j_idx])
            ov = circles_overlap(names_map[l1], names_map[l2])
            if not ov:
                c4 = False
            print(f'C4_{l1}_{l2}_overlap: {ov}')
    print(f'C4_all_overlap: {c4}')
    rm = red_mask_hsv(img)
    ms = {l: circle_mask((h, w), names_map[l]) for l in labels_list}
    quad = ms['A'] & ms['B'] & ms['C'] & ms['D']
    r_quad = color_ratio_in_mask(quad, rm)
    c5 = r_quad >= red_on_thresh
    print(f'C5_ABCD_intersection_red: {r_quad:.3f} ok={c5}')
    c6 = True
    for l in labels_list:
        others = [o for o in labels_list if o != l]
        region = ms[l].copy()
        for o in others:
            region = region & ~ms[o]
        ratio = color_ratio_in_mask(region, rm)
        ok = ratio <= red_off_thresh
        if not ok:
            c6 = False
        print(f'C6_{l}_only_red: {ratio:.3f} ok={ok}')
    for i_idx in range(4):
        for j_idx in range(i_idx + 1, 4):
            l1, l2 = (labels_list[i_idx], labels_list[j_idx])
            others = [o for o in labels_list if o != l1 and o != l2]
            region = ms[l1] & ms[l2]
            for o in others:
                region = region & ~ms[o]
            ratio = color_ratio_in_mask(region, rm)
            ok = ratio <= red_off_thresh
            if not ok:
                c6 = False
            print(f'C6_{l1}_{l2}_only_red: {ratio:.3f} ok={ok}')
    for i_idx in range(4):
        excluded = labels_list[i_idx]
        included = [l for l in labels_list if l != excluded]
        region = ms[included[0]] & ms[included[1]] & ms[included[2]] & ~ms[excluded]
        ratio = color_ratio_in_mask(region, rm)
        ok = ratio <= red_off_thresh
        if not ok:
            c6 = False
        triple_name = '_'.join(included)
        print(f'C6_{triple_name}_only_red: {ratio:.3f} ok={ok}')
    print(f'C6_non_quad_unshaded: {c6}')
    union = ms['A'] | ms['B'] | ms['C'] | ms['D']
    outside = ~(union > 0)
    red_outside = int(np.sum(outside & (rm > 0)))
    total_red = int(np.sum(rm > 0))
    spill = red_outside / total_red if total_red > 0 else 0.0
    c7 = spill <= spill_ratio
    print(f'C7_red_spill: {spill:.4f}')
    print(f'C7_spill_ok: {c7}')
    passed = c2 and c4 and c5 and c6 and c7
    print(f"Result: {('PASS' if passed else 'FAIL')}")
    return passed

def _print_fail(*lines):
    """Print early-exit failure lines."""
    for line in lines:
        print(line)
    print('Result: FAIL')

def evaluate(image_path: str):
    passed = verify_four_set_intersection(image_path)
    return {'passed': bool(passed), 'criteria': {'script_passed': bool(passed)}, 'meta': {'judge': 'set_46'}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen set case 46.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
