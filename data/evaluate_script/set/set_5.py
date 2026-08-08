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
PROMPT_ID = 5
Circle = Tuple[float, float, float]

def yellow_mask_hsv(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([20, 50, 50]), np.array([35, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask

def black_edge_map(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.Canny(inv, 50, 150)

def hough_circles(img, dp, md, p1, p2, rmin, rmax):
    c = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=dp, minDist=md, param1=p1, param2=p2, minRadius=rmin, maxRadius=rmax)
    if c is None:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in c[0]]

def dedupe_circles(circles, ct=25.0, rt=25.0):
    out = []
    for cx, cy, cr in sorted(circles, key=lambda t: t[2], reverse=True):
        if any((np.hypot(cx - sx, cy - sy) < ct and abs(cr - sr) < rt for sx, sy, sr in out)):
            continue
        out.append((cx, cy, cr))
    return out

def circle_perimeter_support(edge, circ, n=360, band=3):
    h, w = edge.shape[:2]
    cx, cy, r = circ
    if r <= 1:
        return 0.0
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.round(cx + r * np.cos(ang)).astype(np.int32)
    ys = np.round(cy + r * np.sin(ang)).astype(np.int32)
    ok = total = 0
    for x, y in zip(xs, ys):
        if x < 0 or x >= w or y < 0 or (y >= h):
            continue
        total += 1
        ok += int(np.any(edge[max(0, y - band):min(h, y + band + 1), max(0, x - band):min(w, x + band + 1)] > 0))
    return ok / total if total else 0.0

def circle_mask(hw, circ):
    m = np.zeros(hw, np.uint8)
    cv2.circle(m, (int(round(circ[0])), int(round(circ[1]))), int(round(circ[2])), 255, -1)
    return m

def circles_overlap(c1, c2):
    return np.hypot(c1[0] - c2[0], c1[1] - c2[1]) < c1[2] + c2[2]

def color_ratio(region, color):
    d = int(cv2.countNonZero(region))
    return float(cv2.countNonZero(cv2.bitwise_and(color, region))) / d if d else 0.0

def classify_abc(circles):
    by_y = sorted(enumerate(circles), key=lambda item: item[1][1])
    c_idx = by_y[-1][0]
    top_two = sorted(by_y[:2], key=lambda item: item[1][0])
    return (top_two[0][0], top_two[1][0], c_idx)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def evaluate(image_path, on_thresh=0.2, off_thresh=0.05, support=0.43, spill_max=0.05):
    if not os.path.exists(image_path):
        print('C1_file_exists: False')
        print('Result: FAIL')
        return False
    img = cv2.imread(image_path)
    if img is None:
        print('C1_image_readable: False')
        print('Result: FAIL')
        return False
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('C1_file_exists: True')
    print('C1_image_readable: True')
    edges = black_edge_map(img)
    md = min(h, w)
    ac = []
    for dp, d, p1, p2, rn, rx in [(1.2, 0.1 * md, 80, 40, int(0.1 * md), int(0.35 * md)), (1.2, 0.1 * md, 80, 35, int(0.3 * md), int(0.8 * md))]:
        ac.extend(hough_circles(edges, dp, d, p1, p2, rn, rx))
    ac = dedupe_circles(ac)
    circles = [c for c in ac if circle_perimeter_support(edges, c) >= support]
    print(f'C2_circles_found: {len(circles)}')
    if len(circles) < 3:
        print('Result: FAIL')
        return False
    circles = circles[:3]
    c3 = all((circles_overlap(circles[i], circles[j]) for i in range(3) for j in range(i + 1, 3)))
    print(f'C3_all_overlap: {c3}')
    if not c3:
        print('Result: FAIL')
        return False
    hits = find_labels_in_circles(image_path, circles, target_letters=['A', 'B', 'C'])
    lmap = {}
    used_circles = set()
    label_ok = True
    for letter in ['A', 'B', 'C']:
        letter_hits = [(l, cx, cy, c) for l, cx, cy, c in hits if l == letter]
        if not letter_hits:
            print(f'C4_label_{letter}_ok: False (not detected)')
            label_ok = False
            continue
        letter_hits.sort(key=lambda h: h[3], reverse=True)
        _, lx, ly, conf = letter_hits[0]
        inside = [i for i, c in enumerate(circles) if point_in_circle(lx, ly, c)]
        if inside and inside[0] not in used_circles:
            used_circles.add(inside[0])
            lmap[letter] = inside[0]
            print(f'C4_label_{letter}_ok: True (conf={conf:.3f})')
        else:
            print(f'C4_label_{letter}_ok: False (not inside unique circle)')
            label_ok = False
    if not label_ok or len(lmap) < 3:
        print('C4_labels_all_ok: False')
        ia, ib, ic = classify_abc(circles)
        print('C4_label_fallback: positional')
    else:
        print('C4_labels_all_ok: True')
        ia, ib, ic = (lmap['A'], lmap['B'], lmap['C'])
    ma = circle_mask((h, w), circles[ia])
    mb = circle_mask((h, w), circles[ib])
    mc_ = circle_mask((h, w), circles[ic])
    ym = yellow_mask_hsv(img)
    triple = ma & mb & mc_
    ab = ma & mb & ~mc_
    ac_ = ma & mc_ & ~mb
    bc = mb & mc_ & ~ma
    ao = ma & ~mb & ~mc_
    bo = mb & ~ma & ~mc_
    co = mc_ & ~ma & ~mb
    regions = [('B_only', bo, True), ('A_only', ao, False), ('C_only', co, False), ('AB_not_C', ab, False), ('AC_not_B', ac_, False), ('BC_not_A', bc, False), ('triple_ABC', triple, False)]
    c5 = True
    for name, reg, should_on in regions:
        r = color_ratio(reg, ym)
        ok = r >= on_thresh if should_on else r <= off_thresh
        if not ok:
            c5 = False
        print(f'C5_{name}_yellow: {r:.3f} ok={ok}')
    union = ma | mb | mc_
    outside = ~(union > 0)
    yo = int(np.sum(outside & (ym > 0)))
    ty = int(np.sum(ym > 0))
    sp = yo / ty if ty else 0.0
    c6 = sp <= spill_max
    print(f'C6_spill: {sp:.4f} ok={c6}')
    passed = c5 and c6
    print(f"Result: {('PASS' if passed else 'FAIL')}")
    return passed

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen set case 5.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
