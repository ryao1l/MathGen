#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'plane'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluator_utils import print_report
from open_scene_common import *
import plane_common as _common
from plane_common import *
PROMPT_ID = 21
PID = 26
TYPE = 'plane'
_PLANE_DIR = Path(__file__).resolve().parents[1] / 'plane'
sys.path.insert(0, str(_PLANE_DIR))

def _dist(p, q):
    return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))

def _circle_ink_coverage(img, center, radius, band_px):
    cx, cy = (float(center[0]), float(center[1]))
    r = float(radius)
    if r <= 1.0:
        return (0.0, 0.0)
    _, bw = _gray_and_ink_mask(img)
    h, w = bw.shape[:2]
    n_full = max(8, int(round(2.0 * math.pi * r / 3.0)))
    hit = 0
    vis = 0
    for i in range(n_full):
        th = 2.0 * math.pi * float(i) / float(n_full)
        x = int(round(cx + r * math.cos(th)))
        y = int(round(cy + r * math.sin(th)))
        if not (0 <= x < w and 0 <= y < h):
            continue
        vis += 1
        ok = False
        for dr in range(-int(band_px), int(band_px) + 1):
            rr = r + float(dr)
            xx = int(round(cx + rr * math.cos(th)))
            yy = int(round(cy + rr * math.sin(th)))
            if 0 <= xx < w and 0 <= yy < h and (bw[yy, xx] > 0):
                ok = True
                break
        if ok:
            hit += 1
    cov = float(hit) / float(max(1, vis))
    vis_ratio = float(vis) / float(max(1, n_full))
    return (cov, vis_ratio)

def _circle_equivalent(ca, cb, min_hw):
    center_tol = scale_px(min_hw, 0.05, floor_px=0.0)
    radius_tol = scale_px(min_hw, 0.045, floor_px=0.0)
    if _dist((ca[0], ca[1]), (cb[0], cb[1])) > center_tol:
        return False
    return abs(float(ca[2]) - float(cb[2])) <= radius_tol

def _dedup_circle_candidates_by_center(candidates, min_hw):
    if not candidates:
        return []
    center_tol = scale_px(min_hw, 0.04, floor_px=0.0)
    out = []
    for item in candidates:
        x, y = (float(item[0]), float(item[1]))
        if any((_dist((x, y), (float(k[0]), float(k[1]))) <= center_tol for k in out)):
            continue
        out.append(item)
    return out

def _detect_circle_candidates(img, min_hw):
    h, w = img.shape[:2]
    min_r = int(round(scale_px(min_hw, 0.03, floor_px=1.0)))
    max_r = float(scale_px(min_hw, 0.62, floor_px=0.0))
    score_th = int(round(scale_px(min_hw, 0.09, floor_px=1.0)))
    band = int(max(1, round(scale_px(min_hw, 0.01, floor_px=0.0))))
    found, scores = _find_top_k_circles(img, k=26, min_r=min_r, max_r=0, seed=0, iters=4000, score_th=score_th)
    x_margin = 0.15 * float(w)
    y_margin = 0.15 * float(h)
    inside_tol = scale_px(min_hw, 0.05, floor_px=0.0)
    raw = []
    for (cx, cy, r), s in zip(found, scores):
        x, y, rr = (float(cx), float(cy), float(r))
        if rr <= 0.0 or rr < min_r or rr > max_r:
            continue
        if x < -x_margin or x > float(w) + x_margin:
            continue
        if y < -y_margin or y > float(h) + y_margin:
            continue
        if min(x - rr, y - rr, float(w) - (x + rr), float(h) - (y + rr)) < -inside_tol:
            continue
        refined = _refine_circle_radius_by_inner_outer_edges(img, (x, y, rr))
        if refined is None:
            continue
        rx, ry, rr2 = [float(v) for v in refined]
        if rr2 <= 0.0 or rr2 < min_r or rr2 > max_r:
            continue
        if min(rx - rr2, ry - rr2, float(w) - (rx + rr2), float(h) - (ry + rr2)) < -inside_tol:
            continue
        cov, vis = _circle_ink_coverage(img, (rx, ry), rr2, band)
        if vis < 0.45 or cov < 0.28:
            continue
        raw.append((rx, ry, rr2, int(s), float(cov), float(vis)))
    if raw:
        raw.sort(key=lambda t: (-t[3], -t[4], -t[2]))
        merged = _merge_circle_candidates([(x, y, r, s) for x, y, r, s, _, _ in raw], center_tol=scale_px(min_hw, 0.03, floor_px=0.0), radius_tol=scale_px(min_hw, 0.03, floor_px=0.0))
        out = []
        for x, y, r, s in merged:
            cov, vis = _circle_ink_coverage(img, (x, y), r, band)
            if vis < 0.45 or cov < 0.28:
                continue
            out.append((float(x), float(y), float(r), int(s), float(cov), float(vis)))
        if out:
            out.sort(key=lambda t: (-t[3], -t[4], -t[2]))
            return _dedup_circle_candidates_by_center(out, min_hw=min_hw)
    fallback = []
    for order in (1, 2, 3, 4, 5, 6):
        c = detect_circle(img, order=order, min_r=min_r, max_r=0)
        if c is None:
            continue
        x, y, r = [float(v) for v in c]
        if r <= 0.0 or r < min_r or r > max_r:
            continue
        if min(x - r, y - r, float(w) - (x + r), float(h) - (y + r)) < -inside_tol:
            continue
        cov, vis = _circle_ink_coverage(img, (x, y), r, band)
        if vis < 0.4 or cov < 0.24:
            continue
        fallback.append((x, y, r, 0, cov, vis))
    dedup = []
    for item in fallback:
        c = (item[0], item[1], item[2])
        if any((_circle_equivalent(c, (q[0], q[1], q[2]), min_hw=min_hw) for q in dedup)):
            continue
        dedup.append(item)
    dedup.sort(key=lambda t: (-t[4], -t[2]))
    return _dedup_circle_candidates_by_center(dedup, min_hw=min_hw)

def _select_disjoint_pair(candidates, min_hw):
    if candidates is None or len(candidates) < 2:
        return None
    best = None
    best_score = None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            if a[2] >= b[2]:
                big = a
                small = b
            else:
                big = b
                small = a
            bx, by, rb, sb, cov_b, vis_b = [float(v) for v in big]
            sx, sy, rs, ss, cov_s, vis_s = [float(v) for v in small]
            size_tol = max(scale_px(min_hw, 0.012, floor_px=0.0), 0.08 * rs)
            if rb - rs <= size_tol:
                continue
            d = _dist((bx, by), (sx, sy))
            contain_tol = max(scale_px(min_hw, 0.012, floor_px=0.0), 0.03 * rs)
            non_overlap_tol = max(scale_px(min_hw, 0.01, floor_px=0.0), 0.03 * rs)
            if d <= abs(rb - rs) + contain_tol:
                continue
            if d < rb + rs - non_overlap_tol:
                continue
            ratio = rs / max(1e-06, rb)
            if ratio >= 0.97:
                continue
            gap = d - (rb + rs)
            score = float(sb + ss) + 16.0 * float(cov_b + cov_s) + 9.0 * float(vis_b + vis_s) + 0.12 * float(d) + 0.2 * float(gap) + 0.45 * float(rb - rs) - 11.0 * float(ratio)
            if best_score is None or score > best_score:
                best_score = score
                best = {'big': (float(bx), float(by), float(rb)), 'small': (float(sx), float(sy), float(rs)), 'd': float(d), 'size_tol': float(size_tol), 'contain_tol': float(contain_tol), 'non_overlap_tol': float(non_overlap_tol)}
    return best

def _bright_segment_ratio(img, p1, p2, trim_ratio=0.18, thickness=2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    otsu_t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_th = int(max(170, min(245, int(otsu_t))))
    bright = (gray >= bright_th).astype(np.uint8)
    x1, y1 = (float(p1[0]), float(p1[1]))
    x2, y2 = (float(p2[0]), float(p2[1]))
    t0 = float(max(0.0, min(0.45, trim_ratio)))
    t1 = 1.0 - t0
    a = (x1 + (x2 - x1) * t0, y1 + (y2 - y1) * t0)
    b = (x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1)
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.line(mask, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))), 255, int(max(1, thickness)))
    den = int(np.count_nonzero(mask))
    if den <= 0:
        return 0.0
    num = int(np.count_nonzero((bright > 0) & (mask > 0)))
    return float(num) / float(den)

def _circle_interior_bright_ratio(img, circle, inner_ratio=0.55):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    otsu_t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_th = int(max(170, min(245, int(otsu_t))))
    cx, cy, r = [float(v) for v in circle]
    rr = int(round(max(1.0, float(r) * float(inner_ratio))))
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), rr, 255, -1)
    den = int(np.count_nonzero(mask))
    if den <= 0:
        return 0.0
    num = int(np.count_nonzero((gray >= bright_th) & (mask > 0)))
    return float(num) / float(den)

def judge_plane_26(img):
    if img is None:
        return [False, 'Input image is None. ']
    min_hw = float(min(img.shape[:2]))
    candidates = _detect_circle_candidates(img, min_hw=min_hw)
    if len(candidates) < 2:
        return [False, 'Failed to detect two valid circle candidates. ']
    best_pair = _select_disjoint_pair(candidates, min_hw=min_hw)
    if best_pair is None:
        return [False, 'Failed to find two non-overlapping circles with distinct sizes. ']
    bx, by, rb = [float(v) for v in best_pair['big']]
    sx, sy, rs = [float(v) for v in best_pair['small']]
    d = float(best_pair['d'])
    size_tol = float(best_pair['size_tol'])
    contain_tol = float(best_pair['contain_tol'])
    non_overlap_tol = float(best_pair['non_overlap_tol'])
    if min(rb, rs) < scale_px(min_hw, 0.035, floor_px=0.0):
        return [False, 'Detected circles are too small. ']
    if rb - rs <= size_tol:
        return [False, 'The two circles are not clearly different in size. ']
    if d <= abs(rb - rs) + contain_tol:
        return [False, 'One circle is contained in or too close to the other. ']
    if d < rb + rs - non_overlap_tol:
        return [False, 'The two circles overlap. ']
    interior_big = _circle_interior_bright_ratio(img, (bx, by, rb), inner_ratio=0.55)
    interior_small = _circle_interior_bright_ratio(img, (sx, sy, rs), inner_ratio=0.55)
    if interior_big <= 0.35 and interior_small <= 0.35:
        center_link_ratio = _bright_segment_ratio(img, (bx, by), (sx, sy), trim_ratio=0.18, thickness=max(2, int(round(scale_px(min_hw, 0.004, floor_px=0.0)))))
        if center_link_ratio >= 0.35:
            return [False, 'Unexpected line segment connecting the two circle centers. ']
    extra_circles = 0
    for cx, cy, rr, _, cov, vis in candidates:
        c = (float(cx), float(cy), float(rr))
        if _circle_equivalent(c, (bx, by, rb), min_hw=min_hw) or _circle_equivalent(c, (sx, sy, rs), min_hw=min_hw):
            continue
        if rr < 0.55 * rs:
            continue
        if vis < 0.48 or cov < 0.3:
            continue
        extra_circles += 1
    if extra_circles > 0:
        return [False, f'Detected extra prominent circle(s): {extra_circles}. ']
    return [True, '']

def evaluate(image_path):
    return evaluate_plane_task(image_path=image_path, pid=PID, judge_fn=judge_plane_26, require_ocr=False, task_type=TYPE)

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen open_scene case 21.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
