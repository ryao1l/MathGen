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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np
try:
    import pytesseract
except Exception:
    pytesseract = None
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluator_utils import print_report
import function_common
from function_common import *
globals().update({name: getattr(function_common, name) for name in dir(function_common) if name.startswith('_')})
PROMPT_ID = 28
sys.path.insert(0, os.path.dirname(__file__))
BLUE_H_LO, BLUE_H_HI = (85, 155)
BLUE_S_MIN, BLUE_V_MIN = (20, 30)
HSV_S_MIN = 15
HSV_V_MIN = 25
AXIS_THICKNESS = 10
XAXIS_GAP = 6
YAXIS_GAP = 8
ASYM_SEARCH_HALF_W = 30
ASYM_MIN_RUN_LEN = 25
ASYM_COL_SMOOTH_K = 41
CROSS_STRIP_HALF_W = 4
CROSS_DENSITY_MAX = 0.12
RUN_MIN_LEN = 8
RUN_ALLOW_GAP = 1
RUN_SEARCH_START_BAND = 8
RUN_SEARCH_MAX_BAND = 220
RUN_SEARCH_STEP = 6

def red_asymptote_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m_red1 = cv2.inRange(hsv, (0, HSV_S_MIN, HSV_V_MIN), (15, 255, 255))
    m_red2 = cv2.inRange(hsv, (160, HSV_S_MIN, HSV_V_MIN), (179, 255, 255))
    m_mag = cv2.inRange(hsv, (135, HSV_S_MIN, HSV_V_MIN), (170, 255, 255))
    mask = cv2.bitwise_or(m_red1, m_red2)
    mask = cv2.bitwise_or(mask, m_mag)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31))
    mask_v = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kv, iterations=1)
    mask = cv2.bitwise_or(mask, mask_v)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k2, iterations=1)
    return mask

def smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    ker = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x.astype(np.float32), ker, mode='same')

def find_vertical_peaks(mask: np.ndarray, n_peaks: int=2) -> List[int]:
    h, w = mask.shape[:2]
    col = (mask > 0).sum(axis=0).astype(np.float32) / float(h)
    col_s = smooth_1d(col, ASYM_COL_SMOOTH_K)
    peaks = []
    temp = col_s.copy()
    for _ in range(n_peaks):
        p = int(np.argmax(temp))
        if temp[p] <= 1e-06:
            break
        peaks.append(p)
        x0 = max(0, p - 40)
        x1 = min(w, p + 41)
        temp[x0:x1] = 0.0
    return peaks

def vertical_run_exists_in_strip(mask: np.ndarray, px_center: int, half_w: int, min_run_len: int) -> Tuple[bool, Dict[str, Any]]:
    h, w = mask.shape[:2]
    x0 = max(0, px_center - half_w)
    x1 = min(w, px_center + half_w + 1)
    strip = mask[:, x0:x1]
    if strip.size == 0:
        return (False, {'best_run': 0, 'best_x': None})
    best_run = 0
    best_x = None
    for xi in range(strip.shape[1]):
        col = (strip[:, xi] > 0).astype(np.uint8)
        run = 0
        max_run = 0
        for v in col:
            if v:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        if max_run > best_run:
            best_run = max_run
            best_x = x0 + xi
    return (best_run >= min_run_len, {'best_run': int(best_run), 'best_x': best_x})

def remove_axes(mask: np.ndarray, x_axis_y: float, y_axis_x: float) -> np.ndarray:
    h, w = mask.shape[:2]
    out = mask.copy()
    ya = int(round(x_axis_y))
    xa = int(round(y_axis_x))
    cv2.line(out, (0, ya), (w - 1, ya), 0, thickness=AXIS_THICKNESS)
    cv2.line(out, (xa, 0), (xa, h - 1), 0, thickness=AXIS_THICKNESS)
    out[max(0, ya - XAXIS_GAP):min(h, ya + XAXIS_GAP + 1), :] = 0
    out[:, max(0, xa - YAXIS_GAP):min(w, xa + YAXIS_GAP + 1)] = 0
    return out

def crossing_density(mask: np.ndarray, px_asym: int) -> float:
    h, w = mask.shape[:2]
    x0 = max(0, px_asym - CROSS_STRIP_HALF_W)
    x1 = min(w, px_asym + CROSS_STRIP_HALF_W + 1)
    strip = mask[:, x0:x1]
    if strip.size == 0:
        return 0.0
    return float(np.sum(strip > 0) / strip.size)

def longest_run_1d(bits: np.ndarray, allow_gap: int=1) -> Tuple[int, Optional[Tuple[int, int]]]:
    n = len(bits)
    best_len = 0
    best_seg = None
    i = 0
    while i < n:
        while i < n and bits[i] == 0:
            i += 1
        if i >= n:
            break
        start = i
        gap = 0
        j = i
        while j < n:
            if bits[j] == 1:
                j += 1
                continue
            gap += 1
            if gap > allow_gap:
                break
            j += 1
        end = j - 1
        while end >= start and bits[end] == 0:
            end -= 1
        seg_len = max(0, end - start + 1)
        if seg_len > best_len:
            best_len = seg_len
            best_seg = (start, end)
        i = j + 1
    return (best_len, best_seg)

def find_run_near_x(curve_mask: np.ndarray, px_center: int, side: str, x_axis_y: float, want_sign: int, min_run_len: int, max_band: int) -> Dict[str, Any]:
    h, w = curve_mask.shape[:2]
    px_center = int(np.clip(px_center, 0, w - 1))
    ya = int(round(x_axis_y))
    if want_sign > 0:
        y0, y1 = (0, max(0, ya - 1))
    else:
        y0, y1 = (min(h - 1, ya + 1), h - 1)
    if y1 <= y0:
        return {'status': 'insufficient'}
    for band in range(RUN_SEARCH_START_BAND, max_band + 1, RUN_SEARCH_STEP):
        if side == 'left':
            x0 = max(0, px_center - band)
            x1 = max(0, px_center - 1)
        else:
            x0 = min(w - 1, px_center + 1)
            x1 = min(w - 1, px_center + band)
        if x1 < x0:
            continue
        roi = curve_mask[y0:y1 + 1, x0:x1 + 1]
        col_indices = range(roi.shape[1] - 1, -1, -1) if side == 'left' else range(0, roi.shape[1])
        for cx in col_indices:
            col = (roi[:, cx] > 0).astype(np.uint8)
            best_len, seg = longest_run_1d(col, allow_gap=RUN_ALLOW_GAP)
            if best_len >= min_run_len and seg is not None:
                x_pick = x0 + cx
                y_mid = y0 + 0.5 * (seg[0] + seg[1])
                got = +1 if y_mid < x_axis_y else -1
                return {'status': 'ok', 'x_pick': int(x_pick), 'run_len': int(best_len), 'got_sign': int(got), 'band': int(band)}
    return {'status': 'insufficient'}

def evaluate(image_path: str) -> Dict[str, Any]:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {'passed': False, 'reason': 'Image not found'}
    h, w = img_bgr.shape[:2]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    x_ay, y_ax = detect_axes(img_gray)
    rmask = red_asymptote_mask(img_bgr)
    peaks = find_vertical_peaks(rmask, n_peaks=2)
    asym_positions = []
    for px in peaks:
        ok, info = vertical_run_exists_in_strip(rmask, px, ASYM_SEARCH_HALF_W, ASYM_MIN_RUN_LEN)
        if ok:
            actual_x = info['best_x'] if info['best_x'] is not None else px
            asym_positions.append(actual_x)
    blue_mask = extract_color_mask(img_hsv, (BLUE_H_LO, BLUE_H_HI), s_min=BLUE_S_MIN, v_min=BLUE_V_MIN)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, k_open, iterations=1)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, k_close, iterations=1)
    curve_mask = remove_axes(blue_mask, x_ay, y_ax)
    blue_pixels = np.column_stack(np.where(blue_mask > 0))
    criteria = {}
    arrow_info = check_single_headed_arrows(img_gray, x_ay, y_ax)
    criteria['single_headed_arrows'] = arrow_info['is_single_headed']
    criteria['asymptote_present'] = len(asym_positions) >= 1
    criteria['blue_curve_exists'] = len(blue_pixels) > 100
    curve_not_crossing = True
    for px_asym in asym_positions:
        if crossing_density(curve_mask, int(px_asym)) >= CROSS_DENSITY_MAX:
            curve_not_crossing = False
            break
    criteria['curve_not_crossing_asymptote'] = curve_not_crossing
    sign_check_ok = False
    if len(asym_positions) >= 1:
        right_asym = max(asym_positions)
        left_res = find_run_near_x(curve_mask, int(right_asym), 'left', x_ay, want_sign=-1, min_run_len=RUN_MIN_LEN, max_band=RUN_SEARCH_MAX_BAND)
        right_res = find_run_near_x(curve_mask, int(right_asym), 'right', x_ay, want_sign=+1, min_run_len=RUN_MIN_LEN, max_band=RUN_SEARCH_MAX_BAND)
        left_ok = left_res['status'] == 'ok' and left_res['got_sign'] == -1
        right_ok = right_res['status'] == 'ok' and right_res['got_sign'] == +1
        sign_check_ok = left_ok or right_ok
    criteria['sign_check'] = sign_check_ok
    x_labels_ok, x_label_detail = _tick_structure_ok(img_gray, x_ay, y_ax, axis='x', img_hsv=img_hsv)
    criteria['x_axis_labels_correct'] = x_labels_ok
    y_labels_ok, y_label_detail = _tick_structure_ok(img_gray, x_ay, y_ax, axis='y', img_hsv=img_hsv)
    criteria['y_axis_labels_correct'] = y_labels_ok
    vlt_ok = True
    vlt_detail = {}
    if len(asym_positions) >= 1:
        boundaries = sorted([0] + [int(p) for p in asym_positions] + [w])
        branch_results = []
        for bi in range(len(boundaries) - 1):
            x0, x1 = (boundaries[bi], boundaries[bi + 1])
            if x1 - x0 < 10:
                continue
            branch_mask = np.zeros_like(blue_mask)
            branch_mask[:, x0:x1] = blue_mask[:, x0:x1]
            b_ok, b_detail = check_vertical_line_test(branch_mask)
            branch_results.append({'range': [x0, x1], 'ok': b_ok, **b_detail})
            if not b_ok:
                vlt_ok = False
        vlt_detail = {'branches': branch_results}
    else:
        vlt_ok, vlt_detail = check_vertical_line_test(blue_mask)
    criteria['vertical_line_test'] = vlt_ok
    if len(blue_pixels) > 0:
        coverage = (np.max(blue_pixels[:, 1]) - np.min(blue_pixels[:, 1])) / w
        criteria['domain_coverage'] = coverage > 0.5
    else:
        criteria['domain_coverage'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 34, 'passed': passed, 'criteria': criteria, 'meta': {'asymptote_positions': [int(p) for p in asym_positions], 'num_asymptotes': len(asym_positions), 'arrow_info': arrow_info, 'vertical_line_test': vlt_detail, 'x_axis_labels': x_label_detail, 'y_axis_labels': y_label_detail}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 28.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
