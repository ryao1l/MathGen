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
PROMPT_ID = 34
CANNY1, CANNY2 = (50, 150)
HOUGH_THRESH = 120
MAX_GAP = 25
AXIS_THICKNESS = 10
XAXIS_GAP = 6
YAXIS_GAP = 8
HSV_S_MIN = 15
HSV_V_MIN = 25
ASYM_SEARCH_HALF_W = 30
ASYM_MIN_RUN_LEN = 25
ASYM_COL_SMOOTH_K = 41
ADAPT_BLOCK = 31
ADAPT_C = 9
CROSS_STRIP_HALF_W = 4
CROSS_DENSITY_MAX = 0.12
RUN_MIN_LEN = 8
RUN_ALLOW_GAP = 1
RUN_SEARCH_START_BAND = 8
RUN_SEARCH_MAX_BAND = 180
RUN_SEARCH_STEP = 6
LEFT_BAND_W = 220
LEFT_DENSITY_MAX = 0.01
MIN_OCR_CONF = 12.0
OCR_SCALE = 3
OCR_ADAPT_BLOCK = 31
OCR_ADAPT_C = 9
EXPECTED_X = list(range(-10, 11))
EXPECTED_Y = list(range(-20, 21))
RANSAC_ITERS = 250
INLIER_TOL_X = 0.35
MIN_INLIERS_X = 2
Y_MODEL_CANDIDATES = 450
Y_INLIER_TOL = 0.75
Y_SLOPE_RANGE = (-0.8, -0.001)
Y_MAX_DUP_PER_VALUE = 2
SAMPLE_XS = [2, 3, 6]
SAMPLE_ERR_TOL = 0.6
SAMPLE_PASS_AT_LEAST = 2

def maybe_set_tesseract_cmd():
    pass

def f(x: float) -> float:
    return float(np.log((x + 3.0) / (x - 1.0)))

def detect_axes_hough(img_bgr: np.ndarray) -> Tuple[float, float]:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY1, CANNY2)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=HOUGH_THRESH, minLineLength=int(0.45 * w), maxLineGap=MAX_GAP)
    x_axis_y = h / 2.0
    y_axis_x = w / 2.0
    if lines is None:
        return (x_axis_y, y_axis_x)
    best_h = (None, -1.0)
    best_v = (None, -1.0)
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = (x2 - x1, y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < 0.4 * w:
            continue
        if abs(dy) <= 4:
            ymid = (y1 + y2) / 2.0
            score = length - 0.5 * abs(ymid - h / 2.0)
            if score > best_h[1]:
                best_h = (ymid, score)
        if abs(dx) <= 4:
            xmid = (x1 + x2) / 2.0
            score = length - 0.5 * abs(xmid - w / 2.0)
            if score > best_v[1]:
                best_v = (xmid, score)
    if best_h[0] is not None:
        x_axis_y = best_h[0]
    if best_v[0] is not None:
        y_axis_x = best_v[0]
    return (x_axis_y, y_axis_x)

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

def find_one_vertical_peak(mask: np.ndarray) -> Optional[int]:
    h, w = mask.shape[:2]
    col = (mask > 0).sum(axis=0).astype(np.float32) / float(h)
    col_s = smooth_1d(col, ASYM_COL_SMOOTH_K)
    p = int(np.argmax(col_s))
    if col_s[p] <= 1e-06:
        return None
    return p

def vertical_run_exists_in_strip(mask: np.ndarray, px_center: int, half_w: int, min_run_len: int) -> Tuple[bool, Dict[str, Any]]:
    h, w = mask.shape[:2]
    x0 = max(0, px_center - half_w)
    x1 = min(w, px_center + half_w + 1)
    strip = mask[:, x0:x1]
    if strip.size == 0:
        return (False, {'best_run': 0, 'best_x': None, 'strip': (x0, x1)})
    best_run = 0
    best_x = None
    for xi in range(strip.shape[1]):
        col = (strip[:, xi] > 0).astype(np.uint8)
        run = 0
        max_run = 0
        for v in col:
            if v:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0
        if max_run > best_run:
            best_run = max_run
            best_x = x0 + xi
    ok = best_run >= min_run_len
    return (ok, {'best_run': int(best_run), 'best_x': best_x, 'strip': (x0, x1)})

def extract_black_mask(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPT_BLOCK, ADAPT_C)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k, iterations=1)
    return bin_img

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

def left_density(mask: np.ndarray, px_asym: int, band_w: int) -> float:
    h, w = mask.shape[:2]
    x0 = max(0, px_asym - band_w)
    x1 = max(0, px_asym - 1)
    if x1 <= x0:
        return 0.0
    roi = mask[:, x0:x1]
    return float(np.sum(roi > 0) / roi.size)

def sign_from_pixel_y(ypix: float, x_axis_y: float) -> int:
    return +1 if ypix < x_axis_y else -1

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
                got = sign_from_pixel_y(y_mid, x_axis_y)
                return {'status': 'ok', 'x_pick': int(x_pick), 'run_len': int(best_len), 'got_sign': int(got), 'band': int(band)}
    return {'status': 'insufficient'}

def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, OCR_ADAPT_BLOCK, OCR_ADAPT_C)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k, iterations=1)
    return bin_img

def ocr_numbers_robust(img_bgr: np.ndarray, roi: Optional[Tuple[int, int, int, int]]=None) -> List[Dict[str, Any]]:
    H, W = img_bgr.shape[:2]
    x0, y0, x1, y1 = (0, 0, W, H)
    if roi is not None:
        x0, y0, x1, y1 = roi
        x0, y0 = (max(0, x0), max(0, y0))
        x1, y1 = (min(W, x1), min(H, y1))
        img_bgr = img_bgr[y0:y1, x0:x1]
    up = cv2.resize(img_bgr, None, fx=OCR_SCALE, fy=OCR_SCALE, interpolation=cv2.INTER_CUBIC)
    ocr_bin = preprocess_for_ocr(up)
    config = '--oem 3 --psm 11 -c tessedit_char_whitelist=-−—–0123456789.,'
    data = pytesseract.image_to_data(ocr_bin, config=config, output_type=pytesseract.Output.DICT)
    tokens = []
    n = len(data['text'])
    for i in range(n):
        txt = (data['text'][i] or '').strip()
        conf = float(data['conf'][i]) if data['conf'][i] != '-1' else -1.0
        if conf < MIN_OCR_CONF:
            continue
        txt = txt.replace('−', '-').replace('—', '-').replace('–', '-')
        txt = txt.strip('., ')
        if txt == '':
            continue
        lx, ty, bw, bh = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        cx, cy = (lx + bw / 2.0, ty + bh / 2.0)
        center = (cx / OCR_SCALE + x0, cy / OCR_SCALE + y0)
        bbox = (lx / OCR_SCALE + x0, ty / OCR_SCALE + y0, bw / OCR_SCALE, bh / OCR_SCALE)
        tokens.append({'text': txt, 'conf': conf, 'center': center, 'bbox': bbox})
    merged = []
    used = set()
    order = sorted(range(len(tokens)), key=lambda i: tokens[i]['center'][0])
    for i in order:
        if i in used:
            continue
        t = tokens[i]
        if t['text'] == '-':
            best_j = None
            best_dx = 1000000000.0
            for j in order:
                if j == i or j in used:
                    continue
                u = tokens[j]
                if abs(u['center'][1] - t['center'][1]) > 14:
                    continue
                dx = u['center'][0] - t['center'][0]
                if 0 < dx < best_dx:
                    best_dx = dx
                    best_j = j
            if best_j is not None:
                u = tokens[best_j]
                if re.fullmatch('\\d+', u['text']) and best_dx < 30:
                    merged.append({'value': int('-' + u['text']), 'conf': min(t['conf'], u['conf']), 'center': ((t['center'][0] + u['center'][0]) / 2.0, (t['center'][1] + u['center'][1]) / 2.0), 'bbox': t['bbox']})
                    used.add(i)
                    used.add(best_j)
                    continue
        if re.fullmatch('-?\\d+', t['text']):
            merged.append({'value': int(t['text']), 'conf': t['conf'], 'center': t['center'], 'bbox': t['bbox']})
            used.add(i)
    return merged

def compute_rois(h: int, w: int, x_axis_y: float, y_axis_x: float) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    band = int(0.18 * h)
    y0 = max(0, int(x_axis_y - band))
    y1 = min(h, int(x_axis_y + band))
    x_roi = (0, y0, w, y1)
    x1 = max(0, int(y_axis_x - 6))
    x0 = max(0, int(y_axis_x - 0.22 * w))
    y_roi = (x0, 0, x1, h)
    return (x_roi, y_roi)

def ransac_fit_v_from_p(points: List[Tuple[float, float]], iters: int, tol: float, min_inliers: int) -> Tuple[Optional[float], Optional[float]]:
    if len(points) < 2:
        return (None, None)
    P = np.array([p for p, _ in points], dtype=np.float64)
    V = np.array([v for _, v in points], dtype=np.float64)
    rng = np.random.default_rng(0)
    best_inliers: List[int] = []
    best_model = (None, None)
    for _ in range(iters):
        i, j = rng.choice(len(points), size=2, replace=False)
        if abs(P[j] - P[i]) < 1e-06:
            continue
        a = (V[j] - V[i]) / (P[j] - P[i])
        b = V[i] - a * P[i]
        pred = a * P + b
        err = np.abs(pred - V)
        inliers = np.where(err <= tol)[0].tolist()
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (float(a), float(b))
    if len(best_inliers) < min_inliers:
        return (None, None)
    inP = P[best_inliers]
    inV = V[best_inliers]
    A = np.vstack([inP, np.ones_like(inP)]).T
    a, b = np.linalg.lstsq(A, inV, rcond=None)[0]
    return (float(a), float(b))

def build_x_mapping(x_dets: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    per_val: Dict[int, List[Dict[str, Any]]] = {}
    for d in x_dets:
        v = int(d['value'])
        if v not in EXPECTED_X:
            continue
        per_val.setdefault(v, []).append(d)
    chosen = []
    for v, lst in per_val.items():
        chosen.extend(sorted(lst, key=lambda z: z['conf'], reverse=True)[:1])
    pts = [(float(d['center'][0]), float(d['value'])) for d in chosen]
    return ransac_fit_v_from_p(pts, RANSAC_ITERS, INLIER_TOL_X, MIN_INLIERS_X)

def filter_y_candidates(y_dets: List[Dict[str, Any]], x_axis_y: float, y_axis_x: float, h: int) -> List[Dict[str, Any]]:
    out = []
    for d in y_dets:
        v = int(d['value'])
        if v not in EXPECTED_Y:
            continue
        cx, cy = d['center']
        x0, y0, bw, bh = d['bbox']
        if abs(cy - x_axis_y) < 0.12 * h:
            continue
        if x0 + bw > y_axis_x - 2:
            continue
        out.append(d)
    per_val: Dict[int, List[Dict[str, Any]]] = {}
    for d in out:
        per_val.setdefault(int(d['value']), []).append(d)
    reduced = []
    for v, lst in per_val.items():
        reduced.extend(sorted(lst, key=lambda z: z['conf'], reverse=True)[:Y_MAX_DUP_PER_VALUE])
    return reduced

def build_y_mapping_multiverify(y_dets: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    if len(y_dets) < 2:
        return (None, None)
    pts = [(float(d['center'][1]), float(d['value']), float(d['conf'])) for d in y_dets]
    rng = np.random.default_rng(1)

    def score_model(ay: float, by: float) -> float:
        if not Y_SLOPE_RANGE[0] < ay < Y_SLOPE_RANGE[1]:
            return -1000000000.0
        inliers = 0
        for py, yv, _ in pts:
            if abs(ay * py + by - yv) <= Y_INLIER_TOL:
                inliers += 1
        return float(inliers)
    best_score = -1e+18
    best_model = (None, None)
    for _ in range(Y_MODEL_CANDIDATES):
        i, j = rng.choice(len(pts), size=2, replace=False)
        py1, y1, _ = pts[i]
        py2, y2, _ = pts[j]
        if abs(py2 - py1) < 1e-06:
            continue
        ay = (y2 - y1) / (py2 - py1)
        by = y1 - ay * py1
        s = score_model(ay, by)
        if s > best_score:
            best_score = s
            best_model = (float(ay), float(by))
    if best_model[0] is None:
        return (None, None)
    ay, by = best_model
    if not Y_SLOPE_RANGE[0] < ay < Y_SLOPE_RANGE[1]:
        return (None, None)
    return (ay, by)

def curve_y_pixel_at_x(mask: np.ndarray, px_x: int, half_window: int=5) -> Optional[float]:
    h, w = mask.shape[:2]
    px_x = int(np.clip(px_x, 0, w - 1))
    x0 = max(0, px_x - half_window)
    x1 = min(w, px_x + half_window + 1)
    strip = mask[:, x0:x1]
    ys = np.where(strip > 0)[0]
    if len(ys) == 0:
        return None
    return float(np.median(ys))

def evaluate_plot(image_path: str) -> Dict[str, object]:
    """Evaluate the plot for the ln((x+3)/(x-1)) task."""
    maybe_set_tesseract_cmd()
    result: Dict[str, object] = {'criteria': {}, 'passed': False}
    img = cv2.imread(image_path)
    if img is None:
        result['criteria'] = {'image_readable': False, 'asymptote_present': False, 'curve_not_crossing_asymptote': False, 'right_branch_positive': False, 'left_branch_empty': False}
        return result
    x_axis_y, y_axis_x = detect_axes_hough(img)
    rmask = red_asymptote_mask(img)
    px_asym = find_one_vertical_peak(rmask)
    asymptote_present = False
    if px_asym is not None:
        asymptote_present, _ = vertical_run_exists_in_strip(rmask, px_asym, ASYM_SEARCH_HALF_W, ASYM_MIN_RUN_LEN)
    black = extract_black_mask(img)
    curve_mask = remove_axes(black, x_axis_y, y_axis_x)
    curve_not_crossing = False
    right_branch_positive = False
    left_branch_empty = False
    if px_asym is not None:
        cross = crossing_density(curve_mask, px_asym)
        curve_not_crossing = cross < CROSS_DENSITY_MAX
        right_res = find_run_near_x(curve_mask, px_asym, 'right', x_axis_y, want_sign=+1, min_run_len=RUN_MIN_LEN, max_band=RUN_SEARCH_MAX_BAND)
        right_branch_positive = right_res['status'] == 'ok' and right_res['got_sign'] == +1
        densL = left_density(curve_mask, px_asym, LEFT_BAND_W)
        left_pos = find_run_near_x(curve_mask, px_asym, 'left', x_axis_y, want_sign=+1, min_run_len=RUN_MIN_LEN, max_band=RUN_SEARCH_MAX_BAND)
        left_neg = find_run_near_x(curve_mask, px_asym, 'left', x_axis_y, want_sign=-1, min_run_len=RUN_MIN_LEN, max_band=RUN_SEARCH_MAX_BAND)
        left_branch_empty = densL <= LEFT_DENSITY_MAX and (left_pos['status'] != 'ok' and left_neg['status'] != 'ok')
    result['criteria'] = {'image_readable': True, 'asymptote_present': bool(asymptote_present), 'curve_not_crossing_asymptote': bool(curve_not_crossing), 'right_branch_positive': bool(right_branch_positive), 'left_branch_empty': bool(left_branch_empty)}
    result['passed'] = all(result['criteria'].values())
    result['meta'] = {'x_axis_y': x_axis_y, 'y_axis_x': y_axis_x, 'asymptote_px': px_asym}
    return result

def evaluate(image_path: str) -> Dict[str, object]:
    """Standardized entry point for the evaluation runner."""
    return evaluate_plot(image_path)

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 34.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
