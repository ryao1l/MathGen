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
PROMPT_ID = 37
DEBUG = False
DEBUG_DIR = 'function_fourier_debug'
BLACK_HSV_LO = (0, 0, 0)
BLACK_HSV_HI = (179, 160, 140)
BLACK_OPEN_K = (3, 3)
KV_LEN = 55
KH_LEN = 55
MIN_AXIS_COL_FRAC = 0.02
MIN_AXIS_ROW_FRAC = 0.02
CANNY1, CANNY2 = (50, 150)
HOUGH_THRESH = 110
MAX_GAP = 30
AXIS_THICKNESS = 12
XAXIS_GAP = 8
YAXIS_GAP = 10
BLUE_H_LO = 85
BLUE_H_HI = 155
BLUE_S_MIN = 20
BLUE_V_MIN = 30
MIN_OCR_CONF = 10.0
OCR_SCALE = 3
OCR_ADAPT_BLOCK = 31
OCR_ADAPT_C = 9
RANSAC_ITERS = 250
INLIER_TOL_X = 0.4
MIN_INLIERS_X = 3
ZERO_TOL_PX = 20.0
KEY_ERR_TOL = 0.3
KEY_PASS_AT_LEAST = 6
PEAK_EXPECT = 13.0 / 15.0
Y_HALF_WINDOW = 6

def ensure_debug_dir():
    if DEBUG:
        os.makedirs(DEBUG_DIR, exist_ok=True)

def dsave(name: str, img: np.ndarray):
    if not DEBUG:
        return
    if img is None or img.size == 0:
        print(f'Debug skip (empty): {name}')
        return
    ensure_debug_dir()
    path = os.path.join(DEBUG_DIR, name)
    cv2.imwrite(path, img)
    print(f'Debug saved: {path}')

def maybe_set_tesseract_cmd():
    pass

def f(x: float) -> float:
    return float(np.sin(x) + 1.0 / 3.0 * np.sin(3.0 * x) + 1.0 / 5.0 * np.sin(5.0 * x))

def extract_blue_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = (BLUE_H_LO, BLUE_S_MIN, BLUE_V_MIN)
    upper = (BLUE_H_HI, 255, 255)
    mask = cv2.inRange(hsv, lower, upper)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)
    return mask

def blackish_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blk = cv2.inRange(hsv, BLACK_HSV_LO, BLACK_HSV_HI)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, BLACK_OPEN_K)
    blk = cv2.morphologyEx(blk, cv2.MORPH_OPEN, k, iterations=1)
    return blk

def erase_mask_to_white(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img_bgr.copy()
    out[mask > 0] = (255, 255, 255)
    return out

def detect_axes_black_projection_no_center(img_bgr: np.ndarray) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    h, w = img_bgr.shape[:2]
    blk = blackish_mask(img_bgr)
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, KV_LEN))
    v = cv2.morphologyEx(blk, cv2.MORPH_OPEN, kv, iterations=1)
    col = (v > 0).sum(axis=0).astype(np.float32) / float(h)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (KH_LEN, 1))
    hh = cv2.morphologyEx(blk, cv2.MORPH_OPEN, kh, iterations=1)
    row = (hh > 0).sum(axis=1).astype(np.float32) / float(w)
    col_max = float(np.max(col)) if col.size else 0.0
    row_max = float(np.max(row)) if row.size else 0.0
    ok = col_max >= MIN_AXIS_COL_FRAC and row_max >= MIN_AXIS_ROW_FRAC
    dbg = {'black': blk, 'v': v, 'hh': hh, 'col_max': col_max, 'row_max': row_max}
    if not ok:
        return (None, None, dbg)
    y_axis_x = float(np.argmax(col))
    x_axis_y = float(np.argmax(row))
    return (x_axis_y, y_axis_x, dbg)

def detect_axes_hough(img_bgr: np.ndarray) -> Tuple[float, float, Dict[str, Any]]:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY1, CANNY2)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=HOUGH_THRESH, minLineLength=int(0.45 * w), maxLineGap=MAX_GAP)
    x_axis_y = h / 2.0
    y_axis_x = w / 2.0
    best_h = (None, -1.0)
    best_v = (None, -1.0)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = (x2 - x1, y2 - y1)
            length = float(np.hypot(dx, dy))
            if length < 0.4 * w:
                continue
            if abs(dy) <= 4:
                ymid = (y1 + y2) / 2.0
                if length > best_h[1]:
                    best_h = (ymid, length)
            if abs(dx) <= 4:
                xmid = (x1 + x2) / 2.0
                if length > best_v[1]:
                    best_v = (xmid, length)
    if best_h[0] is not None:
        x_axis_y = float(best_h[0])
    if best_v[0] is not None:
        y_axis_x = float(best_v[0])
    return (x_axis_y, y_axis_x, {'edges': edges})

def detect_axes(img_bgr: np.ndarray) -> Tuple[float, float]:
    xpy, ypx, dbg = detect_axes_black_projection_no_center(img_bgr)
    if DEBUG:
        dsave('00_input.png', img_bgr)
        dsave('01_black_mask.png', dbg.get('black', np.zeros((1, 1), np.uint8)))
        dsave('02_v_strokes.png', dbg.get('v', np.zeros((1, 1), np.uint8)))
        dsave('03_h_strokes.png', dbg.get('hh', np.zeros((1, 1), np.uint8)))
        print(f"Projection strength: col_max={dbg.get('col_max', 0):.4f}, row_max={dbg.get('row_max', 0):.4f}")
    if xpy is not None and ypx is not None:
        return (float(xpy), float(ypx))
    xpy2, ypx2, hdbg = detect_axes_hough(img_bgr)
    if DEBUG:
        dsave('04_edges_hough.png', hdbg['edges'])
    return (float(xpy2), float(ypx2))

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

def curve_y_pixel_at_x(mask: np.ndarray, px_x: int, half_window: int=Y_HALF_WINDOW) -> Optional[float]:
    h, w = mask.shape[:2]
    px_x = int(np.clip(px_x, 0, w - 1))
    x0 = max(0, px_x - half_window)
    x1 = min(w, px_x + half_window + 1)
    strip = mask[:, x0:x1]
    ys = np.where(strip > 0)[0]
    if len(ys) == 0:
        return None
    return float(np.median(ys))

def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, OCR_ADAPT_BLOCK, OCR_ADAPT_C)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k, iterations=1)
    return bin_img

def ocr_numbers_robust(img_bgr: np.ndarray, roi: Optional[Tuple[int, int, int, int]]=None, debug_prefix: str='') -> List[Dict[str, Any]]:
    H, W = img_bgr.shape[:2]
    rx0, ry0, rx1, ry1 = (0, 0, W, H)
    if roi is not None:
        rx0, ry0, rx1, ry1 = roi
        rx0, ry0 = (max(0, rx0), max(0, ry0))
        rx1, ry1 = (min(W, rx1), min(H, ry1))
        if rx1 <= rx0 or ry1 <= ry0:
            return []
        img_bgr = img_bgr[ry0:ry1, rx0:rx1]
    up = cv2.resize(img_bgr, None, fx=OCR_SCALE, fy=OCR_SCALE, interpolation=cv2.INTER_CUBIC)
    ocr_bin = preprocess_for_ocr(up)
    if DEBUG and debug_prefix:
        dsave(f'{debug_prefix}_ocr_up.png', up)
        dsave(f'{debug_prefix}_ocr_bin.png', ocr_bin)
    configs = ['--oem 3 --psm 6  -c tessedit_char_whitelist=-−—–0123456789', '--oem 3 --psm 11 -c tessedit_char_whitelist=-−—–0123456789']
    raw = []
    for cfg in configs:
        data = pytesseract.image_to_data(ocr_bin, config=cfg, output_type=pytesseract.Output.DICT)
        n = len(data['text'])
        for i in range(n):
            txt = (data['text'][i] or '').strip()
            conf = float(data['conf'][i]) if data['conf'][i] != '-1' else -1.0
            if txt == '':
                continue
            txt_norm = txt.replace('−', '-').replace('—', '-').replace('–', '-').strip('., ')
            lx, ty, bw, bh = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            cx, cy = (lx + bw / 2.0, ty + bh / 2.0)
            center = (cx / OCR_SCALE + rx0, cy / OCR_SCALE + ry0)
            raw.append({'text': txt_norm, 'conf': conf, 'center': center})
    toks = [t for t in raw if t['conf'] >= MIN_OCR_CONF and re.fullmatch('-?\\d+', t['text'] or '')]
    out = []
    for t in sorted(toks, key=lambda z: (-z['conf'], z['center'][0])):
        keep = True
        for e in out:
            if int(t['text']) == int(e['text']) and abs(t['center'][0] - e['center'][0]) < 10 and (abs(t['center'][1] - e['center'][1]) < 10):
                keep = False
                break
        if keep:
            out.append(t)
    return [{'value': int(t['text']), 'conf': float(t['conf']), 'center': t['center'], 'raw': t['text']} for t in out]

def compute_x_roi(h: int, w: int, x_axis_y: float) -> Tuple[int, int, int, int]:
    band = int(0.18 * h)
    y0 = max(0, int(x_axis_y - band))
    y1 = min(h, int(x_axis_y + band))
    if y1 <= y0:
        y0, y1 = (0, h)
    return (0, y0, w, y1)

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

def _mode_bin_1d(vals: List[float], bin_size: float) -> Optional[float]:
    if not vals:
        return None
    bins: Dict[int, List[float]] = {}
    for v in vals:
        k = int(np.floor(v / bin_size))
        bins.setdefault(k, []).append(v)
    best_lst = None
    for lst in bins.values():
        if best_lst is None or len(lst) > len(best_lst):
            best_lst = lst
        elif len(lst) == len(best_lst) and max(lst) - min(lst) < max(best_lst) - min(best_lst):
            best_lst = lst
    return float(np.mean(best_lst)) if best_lst else None

def filter_x_candidates_by_row(x_dets: List[Dict[str, Any]], x_axis_y: float, y_axis_x: float, h: int, w: int, *, y_band_frac: float=0.12, x_margin_frac: float=0.05, row_bin_px: float=8.0, row_tight_px: float=10.0) -> List[Dict[str, Any]]:
    y_tol = float(y_band_frac) * float(h)
    x_margin = float(x_margin_frac) * float(w)
    cand = []
    for d in x_dets:
        cx, cy = d['center']
        if abs(cy - x_axis_y) > y_tol:
            continue
        if abs(cx - y_axis_x) < x_margin:
            continue
        cand.append(d)
    if len(cand) < 2:
        return cand
    y_mode = _mode_bin_1d([d['center'][1] for d in cand], bin_size=row_bin_px)
    if y_mode is None:
        return cand
    row = [d for d in cand if abs(d['center'][1] - y_mode) <= row_tight_px]
    if not row:
        return cand
    per: Dict[int, List[Dict[str, Any]]] = {}
    for d in row:
        per.setdefault(int(d['value']), []).append(d)
    out = []
    for v, lst in per.items():
        lst2 = sorted(lst, key=lambda z: (-z['conf'], abs(z['center'][1] - y_mode)))
        out.append(lst2[0])
    return out

def build_x_mapping_from_ticks(x_dets: List[Dict[str, Any]], x_axis_y: float, y_axis_x: float, h: int, w: int) -> Tuple[Optional[float], Optional[float], List[Dict[str, Any]]]:
    x_use = filter_x_candidates_by_row(x_dets, x_axis_y=x_axis_y, y_axis_x=y_axis_x, h=h, w=w)
    keep = []
    for d in x_use:
        v = int(d['value'])
        if 0 <= v <= 20:
            keep.append(d)
    prefer = {2, 4, 6, 8, 10, 12}
    keep2 = [d for d in keep if int(d['value']) in prefer] or keep
    pts = [(float(d['center'][0]), float(d['value'])) for d in keep2]
    ax, bx = ransac_fit_v_from_p(pts, RANSAC_ITERS, INLIER_TOL_X, MIN_INLIERS_X)
    return (ax, bx, keep2)

def evaluate_plot(image_path: str):
    maybe_set_tesseract_cmd()
    img = cv2.imread(image_path)
    if img is None:
        print('Image not found.')
        return
    h, w = img.shape[:2]
    print(f'\nFourier 3-term evaluator (OCR-x): y=sin(x)+1/3 sin(3x)+1/5 sin(5x), x∈[0,4π] | {os.path.basename(image_path)}')
    print('-' * 100)
    x_axis_y, y_axis_x = detect_axes(img)
    print(f'Axes detected: x-axis y≈{x_axis_y:.1f}, y-axis x≈{y_axis_x:.1f}')
    overlay_axes = img.copy()
    cv2.line(overlay_axes, (0, int(round(x_axis_y))), (w - 1, int(round(x_axis_y))), (0, 255, 255), 2)
    cv2.line(overlay_axes, (int(round(y_axis_x)), 0), (int(round(y_axis_x)), h - 1), (0, 255, 255), 2)
    dsave('05_axes_overlay.png', overlay_axes)
    blue = extract_blue_mask(img)
    dsave('10_blue_mask.png', blue)
    curve_mask = remove_axes(blue, x_axis_y, y_axis_x)
    dsave('12_curve_no_axes.png', curve_mask)
    print(f'curve_pixels(after remove_axes) = {int(np.sum(curve_mask > 0))}')
    print('\n[OCR-X] Fit x = ax*px + bx from x-axis ticks (2,4,6,8,10)')
    img_ocr = erase_mask_to_white(img, blue)
    dsave('20_ocr_cleaned.png', img_ocr)
    x_roi = compute_x_roi(h, w, x_axis_y)
    x0, y0, x1, y1 = x_roi
    dsave('21_x_roi.png', img_ocr[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else np.zeros((1, 1), np.uint8))
    x_dets = ocr_numbers_robust(img_ocr, roi=x_roi, debug_prefix='22_x')
    print(f'\n  [22_x] OCR MERGED INTEGERS (total={len(x_dets)})')
    for d in sorted(x_dets, key=lambda z: z['center'][0]):
        cx, cy = d['center']
        print(f"    value={d['value']} conf={d['conf']:.1f} center=({cx:.1f},{cy:.1f}) raw='{d.get('raw', '')}'")
    print(f"  raw x_dets values={sorted({d['value'] for d in x_dets})}")
    ax, bx, x_use = build_x_mapping_from_ticks(x_dets, x_axis_y, y_axis_x, h, w)
    print('  [X-OCR] points used (filtered row): ' + ', '.join([f"{d['value']}@({d['center'][0]:.1f},{d['center'][1]:.1f}) c{d['conf']:.0f}" for d in sorted(x_use, key=lambda z: z['center'][0])]) if x_use else '  (none)')
    if ax is None:
        print('x-axis OCR mapping failed -> cannot evaluate.')
        print('\n' + '-' * 100)
        print('Final result: FAIL')
        return
    print(f'x mapping: x = {ax:.6f} * px + {bx:.3f}')

    def px_from_x(xv: float) -> int:
        return int(round((float(xv) - bx) / ax))
    px_p = px_from_x(np.pi / 2.0)
    px_n = px_from_x(3.0 * np.pi / 2.0)
    y_p = curve_y_pixel_at_x(curve_mask, px_p)
    y_n = curve_y_pixel_at_x(curve_mask, px_n)
    if y_p is None or y_n is None or abs(y_n - y_p) < 5:
        print('y-scale estimate failed (cannot find peaks).')
        print('\n' + '-' * 100)
        print('Final result: FAIL')
        return
    y_scale = float((y_n - y_p) / (2.0 * PEAK_EXPECT))
    print(f'y-scale(px/unit y) ≈ {y_scale:.2f} (using ±13/15 peaks)')

    def y_meas_at_x(xv: float) -> Optional[float]:
        px = px_from_x(xv)
        ypix = curve_y_pixel_at_x(curve_mask, px)
        if ypix is None:
            return None
        return float((x_axis_y - ypix) / y_scale)
    print('\n[A] Zero checks: y≈0 at x=kπ (pixels near x-axis)')
    okA_all = True
    for k in range(0, 5):
        if k in (0, 4):
            print(f'  x={k}π: SKIP (endpoint)')
            continue
        xv = float(k * np.pi)
        px = px_from_x(xv)
        ypix = curve_y_pixel_at_x(curve_mask, px)
        if ypix is None:
            print(f'  x={k}π: px={px} ypix=None  FAIL')
            okA_all = False
            continue
        dist = abs(ypix - x_axis_y)
        ok = dist <= ZERO_TOL_PX
        okA_all = okA_all and ok
        print(f"  x={k}π: px={px} ypix≈{ypix:.1f} | |ypix-xaxis|={dist:.1f}  {('OK' if ok else 'FAIL')}")
    print('\n[B] Key-point value check: convert y-pixels to values and compare f(x)')
    xs = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi, 5 * np.pi / 2, 3 * np.pi, 7 * np.pi / 2, 4 * np.pi]
    pass_cnt = 0
    key_overlay = img.copy()
    for xv in xs:
        y_exp = f(xv)
        ym = y_meas_at_x(xv)
        if ym is None:
            print(f'  x={xv:.3f}: y_exp={y_exp:+.3f}, y_meas=None  FAIL')
            continue
        err = abs(ym - y_exp)
        ok = err <= KEY_ERR_TOL
        pass_cnt += int(ok)
        print(f"  x={xv:.3f}: y_exp={y_exp:+.3f}, y_meas={ym:+.3f}, err={err:.3f}  {('OK' if ok else 'FAIL')}")
        px = px_from_x(xv)
        ypix = curve_y_pixel_at_x(curve_mask, px)
        if ypix is not None:
            cv2.circle(key_overlay, (int(px), int(round(ypix))), 5, (0, 0, 255), -1)
    dsave('40_keypoints_overlay.png', key_overlay)
    print(f"  -> Result: {pass_cnt}/{len(xs)}  {('OK' if pass_cnt >= KEY_PASS_AT_LEAST else 'FAIL')}  (threshold: >= {KEY_PASS_AT_LEAST})")
    okB = pass_cnt >= KEY_PASS_AT_LEAST
    print('\n[C] Peak sign check: y(π/2)>0 and y(3π/2)<0')
    y1 = y_meas_at_x(np.pi / 2)
    y2 = y_meas_at_x(3 * np.pi / 2)
    okC = y1 is not None and y2 is not None and (y1 > 0) and (y2 < 0)
    if y1 is None or y2 is None:
        print('  insufficient  FAIL')
    else:
        print(f"  y(π/2)≈{y1:+.3f}, y(3π/2)≈{y2:+.3f}  {('OK' if okC else 'FAIL')}")
    core_ok = bool(okA_all and okB and okC)
    print('\n' + '-' * 100)
    print('Final result: PASS' if core_ok else 'Final result: FAIL')

def evaluate(image_path: str):
    if DEBUG:
        result = evaluate_plot(image_path)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate_plot(image_path)
    if isinstance(result, dict):
        result['id'] = 37
        return result
    return {'id': 37, 'passed': False, 'criteria': {'fourier_keypoint_check': False}, 'meta': {'reason': 'legacy OCR evaluator did not produce a structured pass result'}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 37.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
