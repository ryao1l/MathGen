#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'plane'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluator_utils import print_report
import solid_common
import solid_common as _MODULE
import plane_common
import plane_common as _PLANE
from solid_common import *
from plane_common import *
PROMPT_ID = 45
PID = 34
TYPE = 'solid'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
PROJ = Path(__file__).resolve().parent.parent.parent.parent.parent
IMG_ROOT = PROJ / 'data' / 'generated_img'
DEBUG_ROOT = Path(__file__).resolve().parent / 'solid_34_debug'
_MIN_LINES = 2
_MIN_ANGLE_GROUPS = 2
_MIN_ELLIPSES = 2
_MIN_AXES = 2
_AXIS_ANGLE_LOW = 55.0
_AXIS_ANGLE_HIGH = 125.0
_MAX_INK = 0.1
_MAX_FG_SAT = 10.0
_MIN_SYM = 0.955

def _load_and_downscale(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return (None, 0, 0)
    _TARGET_LONG = 1024
    long_side = max(img.shape[:2])
    if long_side > _TARGET_LONG:
        s = _TARGET_LONG / long_side
        img = cv2.resize(img, (int(round(img.shape[1] * s)), int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return (img, img.shape[0], img.shape[1])

def _detect_ellipses(bw, gray, img, h, w):
    """Detect significant ellipses in the image using contour fitting.

    Returns list of dicts with center, axes, angle, edge_ratio, area, etc.
    """
    img_area = h * w
    min_dim = max(15, int(0.025 * max(h, w)))
    max_ell_area = img_area * 0.5
    ve = cv2.Canny(gray, 50, 150) if gray is not None else cv2.Canny(bw, 50, 150)
    ve_d = cv2.dilate(ve, np.ones((5, 5), np.uint8))
    srcs = []
    if gray is not None:
        eg1 = cv2.Canny(gray, 50, 150)
        srcs.append(eg1)
        srcs.append(cv2.dilate(eg1, np.ones((3, 3), np.uint8), iterations=1))
        eg2 = cv2.Canny(gray, 30, 80)
        srcs.append(cv2.dilate(eg2, np.ones((5, 5), np.uint8), iterations=1))
    eb = cv2.Canny(bw, 30, 100)
    srcs.append(eb)
    srcs.append(cv2.dilate(eb, np.ones((3, 3), np.uint8), iterations=1))
    srcs.append(bw)
    if gray is not None:
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 10)
        srcs.append(adaptive)
    candidates = []
    seen = set()
    for src in srcs:
        for mode in [cv2.RETR_LIST, cv2.RETR_EXTERNAL]:
            contours, _ = cv2.findContours(src.copy(), mode, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cnt.shape[0] < 10:
                    continue
                peri = cv2.arcLength(cnt, True)
                if peri < 20:
                    continue
                try:
                    ell = cv2.fitEllipse(cnt)
                except cv2.error:
                    continue
                (ecx, ecy), (ea, eb_ax), eang = ell
                if ea < min_dim or eb_ax < min_dim:
                    continue
                major = max(ea, eb_ax)
                minor = min(ea, eb_ax)
                ell_area = math.pi * (ea / 2) * (eb_ax / 2)
                if ell_area > max_ell_area:
                    continue
                if not (0 <= ecx <= w and 0 <= ecy <= h):
                    continue
                aspect = major / minor if minor > 0 else 999
                if aspect > 10:
                    continue
                key = (round(ecx / 10), round(ecy / 10), round(major / 10), round(minor / 10))
                if key in seen:
                    continue
                seen.add(key)
                bnd = cv2.ellipse2Poly((int(round(ecx)), int(round(ecy))), (max(1, int(round(ea / 2))), max(1, int(round(eb_ax / 2)))), int(round(eang)), 0, 360, 5)
                n_in = 0
                n_edge = 0
                for pt in bnd:
                    px, py = (int(pt[0]), int(pt[1]))
                    if 0 <= px < w and 0 <= py < h:
                        n_in += 1
                        if ve_d[py, px] > 0:
                            n_edge += 1
                if n_in < 10:
                    continue
                edge_ratio = n_edge / n_in
                if edge_ratio < 0.2:
                    continue
                size_frac = ell_area / img_area
                if size_frac < 0.005:
                    continue
                candidates.append({'center': (ecx, ecy), 'axes': (ea, eb_ax), 'angle': eang, 'edge_ratio': round(edge_ratio, 3), 'area': round(ell_area, 1), 'size_frac': round(size_frac, 4), 'aspect': round(aspect, 2)})
    candidates.sort(key=lambda c: -c['area'])
    filtered = []
    for c in candidates:
        cx, cy = c['center']
        overlap = False
        for f in filtered:
            fx, fy = f['center']
            d = math.hypot(cx - fx, cy - fy)
            size_sim = min(c['area'], f['area']) / max(c['area'], f['area'])
            if d < 0.05 * max(h, w) and size_sim > 0.5:
                overlap = True
                break
        if not overlap:
            filtered.append(c)
    return filtered

def _detect_axes(lines, h, w, min_len_ratio=0.12):
    """Detect axis-like straight lines: long relative to image size.

    Returns list of axis candidates with endpoints and angle.
    """
    if not lines:
        return []
    min_len = min_len_ratio * max(h, w)
    axes = []
    for x1, y1, x2, y2 in lines:
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len < min_len:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        axes.append({'p1': (x1, y1), 'p2': (x2, y2), 'length': round(seg_len, 1), 'angle': round(angle, 1)})
    axes.sort(key=lambda a: -a['length'])
    return axes

def _find_perpendicular_axes(axes, angle_low=55.0, angle_high=125.0):
    """Find the best pair of axes that are approximately perpendicular.

    Returns (axis1, axis2, angle_between) or (None, None, None).
    """
    if len(axes) < 2:
        return (None, None, None)
    groups = []
    for ax in axes:
        assigned = False
        for g in groups:
            ref_angle = g[0]['angle']
            diff = abs(ax['angle'] - ref_angle)
            diff = min(diff, 180 - diff)
            if diff < 15:
                g.append(ax)
                assigned = True
                break
        if not assigned:
            groups.append([ax])
    if len(groups) < 2:
        return (None, None, None)
    groups.sort(key=lambda g: -sum((a['length'] for a in g)))
    best = (None, None, None)
    best_score = 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a1 = groups[i][0]
            a2 = groups[j][0]
            diff = abs(a1['angle'] - a2['angle'])
            diff = min(diff, 180 - diff)
            if angle_low <= diff <= angle_high:
                score = a1['length'] + a2['length']
                if score > best_score:
                    best = (a1, a2, round(diff, 1))
                    best_score = score
    return best

def _run_analysis(img, h, w):
    gray, bw, ink_ratio = _MODULE._binarize(img)
    min_len = max(20, int(0.06 * h))
    solid_lines = _MODULE.detect_solid_lines(img, min_length=min_len, max_gap=10, threshold=40)
    lines = solid_lines if solid_lines else []
    n_lines = len(lines)
    diverse, n_groups = _MODULE._has_line_diversity(bw, min_angle_groups=_MIN_ANGLE_GROUPS)
    has_dashed = _MODULE._has_dashed_lines(bw)
    has_curves = _MODULE._has_curves(bw)
    ellipses = _detect_ellipses(bw, gray, img, h, w)
    n_ellipses = len(ellipses)
    axes = _detect_axes(lines, h, w, min_len_ratio=0.12)
    n_axes = len(axes)
    ax1, ax2, axes_angle = _find_perpendicular_axes(axes, angle_low=_AXIS_ANGLE_LOW, angle_high=_AXIS_ANGLE_HIGH)
    axes_perp = ax1 is not None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat_ch = hsv[:, :, 1]
    fg_sat = float(sat_ch[bw > 0].mean()) if np.count_nonzero(bw) > 0 else 0.0
    bw_f = bw.astype(np.float32) / 255.0
    sym_h = 1.0 - float(np.mean(np.abs(bw_f - cv2.flip(bw, 1).astype(np.float32) / 255.0)))
    sym_v = 1.0 - float(np.mean(np.abs(bw_f - cv2.flip(bw, 0).astype(np.float32) / 255.0)))
    return {'gray': gray, 'bw': bw, 'ink_ratio': ink_ratio, 'lines': lines, 'n_lines': n_lines, 'diverse': diverse, 'n_groups': n_groups, 'has_dashed': has_dashed, 'has_curves': has_curves, 'ellipses': ellipses, 'n_ellipses': n_ellipses, 'axes': axes, 'n_axes': n_axes, 'ax1': ax1, 'ax2': ax2, 'axes_angle': axes_angle, 'axes_perp': axes_perp, 'fg_sat': fg_sat, 'sym_h': sym_h, 'sym_v': sym_v}

def judge_steinmetz(image_path: str):
    """Evaluate whether image depicts a Steinmetz solid."""
    _CRITERIA_KEYS = ['image_readable', 'foreground_present', 'line_art_style', 'monochrome', 'sufficient_structure', 'curves_present', 'dashed_lines_present', 'ellipses_detected', 'axes_detected', 'axes_perpendicular', 'bilateral_symmetry']
    _META_KEYS = ['case_id', 'ink_ratio', 'fg_sat', 'sym_h', 'sym_v', 'line_count', 'angle_groups', 'n_ellipses', 'ellipse_details', 'n_axes', 'axis_pair_angle']

    def _result(criteria, meta):
        full_c = {k: criteria.get(k) for k in _CRITERIA_KEYS}
        full_m = {k: meta.get(k) for k in _META_KEYS}
        passed = all((v is True for v in full_c.values()))
        return {'passed': passed, 'criteria': full_c, 'meta': full_m}
    img, h, w = _load_and_downscale(image_path)
    if img is None:
        return _result({'image_readable': False}, {'case_id': PID})
    a = _run_analysis(img, h, w)
    criteria = {'image_readable': True}
    meta = {'case_id': PID, 'ink_ratio': round(a['ink_ratio'], 4)}
    criteria['foreground_present'] = a['ink_ratio'] > 0.003
    if not criteria['foreground_present']:
        return _result(criteria, meta)
    criteria['line_art_style'] = a['ink_ratio'] < _MAX_INK
    meta['fg_sat'] = round(a['fg_sat'], 1)
    criteria['monochrome'] = a['fg_sat'] < _MAX_FG_SAT
    meta['line_count'] = a['n_lines']
    criteria['sufficient_structure'] = a['n_lines'] >= _MIN_LINES
    criteria['curves_present'] = a['has_curves']
    criteria['dashed_lines_present'] = a['has_dashed']
    meta['n_ellipses'] = a['n_ellipses']
    meta['ellipse_details'] = [{'center': [round(e['center'][0], 1), round(e['center'][1], 1)], 'axes': [round(e['axes'][0], 1), round(e['axes'][1], 1)], 'edge_ratio': e['edge_ratio'], 'size_frac': e['size_frac']} for e in a['ellipses'][:5]]
    criteria['ellipses_detected'] = a['n_ellipses'] >= _MIN_ELLIPSES
    meta['n_axes'] = a['n_axes']
    meta['angle_groups'] = a['n_groups']
    criteria['axes_detected'] = a['n_axes'] >= _MIN_AXES
    meta['axis_pair_angle'] = a['axes_angle']
    criteria['axes_perpendicular'] = a['axes_perp']
    meta['sym_h'] = round(a['sym_h'], 3)
    meta['sym_v'] = round(a['sym_v'], 3)
    criteria['bilateral_symmetry'] = a['sym_h'] >= _MIN_SYM and a['sym_v'] >= _MIN_SYM
    return _result(criteria, meta)

def evaluate(image_path: str):
    return judge_steinmetz(image_path)

def draw_debug(image_path: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, str(out_dir / '0_original.png'))
    img, h, w = _load_and_downscale(image_path)
    if img is None:
        print(f'  [SKIP] Cannot read {image_path}')
        return None
    a = _run_analysis(img, h, w)
    cv2.imwrite(str(out_dir / '1_binarized.png'), a['bw'])
    ve = cv2.Canny(a['gray'], 50, 150)
    edge_vis = cv2.cvtColor(ve, cv2.COLOR_GRAY2BGR)
    cv2.putText(edge_vis, 'Validation edge map (Canny)', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(str(out_dir / '2_edges.png'), edge_vis)
    lines_vis = img.copy()
    for idx, (x1, y1, x2, y2) in enumerate(a['lines']):
        seg_len = math.hypot(x2 - x1, y2 - y1)
        cv2.line(lines_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        mx, my = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        cv2.putText(lines_vis, f'#{idx} L={int(seg_len)}', (mx, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    ok_lines = a['n_lines'] >= _MIN_LINES
    cv2.putText(lines_vis, f"lines={a['n_lines']} (need>={_MIN_LINES}: {('OK' if ok_lines else 'FAIL')})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if ok_lines else (0, 0, 255), 1)
    cv2.putText(lines_vis, f"angle_groups={a['n_groups']} (need>={_MIN_ANGLE_GROUPS}: {('OK' if a['diverse'] else 'FAIL')})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if a['diverse'] else (0, 0, 255), 1)
    cv2.imwrite(str(out_dir / '3_lines.png'), lines_vis)
    ell_vis = img.copy()
    for idx, e in enumerate(a['ellipses']):
        cx, cy = e['center']
        ea, eb_ax = e['axes']
        ang = e['angle']
        color = (0, 255, 0) if e['edge_ratio'] >= 0.25 else (0, 200, 255)
        cv2.ellipse(ell_vis, (int(round(cx)), int(round(cy))), (max(1, int(round(ea / 2))), max(1, int(round(eb_ax / 2)))), ang, 0, 360, color, 2)
        cv2.circle(ell_vis, (int(cx), int(cy)), 4, color, -1)
        cv2.putText(ell_vis, f"E{idx} er={e['edge_ratio']:.2f} sf={e['size_frac']:.3f}", (int(cx) + 5, int(cy) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    n_ell = a['n_ellipses']
    cv2.putText(ell_vis, f"ellipses={n_ell} (need>={_MIN_ELLIPSES}: {('OK' if n_ell >= _MIN_ELLIPSES else 'FAIL')})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if n_ell >= _MIN_ELLIPSES else (0, 0, 255), 1)
    cv2.imwrite(str(out_dir / '4_ellipses.png'), ell_vis)
    dashed_vis = img.copy()
    short_lines = cv2.HoughLinesP(a['bw'], 1, np.pi / 180.0, 15, minLineLength=3, maxLineGap=2)
    short_count = 0
    if short_lines is not None:
        short_threshold = max(20, int(min(h, w) * 0.03))
        for l in short_lines:
            x1, y1, x2, y2 = l[0]
            seg_len = math.hypot(x2 - x1, y2 - y1)
            if 3 <= seg_len <= short_threshold:
                cv2.line(dashed_vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
                short_count += 1
    cv2.putText(dashed_vis, f"dashed: short_segs={short_count} ({('OK' if a['has_dashed'] else 'FAIL')})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if a['has_dashed'] else (0, 0, 255), 1)
    cv2.imwrite(str(out_dir / '5_dashed.png'), dashed_vis)
    axes_vis = img.copy()
    for idx, ax in enumerate(a['axes'][:10]):
        x1, y1 = ax['p1']
        x2, y2 = ax['p2']
        color = (200, 200, 200)
        cv2.line(axes_vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
        cv2.putText(axes_vis, f"a{idx} L={ax['length']:.0f} @{ax['angle']:.0f}", (int((x1 + x2) / 2), int((y1 + y2) / 2) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    if a['ax1'] is not None and a['ax2'] is not None:
        for ax, clr in [(a['ax1'], (0, 0, 255)), (a['ax2'], (255, 0, 0))]:
            x1, y1 = ax['p1']
            x2, y2 = ax['p2']
            cv2.line(axes_vis, (int(x1), int(y1)), (int(x2), int(y2)), clr, 3)
    cv2.putText(axes_vis, f"axes={a['n_axes']} (need>={_MIN_AXES}: {('OK' if a['n_axes'] >= _MIN_AXES else 'FAIL')})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if a['n_axes'] >= _MIN_AXES else (0, 0, 255), 1)
    angle_str = f"{a['axes_angle']:.1f}" if a['axes_angle'] is not None else 'N/A'
    cv2.putText(axes_vis, f"axis pair angle={angle_str} (need {_AXIS_ANGLE_LOW:.0f}-{_AXIS_ANGLE_HIGH:.0f}: {('OK' if a['axes_perp'] else 'FAIL')})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if a['axes_perp'] else (0, 0, 255), 1)
    cv2.imwrite(str(out_dir / '6_axes.png'), axes_vis)
    style_vis = img.copy()
    ok_ink = a['ink_ratio'] < _MAX_INK
    ok_sat = a['fg_sat'] < _MAX_FG_SAT
    ok_sym_h = a['sym_h'] >= _MIN_SYM
    ok_sym_v = a['sym_v'] >= _MIN_SYM
    cv2.putText(style_vis, f"ink={a['ink_ratio']:.3f} (need<{_MAX_INK}: {('OK' if ok_ink else 'FAIL')})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if ok_ink else (0, 0, 255), 1)
    cv2.putText(style_vis, f"fg_sat={a['fg_sat']:.1f} (need<{_MAX_FG_SAT}: {('OK' if ok_sat else 'FAIL')})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if ok_sat else (0, 0, 255), 1)
    cv2.putText(style_vis, f"sym_h={a['sym_h']:.3f} (need>={_MIN_SYM}: {('OK' if ok_sym_h else 'FAIL')})", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if ok_sym_h else (0, 0, 255), 1)
    cv2.putText(style_vis, f"sym_v={a['sym_v']:.3f} (need>={_MIN_SYM}: {('OK' if ok_sym_v else 'FAIL')})", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if ok_sym_v else (0, 0, 255), 1)
    cv2.line(style_vis, (w // 2, 0), (w // 2, h), (255, 0, 255), 1)
    cv2.line(style_vis, (0, h // 2), (w, h // 2), (255, 0, 255), 1)
    cv2.imwrite(str(out_dir / '7_style_symmetry.png'), style_vis)
    result = judge_steinmetz(image_path)
    summary_vis = img.copy()
    y_pos = 25
    for k, v in result['criteria'].items():
        color = (0, 255, 0) if v else (0, 0, 255)
        cv2.putText(summary_vis, f'{k}: {v}', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_pos += 22
    verdict = 'PASSED' if result['passed'] else 'FAILED'
    color = (0, 255, 0) if result['passed'] else (0, 0, 255)
    cv2.putText(summary_vis, verdict, (10, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imwrite(str(out_dir / '8_summary.png'), summary_vis)
    with open(str(out_dir / 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return result

def run_debug_all():
    models = sorted((d.name for d in IMG_ROOT.iterdir() if d.is_dir()))
    results = {}
    for model in models:
        img_path = IMG_ROOT / model / 'solid' / f'{PID}.png'
        if not img_path.exists():
            print(f'[SKIP] {model}: no image at {img_path}')
            continue
        out_dir = DEBUG_ROOT / model
        print(f'[{model}] {img_path} -> {out_dir}')
        r = draw_debug(str(img_path), out_dir)
        if r:
            results[model] = r['passed']
    print('\n=== Summary ===')
    for model, passed in sorted(results.items()):
        print(f"  {model:25s} {('PASS' if passed else 'FAIL')}")
    print(f'\n  {sum(results.values())}/{len(results)} passed')

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen solid case 45.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
