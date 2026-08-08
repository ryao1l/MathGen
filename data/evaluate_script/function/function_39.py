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
PROMPT_ID = 39
sys.path.insert(0, os.path.dirname(__file__))

def _vertical_red_peaks(red_mask: np.ndarray, n_peaks: int=3) -> list[int]:
    h, w = red_mask.shape[:2]
    col_score = (red_mask > 0).sum(axis=0).astype(np.float32) / max(float(h), 1.0)
    kernel = np.ones(31, dtype=np.float32) / 31.0
    smooth = np.convolve(col_score, kernel, mode='same')
    peaks = []
    work = smooth.copy()
    for _ in range(n_peaks):
        p = int(np.argmax(work))
        if work[p] < 0.03:
            break
        peaks.append(p)
        work[max(0, p - 45):min(w, p + 46)] = 0
    return sorted(peaks)

def evaluate(image_path: str):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {'passed': False, 'reason': 'Image not found'}
    h, w = img_bgr.shape[:2]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    x_ay, y_ax = detect_axes(img_gray)
    blue_mask = extract_blue_mask(img_hsv)
    red_mask = extract_red_mask(img_hsv)
    blue_pixels = np.column_stack(np.where(blue_mask > 0))
    red_pixels = int(np.sum(red_mask > 0))
    arrow_info = check_single_headed_arrows(img_gray, x_ay, y_ax)
    vlt_ok, vlt_detail = check_vertical_line_test(blue_mask)
    peaks = _vertical_red_peaks(red_mask, 3)
    asym_spacing_ok = False
    asym_position_ok = False
    if len(peaks) >= 3:
        gaps = np.diff(peaks[:3])
        asym_spacing_ok = bool(np.min(gaps) > 0.07 * w and np.max(gaps) / max(np.min(gaps), 1) < 2.0)
        offsets = [(p - y_ax) / max(float(w), 1.0) for p in peaks[:3]]
        asym_position_ok = offsets[0] > 0.02 and offsets[-1] < 0.45
    branch_gaps_ok = False
    if len(peaks) >= 3 and len(blue_pixels) > 100:
        cuts = [0] + peaks[:3] + [w - 1]
        occupied = []
        for left, right in zip(cuts[:-1], cuts[1:]):
            band = blue_pixels[(blue_pixels[:, 1] > left + 8) & (blue_pixels[:, 1] < right - 8)]
            occupied.append(len(band) > 25)
        branch_gaps_ok = sum(occupied) >= 3
    dark = img_gray < 80
    yy, xx = np.where(dark)
    off_axes = (np.abs(yy - x_ay) > 25) & (np.abs(xx - y_ax) > 25) & (yy > 0.05 * h) & (yy < 0.95 * h) & (xx > 0.05 * w) & (xx < 0.95 * w)
    extra_text_ratio = float(np.sum(off_axes)) / max(float(h * w), 1.0)
    criteria = {'single_headed_arrows': bool(arrow_info['is_single_headed']), 'blue_curve_exists': len(blue_pixels) > 100, 'red_asymptotes_three': len(peaks) >= 3 and red_pixels > 300, 'asymptotes_spaced': asym_spacing_ok, 'asymptotes_positioned': asym_position_ok, 'separated_branches': branch_gaps_ok, 'no_extra_formula_text': extra_text_ratio < 0.004, 'vertical_line_test': bool(vlt_ok)}
    criteria = {k: bool(v) for k, v in criteria.items()}
    return {'id': 39, 'passed': all(criteria.values()), 'criteria': criteria, 'meta': {'x_axis_y': float(x_ay), 'y_axis_x': float(y_ax), 'red_pixels': red_pixels, 'asymptote_positions': [int(p) for p in peaks], 'extra_text_ratio': extra_text_ratio, 'arrow_info': arrow_info, 'vertical_line_test': vlt_detail}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 39.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
