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
PROMPT_ID = 38
sys.path.insert(0, os.path.dirname(__file__))

def _red_peaks(red_mask: np.ndarray, n_peaks: int=2) -> list[int]:
    h, w = red_mask.shape[:2]
    score = (red_mask > 0).sum(axis=0).astype(np.float32) / max(float(h), 1.0)
    kernel = np.ones(31, dtype=np.float32) / 31.0
    smooth = np.convolve(score, kernel, mode='same')
    peaks = []
    work = smooth.copy()
    for _ in range(n_peaks):
        p = int(np.argmax(work))
        if work[p] < 0.025:
            break
        peaks.append(p)
        work[max(0, p - 40):min(w, p + 41)] = 0
    return sorted(peaks)

def _crossing_density(mask: np.ndarray, x: int) -> float:
    h, w = mask.shape[:2]
    strip = mask[:, max(0, x - 5):min(w, x + 6)]
    if strip.size == 0:
        return 0.0
    return float(np.sum(strip > 0)) / float(strip.size)

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
    peaks = _red_peaks(red_mask, 2)
    arrow_info = check_single_headed_arrows(img_gray, x_ay, y_ax)
    vlt_ok, vlt_detail = check_vertical_line_test(blue_mask)
    two_sided_asymptotes = False
    symmetric_pm1 = False
    no_crossing = False
    three_branches = False
    if len(peaks) >= 2:
        left, right = (peaks[0], peaks[1])
        axis_for_asym = float(y_ax)
        if axis_for_asym < 0.08 * w or axis_for_asym > 0.92 * w:
            axis_for_asym = 0.5 * (left + right)
        left_offset = (axis_for_asym - left) / max(float(w), 1.0)
        right_offset = (right - axis_for_asym) / max(float(w), 1.0)
        two_sided_asymptotes = left < axis_for_asym < right
        symmetric_pm1 = 0.045 <= left_offset <= 0.11 and 0.045 <= right_offset <= 0.11 and (abs(left_offset - right_offset) < 0.035)
        no_crossing = _crossing_density(blue_mask, left) < 0.08 and _crossing_density(blue_mask, right) < 0.08
        if len(blue_pixels) > 100:
            bands = [blue_pixels[blue_pixels[:, 1] < left - 10], blue_pixels[(blue_pixels[:, 1] > left + 10) & (blue_pixels[:, 1] < right - 10)], blue_pixels[blue_pixels[:, 1] > right + 10]]
            three_branches = all((len(b) > 40 for b in bands))
    criteria = {'single_headed_arrows': bool(arrow_info['is_single_headed']), 'blue_curve_exists': len(blue_pixels) > 100, 'red_asymptotes_two': len(peaks) >= 2 and red_pixels > 200, 'asymptotes_on_both_sides': two_sided_asymptotes, 'asymptotes_at_pm1': symmetric_pm1, 'curve_not_crossing_asymptotes': no_crossing, 'three_branches': three_branches, 'vertical_line_test': bool(vlt_ok)}
    criteria = {k: bool(v) for k, v in criteria.items()}
    return {'id': 38, 'passed': bool(all(criteria.values())), 'criteria': criteria, 'meta': {'x_axis_y': float(x_ay), 'y_axis_x': float(y_ax), 'red_pixels': red_pixels, 'asymptote_positions': [int(p) for p in peaks], 'arrow_info': arrow_info, 'vertical_line_test': vlt_detail}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 38.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
