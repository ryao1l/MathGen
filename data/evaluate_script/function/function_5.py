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
PROMPT_ID = 5
sys.path.insert(0, os.path.dirname(__file__))

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
    red_pixel_count = int(np.sum(red_mask > 0))
    criteria = {}
    arrow_info = check_single_headed_arrows(img_gray, x_ay, y_ax)
    criteria['single_headed_arrows'] = arrow_info['is_single_headed']
    criteria['line_is_blue'] = len(blue_pixels) >= 10000
    criteria['no_red_function_line'] = red_pixel_count < 100
    if len(blue_pixels) > 50:
        rel_x = blue_pixels[:, 1] - y_ax
        x_std = float(np.std(blue_pixels[:, 1]))
        criteria['is_vertical'] = x_std < 15
        mean_x = float(np.mean(rel_x))
        criteria['right_of_y_axis'] = mean_x > 10
        right_range = w - y_ax
        if right_range > 0:
            position_ratio = mean_x / right_range
            criteria['correct_x_position'] = 0.2 < position_ratio < 0.95
        else:
            criteria['correct_x_position'] = False
        y_min = np.min(blue_pixels[:, 0])
        y_max = np.max(blue_pixels[:, 0])
        criteria['vertical_coverage'] = (y_max - y_min) / h > 0.4
    else:
        criteria['is_vertical'] = False
        criteria['right_of_y_axis'] = False
        criteria['correct_x_position'] = False
        criteria['vertical_coverage'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 8, 'passed': passed, 'criteria': criteria, 'meta': {'x_axis_y': x_ay, 'y_axis_x': y_ax, 'blue_pixels': len(blue_pixels), 'red_pixels': red_pixel_count, 'arrow_info': arrow_info}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 5.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
