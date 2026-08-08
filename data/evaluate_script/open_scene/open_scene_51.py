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
PROMPT_ID = 51

def extract_dark_mask(img_bgr):
    """Dark staircase on white wall: grayscale < 110."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return mask

def top_profile(mask):
    """For each column, return the y of the topmost dark pixel (or None)."""
    h, w = mask.shape
    profile = np.full(w, -1, dtype=int)
    for col in range(w):
        rows = np.where(mask[:, col] > 0)[0]
        if len(rows) > 0:
            profile[col] = rows[0]
    return profile

def evaluate(image_path: str):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {'passed': False, 'reason': 'Image not found'}
    h, w = img_bgr.shape[:2]
    mask = extract_dark_mask(img_bgr)
    dark_count = int(np.sum(mask > 0))
    criteria = {}
    n_levels = n_plateaus = 0
    height_cov = width_cov = None
    criteria['silhouette_present'] = dark_count > 0.02 * h * w
    profile = top_profile(mask)
    valid_cols = np.where(profile >= 0)[0]
    if len(valid_cols) > 50:
        prof_valid = profile[valid_cols]
        kernel = np.ones(5) / 5
        prof_smooth = np.convolve(prof_valid.astype(float), kernel, mode='same')
        d = np.abs(np.diff(prof_smooth))
        jump_thresh = max(4.0, 0.015 * h)
        is_jump = d > jump_thresh
        plateau_indices = []
        i = 0
        N = len(prof_smooth)
        while i < N:
            if not (i < N - 1 and is_jump[i]):
                j = i
                while j < N - 1 and (not is_jump[j]):
                    j += 1
                if j - i >= 4:
                    med_y = float(np.median(prof_smooth[i:j + 1]))
                    plateau_indices.append((i, j, med_y))
                i = j + 1
            else:
                i += 1
        n_plateaus = len(plateau_indices)
        if n_plateaus > 0:
            heights_sorted = sorted([p[2] for p in plateau_indices])
            gap_thresh = max(4.0, 0.02 * h)
            n_levels = 1
            for a, b in zip(heights_sorted[:-1], heights_sorted[1:]):
                if abs(b - a) > gap_thresh:
                    n_levels += 1
        criteria['distinct_step_levels'] = n_levels >= 4
        flat_ratio = float(np.mean(d <= jump_thresh))
        criteria['plateaus_and_jumps'] = flat_ratio > 0.55 and np.sum(is_jump) >= 3
        q = max(5, N // 4)
        if N >= 4 * q:
            left_mean = float(np.mean(prof_smooth[:q]))
            right_mean = float(np.mean(prof_smooth[-q:]))
            criteria['ascending_left_to_right'] = left_mean - right_mean > 0.1 * h
        else:
            criteria['ascending_left_to_right'] = False
        if n_plateaus >= 3:
            plateau_ys = np.array([p[2] for p in plateau_indices])
            plateau_xs = np.array([(p[0] + p[1]) / 2 for p in plateau_indices])
            order = np.argsort(plateau_xs)
            plateau_ys = plateau_ys[order]
            jumps = np.abs(np.diff(plateau_ys))
            if len(jumps) >= 2:
                mean_j = float(np.mean(jumps))
                std_j = float(np.std(jumps))
                height_cov = std_j / max(1e-06, mean_j)
                criteria['equal_step_heights'] = height_cov < 0.4
            else:
                criteria['equal_step_heights'] = False
            widths = np.array([p[1] - p[0] for p in plateau_indices])[order]
            if len(widths) >= 3:
                mean_w = float(np.mean(widths))
                std_w = float(np.std(widths))
                width_cov = std_w / max(1e-06, mean_w)
                criteria['equal_tread_widths'] = width_cov < 0.45
            else:
                criteria['equal_tread_widths'] = False
        else:
            criteria['equal_step_heights'] = False
            criteria['equal_tread_widths'] = False
    else:
        criteria['distinct_step_levels'] = False
        criteria['plateaus_and_jumps'] = False
        criteria['ascending_left_to_right'] = False
        criteria['equal_step_heights'] = False
        criteria['equal_tread_widths'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 63, 'passed': passed, 'criteria': criteria, 'meta': {'dark_pixel_count': dark_count, 'num_plateaus': int(n_plateaus), 'num_distinct_levels': int(n_levels), 'height_coefficient_of_variation': height_cov, 'width_coefficient_of_variation': width_cov}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen open_scene case 51.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
