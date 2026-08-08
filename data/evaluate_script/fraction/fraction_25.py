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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluator_utils import print_report
PROMPT_ID = 25
TARGET_RADIUS_RATIO = 0.6
RATIO_TOL = 0.1
CIRCULARITY_TOL = 0.2

def load_image(path: str) -> Optional[np.ndarray]:
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = img[:, :, :3].astype(np.float32)
        white = np.full_like(bgr, 255.0)
        composited = (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)
        return composited
    return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def get_donut_radii(img: np.ndarray) -> Optional[Tuple[float, float]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(sorted_contours) < 2:
        return None
    c_outer = sorted_contours[0]
    c_inner = None
    (x1, y1), r1 = cv2.minEnclosingCircle(c_outer)
    outer_idx = -1
    for i, c in enumerate(contours):
        if c is c_outer:
            outer_idx = i
            break
    c_inner = sorted_contours[1]
    (x2, y2), r2 = cv2.minEnclosingCircle(c_inner)
    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    if dist > r1 * 0.2:
        return None
    if r2 > r1:
        r1, r2 = (r2, r1)
    return (r1, r2)

def check_background_is_white(image_path: str, threshold: int=240) -> Tuple[bool, float]:
    """Check if the image background (border regions) is predominantly white and opaque."""
    if not os.path.isfile(image_path):
        return (False, 0.0)
    raw_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw_img is None:
        return (False, 0.0)
    H, W = raw_img.shape[:2]
    border = max(1, int(min(H, W) * 0.08))
    top = raw_img[:border, :]
    bottom = raw_img[H - border:, :]
    left = raw_img[:, :border]
    right = raw_img[:, W - border:]
    if raw_img.ndim == 3 and raw_img.shape[2] == 4:
        alpha = raw_img[:, :, 3]
        top_a = alpha[:border, :]
        bottom_a = alpha[H - border:, :]
        left_a = alpha[:, :border]
        right_a = alpha[:, W - border:]
        alpha_border = np.concatenate([top_a.ravel(), bottom_a.ravel(), left_a.ravel(), right_a.ravel()])
        mean_alpha = float(np.mean(alpha_border))
        if mean_alpha < 200:
            return (False, mean_alpha)
        img_check = raw_img[:, :, :3]
    else:
        img_check = raw_img
    if img_check.ndim == 3:
        gray = cv2.cvtColor(img_check, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_check
    top_g = gray[:border, :]
    bottom_g = gray[H - border:, :]
    left_g = gray[:, :border]
    right_g = gray[:, W - border:]
    intensity_border = np.concatenate([top_g.ravel(), bottom_g.ravel(), left_g.ravel(), right_g.ravel()])
    median_intensity = float(np.median(intensity_border))
    return (median_intensity >= threshold, median_intensity)

def evaluate(image_path: str) -> Dict[str, object]:
    result = {'criteria': {}, 'passed': False}
    if not os.path.isfile(image_path):
        result['criteria'] = {'image_exists': False, 'passed': False}
        return result
    img = load_image(image_path)
    if img is None:
        result['criteria'] = {'image_exists': True, 'image_readable': False, 'passed': False}
        return result
    c_bg_white, bg_mean = check_background_is_white(image_path)
    radii = get_donut_radii(img)
    if radii is None:
        result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'donut_detected': False, 'ratio_correct': False}
        return result
    r_outer, r_inner = radii
    measured_ratio = r_inner / r_outer if r_outer > 0 else 0
    ratio_ok = abs(measured_ratio - TARGET_RADIUS_RATIO) <= RATIO_TOL
    result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'donut_detected': True, 'ratio_correct': bool(ratio_ok)}
    result['passed'] = bool(ratio_ok and c_bg_white)
    result['meta'] = {'background_median_intensity': round(bg_mean, 1), 'r_outer': round(r_outer, 2), 'r_inner': round(r_inner, 2), 'measured_ratio': round(measured_ratio, 4), 'target_ratio': TARGET_RADIUS_RATIO, 'tolerance': RATIO_TOL}
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen fraction case 25.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
