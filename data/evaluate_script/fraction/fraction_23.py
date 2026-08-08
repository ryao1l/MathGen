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
PROMPT_ID = 23
TARGET_AREA_RATIO = 3.0
RATIO_TOL = 0.1
SQUARE_TOL = 0.15
LOWER_GREEN = np.array([40, 50, 50])
UPPER_GREEN = np.array([80, 255, 255])
LOWER_PURPLE = np.array([125, 50, 50])
UPPER_PURPLE = np.array([155, 255, 255])

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

def get_largest_colored_square_area(img: np.ndarray, lower_bound, upper_bound) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_squares = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        if abs(aspect_ratio - 1.0) <= SQUARE_TOL:
            valid_squares.append((area, (x, y, w, h)))
    if not valid_squares:
        return (0.0, None)
    valid_squares.sort(key=lambda x: x[0], reverse=True)
    return valid_squares[0]

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
    green_area, green_bbox = get_largest_colored_square_area(img, LOWER_GREEN, UPPER_GREEN)
    purple_area, purple_bbox = get_largest_colored_square_area(img, LOWER_PURPLE, UPPER_PURPLE)
    green_detected = green_bbox is not None
    purple_detected = purple_bbox is not None
    if not green_detected or not purple_detected:
        result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'green_square_detected': green_detected, 'purple_square_detected': purple_detected, 'ratio_correct': False}
        return result
    measured_ratio = green_area / purple_area if purple_area > 0 else 0
    ratio_ok = abs(measured_ratio - TARGET_AREA_RATIO) <= TARGET_AREA_RATIO * RATIO_TOL
    result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'green_square_detected': True, 'purple_square_detected': True, 'ratio_correct': bool(ratio_ok)}
    result['passed'] = bool(ratio_ok and c_bg_white)
    result['meta'] = {'background_median_intensity': round(bg_mean, 1), 'green_area': green_area, 'purple_area': purple_area, 'measured_ratio': round(measured_ratio, 4), 'target_ratio': TARGET_AREA_RATIO, 'tolerance': RATIO_TOL}
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen fraction case 23.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
