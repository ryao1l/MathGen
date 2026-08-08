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
PROMPT_ID = 22
TARGET_AREA_RATIO = 2.0
RATIO_TOL = 0.1
SQUARE_TOL = 0.15
LOWER_RED1 = np.array([0, 70, 50])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 70, 50])
UPPER_RED2 = np.array([180, 255, 255])
LOWER_BLUE = np.array([100, 70, 50])
UPPER_BLUE = np.array([130, 255, 255])

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

def get_largest_colored_square_area(img: np.ndarray, lower_bound, upper_bound, lower_bound2=None, upper_bound2=None) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    if lower_bound2 is not None:
        mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
        mask = cv2.bitwise_or(mask, mask2)
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
    red_area, red_bbox = get_largest_colored_square_area(img, LOWER_RED1, UPPER_RED1, LOWER_RED2, UPPER_RED2)
    blue_area, blue_bbox = get_largest_colored_square_area(img, LOWER_BLUE, UPPER_BLUE)
    red_detected = red_bbox is not None
    blue_detected = blue_bbox is not None
    if not red_detected or not blue_detected:
        result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'red_square_detected': red_detected, 'blue_square_detected': blue_detected, 'ratio_correct': False}
        return result
    measured_ratio = red_area / blue_area if blue_area > 0 else 0
    ratio_ok = abs(measured_ratio - TARGET_AREA_RATIO) <= TARGET_AREA_RATIO * RATIO_TOL
    result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'red_square_detected': True, 'blue_square_detected': True, 'ratio_correct': bool(ratio_ok)}
    result['passed'] = bool(ratio_ok and c_bg_white)
    result['meta'] = {'background_median_intensity': round(bg_mean, 1), 'red_area': red_area, 'blue_area': blue_area, 'measured_ratio': round(measured_ratio, 4), 'target_ratio': TARGET_AREA_RATIO, 'tolerance': RATIO_TOL}
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen fraction case 22.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
