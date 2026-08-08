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
PROMPT_ID = 27
TARGET_RATIO = 17 / 24.0
RATIO_TOL = 0.05

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

def get_red_mask(hsv: np.ndarray) -> np.ndarray:
    mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask

def check_background_is_white(image_path: str, threshold: int=240) -> Tuple[bool, float]:
    if not os.path.isfile(image_path):
        return (False, 0.0)
    raw_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw_img is None:
        return (False, 0.0)
    H, W = raw_img.shape[:2]
    border = max(1, int(min(H, W) * 0.08))
    if raw_img.ndim == 3 and raw_img.shape[2] == 4:
        alpha = raw_img[:, :, 3]
        alpha_border = np.concatenate([alpha[:border, :].ravel(), alpha[H - border:, :].ravel(), alpha[:, :border].ravel(), alpha[:, W - border:].ravel()])
        if float(np.mean(alpha_border)) < 200:
            return (False, float(np.mean(alpha_border)))
        img_check = raw_img[:, :, :3]
    else:
        img_check = raw_img
    if img_check.ndim == 3:
        gray = cv2.cvtColor(img_check, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_check
    intensity_border = np.concatenate([gray[:border, :].ravel(), gray[H - border:, :].ravel(), gray[:, :border].ravel(), gray[:, W - border:].ravel()])
    median_intensity = float(np.median(intensity_border))
    return (median_intensity >= threshold, median_intensity)

def evaluate(image_path: str) -> Dict[str, object]:
    result = {'criteria': {}, 'passed': False}
    if not os.path.isfile(image_path):
        result['criteria'] = {'image_exists': False}
        return result
    img = load_image(image_path)
    if img is None:
        result['criteria'] = {'image_exists': True, 'image_readable': False}
        return result
    c_bg_white, bg_mean = check_background_is_white(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'chart_detected': False, 'ratio_correct': False}
        return result
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    circle_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(circle_mask, [c], -1, 255, -1)
    total_pixels = np.count_nonzero(circle_mask)
    if total_pixels < 1000:
        result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'chart_detected': False, 'ratio_correct': False}
        return result
    _, radius = cv2.minEnclosingCircle(c)
    enc_area = np.pi * radius ** 2
    c_is_circle = area / enc_area > 0.75 if enc_area > 0 else False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask = get_red_mask(hsv)
    red_in_chart = cv2.bitwise_and(red_mask, circle_mask)
    red_pixels = np.count_nonzero(red_in_chart)
    ratio = red_pixels / total_pixels if total_pixels > 0 else 0
    c_ratio_ok = abs(ratio - TARGET_RATIO) <= RATIO_TOL
    filled_pixels = np.count_nonzero((gray < 240) & (circle_mask > 0))
    c_pie_filled = filled_pixels > total_pixels * 0.85
    result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'chart_detected': True, 'is_circular': bool(c_is_circle), 'pie_is_filled': bool(c_pie_filled), 'ratio_correct': bool(c_ratio_ok)}
    result['passed'] = bool(c_ratio_ok and c_bg_white and c_is_circle and c_pie_filled)
    result['meta'] = {'background_median_intensity': round(bg_mean, 1), 'red_pixels': red_pixels, 'total_pixels': total_pixels, 'measured_ratio': round(ratio, 4), 'target_ratio': round(TARGET_RATIO, 4), 'tolerance': RATIO_TOL}
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen fraction case 27.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
