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
PROMPT_ID = 42
GREEN_H_LOW, GREEN_H_HIGH = (35, 85)
S_MIN, V_MIN = (50, 50)
TARGET_RATIO = 12 / 100.0
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

def get_green_mask(hsv: np.ndarray) -> np.ndarray:
    lower = np.array([GREEN_H_LOW, S_MIN, V_MIN])
    upper = np.array([GREEN_H_HIGH, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask

def get_grid_bbox(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    c = contours[0]
    if cv2.contourArea(c) < 10000:
        return None
    x, y, w, h = cv2.boundingRect(c)
    H, W = img.shape[:2]
    if w > W * 0.95 and h > H * 0.95:
        if len(contours) < 2:
            return None
        c = contours[1]
        if cv2.contourArea(c) < 10000:
            return None
        x, y, w, h = cv2.boundingRect(c)
    return (x, y, w, h)

def check_cell_count(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
    """Check that the grid is roughly 10x10 by counting square-like contours."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cells = 0
    H, W = img.shape[:2]
    x0, y0, w0, h0 = bbox
    for c in contours:
        area = cv2.contourArea(c)
        if 10 < area < H * W / 20:
            x, y, w, h = cv2.boundingRect(c)
            if x >= x0 and y >= y0 and (x + w <= x0 + w0) and (y + h <= y0 + h0):
                aspect = w / float(h) if h > 0 else 0
                if 0.5 < aspect < 2.0:
                    cells += 1
    return 60 <= cells <= 150

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
    bbox = get_grid_bbox(img)
    c_grid_detected = bbox is not None
    if not c_grid_detected:
        result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'grid_detected': False, 'ratio_correct': False}
        return result
    x, y, w, h = bbox
    grid_area_pixels = w * h
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = get_green_mask(hsv)
    green_roi = green_mask[y:y + h, x:x + w]
    green_pixels = np.count_nonzero(green_roi)
    ratio = green_pixels / grid_area_pixels if grid_area_pixels > 0 else 0
    c_ratio_ok = abs(ratio - TARGET_RATIO) <= RATIO_TOL
    cell_area = w * h / float(100)
    if 'green_mask' in locals():
        blobs, _ = cv2.findContours(green_mask[y:y + h, x:x + w], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        blobs, _ = cv2.findContours(green_mask[y:y + h, x:x + w], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    estimated_cells = 0
    for blob in blobs:
        area = cv2.contourArea(blob)
        if area > cell_area * 0.2:
            estimated_cells += max(1, int(round(area / cell_area)))
    c_cell_count_ok = estimated_cells == 12
    c_grid_10x10 = check_cell_count(img, bbox)
    result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'grid_detected': True, 'is_10x10_grid': bool(c_grid_10x10), 'ratio_correct': bool(c_ratio_ok), 'cell_count_correct': bool(c_cell_count_ok)}
    result['passed'] = bool(c_ratio_ok and c_bg_white and c_grid_10x10 and c_cell_count_ok)
    result['meta'] = {'background_median_intensity': round(bg_mean, 1), 'green_pixels': green_pixels, 'grid_area': grid_area_pixels, 'measured_ratio': round(ratio, 4), 'target_ratio': round(TARGET_RATIO, 4), 'tolerance': RATIO_TOL, 'estimated_cells': estimated_cells}
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen fraction case 42.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
