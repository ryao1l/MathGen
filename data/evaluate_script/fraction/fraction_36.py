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
PROMPT_ID = 36
COLOR_H_LOW1, COLOR_H_HIGH1 = (0, 10)
COLOR_H_LOW2, COLOR_H_HIGH2 = (170, 180)
COLOR_S_MIN = 50
COLOR_V_MIN = 50
TARGET_RATIO = 1 / 9.0
RATIO_TOL = 0.035
TARGET_CELLS = 1

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

def get_target_mask(hsv: np.ndarray) -> np.ndarray:
    l1 = np.array([COLOR_H_LOW1, COLOR_S_MIN, COLOR_V_MIN])
    u1 = np.array([COLOR_H_HIGH1, 255, 255])
    l2 = np.array([COLOR_H_LOW2, COLOR_S_MIN, COLOR_V_MIN])
    u2 = np.array([COLOR_H_HIGH2, 255, 255])
    m1 = cv2.inRange(hsv, l1, u1)
    m2 = cv2.inRange(hsv, l2, u2)
    mask = cv2.bitwise_or(m1, m2)
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

def check_background_is_white(image_path: str, threshold: int=240) -> Tuple[bool, float]:
    if not os.path.isfile(image_path):
        return (False, 0.0)
    raw_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw_img is None:
        return (False, 0.0)
    H, W = raw_img.shape[:2]
    border = max(1, int(min(H, W) * 0.08))
    img_check = raw_img[:, :, :3] if raw_img.ndim == 3 and raw_img.shape[2] == 4 else raw_img
    gray = cv2.cvtColor(img_check, cv2.COLOR_BGR2GRAY) if img_check.ndim == 3 else img_check
    top_g, bottom_g = (gray[:border, :], gray[H - border:, :])
    left_g, right_g = (gray[:, :border], gray[:, W - border:])
    intensity_border = np.concatenate([top_g.ravel(), bottom_g.ravel(), left_g.ravel(), right_g.ravel()])
    median_intensity = float(np.median(intensity_border))
    return (median_intensity >= threshold, median_intensity)

def evaluate(image_path: str, save_debug: bool=False) -> Dict[str, object]:
    result = {'criteria': {}, 'passed': False}
    img = load_image(image_path)
    if img is None:
        result['criteria'] = {'image_exists': False}
        return result
    c_bg_white, bg_mean = check_background_is_white(image_path)
    bbox = get_grid_bbox(img)
    c_grid_detected = bbox is not None
    if not c_grid_detected:
        result['criteria'] = {'grid_detected': False}
        return result
    x, y, w, h = bbox
    grid_area_pixels = w * h
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    target_mask = get_target_mask(hsv)
    roi_mask = target_mask[y:y + h, x:x + w]
    target_pixels = np.count_nonzero(roi_mask)
    ratio = target_pixels / grid_area_pixels if grid_area_pixels > 0 else 0
    c_ratio_ok = 0.075 <= ratio <= 0.13
    cell_area = w * h / 9.0
    blobs, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    estimated_cells = 0
    for blob in blobs:
        area = cv2.contourArea(blob)
        if area > cell_area * 0.2:
            estimated_cells += max(1, int(round(area / cell_area)))
    c_cell_count_ok = estimated_cells == TARGET_CELLS
    result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'grid_detected': True, 'ratio_correct': bool(c_ratio_ok), 'cell_count_correct': bool(c_cell_count_ok)}
    result['passed'] = bool(c_ratio_ok and c_bg_white and c_cell_count_ok)
    result['meta'] = {'background_median_intensity': round(bg_mean, 1), 'target_pixels': target_pixels, 'grid_area': grid_area_pixels, 'measured_ratio': round(ratio, 4), 'target_ratio': round(TARGET_RATIO, 4), 'estimated_cells': estimated_cells}
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen fraction case 36.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
