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
PROMPT_ID = 48
TARGET_RATIO = 99 / 100.0
RATIO_TOL = 0.08
EXPECTED_CELLS = 99
GRID_DIM = 100
GRID_SIDE = 10

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

def get_target_mask(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def sample_grid_cells(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, list]:
    x, y, w, h = bbox
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black_cells = 0
    gray_cells = 0
    reports = []
    for row in range(GRID_SIDE):
        for col in range(GRID_SIDE):
            x0 = x + int(col * w / GRID_SIDE)
            x1 = x + int((col + 1) * w / GRID_SIDE)
            y0 = y + int(row * h / GRID_SIDE)
            y1 = y + int((row + 1) * h / GRID_SIDE)
            padx = max(1, int((x1 - x0) * 0.22))
            pady = max(1, int((y1 - y0) * 0.22))
            cell = gray[y0 + pady:y1 - pady, x0 + padx:x1 - padx]
            if cell.size == 0:
                reports.append({'row': row, 'col': col, 'median': 255.0, 'class': 'empty'})
                continue
            median = float(np.median(cell))
            if median < 70:
                cls = 'black'
                black_cells += 1
            elif 70 <= median < 210:
                cls = 'gray'
                gray_cells += 1
            else:
                cls = 'white'
            reports.append({'row': row, 'col': col, 'median': round(median, 1), 'class': cls})
    return (black_cells, gray_cells, reports)

def get_grid_bbox(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if not contours:
        return None
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
    bbox = get_grid_bbox(img)
    if bbox is None:
        result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'grid_detected': False, 'ratio_correct': False}
        return result
    x, y, w, h = bbox
    grid_area_pixels = w * h
    target_mask = get_target_mask(img)
    roi = target_mask[y:y + h, x:x + w]
    target_pixels = np.count_nonzero(roi)
    ratio = target_pixels / grid_area_pixels if grid_area_pixels > 0 else 0
    c_ratio_ok = abs(ratio - TARGET_RATIO) <= RATIO_TOL
    cell_area = w * h / float(GRID_DIM)
    blobs, _ = cv2.findContours(roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    estimated_cells = 0
    for blob in blobs:
        area = cv2.contourArea(blob)
        if area > cell_area * 0.2:
            estimated_cells += max(1, int(round(area / cell_area)))
    c_cell_count_ok = estimated_cells == EXPECTED_CELLS
    sampled_black_cells, sampled_gray_cells, sampled_cell_reports = sample_grid_cells(img, bbox)
    c_sampled_count_ok = sampled_black_cells >= 98 and sampled_gray_cells >= 1
    if c_sampled_count_ok or c_cell_count_ok:
        c_cell_count_ok = True
        c_ratio_ok = True
    result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'grid_detected': True, 'ratio_correct': bool(c_ratio_ok), 'cell_count_correct': bool(c_cell_count_ok), 'sampled_cell_count_correct': bool(c_sampled_count_ok)}
    result['passed'] = bool(c_ratio_ok and c_bg_white and c_cell_count_ok)
    result['meta'] = {'background_median_intensity': round(bg_mean, 1), 'target_pixels': target_pixels, 'grid_area': grid_area_pixels, 'measured_ratio': round(ratio, 4), 'target_ratio': round(TARGET_RATIO, 4), 'tolerance': RATIO_TOL, 'estimated_cells': estimated_cells, 'sampled_black_cells': sampled_black_cells, 'sampled_gray_cells': sampled_gray_cells, 'sampled_cell_reports': sampled_cell_reports}
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen fraction case 48.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
