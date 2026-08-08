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
PROMPT_ID = 50
COLOR_H_LOW1, COLOR_H_HIGH1 = (0, 10)
COLOR_H_LOW2, COLOR_H_HIGH2 = (170, 180)
COLOR_S_MIN = 50
COLOR_V_MIN = 50
TARGET_RATIO = 1 / 4.0
RATIO_TOL = 0.08
EXPECTED_CELLS = 1
GRID_DIM = 4
MIN_RED_CELL_FRAC = 0.18
MIN_GRAY_CELL_FRAC = 0.45
MAX_GRAY_IN_RED_CELL = 0.35

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

def classify_2x2_cells(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, list]:
    x, y, w, h = bbox
    cell_reports = []
    red_cells = 0
    gray_cells = 0
    white_cells = 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    target_mask = get_target_mask(hsv)
    for row in range(2):
        for col in range(2):
            x0 = x + int(col * w / 2)
            x1 = x + int((col + 1) * w / 2)
            y0 = y + int(row * h / 2)
            y1 = y + int((row + 1) * h / 2)
            padx = max(1, int((x1 - x0) * 0.18))
            pady = max(1, int((y1 - y0) * 0.18))
            roi = img[y0 + pady:y1 - pady, x0 + padx:x1 - padx]
            red_roi = target_mask[y0 + pady:y1 - pady, x0 + padx:x1 - padx]
            if roi.size == 0 or red_roi.size == 0:
                cell_reports.append({'red_fraction': 0.0, 'gray_fraction': 0.0, 'class': 'empty'})
                continue
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            red_frac = float(np.mean(red_roi > 0))
            sat = hsv_roi[:, :, 1]
            val = hsv_roi[:, :, 2]
            gray_frac = float(np.mean((sat < 45) & (val > 70) & (val < 235)))
            white_frac = float(np.mean((sat < 35) & (val >= 235)))
            if red_frac >= MIN_RED_CELL_FRAC:
                cls = 'red'
                red_cells += 1
            elif gray_frac >= MIN_GRAY_CELL_FRAC:
                cls = 'gray'
                gray_cells += 1
            elif white_frac >= 0.55:
                cls = 'white'
                white_cells += 1
            else:
                cls = 'other'
            cell_reports.append({'red_fraction': round(red_frac, 4), 'gray_fraction': round(gray_frac, 4), 'white_fraction': round(white_frac, 4), 'class': cls})
    return (red_cells, gray_cells, white_cells, cell_reports)

def square_bbox_candidates(bbox: Tuple[int, int, int, int], shape: Tuple[int, int]) -> list:
    x, y, w, h = bbox
    H, W = shape
    size = max(w, h)
    cx = x + w / 2.0
    cy = y + h / 2.0
    candidates = [bbox]
    for scale in (1.0, 1.12):
        s = int(round(size * scale))
        nx = int(round(cx - s / 2.0))
        ny = int(round(cy - s / 2.0))
        nx = max(0, min(nx, W - 1))
        ny = max(0, min(ny, H - 1))
        nw = min(s, W - nx)
        nh = min(s, H - ny)
        if nw > 20 and nh > 20:
            candidates.append((nx, ny, nw, nh))
    return candidates

def separator_edge_density(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = bbox
    roi = img[y:y + h, x:x + w]
    if roi.size == 0:
        return (0.0, 0.0)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 35, 130)
    band = max(2, int(min(w, h) * 0.035))
    mid_x = w // 2
    mid_y = h // 2
    vertical = edges[:, max(0, mid_x - band):min(w, mid_x + band + 1)]
    horizontal = edges[max(0, mid_y - band):min(h, mid_y + band + 1), :]
    v_density = float(np.mean(vertical > 0)) if vertical.size else 0.0
    h_density = float(np.mean(horizontal > 0)) if horizontal.size else 0.0
    return (v_density, h_density)

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
    best_layout = None
    for candidate in square_bbox_candidates(bbox, img.shape[:2]):
        cx, cy, cw, ch = candidate
        aspect_candidate = cw / max(ch, 1)
        red_count, gray_count, white_count, reports = classify_2x2_cells(img, candidate)
        red_gray = [report['gray_fraction'] for report in reports if report['class'] == 'red']
        red_cell_clean = bool(red_gray) and max(red_gray) <= MAX_GRAY_IN_RED_CELL
        v_edge, h_edge = separator_edge_density(img, candidate)
        non_red_count = gray_count + white_count
        ok = 0.72 <= aspect_candidate <= 1.38 and red_count == 1 and (non_red_count >= 3) and red_cell_clean
        score = (10 if ok else 0) + non_red_count + v_edge + h_edge - abs(red_count - 1) - abs(1.0 - aspect_candidate)
        item = (score, candidate, aspect_candidate, red_count, gray_count, white_count, reports, ok, v_edge, h_edge)
        if best_layout is None or item[0] > best_layout[0]:
            best_layout = item
    _, layout_bbox, grid_aspect, red_cells_2x2, gray_cells_2x2, white_cells_2x2, cell_reports, c_2x2_cell_layout_ok, v_edge, h_edge = best_layout
    x, y, w, h = layout_bbox
    grid_area_pixels = w * h
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    target_mask = get_target_mask(hsv)
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
    if c_2x2_cell_layout_ok:
        c_cell_count_ok = True
        c_ratio_ok = True
    c_background_ok = c_bg_white or (c_2x2_cell_layout_ok and bg_mean >= 235)
    result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'background_ok': bool(c_background_ok), 'grid_detected': True, 'ratio_correct': bool(c_ratio_ok), 'cell_count_correct': bool(c_cell_count_ok), 'cell_layout_2x2_ok': bool(c_2x2_cell_layout_ok)}
    result['passed'] = bool(c_background_ok and c_2x2_cell_layout_ok)
    result['meta'] = {'background_median_intensity': round(bg_mean, 1), 'target_pixels': target_pixels, 'grid_area': grid_area_pixels, 'measured_ratio': round(ratio, 4), 'target_ratio': round(TARGET_RATIO, 4), 'tolerance': RATIO_TOL, 'estimated_cells': estimated_cells, 'grid_aspect': round(grid_aspect, 4), 'layout_bbox': list(layout_bbox), 'red_cells_2x2': red_cells_2x2, 'gray_cells_2x2': gray_cells_2x2, 'white_cells_2x2': white_cells_2x2, 'vertical_separator_edge_density': round(v_edge, 4), 'horizontal_separator_edge_density': round(h_edge, 4), 'cell_reports': cell_reports}
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen fraction case 50.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
