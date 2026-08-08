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
PROMPT_ID = 7
BLUE_H_LOW, BLUE_H_HIGH = (90, 130)
BLUE_S_MIN = 50
BLUE_V_MIN = 50
TARGET_RATIO = 1 / 3.0
RATIO_TOL = 0.1
MAX_CLUTTER_RATIO = 0.02

def load_image(path: str) -> Optional[np.ndarray]:
    """Load an image (compositing transparent PNGs onto white)."""
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

def blue_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    """Return binary mask of blue pixels."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([BLUE_H_LOW, BLUE_S_MIN, BLUE_V_MIN])
    upper = np.array([BLUE_H_HIGH, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask

def get_liquid_rect_robust(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Returns (x, y, w, h) of the main liquid body using the largest contour."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 100:
        return None
    return cv2.boundingRect(largest_contour)

def get_beaker_vertical_range_robust(img: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Returns (y_top, y_bottom) of the beaker using findNonZero on threshold+edges.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(gray, 50, 150)
    combined = cv2.bitwise_or(binary, edges)
    coords = cv2.findNonZero(combined)
    if coords is None:
        return None
    _, y, _, h = cv2.boundingRect(coords)
    return (y, y + h)

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

def evaluate(image_path: str, save_debug: bool=False) -> Dict[str, object]:
    result = {'criteria': {}, 'passed': False}
    if not os.path.isfile(image_path):
        result['criteria'] = {'image_exists': False, 'passed': False}
        return result
    img = load_image(image_path)
    if img is None:
        result['criteria'] = {'image_exists': True, 'image_readable': False, 'passed': False}
        return result
    c_bg_white, bg_mean = check_background_is_white(image_path)
    mask = blue_mask_hsv(img)
    liquid_rect = get_liquid_rect_robust(mask)
    c_liquid_detected = liquid_rect is not None
    beaker_range = get_beaker_vertical_range_robust(img)
    c_beaker_detected = beaker_range is not None
    c_is_cylinder = False
    if c_beaker_detected:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        edges = cv2.Canny(gray, 50, 150)
        combined = cv2.bitwise_or(binary, edges)
        y_top, y_bot = beaker_range
        h = y_bot - y_top
        ty = y_top + int(h * 0.2)
        by = y_top + int(h * 0.8)
        t_row = combined[ty, :]
        t_px = np.where(t_row > 0)[0]
        t_w = t_px[-1] - t_px[0] if len(t_px) > 0 else 1
        b_row = combined[by, :]
        b_px = np.where(b_row > 0)[0]
        b_w = b_px[-1] - b_px[0] if len(b_px) > 0 else 1
        width_ratio = b_w / max(1, t_w)
        c_is_cylinder = width_ratio < 1.6 and y_bot - y_top <= img.shape[0] * 0.85
    if not c_liquid_detected or not c_beaker_detected or (not c_is_cylinder):
        result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'clean_image': True, 'liquid_detected': c_liquid_detected, 'beaker_detected': c_beaker_detected, 'is_cylinder': bool(c_is_cylinder), 'height_ratio_correct': False}
        return result
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark_mask = (gray < 80).astype(np.uint8) * 255
    non_blue_dark = cv2.bitwise_and(dark_mask, cv2.bitwise_not(mask))
    clutter_ratio = cv2.countNonZero(non_blue_dark) / (img.shape[0] * img.shape[1])
    c_clean_image = clutter_ratio <= MAX_CLUTTER_RATIO
    lx, ly, lw, lh = liquid_rect
    by_top, by_bottom = beaker_range
    beaker_h = by_bottom - by_top
    ratio = lh / beaker_h if beaker_h > 0 else 0
    c_ratio_ok = abs(ratio - TARGET_RATIO) <= RATIO_TOL
    H = img.shape[0]
    is_missing_rim = abs(by_top - ly) < 0.1 * H
    if False and (not c_ratio_ok) and (ratio > 0.8) and is_missing_rim:
        visual_ratio = lh / H
        if abs(visual_ratio - TARGET_RATIO) <= RATIO_TOL + 0.25:
            c_ratio_ok = True
            ratio = visual_ratio
    result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'clean_image': bool(c_clean_image), 'liquid_detected': True, 'beaker_detected': True, 'is_cylinder': True, 'height_ratio_correct': bool(c_ratio_ok)}
    result['passed'] = bool(c_ratio_ok and c_bg_white and c_clean_image)
    result['meta'] = {'background_median_intensity': round(bg_mean, 1), 'liquid_h': lh, 'beaker_h': beaker_h, 'measured_ratio': round(ratio, 4), 'target_ratio': round(TARGET_RATIO, 4), 'tolerance': RATIO_TOL}
    if save_debug:
        pass
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen fraction case 7.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
