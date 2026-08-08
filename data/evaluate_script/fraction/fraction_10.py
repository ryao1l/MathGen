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
PROMPT_ID = 10
RED_H_LOW1, RED_H_HIGH1 = (0, 10)
RED_H_LOW2, RED_H_HIGH2 = (170, 180)
S_MIN, V_MIN = (50, 50)
BLUE_H_LOW, BLUE_H_HIGH = (100, 140)
TARGET_RATIO = 1 / 7.0
RATIO_TOL = 0.02

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

def get_circle_mask(img: np.ndarray) -> np.ndarray:
    """
    Return binary mask of the pie chart circle using Hough Transform.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=50, maxRadius=500)
    mask = np.zeros_like(gray)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest_circle = max(circles[0, :], key=lambda c: c[2])
        center = (largest_circle[0], largest_circle[1])
        radius = largest_circle[2]
        cv2.circle(mask, center, radius, 255, thickness=cv2.FILLED)
    return mask

def get_red_mask(hsv: np.ndarray) -> np.ndarray:
    """Return binary mask of red pixels."""
    lower1 = np.array([RED_H_LOW1, S_MIN, V_MIN])
    upper1 = np.array([RED_H_HIGH1, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    lower2 = np.array([RED_H_LOW2, S_MIN, V_MIN])
    upper2 = np.array([RED_H_HIGH2, 255, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask

def get_blue_mask(hsv: np.ndarray) -> np.ndarray:
    """Return binary mask of blue pixels (the other sector)."""
    lower = np.array([BLUE_H_LOW, S_MIN, V_MIN])
    upper = np.array([BLUE_H_HIGH, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask

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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask = get_red_mask(hsv)
    red_pixels = np.count_nonzero(red_mask)
    c_red_detected = red_pixels > 1000
    blue_mask = get_blue_mask(hsv)
    blue_pixels = np.count_nonzero(blue_mask)
    c_other_detected = blue_pixels > 1000
    if not c_red_detected or not c_other_detected:
        result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'red_sector_detected': c_red_detected, 'other_sector_detected': c_other_detected, 'ratio_correct': False}
        return result
    total_pixels = red_pixels + blue_pixels
    ratio = red_pixels / total_pixels if total_pixels > 0 else 0
    c_ratio_ok = abs(ratio - TARGET_RATIO) <= RATIO_TOL
    result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'red_sector_detected': True, 'other_sector_detected': True, 'ratio_correct': bool(c_ratio_ok)}
    result['passed'] = bool(c_ratio_ok and c_bg_white)
    result['meta'] = {'background_median_intensity': round(bg_mean, 1), 'red_pixels': red_pixels, 'blue_pixels': blue_pixels, 'total_pixels': total_pixels, 'measured_ratio': round(ratio, 4), 'target_ratio': round(TARGET_RATIO, 4), 'tolerance': RATIO_TOL}
    if save_debug:
        base, _ = os.path.splitext(image_path)
        debug_path = base + '_debug.png'
        vis = img.copy()
        vis[red_mask > 0] = [0, 0, 255]
        vis[blue_mask > 0] = [255, 0, 0]
        cv2.putText(vis, f'Ratio: {ratio:.3f} (Tg: {TARGET_RATIO:.3f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imwrite(debug_path, vis)
        result['debug'] = {'debug_path': debug_path}
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen fraction case 10.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
