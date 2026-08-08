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
PROMPT_ID = 28
COLOR_H_LOW1, COLOR_H_HIGH1 = (0, 10)
COLOR_H_LOW2, COLOR_H_HIGH2 = (170, 180)
COLOR_S_MIN = 50
COLOR_V_MIN = 50
TARGET_RATIO = 1 / 4.0
RATIO_TOL = 0.035
MIN_CENTER_FILL = 0.35
MIN_RADIAL_AXES = 2

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

def get_circle_mask(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return mask

def radial_axis_count(img: np.ndarray, circle_mask: np.ndarray) -> Tuple[int, float]:
    contours, _ = cv2.findContours(circle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0.0)
    c = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(c)
    if radius <= 0:
        return (0, 0.0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 140)
    edges = cv2.bitwise_and(edges, circle_mask)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=max(30, int(radius * 0.18)), minLineLength=max(20, int(radius * 0.35)), maxLineGap=max(8, int(radius * 0.08)))
    if lines is None:
        return (0, float(radius))
    axes = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < radius * 0.35:
            continue
        dist = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / max(length, 1.0)
        if dist > radius * 0.12:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
        if not any((min(abs(angle - old), 180.0 - abs(angle - old)) <= 12.0 for old in axes)):
            axes.append(angle)
    return (len(axes), float(radius))

def center_fill_fraction(img: np.ndarray, circle_mask: np.ndarray, radius: float) -> float:
    if radius <= 0:
        return 0.0
    contours, _ = cv2.findContours(circle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    (cx, cy), _ = cv2.minEnclosingCircle(c)
    h, w = circle_mask.shape
    yy, xx = np.ogrid[:h, :w]
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius * 0.16) ** 2
    if not np.any(disc):
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filled = (gray < 235) & (circle_mask > 0)
    return float(np.mean(filled[disc]))

def check_background_is_white(image_path: str, threshold: int=240) -> Tuple[bool, float]:
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
    target_mask = get_target_mask(hsv)
    target_pixels = np.count_nonzero(target_mask)
    c_target_detected = target_pixels > 1000
    circle_mask = get_circle_mask(img)
    total_pixels = np.count_nonzero(circle_mask)
    c_circle_detected = total_pixels > 5000
    if not c_target_detected or not c_circle_detected:
        result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'target_sector_detected': c_target_detected, 'circle_detected': c_circle_detected, 'ratio_correct': False}
        return result
    ratio = target_pixels / total_pixels if total_pixels > 0 else 0
    c_ratio_ok = abs(ratio - TARGET_RATIO) <= RATIO_TOL
    axes_count, radius = radial_axis_count(img, circle_mask)
    center_fill = center_fill_fraction(img, circle_mask, radius)
    c_radial_structure = axes_count == MIN_RADIAL_AXES
    c_center_filled = center_fill >= MIN_CENTER_FILL
    c_is_circle = False
    contours, _ = cv2.findContours(circle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        _, radius = cv2.minEnclosingCircle(c)
        enc_area = np.pi * radius ** 2
        if enc_area > 0 and area / enc_area > 0.75:
            c_is_circle = True
    result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'target_sector_detected': True, 'circle_detected': bool(c_is_circle), 'ratio_correct': bool(c_ratio_ok), 'radial_structure_ok': bool(c_radial_structure), 'center_filled': bool(c_center_filled)}
    result['passed'] = bool(c_ratio_ok and c_bg_white and c_is_circle and c_radial_structure and c_center_filled)
    result['meta'] = {'background_median_intensity': round(bg_mean, 1), 'target_pixels': target_pixels, 'total_pixels': total_pixels, 'measured_ratio': round(ratio, 4), 'target_ratio': round(TARGET_RATIO, 4), 'tolerance': RATIO_TOL, 'radial_axes': axes_count, 'center_fill_fraction': round(center_fill, 4)}
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen fraction case 28.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
