from typing import List, Tuple

import cv2
import numpy as np

CircleI = Tuple[int, int, int]  # (cx, cy, r)


def detect_exactly_three_circles(gray: np.ndarray, h: int, w: int) -> List[CircleI]:
    """Detect and return exactly three circles for standard 3-set Venn layouts."""
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(40.0, min(h, w) * 0.10),
        param1=50.0,
        param2=30.0,
        minRadius=max(20, int(min(h, w) * 0.08)),
        maxRadius=max(30, int(min(h, w) * 0.35)),
    )
    if circles is None:
        return []

    cand = np.uint16(np.around(circles[0]))
    out: List[CircleI] = []
    for x, y, r in cand:
        out.append((int(x), int(y), int(r)))
        if len(out) == 3:
            break
    return out if len(out) == 3 else []


def all_three_overlap(circles: List[CircleI]) -> bool:
    if len(circles) != 3:
        return False
    for i in range(3):
        for j in range(i + 1, 3):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            d = float(np.hypot(x1 - x2, y1 - y2))
            if d >= (r1 + r2):
                return False
    return True


def circle_bool_mask(shape_hw: Tuple[int, int], circle: CircleI) -> np.ndarray:
    h, w = shape_hw
    cx, cy, r = circle
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask.astype(bool)


def classify_abc(circles: List[CircleI]) -> Tuple[CircleI, CircleI, CircleI]:
    """
    Typical 3-set Venn ordering:
    C = lowest circle; A/B = upper-left/upper-right.
    """
    by_y = sorted(circles, key=lambda c: c[1])
    c_circle = by_y[-1]
    top_two = sorted(by_y[:2], key=lambda c: c[0])
    a_circle, b_circle = top_two[0], top_two[1]
    return a_circle, b_circle, c_circle


def color_ratio_in_region(region_mask: np.ndarray, color_mask: np.ndarray) -> float:
    denom = int(np.sum(region_mask))
    if denom == 0:
        return 0.0
    num = int(np.sum(np.logical_and(region_mask, color_mask > 0)))
    return float(num) / float(denom)
