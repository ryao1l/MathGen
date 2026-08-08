import argparse

PID = 20
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _select_circle_pair(circles, min_hw):
    if circles is None or len(circles) < 2:
        return None

    gap = scale_px(min_hw, 0.01, floor_px=0.0)
    best = None
    best_score = None
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            x1, y1, r1, s1 = circles[i]
            x2, y2, r2, s2 = circles[j]
            d = math.hypot(x1 - x2, y1 - y2)
            if d <= (r1 + r2 + gap):
                continue
            score = float(s1 + s2) + 0.20 * d + 0.15 * abs(r1 - r2)
            if best_score is None or score > best_score:
                best_score = score
                best = ((x1, y1, r1), (x2, y2, r2))
    if best is not None:
        return best
    c1 = circles[0]
    c2 = circles[1]
    return ((c1[0], c1[1], c1[2]), (c2[0], c2[1], c2[2]))


def _detect_two_distinct_circles(img):
    h, w = img.shape[:2]
    min_hw = float(min(h, w))

    min_r = int(round(scale_px(min_hw, 0.035, floor_px=1.0)))
    score_th = int(round(scale_px(min_hw, 0.12, floor_px=1.0)))

    found, scores = _find_top_k_circles(
        img,
        k=16,
        min_r=min_r,
        max_r=0,
        seed=0,
        iters=3200,
        score_th=score_th,
    )

    candidates = []
    for (cx, cy, r), s in zip(found, scores):
        candidates.append((float(cx), float(cy), float(r), int(s)))

    if candidates:
        candidates.sort(key=lambda t: (-t[3], -t[2]))
        merged = _merge_circle_candidates(
            candidates,
            center_tol=scale_px(min_hw, 0.03, floor_px=0.0),
            radius_tol=scale_px(min_hw, 0.03, floor_px=0.0),
        )

        max_r = scale_px(min_hw, 0.60, floor_px=0.0)
        x_margin = 0.10 * float(w)
        y_margin = 0.10 * float(h)
        filtered = []
        for x, y, r, s in merged:
            if r <= 0.0 or r > max_r:
                continue
            if x < (-x_margin) or x > (float(w) + x_margin):
                continue
            if y < (-y_margin) or y > (float(h) + y_margin):
                continue

            refined = _refine_circle_radius_by_inner_outer_edges(img, (x, y, r))
            if refined is None:
                continue
            rx, ry, rr = [float(v) for v in refined]
            if rr <= 0.0 or rr > max_r:
                continue
            filtered.append((rx, ry, rr, s))

        filtered.sort(key=lambda t: (-t[3], -t[2]))
        pair = _select_circle_pair(filtered, min_hw)
        if pair is not None:
            return pair

    c1 = detect_circle(img, order=1, min_r=min_r, max_r=0)
    c2 = detect_circle(img, order=2, min_r=min_r, max_r=0)
    return c1, c2


def judge_plane_20(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))

    c1, c2 = _detect_two_distinct_circles(img)
    if c1 is None or c2 is None:
        return [False, "Failed to detect two circles. "]

    x1, y1, r1 = [float(v) for v in c1]
    x2, y2, r2 = [float(v) for v in c2]

    min_r_th = scale_px(min_hw, 0.04, floor_px=0.0)
    if min(r1, r2) < min_r_th:
        return [False, "Detected circles are too small. "]

    disjoint_gap = scale_px(min_hw, 0.02, floor_px=0.0)
    d = math.hypot(x1 - x2, y1 - y2)
    if d <= (r1 + r2 + disjoint_gap):
        return [False, "The two circles are not disjoint. "]

    radius_sep = scale_px(min_hw, 0.04, floor_px=0.0)
    if abs(r1 - r2) <= radius_sep:
        return [False, "The two circles are not clearly different in size. "]

    lines = detect_line_segments(img, min_len_ratio=0.12)
    dominant_len = scale_px(min_hw, 0.35, floor_px=0.0)
    dominant = [ln for ln in lines if float(ln["len"]) >= dominant_len]
    if dominant:
        return [False, f"Unexpected dominant line(s) detected in two-circle task: {len(dominant)}. "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_20,
        require_ocr=False,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
