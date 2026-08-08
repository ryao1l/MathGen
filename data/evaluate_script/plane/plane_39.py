import argparse

PID = 12
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def judge_plane_12(img):
    if img is None:
        return [False, "Input image is None. "]

    h, w = img.shape[:2]
    min_hw = float(min(h, w))

    circle = detect_largest_circle(img)
    if circle is None:
        return [False, "Failed to detect the main circle. "]
    cx, cy, r = [float(v) for v in circle]
    min_radius = scale_px(min_hw, 0.12, floor_px=0.0)
    if r < min_radius:
        return [False, f"Detected circle is too small (r={r:.1f}). "]

    def _label_ink_near(point, line_item=None):
        px, py = float(point[0]), float(point[1])
        _, bw = _gray_and_ink_mask(img)
        win = 0.24 * r
        x1 = int(max(0, px - win))
        x2 = int(min(w, px + win))
        y1 = int(max(0, py - win))
        y2 = int(min(h, py + win))
        if x2 <= x1 or y2 <= y1:
            return 0
        roi = bw[y1:y2, x1:x2] > 0
        yy, xx = np.ogrid[y1:y2, x1:x2]
        point_disk = ((xx.astype(np.float32) - px) ** 2 + (yy.astype(np.float32) - py) ** 2) <= (0.055 * r) ** 2
        if line_item is not None:
            a, b, c = [float(v) for v in line_item["abc"]]
            den = max(1e-6, math.hypot(a, b))
            line_band = np.abs(a * xx.astype(np.float32) + b * yy.astype(np.float32) + c) / den <= max(3.0, 0.016 * r)
        else:
            line_band = np.zeros_like(roi, dtype=bool)
        label_mask = roi & (~point_disk) & (~line_band)
        if int(label_mask.sum()) <= 0:
            return 0
        num, _, stats, _ = cv2.connectedComponentsWithStats(label_mask.astype(np.uint8), connectivity=8)
        min_area = max(8, int(0.000005 * h * w))
        max_area = max(450, int(0.0045 * h * w))
        areas = []
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            bw_box = int(stats[i, cv2.CC_STAT_WIDTH])
            bh_box = int(stats[i, cv2.CC_STAT_HEIGHT])
            if area < min_area or area > max_area:
                continue
            if bw_box > 0.24 * r or bh_box > 0.24 * r:
                continue
            areas.append(area)
        return max(areas) if areas else 0

    lines = detect_line_segments(img, min_len_ratio=0.08)
    if not lines:
        return [False, "Failed to detect radius candidate. "]

    center_tol = 0.06 * r
    near_center_tol = 0.18 * r
    best = None
    best_score = None
    radius_like_count = 0
    for ln in lines:
        if point_line_distance((cx, cy), ln["abc"]) > center_tol:
            continue
        x1, y1, x2, y2 = [float(v) for v in ln["seg"]]
        d1 = math.hypot(x1 - cx, y1 - cy)
        d2 = math.hypot(x2 - cx, y2 - cy)
        if min(d1, d2) > near_center_tol:
            continue
        far_d = max(d1, d2)
        if far_d < 0.55 * r or far_d > 2.10 * r:
            continue
        radius_like_count += 1

        A_geo = (x1, y1) if d1 >= d2 else (x2, y2)
        inters = circle_line_intersections(circle, ln["abc"])
        if inters:
            A_geo = min(inters, key=lambda p: math.hypot(float(p[0]) - A_geo[0], float(p[1]) - A_geo[1]))

        ok_ca, ratio_ca = has_segment_between_points(img, (cx, cy), A_geo, ratio_th=0.14, thickness=2, trim_ratio=0.03)
        if not ok_ca:
            continue
        score = float(ln["len"]) + ratio_ca * r
        if best_score is None or score > best_score:
            best_score = score
            best = (ln, A_geo)

    if best is None:
        return [False, "Failed to detect a valid radius segment from center to circle. "]

    radius_line, A_geo = best
    if radius_like_count > 2:
        return [False, f"Detected multiple radius-like segments through C ({radius_like_count}). "]
    extra_th = scale_px(min_hw, 0.26, floor_px=0.0)
    extras = []
    for ln in lines:
        if float(ln["len"]) < extra_th:
            continue
        if line_equivalent(ln, radius_line, min_hw):
            continue
        extras.append(ln)
    if len(extras) > 12:
        return [False, f"Detected extra dominant line(s) outside radius CA: {len(extras)}. "]

    tokens = extract_global_letter_tokens(img, whitelist="AC", min_conf=0.10)
    C_lbl = select_token_near_point(tokens, expected_char="C", point=(cx, cy), max_dist=0.24 * r)
    C_ink = _label_ink_near((cx, cy), radius_line)
    ink_th = max(10, int(0.000006 * h * w))
    if C_lbl is None and C_ink < ink_th:
        return [False, f"Failed to detect center label C near the circle center (label_ink={C_ink}). "]
    A_lbl = select_token_near_point(tokens, expected_char="A", point=A_geo, max_dist=0.20 * r)
    A_ink = _label_ink_near(A_geo, radius_line)
    if A_lbl is None and A_ink < ink_th:
        return [False, f"Failed to detect endpoint label A on the radius (label_ink={A_ink}). "]

    if C_lbl is None and A_lbl is None and horizontal_error_deg(radius_line["ang"]) > 70.0:
        return [False, "Labels are ambiguous on a near-vertical radius/diameter. "]

    crossing_extras = []
    for ln in extras:
        if float(ln["len"]) < 0.24 * min_hw:
            continue
        if angle_diff_deg(ln["ang"], radius_line["ang"]) >= 24.0:
            crossing_extras.append(ln)
    if len(crossing_extras) > 1:
        return [False, f"Detected too many non-radius line structures ({len(crossing_extras)}). "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_12,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
