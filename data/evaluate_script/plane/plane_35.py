import argparse

PID = 8
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _select_diameter_candidate(img, circle, lines, raw_lines, min_hw):
    cx, cy, r = [float(v) for v in circle]
    center_tol = 0.10 * r
    best = None
    best_score = None

    for ln in lines:
        if point_line_distance((cx, cy), ln["abc"]) > center_tol:
            continue
        inters = circle_line_intersections(circle, ln["abc"])
        if len(inters) < 2:
            continue
        p1, p2 = inters[0], inters[1]
        chord_len = math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))
        if chord_len < 1.65 * r:
            continue

        A_geo, B_geo = sorted([p1, p2], key=lambda p: float(p[0]))
        support = collinear_support_stats_on_segment(
            lines=raw_lines,
            reference_line=ln,
            base_seg=(A_geo[0], A_geo[1], B_geo[0], B_geo[1]),
            min_hw=min_hw,
            angle_tol_deg=3.0,
            offset_ratio=0.025,
            offset_floor_px=8.0,
            reach_left_t=0.12,
            reach_right_t=0.88,
        )
        if support["coverage"] < 0.58:
            continue
        if not support["reaches_left"] or not support["reaches_right"]:
            continue
        if (not support["center_supported"]) and support["center_gap"] > 0.06:
            continue

        ok_ab, ratio_ab = has_segment_between_points(img, A_geo, B_geo, ratio_th=0.15, trim_ratio=0.04)
        if not ok_ab:
            continue

        score = (
            float(ln["len"])
            + 0.15 * chord_len
            + 0.30 * r * float(support["coverage"])
            + 0.20 * r * float(ratio_ab)
            - 0.50 * r * float(support["center_gap"])
        )
        if best_score is None or score > best_score:
            best_score = score
            best = (ln, A_geo, B_geo, ratio_ab)
    return best


def _label_ink_near_point(img, point, line_item, radius, side="any"):
    if img is None or point is None:
        return 0
    h, w = img.shape[:2]
    px, py = float(point[0]), float(point[1])
    r = float(radius)
    _, bw = _gray_and_ink_mask(img)
    win = 0.26 * r
    x1 = int(max(0, px - win))
    x2 = int(min(w, px + win))
    y1 = int(max(0, py - win))
    y2 = int(min(h, py + win))
    if x2 <= x1 or y2 <= y1:
        return 0
    roi = bw[y1:y2, x1:x2] > 0
    yy, xx = np.ogrid[y1:y2, x1:x2]

    if line_item is not None:
        a, b, c = [float(v) for v in line_item["abc"]]
        den = max(1e-6, math.hypot(a, b))
        line_band = np.abs(a * xx.astype(np.float32) + b * yy.astype(np.float32) + c) / den <= max(3.0, 0.018 * r)
    else:
        line_band = np.zeros_like(roi, dtype=bool)
    point_disk = ((xx.astype(np.float32) - px) ** 2 + (yy.astype(np.float32) - py) ** 2) <= (0.055 * r) ** 2

    side_mask = np.ones_like(roi, dtype=bool)
    if side == "left":
        side_mask = xx.astype(np.float32) <= px - 0.02 * r
    elif side == "right":
        side_mask = xx.astype(np.float32) >= px + 0.02 * r

    label_mask = roi & (~line_band) & (~point_disk) & side_mask
    if int(label_mask.sum()) <= 0:
        return 0
    num, _, stats, _ = cv2.connectedComponentsWithStats(label_mask.astype(np.uint8), connectivity=8)
    min_area = max(8, int(0.000005 * h * w))
    max_area = max(500, int(0.0045 * h * w))
    areas = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw_box = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_box = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < min_area or area > max_area:
            continue
        if bw_box > 0.28 * r or bh_box > 0.28 * r:
            continue
        areas.append(area)
    return max(areas) if areas else 0


def judge_plane_8(img):
    if img is None:
        return [False, "Input image is None. "]

    h, w = img.shape[:2]
    min_hw = float(min(h, w))

    circle = detect_largest_circle(img)
    if circle is None:
        return [False, "Failed to detect the main circle. "]
    cx, cy, r = [float(v) for v in circle]
    if r < 0.12 * min_hw:
        return [False, f"Detected circle is too small (r={r:.1f}). "]

    lines = detect_line_segments(img, min_len_ratio=0.10)
    raw_lines = detect_line_segments_raw(img, min_len_ratio=0.10)
    if not lines:
        return [False, "Failed to detect diameter candidate. "]
    if not raw_lines:
        raw_lines = lines

    best = _select_diameter_candidate(img=img, circle=circle, lines=lines, raw_lines=raw_lines, min_hw=min_hw)
    if best is None:
        return [False, "Failed to detect a valid diameter through the circle center. "]

    diameter_line, A_geo, B_geo, _ = best

    extra_th = 0.26 * min_hw
    extras = []
    for ln in lines:
        if float(ln["len"]) < extra_th:
            continue
        if line_equivalent(ln, diameter_line, min_hw):
            continue
        extras.append(ln)

    tokens = extract_global_letter_tokens(img, whitelist="ABC", min_conf=0.10)
    center_tok = select_token_near_point(tokens, expected_char="C", point=(cx, cy), max_dist=0.28 * r)
    center_ink = _label_ink_near_point(img, (cx, cy), diameter_line, r, side="any")
    label_ink_th = max(10, int(0.000006 * h * w))
    if center_tok is None and center_ink < label_ink_th:
        return [False, f"Failed to detect center label near C (label_ink={center_ink}). "]

    endpoint_radius = 0.24 * r
    A_lbl = select_token_near_point(tokens, expected_char="A", point=A_geo, max_dist=endpoint_radius)
    B_lbl = select_token_near_point(tokens, expected_char="B", point=B_geo, max_dist=endpoint_radius)
    A_ink = _label_ink_near_point(img, A_geo, diameter_line, r, side="left")
    B_ink = _label_ink_near_point(img, B_geo, diameter_line, r, side="right")
    A_ink_any = _label_ink_near_point(img, A_geo, diameter_line, r, side="any")
    B_ink_any = _label_ink_near_point(img, B_geo, diameter_line, r, side="any")

    crossing_extras = []
    for ln in extras:
        angle_delta = angle_diff_deg(ln["ang"], diameter_line["ang"])
        if float(ln["len"]) >= 0.24 * min_hw and angle_delta >= 28.0:
            crossing_extras.append(ln)

    strong_label_ink_th = max(900, int(0.00050 * h * w))
    strong_label_fallback = (
        center_ink >= strong_label_ink_th
        and A_ink_any >= strong_label_ink_th
        and B_ink_any >= strong_label_ink_th
        and len(crossing_extras) <= 12
        and horizontal_error_deg(diameter_line["ang"]) <= 10.0
    )
    if (A_lbl is None and A_ink < label_ink_th) or (B_lbl is None and B_ink < label_ink_th):
        if not strong_label_fallback:
            return [
                False,
                f"Failed to detect endpoint labels A/B (got A={None if A_lbl is None else A_lbl.get('char')}, "
                f"B={None if B_lbl is None else B_lbl.get('char')}, label_ink=({A_ink},{B_ink})). ",
            ]

    if crossing_extras and not strong_label_fallback:
        return [False, f"Detected extra dominant crossing line(s) outside diameter AB: {len(crossing_extras)}. "]

    if center_tok is None and center_ink < max(300, int(0.00020 * h * w)):
        return [False, f"Failed to find reliable center C label evidence (label_ink={center_ink}). "]

    diameter_horizontal_error = horizontal_error_deg(diameter_line["ang"])
    if A_lbl is None and B_lbl is None and diameter_horizontal_error > 55.0:
        return [False, "Endpoint labels are ambiguous on a near-vertical diameter. "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_8,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
