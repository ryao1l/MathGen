import argparse

PID = 16
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _axis_aligned_like(line_item):
    return min(horizontal_error_deg(line_item["ang"]), abs(angle_diff_deg(line_item["ang"], 90.0))) <= 8.0


def _intersection_label_ink_score(img, point, main_lines, min_hw):
    h, w = img.shape[:2]
    px, py = float(point[0]), float(point[1])
    _, bw = _gray_and_ink_mask(img)
    win = 0.18 * float(min_hw)
    x1 = int(max(0, px - win))
    x2 = int(min(w, px + win))
    y1 = int(max(0, py - win))
    y2 = int(min(h, py + win))
    if x2 <= x1 or y2 <= y1:
        return 0

    roi = bw[y1:y2, x1:x2] > 0
    yy, xx = np.ogrid[y1:y2, x1:x2]
    line_band = np.zeros_like(roi, dtype=bool)
    for ln in main_lines:
        a, b, c = [float(v) for v in ln["abc"]]
        den = max(1e-6, math.hypot(a, b))
        line_band |= (
            np.abs(a * xx.astype(np.float32) + b * yy.astype(np.float32) + c) / den
            <= max(3.0, 0.014 * float(min_hw))
        )
    center_disk = ((xx - px) ** 2 + (yy - py) ** 2) <= (0.035 * float(min_hw)) ** 2
    label_mask = roi & (~line_band) & (~center_disk)
    if not bool(label_mask.any()):
        return 0

    num, _, stats, _ = cv2.connectedComponentsWithStats(label_mask.astype(np.uint8), connectivity=8)
    min_area = max(8, int(0.000005 * h * w))
    max_area = max(900, int(0.004 * h * w))
    areas = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw_box = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_box = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < min_area or area > max_area:
            continue
        if bw_box > 0.18 * float(min_hw) or bh_box > 0.18 * float(min_hw):
            continue
        areas.append(area)
    return max(areas) if areas else 0


def judge_plane_16(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.12)
    if len(lines) < 2:
        return [False, "Insufficient line structure. "]

    long_th = scale_px(min_hw, 0.38, floor_px=0.0)
    candidates = [it for it in lines if float(it["len"]) >= long_th]
    if len(candidates) < 2:
        return [False, "Failed to detect two sufficiently long lines. "]

    best = None
    best_score = None
    margin = scale_px(min_hw, 0.12, floor_px=0.0)
    h, w = img.shape[:2]
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            li = candidates[i]
            lj = candidates[j]
            if angle_diff_deg(li["ang"], lj["ang"]) < 15.0:
                continue
            p = line_intersection_from_abc(li["abc"], lj["abc"])
            if p is None:
                continue
            x, y = float(p[0]), float(p[1])
            if x < -margin or x > (w + margin) or y < -margin or y > (h + margin):
                continue
            ti = segment_projection_t(li["seg"], p)
            tj = segment_projection_t(lj["seg"], p)
            if not (-0.12 <= ti <= 1.12 and -0.12 <= tj <= 1.12):
                continue
            score = float(li["len"]) + float(lj["len"]) + 0.5 * angle_diff_deg(li["ang"], lj["ang"])
            if best_score is None or score > best_score:
                best_score = score
                best = (li, lj, p)

    if best is None:
        return [False, "Failed to detect a valid pair of intersecting lines. "]
    l1, l2, P_geo = best

    extra_th = scale_px(min_hw, 0.30, floor_px=0.0)
    extras = []
    for ln in lines:
        if float(ln["len"]) < extra_th:
            continue
        if line_equivalent(ln, l1, min_hw) or line_equivalent(ln, l2, min_hw):
            continue
        extras.append(ln)
    if len(extras) > 6:
        return [False, f"Detected extra dominant line(s) outside target pair: {len(extras)}. "]

    tokens = extract_global_letter_tokens(img, whitelist="P", min_conf=0.10)
    P_lbl = select_token_near_point(
        tokens,
        expected_char="P",
        point=P_geo,
        max_dist=scale_px(min_hw, 0.16, floor_px=0.0),
    )
    if P_lbl is None and extras:
        return [False, "Failed to detect label P near the line intersection. "]
    if P_lbl is None:
        label_ink = _intersection_label_ink_score(img, P_geo, [l1, l2], min_hw)
        label_ink_th = max(120, int(0.00008 * img.shape[0] * img.shape[1]))
        if (
            label_ink < label_ink_th
            or (_axis_aligned_like(l1) and _axis_aligned_like(l2))
        ):
            return [False, "Failed to detect label P near the line intersection. "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_16,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
