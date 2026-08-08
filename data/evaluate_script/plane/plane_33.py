import argparse

PID = 6
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def judge_plane_6(img):
    if img is None:
        return [False, "Input image is None. "]

    h, w = img.shape[:2]
    min_hw = float(min(h, w))
    if img.ndim == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturated = (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 80)
        if float(saturated.sum()) / float(h * w) > 0.025:
            return [False, "Detected substantial colored non-AB content in the drawing. "]

    lines = detect_line_segments(img, min_len_ratio=0.10)
    if not lines:
        return [False, "Failed to detect line structure. "]

    long_th = 0.40 * min_hw
    h_candidates = [it for it in lines if horizontal_error_deg(it["ang"]) <= 8.0 and float(it["len"]) >= long_th]
    if not h_candidates:
        return [False, "Failed to detect a sufficiently long horizontal segment. "]

    main = max(h_candidates, key=lambda t: float(t["len"]))
    seg = main["seg"]
    left_pt, right_pt = segment_endpoints_lr(seg)

    def _endpoint_label_ink(point, side):
        _, bw = _gray_and_ink_mask(img)
        px, py = float(point[0]), float(point[1])
        x1 = int(max(0, px - 0.16 * min_hw))
        x2 = int(min(w, px + 0.16 * min_hw))
        y1 = int(max(0, py - 0.16 * min_hw))
        y2 = int(min(h, py + 0.16 * min_hw))
        if x2 <= x1 or y2 <= y1:
            return 0
        roi = (bw[y1:y2, x1:x2] > 0)
        yy, xx = np.ogrid[y1:y2, x1:x2]
        line_band = np.abs(yy.astype(np.float32) - py) <= max(3.0, 0.018 * min_hw)
        endpoint_dot = ((xx.astype(np.float32) - px) ** 2 + (yy.astype(np.float32) - py) ** 2) <= (0.035 * min_hw) ** 2
        label_mask = roi & (~line_band) & (~endpoint_dot)
        if int(label_mask.sum()) <= 0:
            return 0
        num, _, stats, _ = cv2.connectedComponentsWithStats(label_mask.astype(np.uint8), connectivity=8)
        min_area = max(10, int(0.000008 * h * w))
        max_area = max(500, int(0.0045 * h * w))
        areas = []
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            bw_box = int(stats[i, cv2.CC_STAT_WIDTH])
            bh_box = int(stats[i, cv2.CC_STAT_HEIGHT])
            if area < min_area or area > max_area:
                continue
            if bw_box > 0.13 * min_hw or bh_box > 0.13 * min_hw:
                continue
            areas.append(area)
        if not areas:
            return 0
        return max(areas)

    extra_th = 0.25 * min_hw
    extras = []
    for ln in lines:
        if float(ln["len"]) < extra_th:
            continue
        if line_equivalent(ln, main, min_hw):
            continue
        angle_delta = abs(((float(ln["ang"]) - float(main["ang"]) + 90.0) % 180.0) - 90.0)
        if float(ln["len"]) >= 0.48 * min_hw and angle_delta >= 35.0:
            return [False, "Detected a long crossing/non-horizontal line in the drawing. "]
        extras.append(ln)
    if len(extras) > 12:
        return [False, f"Detected extra dominant line(s) outside AB segment: {len(extras)}. "]

    sample_min = max(1, int(round(0.12 * min_hw)))
    mid_gap_px = middle_gap_px_on_segment(img, seg, t_lo=0.35, t_hi=0.65, sample_min=sample_min)
    mid_gap_th = 0.025 * min_hw
    if mid_gap_px > mid_gap_th:
        return [False, f"Detected a visible break in the middle of AB segment (gap={mid_gap_px:.1f}px). "]

    tokens = extract_global_letter_tokens(img, whitelist="AB", min_conf=0.10)
    max_dist = 0.14 * min_hw
    A_lbl = select_token_near_point(tokens, expected_char="A", point=left_pt, max_dist=max_dist)
    B_lbl = select_token_near_point(tokens, expected_char="B", point=right_pt, max_dist=max_dist)
    left_ink = _endpoint_label_ink(left_pt, "left")
    right_ink = _endpoint_label_ink(right_pt, "right")
    ink_th = max(60, int(0.000008 * h * w))
    left_ok = A_lbl is not None or left_ink >= ink_th
    right_ok = B_lbl is not None or right_ink >= ink_th
    if not (left_ok and right_ok):
        return [
            False,
            f"Failed to detect endpoint labels A/B (got A={None if A_lbl is None else A_lbl.get('char')}, "
            f"B={None if B_lbl is None else B_lbl.get('char')}, endpoint_ink=({left_ink},{right_ink})). ",
        ]

    violated, info = has_excess_outside_ink(
        img=img,
        allowed_lines=[main],
        anchor_points=[left_pt, right_pt],
        max_outside_ratio=0.05,
        max_outside_px_ratio=0.00003,
        max_outside_px_floor=0,
    )
    if violated and float(info.get("outside_ratio", 0.0)) > 0.90:
        return [
            False,
            (
                "Detected extra drawing content outside target segment AB "
                f"(outside={info['outside_px']}, ratio={info['outside_ratio']:.3f}). "
            ),
        ]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_6,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
