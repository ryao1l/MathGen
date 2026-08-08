import argparse

PID = 7
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _line_anchor_samples(line_item):
    left, right = segment_endpoints_lr(line_item["seg"])
    out = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        out.append(
            (
                left[0] + (right[0] - left[0]) * t,
                left[1] + (right[1] - left[1]) * t,
            )
        )
    return out


def judge_plane_7(img):
    if img is None:
        return [False, "Input image is None. "]

    h, w = img.shape[:2]
    min_hw = float(min(h, w))

    def _line_label_stats(line_item):
        _, bw = _gray_and_ink_mask(img)
        left, right = segment_endpoints_lr(line_item["seg"])
        y_line = 0.5 * (float(left[1]) + float(right[1]))
        x1 = int(max(0, min(left[0], right[0]) - 0.12 * min_hw))
        x2 = int(min(w, max(left[0], right[0]) + 0.12 * min_hw))
        y1 = int(max(0, y_line - 0.14 * min_hw))
        y2 = int(min(h, y_line + 0.14 * min_hw))
        if x2 <= x1 or y2 <= y1:
            return 0
        roi = bw[y1:y2, x1:x2] > 0
        yy, _ = np.ogrid[y1:y2, x1:x2]
        line_band = np.abs(yy.astype(np.float32) - y_line) <= max(3.0, 0.014 * min_hw)
        label_mask = roi & (~line_band)
        if int(label_mask.sum()) <= 0:
            return 0, 0
        num, _, stats, _ = cv2.connectedComponentsWithStats(label_mask.astype(np.uint8), connectivity=8)
        min_area = max(10, int(0.000006 * h * w))
        max_area = max(400, int(0.0040 * h * w))
        areas = []
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            bw_box = int(stats[i, cv2.CC_STAT_WIDTH])
            bh_box = int(stats[i, cv2.CC_STAT_HEIGHT])
            if area < min_area or area > max_area:
                continue
            if bw_box > 0.16 * min_hw or bh_box > 0.16 * min_hw:
                continue
            areas.append(area)
        return (max(areas), len(areas)) if areas else (0, 0)

    lines = detect_line_segments(img, min_len_ratio=0.10)
    if len(lines) < 2:
        return [False, "Insufficient line structure. "]

    long_th = 0.35 * min_hw
    h_candidates = [it for it in lines if horizontal_error_deg(it["ang"]) <= 8.0 and float(it["len"]) >= long_th]
    if len(h_candidates) < 2:
        return [False, "Failed to detect two long horizontal lines. "]

    h_candidates = sorted(h_candidates, key=lambda t: float(t["len"]), reverse=True)[:6]
    best = None
    best_score = None
    for i in range(len(h_candidates)):
        for j in range(i + 1, len(h_candidates)):
            li = h_candidates[i]
            lj = h_candidates[j]
            if angle_diff_deg(li["ang"], lj["ang"]) > 5.0:
                continue
            yi = 0.5 * (float(li["seg"][1]) + float(li["seg"][3]))
            yj = 0.5 * (float(lj["seg"][1]) + float(lj["seg"][3]))
            sep = abs(yi - yj)
            if sep < 0.055 * min_hw:
                continue
            if sep > 0.38 * min_hw:
                continue
            top_probe, bottom_probe = (li, lj) if yi <= yj else (lj, li)
            top_ink_probe, top_parts_probe = _line_label_stats(top_probe)
            bottom_ink_probe, bottom_parts_probe = _line_label_stats(bottom_probe)
            label_bonus = 0.0
            for ink_probe, parts_probe in ((top_ink_probe, top_parts_probe), (bottom_ink_probe, bottom_parts_probe)):
                if 1 <= parts_probe <= 6 and ink_probe >= max(30, int(0.000008 * h * w)):
                    label_bonus += 0.35 * min_hw
                elif parts_probe > 20:
                    label_bonus -= 0.50 * min_hw
            score = float(li["len"]) + float(lj["len"]) - 0.60 * sep + label_bonus
            if best_score is None or score > best_score:
                best_score = score
                best = (li, lj)
    if best is None:
        return [False, "Failed to find two separated parallel horizontal lines. "]

    l1, l2 = best
    y1 = 0.5 * (float(l1["seg"][1]) + float(l1["seg"][3]))
    y2 = 0.5 * (float(l2["seg"][1]) + float(l2["seg"][3]))
    top_line, bottom_line = (l1, l2) if y1 <= y2 else (l2, l1)

    extra_th = 0.28 * min_hw
    extras = []
    for ln in lines:
        if float(ln["len"]) < extra_th:
            continue
        if line_equivalent(ln, top_line, min_hw) or line_equivalent(ln, bottom_line, min_hw):
            continue
        angle_delta = min(angle_diff_deg(ln["ang"], top_line["ang"]), angle_diff_deg(ln["ang"], bottom_line["ang"]))
        if float(ln["len"]) >= 0.45 * min_hw and angle_delta >= 30.0:
            return [False, "Detected a long non-horizontal structure outside the two parallel lines. "]
        extras.append(ln)
    if len(extras) > 8:
        return [False, f"Detected extra dominant line(s) outside target pair: {len(extras)}. "]

    sample_min = max(1, int(round(0.15 * min_hw)))
    gap_top = middle_gap_px_on_line(img, top_line, t_lo=0.30, t_hi=0.70, sample_min=sample_min)
    gap_bottom = middle_gap_px_on_line(img, bottom_line, t_lo=0.30, t_hi=0.70, sample_min=sample_min)
    gap_th = 0.035 * min_hw
    if gap_top > gap_th or gap_bottom > gap_th:
        return [False, f"Detected broken/non-straight line structure (gap={max(gap_top, gap_bottom):.1f}px). "]

    tokens = extract_global_letter_tokens(img, whitelist="TB", min_conf=0.10)
    max_perp = 0.10 * min_hw
    lbl_t = select_token_near_line(tokens, expected_char="T", line_item=top_line, max_perp=max_perp, t_margin=0.30)
    lbl_b = select_token_near_line(tokens, expected_char="B", line_item=bottom_line, max_perp=max_perp, t_margin=0.30)
    top_ink, top_label_parts = _line_label_stats(top_line)
    bottom_ink, bottom_label_parts = _line_label_stats(bottom_line)
    label_ink_th = max(30, int(0.000008 * h * w))
    if lbl_t is None and top_ink < label_ink_th:
        return [False, f"Failed to detect label 't' near the top line (label_ink={top_ink}). "]
    if lbl_b is None and bottom_ink < label_ink_th:
        return [False, f"Failed to detect label 'b' near the bottom line (label_ink={bottom_ink}). "]

    top_span = abs(float(top_line["seg"][2]) - float(top_line["seg"][0]))
    bottom_span = abs(float(bottom_line["seg"][2]) - float(bottom_line["seg"][0]))
    if (
        lbl_t is None
        and lbl_b is None
        and len(h_candidates) >= 3
        and top_span > 0.94 * w
        and bottom_span > 0.94 * w
    ):
        return [False, "Detected full-width separator lines without reliable T/B labels. "]

    anchors = _line_anchor_samples(top_line) + _line_anchor_samples(bottom_line)
    violated, info = has_excess_outside_ink(
        img=img,
        allowed_lines=[top_line, bottom_line],
        anchor_points=anchors,
        max_outside_ratio=0.10,
        max_outside_px_ratio=0.00005,
        max_outside_px_floor=0,
    )
    if violated and lbl_t is None and lbl_b is None and float(info.get("outside_ratio", 0.0)) > 0.30:
        return [
            False,
            (
                "Detected substantial ambiguous label/content clutter around the target lines "
                f"(outside={info['outside_px']}, ratio={info['outside_ratio']:.3f}). "
            ),
        ]
    if violated and float(info.get("outside_ratio", 0.0)) > 0.88:
        return [
            False,
            (
                "Detected extra drawing content outside target parallel lines "
                f"(outside={info['outside_px']}, ratio={info['outside_ratio']:.3f}). "
            ),
        ]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_7,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
