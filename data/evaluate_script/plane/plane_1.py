import argparse
import math

PID = 3
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _vertical_error_deg(ang):
    return angle_diff_deg(float(ang), 90.0)


def _select_main_perpendicular_pair(lines, min_hw, img_shape):
    h, w = img_shape[:2]
    long_th = 0.35 * min_hw

    h_candidates = [it for it in lines if horizontal_error_deg(it["ang"]) <= 8.0 and float(it["len"]) >= long_th]
    v_candidates = [it for it in lines if _vertical_error_deg(it["ang"]) <= 8.0 and float(it["len"]) >= long_th]
    if not h_candidates or not v_candidates:
        return None

    margin = 0.10 * min_hw
    best = None
    best_score = None
    for h_ln in h_candidates:
        for v_ln in v_candidates:
            orth_err = abs(angle_diff_deg(h_ln["ang"], v_ln["ang"]) - 90.0)
            if orth_err > 8.0:
                continue
            inter = line_intersection_from_abc(h_ln["abc"], v_ln["abc"])
            if inter is None:
                continue
            ix, iy = float(inter[0]), float(inter[1])
            if ix < -margin or ix > (w + margin) or iy < -margin or iy > (h + margin):
                continue
            t_h = segment_projection_t(h_ln["seg"], inter)
            t_v = segment_projection_t(v_ln["seg"], inter)
            if not (-0.15 <= t_h <= 1.10 and -0.15 <= t_v <= 1.10):
                continue
            score = float(h_ln["len"]) + float(v_ln["len"]) - 6.0 * orth_err
            if best_score is None or score > best_score:
                best_score = score
                best = (h_ln, v_ln, inter)
    return best


def _line_label_ink_score(img, line_item, min_hw):
    gray, bw = _gray_and_ink_mask(img)
    h, w = gray.shape[:2]
    ys, xs = np.where(bw > 0)
    if len(xs) == 0:
        return 0

    pts = np.column_stack([xs, ys]).astype(np.float32)
    a, b, c = line_item["abc"]
    denom = max(1e-6, math.hypot(float(a), float(b)))
    seg = line_item["seg"]

    dists = np.abs(a * pts[:, 0] + b * pts[:, 1] + c) / denom
    ts = np.array([segment_projection_t(seg, (float(x), float(y))) for x, y in pts], dtype=np.float32)
    near_label_band = (
        (dists >= max(4.0, 0.012 * min_hw))
        & (dists <= 0.16 * min_hw)
        & (ts >= -0.28)
        & (ts <= 1.28)
    )
    if not bool(near_label_band.any()):
        return 0

    label_mask = np.zeros_like(bw, dtype=np.uint8)
    label_mask[pts[near_label_band, 1].astype(int), pts[near_label_band, 0].astype(int)] = 255
    num, _, stats, _ = cv2.connectedComponentsWithStats(label_mask, connectivity=8)
    min_area = max(8, int(0.000006 * h * w))
    max_area = max(500, int(0.004 * h * w))
    total = 0
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw_box = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_box = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < min_area or area > max_area:
            continue
        if bw_box > 0.14 * min_hw or bh_box > 0.14 * min_hw:
            continue
        total += area
    return int(total)


def judge_plane_3(img):
    if img is None:
        return [False, "Input image is None. "]

    h, w = img.shape[:2]
    min_hw = float(min(h, w))
    lines = detect_line_segments(img, min_len_ratio=0.12)
    if len(lines) < 2:
        return [False, "Insufficient line structure. "]

    pair = _select_main_perpendicular_pair(lines, min_hw, img.shape)
    if pair is None:
        return [False, "Failed to detect valid perpendicular horizontal/vertical main lines. "]
    h_main, v_main, _ = pair

    main_th = 0.45 * min_hw
    if float(h_main["len"]) < main_th or float(v_main["len"]) < main_th:
        return [
            False,
            f"Main lines are too short (h={float(h_main['len']):.1f}, v={float(v_main['len']):.1f}). ",
        ]

    extra_th = 0.28 * min_hw
    extras = []
    for ln in lines:
        if float(ln["len"]) < extra_th:
            continue
        if line_equivalent(ln, h_main, min_hw) or line_equivalent(ln, v_main, min_hw):
            continue
        extras.append(ln)
    if extras:
        return [False, f"Detected extra dominant line(s) outside target pair: {len(extras)}. "]

    tokens = extract_global_letter_tokens(img, whitelist="FG", min_conf=0.10)
    f_tok = select_token_near_line(tokens, expected_char="F", line_item=h_main, max_perp=0.10 * min_hw, t_margin=0.25)
    g_tok = select_token_near_line(tokens, expected_char="G", line_item=v_main, max_perp=0.10 * min_hw, t_margin=0.25)
    if f_tok is None or g_tok is None:
        h_ink = _line_label_ink_score(img, h_main, min_hw)
        v_ink = _line_label_ink_score(img, v_main, min_hw)
        ink_th = max(18, int(0.000018 * h * w))
        if h_ink < ink_th or v_ink < ink_th:
            missing = "F" if f_tok is None else "G"
            return [
                False,
                (
                    f"Failed to detect label '{missing}' near the expected line "
                    f"(fallback_ink=({h_ink},{v_ink}), threshold={ink_th}). "
                ),
            ]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_3,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
