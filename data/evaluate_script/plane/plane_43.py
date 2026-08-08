import argparse

PID = 17
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def judge_plane_17(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    circle = detect_largest_circle(img)
    if circle is None:
        return [False, "Failed to detect the main circle. "]
    cx, cy, r = [float(v) for v in circle]
    min_radius = scale_px(min_hw, 0.10, floor_px=0.0)
    if r < min_radius:
        return [False, f"Detected circle is too small (r={r:.1f}). "]

    lines = detect_line_segments(img, min_len_ratio=0.10)
    if not lines:
        return [False, "Failed to detect line structure. "]

    long_th = scale_px(min_hw, 0.12, floor_px=0.0)
    candidates = [it for it in lines if float(it["len"]) >= long_th]
    if not candidates:
        return [False, "No sufficiently long tangent candidate detected. "]

    tang_tol = 0.06 * r
    valid = []
    for ln in candidates:
        d = point_line_distance((cx, cy), ln["abc"])
        err = abs(d - r)
        if err > tang_tol:
            continue
        foot = project_point_to_line((cx, cy), ln["abc"])
        t = segment_projection_t(ln["seg"], foot)
        if not (-0.15 <= t <= 1.10):
            continue
        inters = circle_line_intersections(circle, ln["abc"])
        if len(inters) >= 2:
            sep = math.hypot(float(inters[0][0]) - float(inters[1][0]), float(inters[0][1]) - float(inters[1][1]))
            if sep > 0.40 * r:
                continue
        valid.append((ln, foot, err))

    if not valid:
        return [False, "Failed to detect a valid tangent line touching circle at one point. "]

    max_len = max(float(it[0]["len"]) for it in valid)
    dominant = [it for it in valid if float(it[0]["len"]) >= 0.60 * max_len]
    tangent_line, T_geo, _ = min(dominant, key=lambda it: (float(it[2]), -float(it[0]["len"])))

    extra_th = scale_px(min_hw, 0.28, floor_px=0.0)
    extras = []
    for ln in lines:
        if float(ln["len"]) < extra_th:
            continue
        if line_equivalent(ln, tangent_line, min_hw):
            continue
        extras.append(ln)
    if extras:
        return [False, f"Detected extra dominant line(s) outside tangent structure: {len(extras)}. "]

    tokens = extract_global_letter_tokens(img, whitelist="T", min_conf=0.10)
    T_lbl = select_token_near_point(
        tokens,
        expected_char="T",
        point=T_geo,
        max_dist=scale_px(min_hw, 0.16, floor_px=0.0),
    )
    if T_lbl is None:
        return [False, "Failed to detect label T near tangent point. "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_17,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
