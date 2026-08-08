import argparse

PID = 18
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _dist(p, q):
    return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))


def judge_plane_18(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.10)
    if len(lines) < 4:
        return [False, "Insufficient line structure for a quadrilateral. "]

    quad = extract_polygon_from_lines(
        lines=lines,
        img_shape=img.shape,
        sides=4,
        min_len_ratio=0.18,
        top_k=10,
        min_angle_sep_deg=12.0,
        margin_ratio=0.15,
        point_tol_ratio=0.04,
        min_area_ratio=0.008,
        support_t=(-0.45, 1.35),
    )
    if quad is None:
        return [False, "Failed to reconstruct a closed quadrilateral. "]

    v = quad["vertices"]
    e = [
        {"p1": v[0], "p2": v[1], "ang": segment_angle_deg((v[0][0], v[0][1], v[1][0], v[1][1])), "len": _dist(v[0], v[1])},
        {"p1": v[1], "p2": v[2], "ang": segment_angle_deg((v[1][0], v[1][1], v[2][0], v[2][1])), "len": _dist(v[1], v[2])},
        {"p1": v[2], "p2": v[3], "ang": segment_angle_deg((v[2][0], v[2][1], v[3][0], v[3][1])), "len": _dist(v[2], v[3])},
        {"p1": v[3], "p2": v[0], "ang": segment_angle_deg((v[3][0], v[3][1], v[0][0], v[0][1])), "len": _dist(v[3], v[0])},
    ]

    pair_a_err = angle_diff_deg(e[0]["ang"], e[2]["ang"])
    pair_b_err = angle_diff_deg(e[1]["ang"], e[3]["ang"])
    if pair_a_err <= pair_b_err:
        top_bottom_pair = (e[0], e[2])
        non_par_err = pair_b_err
        par_err = pair_a_err
    else:
        top_bottom_pair = (e[1], e[3])
        non_par_err = pair_a_err
        par_err = pair_b_err

    if par_err > 10.0:
        return [False, f"No clear pair of parallel opposite sides for trapezoid (err={par_err:.1f} deg). "]
    if non_par_err <= 9.0:
        return [False, "Both opposite-side pairs look parallel (parallelogram-like, not trapezoid). "]

    s1, s2 = top_bottom_pair
    y1 = 0.5 * (float(s1["p1"][1]) + float(s1["p2"][1]))
    y2 = 0.5 * (float(s2["p1"][1]) + float(s2["p2"][1]))
    top, bottom = (s1, s2) if y1 <= y2 else (s2, s1)

    if float(bottom["len"]) <= float(top["len"]) + scale_px(min_hw, 0.05, floor_px=0.0):
        return [False, f"Bottom side is not longer than top side (top={top['len']:.1f}, bottom={bottom['len']:.1f}). "]

    extra_th = scale_px(min_hw, 0.22, floor_px=0.0)
    extras = []
    for ln in lines:
        if float(ln["len"]) < extra_th:
            continue
        if any(line_equivalent(ln, ref, min_hw) for ref in quad["lines"]):
            continue
        extras.append(ln)
    if len(extras) > 4:
        return [False, f"Detected too many extra dominant line(s) outside trapezoid: {len(extras)}. "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_18,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
