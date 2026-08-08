import argparse

PID = 9
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _triangle_angle_deg(a, b, c):
    bax = float(a[0]) - float(b[0])
    bay = float(a[1]) - float(b[1])
    bcx = float(c[0]) - float(b[0])
    bcy = float(c[1]) - float(b[1])
    n1 = math.hypot(bax, bay)
    n2 = math.hypot(bcx, bcy)
    if n1 <= 1e-6 or n2 <= 1e-6:
        return 0.0
    cosv = (bax * bcx + bay * bcy) / (n1 * n2)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))


def _vertex_label_ink_score(img, point, edge_lines, min_hw):
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
    for ln in edge_lines:
        a, b, c = [float(v) for v in ln["abc"]]
        den = max(1e-6, math.hypot(a, b))
        line_band |= (
            np.abs(a * xx.astype(np.float32) + b * yy.astype(np.float32) + c) / den
            <= max(3.0, 0.012 * float(min_hw))
        )
    point_disk = ((xx - px) ** 2 + (yy - py) ** 2) <= (0.035 * float(min_hw)) ** 2
    label_mask = roi & (~line_band) & (~point_disk)
    if not bool(label_mask.any()):
        return 0

    num, _, stats, _ = cv2.connectedComponentsWithStats(label_mask.astype(np.uint8), connectivity=8)
    min_area = max(8, int(0.000005 * h * w))
    max_area = max(1200, int(0.004 * h * w))
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


def judge_plane_9(img):
    if img is None:
        return [False, "Input image is None. "]

    h, w = img.shape[:2]
    min_hw = float(min(h, w))
    lines = detect_line_segments(img, min_len_ratio=0.10)
    if len(lines) < 3:
        return [False, "Insufficient line structure for a triangle. "]

    tri = extract_polygon_from_lines(
        lines=lines,
        img_shape=img.shape,
        sides=3,
        min_len_ratio=0.20,
        top_k=8,
        min_angle_sep_deg=12.0,
        margin_ratio=0.15,
        point_tol_ratio=0.04,
        min_area_ratio=0.008,
        support_t=(-0.15, 1.10),
    )
    if tri is None:
        return [False, "Failed to reconstruct a closed triangle from detected lines. "]

    edge_lines = tri["lines"]
    verts = tri["vertices"]

    extra_th = 0.25 * min_hw
    extras = []
    for ln in lines:
        if float(ln["len"]) < extra_th:
            continue
        if any(line_equivalent(ln, ref, min_hw) for ref in edge_lines):
            continue
        extras.append(ln)
    if extras:
        return [False, f"Detected extra dominant line(s) outside triangle: {len(extras)}. "]

    best_idx = -1
    best_err = None
    for i in range(3):
        b = verts[i]
        a = verts[(i + 1) % 3]
        c = verts[(i + 2) % 3]
        ang = _triangle_angle_deg(a, b, c)
        err = abs(ang - 90.0)
        if best_err is None or err < best_err:
            best_err = err
            best_idx = i
    if best_idx < 0 or best_err is None or best_err > 14.0:
        return [False, f"No clear right angle detected (best error={0.0 if best_err is None else best_err:.1f} deg). "]

    C_geo = verts[best_idx]
    others = [verts[i] for i in range(3) if i != best_idx]

    label_radius = 0.14 * min_hw
    tokens = extract_global_letter_tokens(img, whitelist="ABC", min_conf=0.10)
    ink_scores = [_vertex_label_ink_score(img, v, edge_lines, min_hw) for v in verts]
    strong_ink_th = max(320, int(0.00006 * h * w))
    strong_vertex_label_ink = sum(score >= strong_ink_th for score in ink_scores) >= 3

    c_tok = select_token_near_point(tokens, expected_char="C", point=C_geo, max_dist=label_radius)
    if c_tok is None and not strong_vertex_label_ink:
        return [False, "Failed to detect label C at the right-angle vertex. "]

    o1_tok = select_token_near_point(tokens, expected_char=None, point=others[0], max_dist=label_radius)
    o2_tok = select_token_near_point(tokens, expected_char=None, point=others[1], max_dist=label_radius)
    o1 = None if o1_tok is None else str(o1_tok.get("char", "")).upper()
    o2 = None if o2_tok is None else str(o2_tok.get("char", "")).upper()
    if {o1, o2} != {"A", "B"} and not strong_vertex_label_ink:
        return [False, f"Failed to detect A/B on the other two vertices (got {o1}, {o2}). "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_9,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
