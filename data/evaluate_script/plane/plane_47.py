import argparse

PID = 23
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _dist(p, q):
    return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))


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


def _segment_item(p, q):
    seg = (float(p[0]), float(p[1]), float(q[0]), float(q[1]))
    abc = segment_to_abc(seg)
    if abc is None:
        raise RuntimeError("Failed to build line from segment points.")
    return {
        "seg": seg,
        "abc": abc,
        "ang": segment_angle_deg(seg),
        "len": float(_dist(p, q)),
    }


def _perp_err_deg(seg_a, seg_b):
    d = angle_diff_deg(segment_angle_deg(seg_a), segment_angle_deg(seg_b))
    if d > 90.0:
        d = 180.0 - d
    return abs(d - 90.0)


def judge_plane_23(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.08)
    if len(lines) < 4:
        return [False, "Insufficient line structure for hard case 23. "]

    tokens = extract_global_letter_tokens(img, whitelist="ABCDM", min_conf=0.10)
    best_by_char = pick_best_tokens_by_char(tokens, ["A", "B", "C", "D", "M"], min_conf=0.10)
    missing = [ch for ch in ["A", "B", "C", "D", "M"] if ch not in best_by_char]
    if missing:
        return [False, f"Missing labels from OCR: {','.join(missing)}. "]

    tri_cands = extract_triangle_candidates_from_lines(
        lines=lines,
        img_shape=img.shape,
        min_len_ratio=0.18,
        top_k=12,
        min_angle_sep_deg=10.0,
        margin_ratio=0.22,
        point_tol_ratio=0.045,
        min_area_ratio=0.008,
        support_t=(-0.75, 1.35),
    )
    if not tri_cands:
        return [False, "Failed to reconstruct triangle candidates from detected lines. "]

    best_fail_stage = -1
    best_fail_reason = "No candidate triangle satisfied all constraints."

    def _record_fail(stage, reason):
        nonlocal best_fail_stage, best_fail_reason
        if stage > best_fail_stage:
            best_fail_stage = int(stage)
            best_fail_reason = str(reason)

    for cand in tri_cands:
        verts = cand["vertices"]
        assign, dists = assign_labels_to_vertices_min_cost(best_by_char, verts, ["A", "B", "C"])
        if assign is None or dists is None:
            _record_fail(1, "Failed to assign A/B/C to triangle vertices.")
            continue

        a_geo = verts[assign["A"]]
        b_geo = verts[assign["B"]]
        c_geo = verts[assign["C"]]

        abc_tol = scale_px(min_hw, 0.10, floor_px=0.0)
        if any(float(dists[ch]) > abc_tol for ch in ["A", "B", "C"]):
            _record_fail(2, "A/B/C labels are too far from triangle vertices.")
            continue

        ang_c = _triangle_angle_deg(a_geo, c_geo, b_geo)
        if abs(ang_c - 90.0) > 16.0:
            _record_fail(3, f"Right-angle at C not satisfied (angle={ang_c:.1f}). ")
            continue

        top_gap = scale_px(min_hw, 0.02, floor_px=0.0)
        side_gap = scale_px(min_hw, 0.07, floor_px=0.0)
        if not (float(a_geo[1]) < float(c_geo[1]) - top_gap):
            _record_fail(3, "A is not above C as required.")
            continue
        if not (float(a_geo[0]) <= float(c_geo[0]) + side_gap):
            _record_fail(3, "A is not at the left side relative to C.")
            continue
        if not (float(b_geo[0]) >= float(c_geo[0]) + side_gap):
            _record_fail(3, "B is not to the right of C.")
            continue
        if not (float(b_geo[1]) >= float(c_geo[1]) - side_gap):
            _record_fail(3, "B is not at the lower side relative to C.")
            continue

        ok_ab, r_ab = has_segment_between_points(img, a_geo, b_geo, ratio_th=0.14, thickness=2, trim_ratio=0.02)
        ok_bc, r_bc = has_segment_between_points(img, b_geo, c_geo, ratio_th=0.14, thickness=2, trim_ratio=0.02)
        ok_ac, r_ac = has_segment_between_points(img, a_geo, c_geo, ratio_th=0.14, thickness=2, trim_ratio=0.02)
        if not (ok_ab and ok_bc and ok_ac):
            _record_fail(4, f"Triangle boundary incomplete (AB={r_ab:.2f}, BC={r_bc:.2f}, AC={r_ac:.2f}). ")
            continue

        ab = _segment_item(a_geo, b_geo)

        d_tok = select_token_near_line(
            tokens,
            expected_char="D",
            line_item=ab,
            max_perp=scale_px(min_hw, 0.08, floor_px=0.0),
            t_margin=0.20,
        )
        if d_tok is None:
            _record_fail(5, "Failed to detect D on hypotenuse AB.")
            continue

        d_lbl = (float(d_tok["center"][0]), float(d_tok["center"][1]))
        d_proj = project_point_to_line(c_geo, ab["abc"])
        d_label_tol = scale_px(min_hw, 0.12, floor_px=0.0)

        d_candidates = [d_lbl]
        if d_proj is not None:
            d_proj = (float(d_proj[0]), float(d_proj[1]))
            if _dist(d_proj, d_lbl) <= d_label_tol:
                d_candidates.append(d_proj)

        d_ok = False
        best_cd_ratio = -1.0
        best_cd_reason = ""
        for d_geo in d_candidates:
            t_d = segment_projection_t(ab["seg"], d_geo)
            if not (-0.08 <= t_d <= 1.08):
                continue
            ok_cd, r_cd = has_segment_between_points(img, c_geo, d_geo, ratio_th=0.12, thickness=2, trim_ratio=0.02)
            pe = _perp_err_deg((c_geo[0], c_geo[1], d_geo[0], d_geo[1]), ab["seg"])
            best_cd_ratio = max(best_cd_ratio, float(r_cd))
            if ok_cd and pe <= 14.0:
                d_ok = True
                break
            best_cd_reason = f"CD check failed (ratio={r_cd:.2f}, perp_err={pe:.1f}). "

        if not d_ok:
            _record_fail(6, best_cd_reason if best_cd_reason else "Failed CD/altitude constraints.")
            continue

        m_expect = ((float(b_geo[0]) + float(c_geo[0])) * 0.5, (float(b_geo[1]) + float(c_geo[1])) * 0.5)
        m_tok = select_token_near_point(
            tokens,
            expected_char="M",
            point=m_expect,
            max_dist=scale_px(min_hw, 0.10, floor_px=0.0),
        )
        if m_tok is None:
            _record_fail(7, "Failed to detect midpoint label M on BC.")
            continue
        m_geo = (float(m_tok["center"][0]), float(m_tok["center"][1]))

        d_mb = _dist(m_geo, b_geo)
        d_mc = _dist(m_geo, c_geo)
        rel_mid = abs(d_mb - d_mc) / max(1e-6, 0.5 * (d_mb + d_mc))
        if rel_mid > 0.22:
            _record_fail(8, f"M is not close enough to midpoint of BC (rel_mid={rel_mid:.3f}). ")
            continue

        return [True, ""]

    return [False, best_fail_reason]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_23,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
