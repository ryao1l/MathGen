import argparse

PID = 30
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _dist(p, q):
    return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))


def _circle_ink_coverage(img, center, radius, band_px):
    cx, cy = float(center[0]), float(center[1])
    r = float(radius)
    if r <= 1.0:
        return 0.0, 0.0

    _, bw = _gray_and_ink_mask(img)
    h, w = bw.shape[:2]
    n_full = max(8, int(round(2.0 * math.pi * r / 3.0)))
    hit = 0
    vis = 0

    for i in range(n_full):
        th = (2.0 * math.pi * float(i)) / float(n_full)
        x = int(round(cx + r * math.cos(th)))
        y = int(round(cy + r * math.sin(th)))
        if not (0 <= x < w and 0 <= y < h):
            continue
        vis += 1
        ok = False
        for dr in range(-int(band_px), int(band_px) + 1):
            rr = r + float(dr)
            xx = int(round(cx + rr * math.cos(th)))
            yy = int(round(cy + rr * math.sin(th)))
            if 0 <= xx < w and 0 <= yy < h and bw[yy, xx] > 0:
                ok = True
                break
        if ok:
            hit += 1

    cov = float(hit) / float(max(1, vis))
    vis_ratio = float(vis) / float(max(1, n_full))
    return cov, vis_ratio


def _pin_to_marker(p, markers, tol):
    if not markers:
        return (float(p[0]), float(p[1]))
    q = nearest_point(markers, p)
    if q is None:
        return (float(p[0]), float(p[1]))
    if _dist(q, p) <= float(tol):
        return (float(q[0]), float(q[1]))
    return (float(p[0]), float(p[1]))


def _project_to_circle(point, center, radius):
    px, py = float(point[0]), float(point[1])
    cx, cy = float(center[0]), float(center[1])
    r = float(radius)
    vx = px - cx
    vy = py - cy
    norm = math.hypot(vx, vy)
    if norm <= 1e-6:
        return (cx, cy - r)
    return (cx + r * vx / norm, cy + r * vy / norm)


def _radius_present(img, lines, center, endpoint, min_hw):
    ok_seg, ratio = has_segment_between_points(
        img,
        center,
        endpoint,
        ratio_th=0.10,
        thickness=2,
        trim_ratio=0.02,
    )
    if ok_seg:
        return True, float(ratio)
    ok_line = has_support_line(
        lines,
        center,
        endpoint,
        min_hw=min_hw,
        ang_tol_deg=10.0,
        dist_ratio=0.05,
        dist_floor_px=0.0,
    )
    return bool(ok_line), float(ratio)


def _labels_free_geometry_ok(img, outer, inner, min_hw):
    ox, oy, ro = [float(v) for v in outer]
    ix, iy, ri = [float(v) for v in inner]

    if ro <= scale_px(min_hw, 0.16, floor_px=0.0):
        return False
    if ri <= scale_px(min_hw, 0.05, floor_px=0.0):
        return False
    if math.hypot(ox - ix, oy - iy) > scale_px(min_hw, 0.06, floor_px=0.0):
        return False
    if ri >= 0.72 * ro:
        return False

    band = int(max(1, round(scale_px(min_hw, 0.012, floor_px=0.0))))
    cov_o, vis_o = _circle_ink_coverage(img, (ox, oy), ro, band)
    cov_i, vis_i = _circle_ink_coverage(img, (ox, oy), ri, band)
    if vis_o < 0.80 or vis_i < 0.80:
        return False
    if cov_o < 0.55 or cov_i < 0.45:
        return False

    lines = detect_line_segments(img, min_len_ratio=0.07)
    if len(lines) < 6:
        return False

    triangles = extract_triangle_candidates_from_lines(
        lines=lines,
        img_shape=img.shape,
        min_len_ratio=0.10,
        top_k=14,
        min_angle_sep_deg=10.0,
        margin_ratio=0.18,
        point_tol_ratio=0.045,
        min_area_ratio=0.006,
        support_t=(-0.45, 1.30),
    )
    if not triangles:
        return False

    center = (ox, oy)
    on_tol = scale_px(min_hw, 0.08, floor_px=0.0)
    for tri in triangles:
        verts = tri.get("vertices")
        if not (isinstance(verts, list) and len(verts) == 3):
            continue
        if any(abs(_dist(v, center) - ro) > on_tol for v in verts):
            continue

        sides = [_dist(verts[i], verts[(i + 1) % 3]) for i in range(3)]
        mean_side = sum(sides) / 3.0
        rel = max(abs(s - mean_side) for s in sides) / max(1e-6, mean_side)
        if rel > 0.18:
            continue

        if not all(
            has_support_line(
                lines,
                verts[i],
                verts[(i + 1) % 3],
                min_hw=min_hw,
                ang_tol_deg=12.0,
                dist_ratio=0.08,
            )
            for i in range(3)
        ):
            continue

        if not all(
            _radius_present(img, lines, center, _project_to_circle(v, center, ro), min_hw)[0]
            for v in verts
        ):
            continue

        side_lines = [
            segment_to_abc((verts[i][0], verts[i][1], verts[(i + 1) % 3][0], verts[(i + 1) % 3][1]))
            for i in range(3)
        ]
        if any(line is None for line in side_lines):
            continue

        dmin = min(point_line_distance(center, line) for line in side_lines)
        if ri > dmin + scale_px(min_hw, 0.03, floor_px=0.0):
            continue
        inscribe_tol = max(0.06 * dmin, scale_px(min_hw, 0.015, floor_px=0.0))
        if abs(ri - dmin) > inscribe_tol:
            continue
        return True

    return False


def judge_plane_30(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))

    outer = detect_circle(
        img,
        order=1,
        min_r=int(round(scale_px(min_hw, 0.10, floor_px=1.0))),
        max_r=0,
    )
    inner = detect_circle(
        img,
        order=2,
        min_r=int(round(scale_px(min_hw, 0.05, floor_px=1.0))),
        max_r=0,
    )
    if outer is None or inner is None:
        return [False, "Failed to detect two circles (outer and inner). "]

    ox, oy, ro = [float(v) for v in outer]
    ix, iy, ri = [float(v) for v in inner]

    if ro <= scale_px(min_hw, 0.16, floor_px=0.0):
        return [False, f"Outer circle is too small (r={ro:.1f}). "]
    if ri <= scale_px(min_hw, 0.05, floor_px=0.0):
        return [False, f"Inner circle is too small (r={ri:.1f}). "]

    cdist = math.hypot(ox - ix, oy - iy)
    if cdist > scale_px(min_hw, 0.06, floor_px=0.0):
        return [False, f"Inner and outer circles are not concentric enough (center_dist={cdist:.1f}). "]

    if ri >= 0.72 * ro:
        return [False, f"Inner circle is not clearly smaller than outer circle (ri={ri:.1f}, ro={ro:.1f}). "]

    band = int(max(1, round(scale_px(min_hw, 0.012, floor_px=0.0))))
    cov_o, vis_o = _circle_ink_coverage(img, (ox, oy), ro, band)
    cov_i, vis_i = _circle_ink_coverage(img, (ox, oy), ri, band)
    if vis_o < 0.80 or vis_i < 0.80:
        return [False, "Circle visibility is too low for verification. "]
    if cov_o < 0.55 or cov_i < 0.45:
        return [False, f"Missing outer/inner circle traces (outer={cov_o:.2f}, inner={cov_i:.2f}). "]

    tokens = extract_global_letter_tokens(img, whitelist="ABCH", min_conf=0.10)
    best = pick_best_tokens_by_char(tokens, ["A", "B", "C", "H"], min_conf=0.10)
    missing = [ch for ch in ["A", "B", "C", "H"] if ch not in best]
    if missing:
        if _labels_free_geometry_ok(img, outer, inner, min_hw):
            return [True, ""]
        return [False, f"Missing labels from OCR: {','.join(missing)}. "]

    markers = detect_marker_points(img)
    pin_tol = scale_px(min_hw, 0.10, floor_px=0.0)
    a_geo = _pin_to_marker(best["A"]["center"], markers, pin_tol)
    b_geo = _pin_to_marker(best["B"]["center"], markers, pin_tol)
    c_geo = _pin_to_marker(best["C"]["center"], markers, pin_tol)
    h_lbl = (float(best["H"]["center"][0]), float(best["H"]["center"][1]))

    center = (ox, oy)
    on_tol = scale_px(min_hw, 0.07, floor_px=0.0)
    if abs(_dist(a_geo, center) - ro) > on_tol:
        return [False, "A is not on the outer circle. "]
    if abs(_dist(b_geo, center) - ro) > on_tol:
        return [False, "B is not on the outer circle. "]
    if abs(_dist(c_geo, center) - ro) > on_tol:
        return [False, "C is not on the outer circle. "]

    s_ab = _dist(a_geo, b_geo)
    s_bc = _dist(b_geo, c_geo)
    s_ca = _dist(c_geo, a_geo)
    s_mean = (s_ab + s_bc + s_ca) / 3.0
    rel = max(abs(s_ab - s_mean), abs(s_bc - s_mean), abs(s_ca - s_mean)) / max(1e-6, s_mean)
    if rel > 0.16:
        return [False, f"ABC is not equilateral enough (rel_dev={rel:.3f}). "]

    lines = detect_line_segments(img, min_len_ratio=0.07)
    if len(lines) < 6:
        return [False, "Insufficient line structure for triangle/radii. "]

    if not has_support_line(lines, a_geo, b_geo, min_hw=min_hw, ang_tol_deg=12.0, dist_ratio=0.08):
        return [False, "Missing side AB support line. "]
    if not has_support_line(lines, b_geo, c_geo, min_hw=min_hw, ang_tol_deg=12.0, dist_ratio=0.08):
        return [False, "Missing side BC support line. "]
    if not has_support_line(lines, c_geo, a_geo, min_hw=min_hw, ang_tol_deg=12.0, dist_ratio=0.08):
        return [False, "Missing side CA support line. "]

    a_rad = _project_to_circle(a_geo, center, ro)
    b_rad = _project_to_circle(b_geo, center, ro)
    c_rad = _project_to_circle(c_geo, center, ro)
    ok_ha, r_ha = _radius_present(img, lines, center, a_rad, min_hw)
    ok_hb, r_hb = _radius_present(img, lines, center, b_rad, min_hw)
    ok_hc, r_hc = _radius_present(img, lines, center, c_rad, min_hw)
    if not ok_ha:
        return [False, f"Missing radius HA line segment (ratio={r_ha:.2f}). "]
    if not ok_hb:
        return [False, f"Missing radius HB line segment (ratio={r_hb:.2f}). "]
    if not ok_hc:
        return [False, f"Missing radius HC line segment (ratio={r_hc:.2f}). "]

    if _dist(h_lbl, center) > scale_px(min_hw, 0.05, floor_px=0.0):
        return [False, "H label is not near the common center. "]

    l_ab = segment_to_abc((a_geo[0], a_geo[1], b_geo[0], b_geo[1]))
    l_bc = segment_to_abc((b_geo[0], b_geo[1], c_geo[0], c_geo[1]))
    l_ca = segment_to_abc((c_geo[0], c_geo[1], a_geo[0], a_geo[1]))
    if l_ab is None or l_bc is None or l_ca is None:
        return [False, "Failed to build triangle side lines. "]

    dmin = min(
        point_line_distance(center, l_ab),
        point_line_distance(center, l_bc),
        point_line_distance(center, l_ca),
    )
    if ri > dmin + scale_px(min_hw, 0.03, floor_px=0.0):
        return [False, "Inner circle exceeds triangle interior near at least one side. "]

    inscribe_tol = max(0.06 * dmin, scale_px(min_hw, 0.015, floor_px=0.0))
    if abs(ri - dmin) > inscribe_tol:
        return [
            False,
            (
                "Inner circle is not inscribed with respect to triangle sides "
                f"(ri={ri:.1f}, expected~{dmin:.1f}, tol={inscribe_tol:.1f}). "
            ),
        ]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_30,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
