import argparse

PID = 85
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _dist(p, q):
    return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))


def _segment_item(p, q):
    seg = (float(p[0]), float(p[1]), float(q[0]), float(q[1]))
    abc = segment_to_abc(seg)
    if abc is None:
        return None
    return {
        "seg": seg,
        "abc": abc,
        "ang": segment_angle_deg(seg),
        "len": float(_dist(p, q)),
    }


def _segment_presence(img, lines, p1, p2, min_hw, ratio_th=0.12, trim_ratio=0.02, dist_ratio=0.07):
    ok_seg, ratio = has_segment_between_points(
        img,
        p1,
        p2,
        ratio_th=float(ratio_th),
        thickness=2,
        trim_ratio=float(trim_ratio),
    )
    line_ref = find_support_line(
        lines,
        p1,
        p2,
        min_hw=min_hw,
        ang_tol_deg=11.0,
        dist_ratio=float(dist_ratio),
        dist_floor_px=0.0,
    )
    ratio = float(ratio)
    present = bool(ratio >= float(ratio_th) or (line_ref is not None and ratio >= max(0.08, 0.75 * float(ratio_th))))
    return present, ratio, line_ref


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


def _circle_circle_intersections(c1, c2, eps=1e-6):
    x1, y1, r1 = [float(v) for v in c1]
    x2, y2, r2 = [float(v) for v in c2]
    dx = x2 - x1
    dy = y2 - y1
    d = math.hypot(dx, dy)
    if d <= float(eps):
        return []

    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h2 = r1 * r1 - a * a
    if h2 < -float(eps):
        return []
    h = math.sqrt(max(0.0, h2))

    xm = x1 + a * dx / d
    ym = y1 + a * dy / d
    rx = -dy * (h / d)
    ry = dx * (h / d)

    if h <= float(eps):
        return [(float(xm), float(ym))]
    return [
        (float(xm + rx), float(ym + ry)),
        (float(xm - rx), float(ym - ry)),
    ]


def _circle_equivalent(ca, cb, min_hw):
    center_tol = scale_px(min_hw, 0.08, floor_px=0.0)
    radius_tol = scale_px(min_hw, 0.06, floor_px=0.0)
    if _dist((ca[0], ca[1]), (cb[0], cb[1])) > center_tol:
        return False
    return abs(float(ca[2]) - float(cb[2])) <= radius_tol


def _detect_circle_candidates(img):
    h, w = img.shape[:2]
    min_hw = float(min(h, w))
    min_r = int(round(scale_px(min_hw, 0.045, floor_px=1.0)))
    max_r = scale_px(min_hw, 0.62, floor_px=0.0)
    score_th = int(round(scale_px(min_hw, 0.10, floor_px=1.0)))
    band = int(max(1, round(scale_px(min_hw, 0.010, floor_px=0.0))))

    found, scores = _find_top_k_circles(
        img,
        k=20,
        min_r=min_r,
        max_r=0,
        seed=0,
        iters=3600,
        score_th=score_th,
    )

    x_margin = 0.12 * float(w)
    y_margin = 0.12 * float(h)
    candidates = []
    for (cx, cy, r), s in zip(found, scores):
        x, y, rr = float(cx), float(cy), float(r)
        if rr <= 0.0 or rr > max_r:
            continue
        if x < (-x_margin) or x > (float(w) + x_margin):
            continue
        if y < (-y_margin) or y > (float(h) + y_margin):
            continue
        refined = _refine_circle_radius_by_inner_outer_edges(img, (x, y, rr))
        if refined is None:
            continue
        rx, ry, rr2 = [float(v) for v in refined]
        if rr2 <= 0.0 or rr2 > max_r:
            continue
        if rr2 < scale_px(min_hw, 0.055, floor_px=0.0):
            continue
        cov, vis = _circle_ink_coverage(img, (rx, ry), rr2, band)
        if vis < 0.45 or cov < 0.28:
            continue
        candidates.append((rx, ry, rr2, int(s), float(cov), float(vis)))

    if candidates:
        candidates.sort(key=lambda t: (-t[3], -t[2]))
        merged = _merge_circle_candidates(
            [(x, y, r, s) for (x, y, r, s, _, _) in candidates],
            center_tol=scale_px(min_hw, 0.03, floor_px=0.0),
            radius_tol=scale_px(min_hw, 0.03, floor_px=0.0),
        )
        merged_out = []
        for x, y, r, s in merged:
            cov, vis = _circle_ink_coverage(img, (x, y), r, band)
            if vis < 0.45 or cov < 0.28:
                continue
            merged_out.append((float(x), float(y), float(r), int(s), float(cov), float(vis)))
        if merged_out:
            merged_out.sort(key=lambda t: (-t[3], -t[4], -t[2]))
            return merged_out

    fallback = []
    for order in (1, 2, 3):
        c = detect_circle(img, order=order, min_r=min_r, max_r=0)
        if c is None:
            continue
        x, y, r = [float(v) for v in c]
        if r <= 0.0 or r > max_r:
            continue
        cov, vis = _circle_ink_coverage(img, (x, y), r, band)
        if vis < 0.45 or cov < 0.28:
            continue
        fallback.append((x, y, r, 0, cov, vis))

    dedup = []
    for it in fallback:
        c = (it[0], it[1], it[2])
        if any(_circle_equivalent(c, (q[0], q[1], q[2]), min_hw) for q in dedup):
            continue
        dedup.append(it)
    return dedup


def _match_ab_labels(tokens, p1, p2, tol, min_sep):
    p1_a = select_token_near_point(tokens, expected_char="A", point=p1, max_dist=tol)
    p1_b = select_token_near_point(tokens, expected_char="B", point=p1, max_dist=tol)
    p2_a = select_token_near_point(tokens, expected_char="A", point=p2, max_dist=tol)
    p2_b = select_token_near_point(tokens, expected_char="B", point=p2, max_dist=tol)

    def _valid_pair(ta, tb):
        if ta is None or tb is None:
            return False
        if ta is tb:
            return False
        ca = ta.get("center")
        cb = tb.get("center")
        if ca is None or cb is None:
            return False
        return _dist(ca, cb) >= float(min_sep)

    if _valid_pair(p1_a, p2_b):
        return True, ""
    if _valid_pair(p2_a, p1_b):
        return True, ""

    return (
        False,
        (
            "Failed to match labels A/B to the two circle intersections "
            f"(near_A={p1_a is not None or p2_a is not None}, near_B={p1_b is not None or p2_b is not None}). "
        ),
    )


def _select_intersecting_pair(candidates, tokens, min_hw, img_shape):
    if candidates is None or len(candidates) < 2:
        return None

    h, w = img_shape[:2]
    margin = scale_px(min_hw, 0.08, floor_px=0.0)
    label_tol = scale_px(min_hw, 0.14, floor_px=0.0)
    label_sep = scale_px(min_hw, 0.02, floor_px=0.0)

    best = None
    best_score = None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            x1, y1, r1, s1, cov1, vis1 = candidates[i]
            x2, y2, r2, s2, cov2, vis2 = candidates[j]
            c1 = (x1, y1, r1)
            c2 = (x2, y2, r2)

            d = _dist((x1, y1), (x2, y2))
            sep_tol = max(scale_px(min_hw, 0.010, floor_px=0.0), 0.03 * min(r1, r2))
            if d <= abs(r1 - r2) + sep_tol:
                continue
            if d >= (r1 + r2) - sep_tol:
                continue

            inters = _circle_circle_intersections(c1, c2, eps=sep_tol)
            if len(inters) != 2:
                continue
            p1, p2 = inters[0], inters[1]
            if _dist(p1, p2) <= max(scale_px(min_hw, 0.010, floor_px=0.0), 0.015 * min(r1, r2)):
                continue
            if any(
                (
                    float(p[0]) < -margin
                    or float(p[0]) > (float(w) + margin)
                    or float(p[1]) < -margin
                    or float(p[1]) > (float(h) + margin)
                )
                for p in (p1, p2)
            ):
                continue

            label_bonus = 0.0
            if tokens:
                ok_label, _ = _match_ab_labels(tokens, p1, p2, tol=label_tol, min_sep=label_sep)
                if ok_label:
                    label_bonus += 1.0
                else:
                    a_ok = (
                        select_token_near_point(tokens, "A", p1, label_tol) is not None
                        or select_token_near_point(tokens, "A", p2, label_tol) is not None
                    )
                    b_ok = (
                        select_token_near_point(tokens, "B", p1, label_tol) is not None
                        or select_token_near_point(tokens, "B", p2, label_tol) is not None
                    )
                    if a_ok:
                        label_bonus += 0.35
                    if b_ok:
                        label_bonus += 0.35

            score = (
                float(s1 + s2)
                + 0.20 * float(r1 + r2)
                + 0.15 * _dist(p1, p2)
                + 40.0 * label_bonus
                + 16.0 * (cov1 + cov2)
                + 8.0 * (vis1 + vis2)
                - 0.08 * abs(r1 - r2)
            )
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "c1": c1,
                    "c2": c2,
                    "p1": (float(p1[0]), float(p1[1])),
                    "p2": (float(p2[0]), float(p2[1])),
                    "sep_tol": float(sep_tol),
                }
    return best


def judge_plane_85(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))

    tokens = extract_global_letter_tokens(img, whitelist="AB", min_conf=0.08)
    candidates = _detect_circle_candidates(img)
    if len(candidates) < 2:
        return [False, "Failed to detect two valid circle candidates. "]

    best = _select_intersecting_pair(
        candidates=candidates,
        tokens=tokens,
        min_hw=min_hw,
        img_shape=img.shape,
    )
    if best is None:
        return [False, "Failed to find two circles intersecting at exactly two points. "]

    c1 = best["c1"]
    c2 = best["c2"]
    p1 = best["p1"]
    p2 = best["p2"]
    sep_tol = float(best["sep_tol"])

    x1, y1, r1 = [float(v) for v in c1]
    x2, y2, r2 = [float(v) for v in c2]
    d = _dist((x1, y1), (x2, y2))

    min_r_th = scale_px(min_hw, 0.055, floor_px=0.0)
    if min(r1, r2) < min_r_th:
        return [False, "Detected circles are too small. "]

    if d <= abs(r1 - r2) + sep_tol:
        return [False, "One circle is contained in/tangent to the other; no two-point intersection. "]
    if d >= (r1 + r2) - sep_tol:
        return [False, "The two circles are disjoint or tangent; not intersecting at two points. "]

    inters = _circle_circle_intersections(c1, c2, eps=sep_tol)
    if len(inters) != 2:
        return [False, "Failed to derive two stable intersection points from detected circles. "]

    lines = detect_line_segments(img, min_len_ratio=0.07)
    ab_ok, ab_ratio, ab_line = _segment_presence(
        img,
        lines,
        p1,
        p2,
        min_hw=min_hw,
        ratio_th=0.12,
        trim_ratio=0.02,
        dist_ratio=0.07,
    )
    if not ab_ok:
        return [False, f"Missing segment AB between the two circle intersections (ratio={ab_ratio:.2f}). "]

    if not tokens:
        return [False, "Failed to detect labels A/B. "]
    label_tol = max(scale_px(min_hw, 0.14, floor_px=0.0), 0.10 * min(r1, r2))
    label_sep = scale_px(min_hw, 0.02, floor_px=0.0)
    ok_labels, msg = _match_ab_labels(tokens, p1, p2, tol=label_tol, min_sep=label_sep)
    if not ok_labels:
        a_near = (
            select_token_near_point(tokens, "A", p1, label_tol) is not None
            or select_token_near_point(tokens, "A", p2, label_tol) is not None
        )
        b_near = (
            select_token_near_point(tokens, "B", p1, label_tol) is not None
            or select_token_near_point(tokens, "B", p2, label_tol) is not None
        )
        if not (a_near or b_near):
            return [False, msg]

    ab_ref = ab_line if ab_line is not None else _segment_item(p1, p2)
    dominant_len = max(scale_px(min_hw, 0.33, floor_px=0.0), 0.60 * _dist(p1, p2))
    dominant_extras = []
    for ln in lines:
        if float(ln.get("len", 0.0)) < dominant_len:
            continue
        if ab_ref is not None and line_equivalent(ln, ab_ref, min_hw=min_hw):
            continue
        dominant_extras.append(ln)
    if dominant_extras:
        return [False, f"Detected extra dominant line(s) outside AB structure: {len(dominant_extras)}. "]

    extra_circle_count = 0
    for cx, cy, rr, _, cov, vis in candidates:
        if vis < 0.45 or cov < 0.30:
            continue
        c = (float(cx), float(cy), float(rr))
        if _circle_equivalent(c, c1, min_hw) or _circle_equivalent(c, c2, min_hw):
            continue
        if rr < 0.75 * min(r1, r2):
            continue
        extra_circle_count += 1
    if extra_circle_count > 0:
        return [False, f"Detected extra prominent circle(s): {extra_circle_count}. "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_85,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
