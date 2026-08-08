import argparse

PID = 2
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _token_effective_distance_to_point(token, point, r):
    if token is None or point is None:
        return float("inf")
    return token_edge_distance_to_point(token, point)


def _circle_ring_connectivity_ok(img, circle):
    h, w = img.shape[:2]
    cx, cy, r = [float(v) for v in circle]

    _, bw = _gray_and_ink_mask(img)
    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    ink = (bw > 0).astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    band = 0.045 * r
    ring_mask = (np.abs(dist - r) <= band) & (ink > 0)

    ring_px = int(ring_mask.sum())
    if ring_px <= 0:
        return False, 0, 0.0

    ids = np.unique(labels[ring_mask])
    ids = [int(i) for i in ids if int(i) != 0]
    if not ids:
        return False, 0, 0.0

    comp_areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in ids]
    largest = max(comp_areas) if comp_areas else 0
    largest_share = float(largest) / float(max(1, ring_px))
    ok = (len(ids) <= 4) and (largest_share >= 0.55)
    return ok, len(ids), largest_share


def judge_plane_2(img):
    if img is None:
        return [False, "Input image is None. "]

    h, w = img.shape[:2]
    min_hw = float(min(h, w))

    circle = detect_largest_circle(img)
    if circle is None:
        return [False, "Failed to detect the main circle. "]
    cx, cy, r = [float(v) for v in circle]
    if r < 0.12 * min_hw:
        return [False, f"Detected circle is too small (r={r:.1f}). "]

    ring_ok, ring_comp_cnt, ring_share = _circle_ring_connectivity_ok(img, circle)
    if not ring_ok:
        return [
            False,
            f"Detected circle is not a continuous stroke (ring_components={ring_comp_cnt}, largest_share={ring_share:.2f}). ",
        ]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if float(np.mean(hsv[:, :, 1] > 60)) > 0.05:
        return [False, "Detected a colorful/complex diagram rather than a simple labeled circle. "]

    lines = detect_line_segments(img, min_len_ratio=0.10)
    dominant_len = 0.30 * min_hw
    dominant = [it for it in lines if float(it["len"]) >= dominant_len]
    if len(dominant) > 15:
        return [False, f"Too many dominant line(s) for a simple centered circle: {len(dominant)}. "]

    pts = detect_marker_points(img)
    center_tol = 0.14 * r
    near_center = [p for p in pts if math.hypot(float(p[0]) - cx, float(p[1]) - cy) <= center_tol]
    center_pt = min(near_center, key=lambda p: math.hypot(float(p[0]) - cx, float(p[1]) - cy)) if near_center else (cx, cy)

    tokens = extract_global_letter_tokens(img, whitelist="C", min_conf=0.10)
    cands = []
    for t in tokens:
        letters = str(t.get("letters", "")).upper()
        if not letters:
            letters = str(t.get("char", "")).upper()
        if "C" not in letters:
            continue
        eff_d = _token_effective_distance_to_point(t, center_pt, r)
        cands.append((eff_d, -float(t.get("conf", 0.0)), t))
    cands.sort(key=lambda x: (x[0], x[1]))
    c_tok = cands[0][2] if cands else None
    c_eff_d = cands[0][0] if cands else float("inf")

    rough_circles = []
    for order in range(1, 5):
        c = detect_circle(img, order=order, min_r=int(0.05 * min_hw), max_r=0)
        if c is not None:
            rough_circles.append(c)

    if c_tok is None or c_eff_d > 0.24 * r:
        _, bw = _gray_and_ink_mask(img)
        yy, xx = np.ogrid[:h, :w]
        center_disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (0.24 * r) ** 2
        inner_dot = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (0.035 * r) ** 2
        center_ink = int(np.logical_and(center_disk, bw > 0).sum())
        dot_ink = int(np.logical_and(inner_dot, bw > 0).sum())
        if len(rough_circles) >= 4 and center_ink < 1000 and (r / max(1.0, min_hw)) > 0.30:
            return [False, "Detected multiple circle-like structures without a reliable centered C label. "]
        if len(rough_circles) >= 3 and center_ink < 650 and len(dominant) == 0 and (r / max(1.0, min_hw)) > 0.25:
            return [False, "Center evidence is too weak for a simple circle centered at C. "]
        if center_ink < max(10, int(0.00001 * h * w)) and dot_ink < 2:
            return [False, "Failed to detect center label/mark near the circle center. "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_2,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
