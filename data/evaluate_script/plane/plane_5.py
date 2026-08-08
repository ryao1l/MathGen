import argparse

PID = 26
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _build_token_allow_mask(img_shape, tokens, min_hw, pad_ratio=0.015):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pad = int(round(scale_px(min_hw, pad_ratio, floor_px=1.0)))
    for tok in tokens if isinstance(tokens, list) else []:
        if not isinstance(tok, dict):
            continue
        bbox = tok.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except Exception:
            continue
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        xa = max(0, int(round(x1 - pad)))
        ya = max(0, int(round(y1 - pad)))
        xb = min(w - 1, int(round(x2 + pad)))
        yb = min(h - 1, int(round(y2 + pad)))
        if xb <= xa or yb <= ya:
            continue
        cv2.rectangle(mask, (xa, ya), (xb, yb), 255, -1)
    return mask


def _horizontal_error_deg(ang):
    return min(angle_diff_deg(float(ang), 0.0), angle_diff_deg(float(ang), 180.0))


def _pick_top_bottom_lines(lines, min_hw):
    long_th = scale_px(min_hw, 0.34, floor_px=0.0)
    cands = [it for it in lines if _horizontal_error_deg(it["ang"]) <= 10.0 and float(it["len"]) >= long_th]
    if len(cands) < 2:
        return None
    cands = sorted(cands, key=lambda t: float(t["len"]), reverse=True)[:8]

    best = None
    best_score = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            li = cands[i]
            lj = cands[j]
            if angle_diff_deg(li["ang"], lj["ang"]) > 5.0:
                continue
            yi = 0.5 * (float(li["seg"][1]) + float(li["seg"][3]))
            yj = 0.5 * (float(lj["seg"][1]) + float(lj["seg"][3]))
            sep = abs(yi - yj)
            if sep < scale_px(min_hw, 0.12, floor_px=0.0):
                continue
            score = float(li["len"]) + float(lj["len"]) + 0.3 * sep
            if best_score is None or score > best_score:
                best_score = score
                best = (li, lj)

    if best is None:
        return None

    l1, l2 = best
    y1 = 0.5 * (float(l1["seg"][1]) + float(l1["seg"][3]))
    y2 = 0.5 * (float(l2["seg"][1]) + float(l2["seg"][3]))
    return (l1, l2) if y1 <= y2 else (l2, l1)


def _pick_transversal_pair(lines, top_line, bottom_line, min_hw, img_shape):
    h, w = img_shape[:2]
    margin = 0.20 * min_hw

    cands = []
    long_th = scale_px(min_hw, 0.20, floor_px=0.0)
    for ln in lines:
        if float(ln["len"]) < long_th:
            continue
        if line_equivalent(ln, top_line, min_hw) or line_equivalent(ln, bottom_line, min_hw):
            continue
        if _horizontal_error_deg(ln["ang"]) <= 14.0:
            continue

        p_top = line_intersection_from_abc(ln["abc"], top_line["abc"])
        p_bot = line_intersection_from_abc(ln["abc"], bottom_line["abc"])
        if p_top is None or p_bot is None:
            continue
        xt, yt = float(p_top[0]), float(p_top[1])
        xb, yb = float(p_bot[0]), float(p_bot[1])

        if xt < -margin or xt > (float(w) + margin) or yt < -margin or yt > (float(h) + margin):
            continue
        if xb < -margin or xb > (float(w) + margin) or yb < -margin or yb > (float(h) + margin):
            continue

        if not point_on_segment_support(ln, p_top, min_hw, dist_ratio=0.02, dist_floor_px=0.0, t_min=-0.65, t_max=1.45):
            continue
        if not point_on_segment_support(ln, p_bot, min_hw, dist_ratio=0.02, dist_floor_px=0.0, t_min=-0.65, t_max=1.45):
            continue
        if not point_on_segment_support(top_line, p_top, min_hw, dist_ratio=0.02, dist_floor_px=0.0, t_min=-0.65, t_max=1.45):
            continue
        if not point_on_segment_support(bottom_line, p_bot, min_hw, dist_ratio=0.02, dist_floor_px=0.0, t_min=-0.65, t_max=1.45):
            continue

        span = math.hypot(xb - xt, yb - yt)
        if span < scale_px(min_hw, 0.14, floor_px=0.0):
            continue

        cands.append((ln, (xt, yt), (xb, yb), span))

    if len(cands) < 2:
        return None

    best = None
    best_score = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            li, pti_top, pti_bot, spi = cands[i]
            lj, ptj_top, ptj_bot, spj = cands[j]
            top_sep = abs(float(pti_top[0]) - float(ptj_top[0]))
            bot_sep = abs(float(pti_bot[0]) - float(ptj_bot[0]))
            if top_sep < scale_px(min_hw, 0.05, floor_px=0.0):
                continue
            if bot_sep < scale_px(min_hw, 0.05, floor_px=0.0):
                continue
            score = spi + spj + 0.35 * (top_sep + bot_sep)
            if best_score is None or score > best_score:
                best_score = score
                best = ((li, pti_top, pti_bot), (lj, ptj_top, ptj_bot))

    return best


def _assign_two_labels(tokens, ch1, ch2, p1, p2, tol):
    t11 = select_token_near_point(tokens, expected_char=ch1, point=p1, max_dist=tol)
    t12 = select_token_near_point(tokens, expected_char=ch1, point=p2, max_dist=tol)
    t21 = select_token_near_point(tokens, expected_char=ch2, point=p1, max_dist=tol)
    t22 = select_token_near_point(tokens, expected_char=ch2, point=p2, max_dist=tol)
    return (t11 is not None and t22 is not None) or (t12 is not None and t21 is not None)


def _line_end_label_ink(img, line_item, min_hw):
    x1, y1, x2, y2 = [float(v) for v in line_item["seg"]]
    px, py = (x1, y1) if x1 <= x2 else (x2, y2)
    _, bw = _gray_and_ink_mask(img)
    h, w = bw.shape[:2]
    win = scale_px(min_hw, 0.11, floor_px=0.0)
    x0 = int(max(0, px - win))
    x3 = int(min(w, px + win))
    y0 = int(max(0, py - win))
    y3 = int(min(h, py + win))
    if x3 <= x0 or y3 <= y0:
        return 0

    roi = bw[y0:y3, x0:x3] > 0
    yy, xx = np.ogrid[y0:y3, x0:x3]
    a, b, c = [float(v) for v in line_item["abc"]]
    line_band = np.abs(a * xx.astype(np.float32) + b * yy.astype(np.float32) + c) / max(1e-6, math.hypot(a, b)) <= scale_px(min_hw, 0.014, floor_px=0.0)
    label_mask = roi & (~line_band)
    if int(label_mask.sum()) <= 0:
        return 0

    num, _, stats, _ = cv2.connectedComponentsWithStats(label_mask.astype(np.uint8), connectivity=8)
    best_area = 0
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw_box = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_box = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < max(8, int(0.000004 * h * w)):
            continue
        if area > max(900, int(0.0040 * h * w)):
            continue
        if bw_box > 0.16 * min_hw or bh_box > 0.16 * min_hw:
            continue
        best_area = max(best_area, area)
    return best_area


def _intersection_label_ink(img, point, lines_to_exclude, min_hw):
    px, py = float(point[0]), float(point[1])
    _, bw = _gray_and_ink_mask(img)
    h, w = bw.shape[:2]
    win = scale_px(min_hw, 0.11, floor_px=0.0)
    x0 = int(max(0, px - win))
    x3 = int(min(w, px + win))
    y0 = int(max(0, py - win))
    y3 = int(min(h, py + win))
    if x3 <= x0 or y3 <= y0:
        return 0

    roi = bw[y0:y3, x0:x3] > 0
    yy, xx = np.ogrid[y0:y3, x0:x3]
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    point_disk = (xx - px) ** 2 + (yy - py) ** 2 <= scale_px(min_hw, 0.035, floor_px=0.0) ** 2
    line_band = np.zeros_like(roi, dtype=bool)
    for ln in lines_to_exclude:
        a, b, c = [float(v) for v in ln["abc"]]
        line_band |= np.abs(a * xx + b * yy + c) / max(1e-6, math.hypot(a, b)) <= scale_px(min_hw, 0.014, floor_px=0.0)
    label_mask = roi & (~point_disk) & (~line_band)
    if int(label_mask.sum()) <= 0:
        return 0

    num, _, stats, _ = cv2.connectedComponentsWithStats(label_mask.astype(np.uint8), connectivity=8)
    best_area = 0
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw_box = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_box = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < max(8, int(0.000004 * h * w)):
            continue
        if area > max(900, int(0.0040 * h * w)):
            continue
        if bw_box > 0.16 * min_hw or bh_box > 0.16 * min_hw:
            continue
        best_area = max(best_area, area)
    return best_area


def judge_plane_26(img):
    if img is None:
        return [False, "Input image is None. "]

    h, w = img.shape[:2]
    min_hw = float(min(h, w))

    lines = detect_line_segments(img, min_len_ratio=0.08)
    if len(lines) < 4:
        return [False, "Insufficient line structure for hard case 26. "]

    top_bottom = _pick_top_bottom_lines(lines, min_hw)
    if top_bottom is None:
        return [False, "Failed to detect two long horizontal parallel lines. "]
    top_line, bottom_line = top_bottom

    pair = _pick_transversal_pair(lines, top_line, bottom_line, min_hw, img.shape)
    if pair is None:
        return [False, "Failed to detect two transversals crossing both horizontal lines. "]

    (l1, top1, bot1), (l2, top2, bot2) = pair

    ok_t1, r_t1 = has_segment_between_points(img, top1, bot1, ratio_th=0.14, thickness=2, trim_ratio=0.02)
    ok_t2, r_t2 = has_segment_between_points(img, top2, bot2, ratio_th=0.14, thickness=2, trim_ratio=0.02)
    if not (ok_t1 and ok_t2):
        return [False, f"Missing transversal segments (t1={r_t1:.2f}, t2={r_t2:.2f}). "]

    tokens = extract_global_letter_tokens(img, whitelist="TBPQHF", min_conf=0.10)
    line_tol = scale_px(min_hw, 0.11, floor_px=0.0)

    lbl_t = select_token_near_line(tokens, expected_char="T", line_item=top_line, max_perp=line_tol, t_margin=0.35)
    lbl_b = select_token_near_line(tokens, expected_char="B", line_item=bottom_line, max_perp=line_tol, t_margin=0.35)
    line_labels_ok = lbl_t is not None and lbl_b is not None

    top_pts = sorted([top1, top2], key=lambda p: float(p[0]))
    bot_pts = sorted([bot1, bot2], key=lambda p: float(p[0]))

    cpt = line_intersection_from_abc(l1["abc"], l2["abc"])
    anchors = list(top_pts) + list(bot_pts)
    if cpt is not None:
        anchors.append((float(cpt[0]), float(cpt[1])))



    allowed_line_abc = [
        {"abc": top_line["abc"]},
        {"abc": bottom_line["abc"]},
        {"abc": l1["abc"]},
        {"abc": l2["abc"]},
    ]
    token_allow = _build_token_allow_mask(
        img_shape=img.shape,
        tokens=tokens,
        min_hw=min_hw,
    )

    violated, info = has_excess_outside_ink(
        img=img,
        allowed_lines=allowed_line_abc,
        anchor_points=anchors,
        max_outside_ratio=0.03,
        max_outside_px_ratio=0.00008,
        max_outside_px_floor=0,
        extra_allow_mask=token_allow,
    )
    if violated:
        return [
            False,
            (
                "Detected extra drawing content outside target hard-26 structure "
                f"(outside={info['outside_px']}, ratio={info['outside_ratio']:.3f}). "
            ),
        ]

    point_tol = scale_px(min_hw, 0.12, floor_px=0.0)
    ok_top = _assign_two_labels(tokens, "P", "Q", top_pts[0], top_pts[1], point_tol)
    ok_bot = _assign_two_labels(tokens, "H", "F", bot_pts[0], bot_pts[1], point_tol)
    if not ok_top or not ok_bot:
        ink_th = max(10, int(0.000006 * img.shape[0] * img.shape[1]))
        exclude_lines = [top_line, bottom_line, l1, l2]
        point_inks = [
            _intersection_label_ink(img, top_pts[0], exclude_lines, min_hw),
            _intersection_label_ink(img, top_pts[1], exclude_lines, min_hw),
            _intersection_label_ink(img, bot_pts[0], exclude_lines, min_hw),
            _intersection_label_ink(img, bot_pts[1], exclude_lines, min_hw),
        ]
        if any(v < ink_th for v in point_inks):
            return [False, f"Failed to detect intersection labels on top/bottom lines (top={ok_top}, bottom={ok_bot}). "]
    if not line_labels_ok:
        ink_th = max(10, int(0.000006 * img.shape[0] * img.shape[1]))
        t_ink = _line_end_label_ink(img, top_line, min_hw)
        b_ink = _line_end_label_ink(img, bottom_line, min_hw)
        if t_ink < ink_th or b_ink < ink_th:
            return [False, f"Failed to detect line labels T/B (T={lbl_t is not None}, B={lbl_b is not None}). "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_26,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
