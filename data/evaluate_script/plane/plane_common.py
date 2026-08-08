import argparse
import itertools
import os
import sys
import cv2
import math
import numpy as np
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


_EASYOCR_READER = None
_EASYOCR_INIT_ATTEMPTED = False
_EASYOCR_INIT_ERROR = ""


def _easyocr_log(message):
    print(f"[plane_common] EasyOCR: {message}", file=sys.stderr, flush=True)


def _get_easyocr_reader():
    global _EASYOCR_READER, _EASYOCR_INIT_ATTEMPTED, _EASYOCR_INIT_ERROR
    if _EASYOCR_INIT_ATTEMPTED:
        return _EASYOCR_READER
    _EASYOCR_INIT_ATTEMPTED = True
    try:
        import easyocr
    except Exception as e:
        _EASYOCR_INIT_ERROR = f"import easyocr failed: {e!r}"
        _EASYOCR_READER = None
        _easyocr_log(_EASYOCR_INIT_ERROR)
        return None

    auto_download = os.getenv("PLANE_EASYOCR_AUTO_DOWNLOAD", "1").strip().lower() not in {"0", "false", "no"}
    attempts = [(True, False), (False, False)]
    if auto_download:
        attempts += [(True, True), (False, True)]

    _easyocr_log(
        f"initializing reader (auto_download={'on' if auto_download else 'off'}, attempts={len(attempts)})"
    )

    errs = []
    init_t0 = time.perf_counter()
    for use_gpu, allow_download in attempts:
        attempt_t0 = time.perf_counter()
        _easyocr_log(f"trying gpu={use_gpu} download_enabled={allow_download}")
        try:
            _EASYOCR_READER = easyocr.Reader(["en"], gpu=use_gpu, download_enabled=allow_download)
            _EASYOCR_INIT_ERROR = ""
            elapsed = time.perf_counter() - attempt_t0
            total_elapsed = time.perf_counter() - init_t0
            _easyocr_log(
                f"reader ready (gpu={use_gpu}, download_enabled={allow_download}, "
                f"attempt_elapsed={elapsed:.1f}s, total_elapsed={total_elapsed:.1f}s)"
            )
            return _EASYOCR_READER
        except Exception as e:
            elapsed = time.perf_counter() - attempt_t0
            err = f"gpu={use_gpu},download={allow_download}: {e!r}"
            errs.append(err)
            _easyocr_log(f"attempt failed after {elapsed:.1f}s ({err})")

    _EASYOCR_READER = None
    _EASYOCR_INIT_ERROR = " | ".join(errs[-3:])
    total_elapsed = time.perf_counter() - init_t0
    _easyocr_log(f"reader unavailable after {total_elapsed:.1f}s ({_EASYOCR_INIT_ERROR})")
    return _EASYOCR_READER


def _extract_first_allowed_letter(text, allowed_set):
    for ch in str(text).upper():
        if ch in allowed_set:
            return ch
    return None


def _extract_allowed_letters(text, allowed_set):
    seen = set()
    out = []
    for ch in str(text).upper():
        if ch in allowed_set and ch not in seen:
            seen.add(ch)
            out.append(ch)
    return "".join(out)


def _normalize_letter_whitelist(whitelist):
    seen = set()
    out = []
    for ch in str(whitelist).upper():
        if "A" <= ch <= "Z" and ch not in seen:
            seen.add(ch)
            out.append(ch)
    return "".join(out)


def _extract_global_letter_tokens_tesseract(img, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_conf=0.08):
    if img is None:
        return []
    try:
        import pytesseract
    except Exception:
        return []

    h, w = img.shape[:2]
    min_hw = float(min(h, w))
    allowed = _normalize_letter_whitelist(whitelist)
    if not allowed:
        return []
    allowed_set = set(allowed)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    variants = []
    for src in (gray, 255 - gray):
        thr = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        variants.extend([src, thr])

    tokens = []
    config = f"--psm 11 --oem 3 -c tessedit_char_whitelist={allowed}{allowed.lower()}"
    for proc in variants:
        try:
            data = pytesseract.image_to_data(proc, config=config, output_type=pytesseract.Output.DICT)
        except Exception:
            continue
        n = len(data.get("text", []))
        for i in range(n):
            txt = str(data["text"][i] or "")
            letters = _extract_allowed_letters(txt, allowed_set)
            if not letters:
                continue
            try:
                conf_raw = float(data.get("conf", ["-1"])[i])
            except Exception:
                conf_raw = -1.0
            conf = max(0.0, min(1.0, conf_raw / 100.0))
            if conf < float(min_conf):
                continue
            x = float(data["left"][i])
            y = float(data["top"][i])
            ww = float(data["width"][i])
            hh = float(data["height"][i])
            if ww <= 1 or hh <= 1:
                continue
            if ww > 0.20 * min_hw or hh > 0.20 * min_hw:
                continue
            ordered_letters = [ch for ch in str(txt).upper() if ch in allowed_set]
            if len(ordered_letters) <= 1:
                tokens.append(
                    {
                        "char": letters[0],
                        "letters": letters,
                        "raw_text": txt,
                        "conf": float(conf),
                        "center": (float(x + 0.5 * ww), float(y + 0.5 * hh)),
                        "bbox": (float(x), float(y), float(x + ww), float(y + hh)),
                    }
                )
                continue

            char_w = ww / float(len(ordered_letters))
            for j, ch in enumerate(ordered_letters):
                cx1 = x + j * char_w
                cx2 = x + (j + 1) * char_w
                tokens.append(
                    {
                        "char": ch,
                        "letters": ch,
                        "raw_text": txt,
                        "conf": float(conf),
                        "center": (float(0.5 * (cx1 + cx2)), float(y + 0.5 * hh)),
                        "bbox": (float(cx1), float(y), float(cx2), float(y + hh)),
                    }
                )

    dedup = []
    merge_tol = max(8.0, 0.015 * min_hw)
    for t in sorted(tokens, key=lambda z: float(z["conf"]), reverse=True):
        cx, cy = t["center"]
        if any(t["char"] == q["char"] and math.hypot(cx - q["center"][0], cy - q["center"][1]) <= merge_tol for q in dedup):
            continue
        dedup.append(t)
    return dedup


def extract_global_letter_tokens(img, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_conf=0.08):

    if img is None:
        return []
    fast_tokens = _extract_global_letter_tokens_tesseract(img, whitelist=whitelist, min_conf=min_conf)
    if fast_tokens:
        return fast_tokens
    if os.getenv("PLANE_EASYOCR_FALLBACK", "0").strip().lower() not in {"1", "true", "yes"}:
        return []
    reader = _get_easyocr_reader()
    if reader is None:
        return []

    h, w = img.shape[:2]
    min_hw = float(min(h, w))
    allowed = _normalize_letter_whitelist(whitelist)
    if not allowed:
        return []
    allowed_set = set(allowed)

    seen = set()
    allowlist_ocr = []
    for ch in allowed:
        up = ch.upper()
        lo = ch.lower()
        if up not in seen:
            allowlist_ocr.append(up)
            seen.add(up)
        if lo not in seen:
            allowlist_ocr.append(lo)
            seen.add(lo)
    allowlist_ocr = "".join(allowlist_ocr)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    num, _, stats, cents = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), 8)
    area_min = max(6, int(0.000003 * h * w))
    area_max = max(area_min + 1, int(0.006 * h * w))
    side_max = max(24, int(0.14 * min_hw))
    side_min = max(5, int(0.006 * min_hw))

    candidates = []
    for i in range(1, num):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < area_min or area > area_max:
            continue
        if ww < side_min or hh < side_min or ww > side_max or hh > side_max:
            continue

        aspect = ww / float(max(1, hh))
        density = area / float(max(1, ww * hh))
        if not (0.10 <= aspect <= 6.0):
            continue
        if not (0.05 <= density <= 0.95):
            continue

        candidates.append((i, x, y, ww, hh, area))


    candidates.sort(key=lambda t: t[5], reverse=True)
    candidates = candidates[:40]

    def _ocr_crop(crop_gray):
        if crop_gray is None or crop_gray.size == 0:
            return None, 0.0, "", ""
        proc0 = cv2.copyMakeBorder(crop_gray, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
        best_char = None
        best_conf = -1.0
        best_letters = ""
        best_raw_text = ""
        for target_short in (140.0, 220.0):
            proc = proc0
            short_side = float(max(1, min(proc.shape[:2])))
            scale = max(1.0, min(8.0, target_short / short_side))
            if scale > 1.01:
                proc = cv2.resize(proc, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            thr = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            for v in (proc, 255 - proc, thr):
                try:
                    result = reader.readtext(
                        cv2.cvtColor(v, cv2.COLOR_GRAY2BGR),
                        detail=1,
                        paragraph=False,
                        allowlist=allowlist_ocr,
                    )
                except Exception:
                    result = []
                for item in result if isinstance(result, list) else []:
                    if not (isinstance(item, (list, tuple)) and len(item) >= 3):
                        continue
                    txt = str(item[1])
                    try:
                        conf = float(item[2])
                    except Exception:
                        conf = 0.0
                    letters = _extract_allowed_letters(txt, allowed_set)
                    if not letters:
                        continue
                    if conf >= float(min_conf) and conf > best_conf:
                        best_conf = conf
                        best_char = letters[0]
                        best_letters = letters
                        best_raw_text = txt
                if best_char is not None and best_conf >= max(0.35, float(min_conf)):
                    return best_char, best_conf, best_letters, best_raw_text
        return best_char, best_conf, best_letters, best_raw_text

    tokens = []
    for i, x, y, ww, hh, _ in candidates:
        best_char = None
        best_conf = -1.0
        best_letters = ""
        best_raw_text = ""
        best_box = None


        for pmul in (0.8, 1.1, 1.4):
            pad = int(max(4, pmul * max(ww, hh)))
            x1 = int(max(0, x - pad))
            y1 = int(max(0, y - pad))
            x2 = int(min(w, x + ww + pad))
            y2 = int(min(h, y + hh + pad))
            if x2 <= x1 or y2 <= y1:
                continue
            ch, conf, letters, raw_text = _ocr_crop(gray[y1:y2, x1:x2])
            if ch is None:
                continue
            if conf > best_conf:
                best_char = ch
                best_conf = conf
                best_letters = letters
                best_raw_text = raw_text
                best_box = (float(x1), float(y1), float(x2), float(y2))

        if best_char is None or best_conf < float(min_conf):
            continue
        tokens.append(
            {
                "char": best_char,
                "letters": best_letters if best_letters else best_char,
                "raw_text": best_raw_text,
                "conf": float(best_conf),
                "center": (float(cents[i][0]), float(cents[i][1])),
                "bbox": (float(x), float(y), float(x + ww), float(y + hh)),
                "ocr_crop_bbox": best_box,
            }
        )



    try:
        full_result = reader.readtext(
            img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
            detail=1,
            paragraph=False,
            allowlist=allowlist_ocr,
        )
    except Exception:
        full_result = []

    for item in full_result if isinstance(full_result, list) else []:
        if not (isinstance(item, (list, tuple)) and len(item) >= 3):
            continue
        bbox = item[0]
        txt = str(item[1])
        try:
            conf = float(item[2])
        except Exception:
            conf = 0.0
        letters = _extract_allowed_letters(txt, allowed_set)
        if not letters or conf < float(min_conf):
            continue
        try:
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        except Exception:
            continue
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        tokens.append(
            {
                "char": letters[0],
                "letters": letters,
                "raw_text": txt,
                "conf": float(conf),
                "center": (float(cx), float(cy)),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
            }
        )

    if not tokens:
        return []

    dedup = []
    merge_tol = max(8.0, 0.015 * min_hw)
    for t in sorted(tokens, key=lambda z: float(z["conf"]), reverse=True):
        cx, cy = t["center"]
        merged = False
        for q in dedup:
            if t["char"] != q["char"]:
                continue
            qx, qy = q["center"]
            if math.hypot(cx - qx, cy - qy) <= merge_tol:
                merged = True
                break
        if not merged:
            dedup.append(t)
    return dedup


def segment_projection_t(seg, p):
    x1, y1, x2, y2 = [float(v) for v in seg]
    px, py = float(p[0]), float(p[1])
    dx = x2 - x1
    dy = y2 - y1
    den = dx * dx + dy * dy
    if den <= 1e-9:
        return 0.0
    return ((px - x1) * dx + (py - y1) * dy) / den


def _token_bbox_xyxy(token):
    if not isinstance(token, dict):
        return None
    bbox = token.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            x_lo, x_hi = (x1, x2) if x1 <= x2 else (x2, x1)
            y_lo, y_hi = (y1, y2) if y1 <= y2 else (y2, y1)
            if np.isfinite([x_lo, y_lo, x_hi, y_hi]).all():
                return (x_lo, y_lo, x_hi, y_hi)
        except Exception:
            pass
    c = token.get("center")
    if isinstance(c, (list, tuple)) and len(c) >= 2:
        try:
            cx, cy = float(c[0]), float(c[1])
            if np.isfinite([cx, cy]).all():
                return (cx, cy, cx, cy)
        except Exception:
            pass
    return None


def _bbox_corners_xyxy(bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return ((x1, y1), (x1, y2), (x2, y1), (x2, y2))


def _point_to_bbox_distance(point, bbox):
    px, py = float(point[0]), float(point[1])
    x1, y1, x2, y2 = [float(v) for v in bbox]
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def _point_to_token_edge_distance(point, token):
    bbox = _token_bbox_xyxy(token)
    if bbox is None or point is None:
        return float("inf")
    return _point_to_bbox_distance(point, bbox)


def _line_to_bbox_distance(abc, bbox):
    a, b, c = [float(v) for v in abc]
    corners = _bbox_corners_xyxy(bbox)
    vals = [a * float(x) + b * float(y) + c for x, y in corners]
    if min(vals) <= 0.0 <= max(vals):
        return 0.0
    return min(abs(v) for v in vals)


def _line_to_token_edge_distance(abc, token):
    bbox = _token_bbox_xyxy(token)
    if bbox is None or abc is None:
        return float("inf")
    return _line_to_bbox_distance(abc, bbox)


def _token_projection_t_interval(seg, token):
    bbox = _token_bbox_xyxy(token)
    if bbox is None:
        return None
    ts = [segment_projection_t(seg, p) for p in _bbox_corners_xyxy(bbox)]
    if not ts:
        return None
    return (min(ts), max(ts))


def _interval_distance_to_value(lo, hi, value):
    lo = float(lo)
    hi = float(hi)
    if lo > hi:
        lo, hi = hi, lo
    value = float(value)
    if lo <= value <= hi:
        return 0.0
    return min(abs(value - lo), abs(value - hi))


def token_edge_distance_to_point(token, point):

    return _point_to_token_edge_distance(point, token)


def token_edge_distance_to_line(token, abc):

    return _line_to_token_edge_distance(abc, token)


def token_projection_t_interval(token, seg):

    return _token_projection_t_interval(seg, token)


def select_token_near_point(tokens, expected_char, point, max_dist):
    if not tokens or point is None:
        return None
    expected = None if expected_char is None else str(expected_char).strip().upper()[:1]
    px, py = float(point[0]), float(point[1])
    best = None
    best_key = None
    for t in tokens:
        letters = str(t.get("letters", "")).upper()
        if not letters:
            letters = str(t.get("char", "")).upper()
        if expected is not None and expected not in letters:
            continue
        d = _point_to_token_edge_distance((px, py), t)
        if d > float(max_dist):
            continue
        key = (d, -float(t.get("conf", 0.0)))
        if best_key is None or key < best_key:
            best_key = key
            best = t
    return best


def select_token_near_line(tokens, expected_char, line_item, max_perp, t_margin=0.20):
    if not tokens or line_item is None:
        return None
    if "abc" not in line_item or "seg" not in line_item:
        return None
    expected = None if expected_char is None else str(expected_char).strip().upper()[:1]
    best = None
    best_key = None
    for t in tokens:
        letters = str(t.get("letters", "")).upper()
        if not letters:
            letters = str(t.get("char", "")).upper()
        if expected is not None and expected not in letters:
            continue
        perp = _line_to_token_edge_distance(line_item["abc"], t)
        if perp > float(max_perp):
            continue
        t_int = _token_projection_t_interval(line_item["seg"], t)
        if t_int is None:
            continue
        tt_lo, tt_hi = float(t_int[0]), float(t_int[1])
        if tt_hi < -float(t_margin) or tt_lo > (1.0 + float(t_margin)):
            continue
        off = _interval_distance_to_value(tt_lo, tt_hi, 0.5)
        key = (perp, off, -float(t.get("conf", 0.0)))
        if best_key is None or key < best_key:
            best_key = key
            best = t
    return best







def _refine_circle_radius_by_inner_outer_edges(img, circle, n_angles=180):
    if circle is None:
        return None
    cx, cy, r0 = float(circle[0]), float(circle[1]), float(circle[2])
    if r0 <= 1.0:
        return (cx, cy, r0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    h, w = bw.shape[:2]

    search = float(max(8.0, 0.12 * r0))
    r_min = max(2.0, r0 - search)
    r_max = min(float(max(h, w)) * 2.0, r0 + search)
    if r_max <= r_min + 2.0:
        return (cx, cy, r0)

    ang = np.linspace(0.0, 2.0 * math.pi, int(max(30, n_angles)), endpoint=False)
    inner_rs = []
    outer_rs = []


    rs = np.arange(r_min, r_max + 1.0, 1.0, dtype=np.float32)

    min_thick = int(max(2.0, 0.01 * r0))

    for th in ang:
        xs = cx + rs * math.cos(th)
        ys = cy + rs * math.sin(th)
        xi = np.round(xs).astype(np.int32)
        yi = np.round(ys).astype(np.int32)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        if ok.sum() < max(8, min_thick + 2):
            continue
        xi = xi[ok]
        yi = yi[ok]
        rsv = rs[ok]
        vals = (bw[yi, xi] > 0).astype(np.uint8)
        if vals.max() == 0:
            continue


        idx = np.where(vals > 0)[0]
        if idx.size == 0:
            continue
        cuts = np.where(np.diff(idx) > 1)[0]
        runs = np.split(idx, cuts + 1)

        best_run = None
        best_d = None
        for run in runs:
            if run.size < min_thick:
                continue
            rin = float(rsv[run[0]])
            rout = float(rsv[run[-1]])
            rmid = 0.5 * (rin + rout)
            d = abs(rmid - r0)
            if best_d is None or d < best_d:
                best_d = d
                best_run = (rin, rout)
        if best_run is None:
            continue
        rin, rout = best_run
        if rout <= rin:
            continue
        inner_rs.append(rin)
        outer_rs.append(rout)

    if len(inner_rs) < 12 or len(outer_rs) < 12:
        return (cx, cy, r0)

    r_in = float(np.median(np.array(inner_rs, dtype=np.float32)))
    r_out = float(np.median(np.array(outer_rs, dtype=np.float32)))
    if not (r_out > r_in + 0.5):
        return (cx, cy, r0)
    r_mid = 0.5 * (r_in + r_out)

    if abs(r_mid - r0) > max(12.0, 0.20 * r0):
        return (cx, cy, r0)
    return (cx, cy, r_mid)


def _merge_circle_candidates(candidates, center_tol, radius_tol):
    merged = []
    for x, y, r, s in candidates:
        placed = False
        for i, (mx, my, mr, ms) in enumerate(merged):
            if math.hypot(x - mx, y - my) <= center_tol and abs(r - mr) <= radius_tol:
                if s > ms:
                    merged[i] = (x, y, r, s)
                placed = True
                break
        if not placed:
            merged.append((x, y, r, s))
    return merged


def _extract_edge_points(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    ys, xs = np.where(edges > 0)
    if xs.size < 300:
        return None, None, None, None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    max_pts = int(os.getenv("PLANE_RANSAC_MAX_EDGE_POINTS", "2500"))
    if max_pts > 0 and pts.shape[0] > max_pts:
        step = int(math.ceil(pts.shape[0] / float(max_pts)))
        pts = pts[::step]
    h, w = gray.shape[:2]
    return pts, h, w, min(h, w)


def _circle_from_3pts(a, b, c):
    x1, y1 = a;
    x2, y2 = b;
    x3, y3 = c
    A = x1 - x2;
    B = y1 - y2;
    C = x1 - x3;
    D = y1 - y3
    E = (x1 * x1 - x2 * x2 + y1 * y1 - y2 * y2) / 2.0
    F = (x1 * x1 - x3 * x3 + y1 * y1 - y3 * y3) / 2.0
    det = A * D - B * C
    if abs(det) < 1e-6:
        return None
    cx = (D * E - B * F) / det
    cy = (-C * E + A * F) / det
    r = math.hypot(cx - x1, cy - y1)
    return float(cx), float(cy), float(r)


def _best_circle_from_pts(pts, rng, min_r, max_r, iters=3000, score_th=120):
    if pts is None or pts.shape[0] < 200:
        return None, None, 0
    best = None
    best_mask = None
    best_score = 0
    iters = int(min(int(iters), int(os.getenv("PLANE_RANSAC_MAX_ITERS", "1200"))))
    early_score = max(int(score_th), int(round(0.18 * float(pts.shape[0]))))
    for _ in range(iters):
        i1, i2, i3 = rng.choice(pts.shape[0], 3, replace=False)
        c = _circle_from_3pts(pts[i1], pts[i2], pts[i3])
        if c is None:
            continue
        cx, cy, r = c
        if r < min_r or r > max_r:
            continue
        d = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        tol = max(2.5, 0.01 * r)
        mask_score = np.abs(d - r) <= tol
        score = int(np.sum(mask_score))
        if score > best_score:
            best_score = score
            best = (cx, cy, r)
            band = max(4.0, 0.03 * r)
            best_mask = np.abs(d - r) <= band
            if best_score >= early_score:
                break
    if best is None or best_score < score_th:
        return None, None, 0
    return best, best_mask, best_score


def _find_top_k_circles_from_pts(pts, m, k, min_r=6, max_r=0, seed=0, iters=3000, score_th=120):
    if pts is None:
        return [], []
    if max_r == 0:
        max_r = int(0.60 * m)
    rng = np.random.default_rng(seed)
    found = []
    scores = []
    cur = pts
    for _ in range(max(1, k)):
        c, mask, score = _best_circle_from_pts(cur, rng, min_r, max_r, iters=iters, score_th=score_th)
        if c is None:
            break
        found.append(c)
        scores.append(score)
        cur = cur[~mask]
    return found, scores


def _find_top_k_circles(img, k, min_r=6, max_r=0, seed=0, iters=3000, score_th=120):
    pts, h, w, m = _extract_edge_points(img)
    if pts is None:
        return [], []
    return _find_top_k_circles_from_pts(pts, m, k, min_r=min_r, max_r=max_r, seed=seed, iters=iters, score_th=score_th)


def detect_circle(img, order=1, min_r=6, max_r=0):
    found, scores = _find_top_k_circles(img, k=order, min_r=min_r, max_r=max_r)
    idx = order - 1
    if idx < 0 or idx >= len(found):
        return None
    cx, cy, r = found[idx]
    cx, cy, r = _refine_circle_radius_by_inner_outer_edges(img, (cx, cy, r))
    return (float(cx), float(cy), float(r))


def detecting_concentric_circles(img, t_circle, order):
    x0, y0, r0 = t_circle
    found, scores = _find_top_k_circles(img, k=80, min_r=6, max_r=0, seed=1, iters=4500, score_th=25)
    center_tol = max(4.0, 0.02 * r0)
    band = max(6.0, 0.035 * r0)
    raw = []
    for (cx, cy, r), s in zip(found, scores):
        if math.hypot(cx - x0, cy - y0) > center_tol:
            continue
        if abs(r - r0) <= band:
            continue
        if r >= r0:
            continue
        raw.append((float(cx), float(cy), float(r), int(s)))
    raw.sort(key=lambda t: (r0 - t[2], -t[3]))
    merged = _merge_circle_candidates(raw, center_tol=center_tol, radius_tol=band)
    merged.sort(key=lambda t: (r0 - t[2], -t[3]))
    idx = order - 1
    if idx < 0 or idx >= len(merged):
        return None

    refined = _refine_circle_radius_by_inner_outer_edges(img, (merged[idx][0], merged[idx][1], merged[idx][2]))
    return refined


def detect_circles_between_two_circles(img, c1, c2):
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    R = max(r1, r2)
    r = min(r1, r2)
    found, scores = _find_top_k_circles(img, k=120, min_r=6, max_r=0, seed=2, iters=4500, score_th=25)
    tol_band = max(6.0, 0.03 * R)
    center_tol = max(5.0, 0.02 * R)
    rad_merge = max(6.0, 0.04 * (R - r))
    circles = []
    for (x, y, rr), s in zip(found, scores):
        d = math.hypot(x - cx, y - cy)
        if (d - rr) < (r - tol_band):
            continue
        if (d + rr) > (R + tol_band):
            continue
        circles.append((float(x), float(y), float(rr), int(s)))
    circles.sort(key=lambda t: (-t[3], -t[2]))
    merged = _merge_circle_candidates(circles, center_tol=center_tol, radius_tol=rad_merge)
    merged.sort(key=lambda t: t[2], reverse=True)
    merged = [(x, y, rr) for (x, y, rr, s) in merged]
    refined_merged = [_refine_circle_radius_by_inner_outer_edges(img, c) for c in merged]
    refined_merged = [c for c in refined_merged if c is not None]
    return refined_merged


def detect_largest_circle(img):
    return detect_circle(img, order=1)


def detect_second_largest_circle(img):
    return detect_circle(img, order=2)


def detect_third_largest_circle(img):
    return detect_circle(img, order=3)


def detect_annulus(img):
    outer = detect_largest_circle(img)
    if outer is None:
        return None
    inner = detecting_concentric_circles(img, outer, order=1)
    if inner is None:
        return None
    return {"outer": outer, "inner": inner}


def detect_points_on_circle(img, circle):
    x0, y0, r0 = circle

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    h, w = bw.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    bi, bo = float(max(12.0, 0.03 * r0)), float(max(6.0, 0.03 * r0))
    ring = (((dist >= (r0 - bi)) & (dist <= (r0 + bo))).astype(np.uint8) * 255)
    bw = cv2.bitwise_and(bw, ring)

    rd = float(max(4.0, 0.03 * r0))
    n = int(max(60, round(2.0 * math.pi * r0 / max(1.0, 0.6 * rd))))
    ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    score = np.zeros(n, np.float32)

    all_c = []
    all_x = []
    all_y = []
    all_score = []
    for i, th in enumerate(ang):
        cx, cy = x0 + r0 * math.cos(th), y0 + r0 * math.sin(th)
        x1, x2 = max(0, int(cx - rd - 1)), min(w, int(cx + rd + 2))
        y1, y2 = max(0, int(cy - rd - 1)), min(h, int(cy + rd + 2))
        if x2 <= x1 or y2 <= y1:
            continue
        yy, xx = np.ogrid[y1:y2, x1:x2]
        disk = ((xx - cx) ** 2 + (yy - cy) ** 2 <= rd * rd)
        den = int(disk.sum())
        if den:
            score[i] = float((bw[y1:y2, x1:x2] > 0)[disk].sum()) / float(den)
        all_c.append((cx, cy))
        all_x.append(cx)
        all_y.append(cy)
        all_score.append(score[i])

    threshold = 0.95
    m = score >= threshold

    if not m.any():
        dbg = img.copy()
        cv2.circle(dbg, (int(x0), int(y0)), int(r0), (0, 255, 0), 2)
        return []

    idx = np.where(m)[0]
    cuts = np.where(np.diff(idx) > 1)[0]
    segs = np.split(idx, cuts + 1)
    if m[0] and m[-1] and len(segs) > 1:
        segs = [np.r_[segs[-1], segs[0]]] + segs[1:-1]

    pts = []
    for s in segs:
        ss = np.sort(s).astype(int)
        if m[0] and m[-1] and len(segs) > 1 and ((ss == 0).any() or (ss == (len(ang) - 1)).any()):
            ss = np.where(ss < (len(ang) // 2), ss + len(ang), ss)
            ss = np.sort(ss)
        i1, i2 = int(ss[0] % len(ang)), int(ss[-1] % len(ang))
        th1, th2 = float(ang[i1]), float(ang[i2])
        if th2 < th1:
            th2 += 2.0 * math.pi
        thm = 0.5 * (th1 + th2)
        pts.append((x0 + r0 * math.cos(thm), y0 + r0 * math.sin(thm)))

    pts.sort(key=lambda p: -math.atan2(p[1] - y0, p[0] - x0))
    return pts



def detect_points_on_line(img, line, mode="internal"):
    (x1, y1), (x2, y2) = line
    x1 = float(x1);
    y1 = float(y1);
    x2 = float(x2);
    y2 = float(y2)
    vx = x2 - x1
    vy = y2 - y1
    denom = vx * vx + vy * vy
    if img is None or denom <= 1e-6:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    h, w = bw.shape[:2]
    Y, X = np.ogrid[:h, :w]
    seg_len = float(math.hypot(vx, vy))
    band = float(max(12.0, 0.03 * seg_len))
    dist_line = np.abs((X - x1) * vy - (Y - y1) * vx) / float(max(1e-6, seg_len))
    tmap = (((X - x1) * vx + (Y - y1) * vy) / float(denom)).astype(np.float32)
    margin = float(max(0.08, band / float(max(1.0, seg_len))))

    if mode == "internal":
        keep_t = (tmap >= (0.0 - margin)) & (tmap <= (1.0 + margin))
    elif mode == "external":
        keep_t = (tmap < (0.0 - margin)) | (tmap > (1.0 + margin))
    elif mode == "all":
        keep_t = np.ones_like(tmap, dtype=bool)
    else:
        raise ValueError("Invalid mode. Use 'internal', 'external', or 'all'.")

    band_mask = ((dist_line <= band) & keep_t).astype(np.uint8) * 255
    bw = cv2.bitwise_and(bw, band_mask)

    rd = float(max(6.0, 0.04 * seg_len))
    n = int(max(80, round(seg_len / max(1.0, 0.6 * rd))))
    ts = np.linspace(0.0, 1.0, n, endpoint=True)
    score = np.zeros(n, np.float32)

    for i, t in enumerate(ts):
        cx = x1 + t * vx
        cy = y1 + t * vy
        xL, xR = max(0, int(cx - rd - 1)), min(w, int(cx + rd + 2))
        yT, yB = max(0, int(cy - rd - 1)), min(h, int(cy + rd + 2))
        if xR <= xL or yB <= yT:
            continue
        yy, xx = np.ogrid[yT:yB, xL:xR]
        disk = ((xx - cx) ** 2 + (yy - cy) ** 2 <= rd * rd)
        den = int(disk.sum())
        if den:
            score[i] = float((bw[yT:yB, xL:xR] > 0)[disk].sum()) / float(den)

    thr = 0.95
    m = score >= thr

    pts = []
    if m.any():
        idx = np.where(m)[0]
        cuts = np.where(np.diff(idx) > 1)[0]
        segs = np.split(idx, cuts + 1)
        for s in segs:
            ss = np.sort(s).astype(int)
            tmid = float(0.5 * (ts[ss[0]] + ts[ss[-1]]))
            pts.append((x1 + tmid * vx, y1 + tmid * vy))
        pts.sort(key=lambda p: ((p[0] - x1) * vx + (p[1] - y1) * vy))







    return pts


def has_point_at_point(img, point):
    if point is None: return False
    px, py = int(round(float(point[0]))), int(round(float(point[1])))
    h, w = img.shape[:2]
    if not (0 <= px < w and 0 <= py < h): return False

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    g = cv2.GaussianBlur(g, (3, 3), 0)
    bw = cv2.bitwise_not(cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    s = float(min(h, w))
    r = int(max(1.5, 0.008 * s))
    n = 96
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = (px + r * np.cos(th)).round().astype(np.int32)
    ys = (py + r * np.sin(th)).round().astype(np.int32)
    m = (0 <= xs) & (xs < w) & (0 <= ys) & (ys < h)
    cover = float((bw[ys[m], xs[m]] > 0).mean()) if m.any() else 0.0
    ok = cover >= 0.95
    return ok



def circle_tangent_to_circle(c1, c2, eps, mode):
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    d = math.hypot(x1 - x2, y1 - y2)
    e_ext = abs(d - (r1 + r2))
    e_int = abs(d - abs(r1 - r2))
    if mode == "external":
        return (e_ext <= eps), e_ext
    elif mode == "internal":
        return (e_int <= eps), e_int
    else:
        raise ValueError("Invalid mode for tangency check. Use 'external' or 'internal'. ")


def get_mid_point(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def get_perpendicular_line_through_point(start, end, order, d=50.0):
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    dx = x2 - x1
    dy = y2 - y1
    n = math.hypot(dx, dy)
    if n <= 1e-9:
        return (x2, y2), (x2, y2)
    o = order.strip().lower()
    cw = ("cw", "clockwise")
    ccw = ("ccw", "counterclockwise")
    if o in cw:
        px, py = (-dy / n), (dx / n)
    elif o in ccw:
        px, py = (dy / n), (-dx / n)
    else:
        raise ValueError("Invalid order. Use 'clockwise'/'counterclockwise' (or 'cw'/'ccw'). ")
    ex = x2 + px * float(d)
    ey = y2 + py * float(d)
    return (x2, y2), (ex, ey)


def get_intersection_of_lines(line1, line2, tol=1e-9):
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    x1 = float(x1);
    y1 = float(y1);
    x2 = float(x2);
    y2 = float(y2)
    x3 = float(x3);
    y3 = float(y3);
    x4 = float(x4);
    y4 = float(y4)
    rx = x2 - x1
    ry = y2 - y1
    sx = x4 - x3
    sy = y4 - y3
    nr = math.hypot(rx, ry)
    ns = math.hypot(sx, sy)
    if nr <= 1e-12 or ns <= 1e-12:
        return None
    denom = rx * sy - ry * sx
    if abs(denom) <= tol * nr * ns:
        return None
    qpx = x3 - x1
    qpy = y3 - y1
    t = (qpx * sy - qpy * sx) / denom
    return (x1 + t * rx, y1 + t * ry)


def has_line_between_points(img, p1, p2, allow_dash=True):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    h, w = img.shape[:2]
    dx = x2 - x1
    dy = y2 - y1
    L = math.hypot(dx, dy)
    if L <= 1e-6:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    ink = (bw > 0).astype(np.uint8) * 255
    dt = cv2.distanceTransform(255 - ink, cv2.DIST_L2, 3)
    r = float(max(2.5, 0.006 * min(h, w)))
    n = int(max(25.0, round(L)))
    hit = 0
    for i in range(n + 1):
        t = float(i) / float(max(1, n))
        x = x1 + t * dx
        y = y1 + t * dy
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            continue
        if float(dt[yi, xi]) <= r:
            hit += 1

    total = int(n + 1)
    hit_ratio = float(hit) / float(max(1, total))
    thr = 0.40 if allow_dash else 0.90
    ok = hit_ratio >= thr
    return ok


def get_point_position_relative_to_circle(point, circle):
    px, py = float(point[0]), float(point[1])
    cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
    d = math.hypot(px - cx, py - cy)
    if abs(d - r) <= 1e-6:
        return "on"
    elif d < r:
        return "inside"
    else:
        return "outside"


def _gray_and_ink_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]



    if img.ndim == 3:
        b = img[:, :, 0].astype(np.int16)
        g = img[:, :, 1].astype(np.int16)
        r = img[:, :, 2].astype(np.int16)
        chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype(np.int16)
        v = hsv[:, :, 2].astype(np.int16)

        color_mask = (
            (chroma >= 22)
            & (s >= 35)
            & (v <= 245)
            & (gray <= 250)
        )
        bw_color = (color_mask.astype(np.uint8) * 255)
        bw = cv2.bitwise_or(bw_gray, bw_color)
    else:
        bw = bw_gray

    return gray, bw


def dedup_points(points, tol=8.0):
    deduped = []
    for p in points:
        x, y = float(p[0]), float(p[1])
        if all(math.hypot(x - q[0], y - q[1]) > float(tol) for q in deduped):
            deduped.append((x, y))
    return deduped


def detect_marker_points(img):

    h, w = img.shape[:2]
    _, bw = _gray_and_ink_mask(img)
    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    num, labels, stats, cents = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), 8)
    pts = []
    area_min = max(20, int(0.000025 * h * w))
    area_max = max(area_min + 1, int(0.0045 * h * w))
    max_side = max(30, int(0.12 * min(h, w)))
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < area_min or area > area_max:
            continue
        if ww <= 0 or hh <= 0 or ww > max_side or hh > max_side:
            continue
        fill = area / float(max(1, ww * hh))
        aspect = ww / float(max(1, hh))
        if fill < 0.30:
            continue
        if not (0.45 <= aspect <= 2.2):
            continue
        x, y = float(cents[i][0]), float(cents[i][1])
        pts.append((x, y))
    return dedup_points(pts, tol=max(6.0, 0.018 * min(h, w)))


def segment_angle_deg(seg):
    x1, y1, x2, y2 = [float(v) for v in seg]
    return (math.degrees(math.atan2(y2 - y1, x2 - x1)) + 180.0) % 180.0


def angle_diff_deg(a1, a2):
    d = abs(float(a1) - float(a2)) % 180.0
    return min(d, 180.0 - d)


def horizontal_error_deg(ang):
    return min(angle_diff_deg(float(ang), 0.0), angle_diff_deg(float(ang), 180.0))


def line_equivalent(a, b, min_hw, angle_tol_deg=4.0, offset_tol_px=None):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if "ang" not in a or "ang" not in b or "abc" not in a or "abc" not in b:
        return False
    if angle_diff_deg(a["ang"], b["ang"]) > float(angle_tol_deg):
        return False
    tol = float(offset_tol_px) if offset_tol_px is not None else (0.03 * float(min_hw))
    try:
        ca = float(a["abc"][2])
        cb = float(b["abc"][2])
    except Exception:
        return False
    return abs(ca - cb) <= tol


def segment_endpoints_lr(seg):
    x1, y1, x2, y2 = [float(v) for v in seg]
    if x1 < x2:
        return (x1, y1), (x2, y2)
    if x1 > x2:
        return (x2, y2), (x1, y1)
    if y1 <= y2:
        return (x1, y1), (x2, y2)
    return (x2, y2), (x1, y1)


def middle_gap_px_on_segment(
    img,
    seg,
    t_lo=0.35,
    t_hi=0.65,
    probe_half_width=2,
    sample_min=80,
):
    if img is None or seg is None:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in seg]
    L = float(math.hypot(x2 - x1, y2 - y1))
    if L <= 1.0:
        return 0.0
    _, bw = _gray_and_ink_mask(img)
    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    bin_img = (bw > 0).astype(np.uint8)
    n = max(int(sample_min), int(round(L)))
    th = int(max(1, probe_half_width))
    miss = 0
    max_mid_miss = 0
    for i in range(n + 1):
        t = float(i) / float(max(1, n))
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        xL = max(0, int(round(x - th)))
        xR = min(bin_img.shape[1], int(round(x + th + 1)))
        yT = max(0, int(round(y - th)))
        yB = min(bin_img.shape[0], int(round(y + th + 1)))
        has_ink = bool(bin_img[yT:yB, xL:xR].any())
        if has_ink:
            miss = 0
        else:
            miss += 1
            if float(t_lo) <= t <= float(t_hi) and miss > max_mid_miss:
                max_mid_miss = miss
    return float(max_mid_miss) * L / float(max(1, n))


def middle_gap_px_on_line(img, line_item, **kwargs):
    if not isinstance(line_item, dict) or "seg" not in line_item:
        return 0.0
    return middle_gap_px_on_segment(img, line_item["seg"], **kwargs)


def has_non_horizontal_endpoint_branches(
    img,
    line_items,
    near_r_px=None,
    min_len_px=None,
    max_len_px=None,
    horizontal_tol_deg=15.0,
    min_total_non_h=4,
    min_endpoint_hits=3,
):
    if img is None or not line_items:
        return False
    h, w = img.shape[:2]
    min_hw = float(min(h, w))
    endpoints = []
    for ln in line_items:
        seg = ln["seg"] if isinstance(ln, dict) and "seg" in ln else ln
        try:
            left, right = segment_endpoints_lr(seg)
        except Exception:
            continue
        endpoints.append(left)
        endpoints.append(right)
    if not endpoints:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    edges = cv2.Canny(gray, 60, 160)
    raw = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=18,
        minLineLength=max(8, int(0.015 * min_hw)),
        maxLineGap=max(2, int(0.008 * min_hw)),
    )
    if raw is None:
        return False

    near_r = float(near_r_px) if near_r_px is not None else max(34.0, 0.045 * min_hw)
    min_len = float(min_len_px) if min_len_px is not None else max(9.0, 0.012 * min_hw)
    max_len = float(max_len_px) if max_len_px is not None else max(40.0, 0.11 * min_hw)
    endpoint_hits = [0 for _ in endpoints]
    total_non_h = 0

    for seg in raw[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in seg]
        L = float(math.hypot(x2 - x1, y2 - y1))
        if L < min_len or L > max_len:
            continue
        ang = float(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if horizontal_error_deg(ang) <= float(horizontal_tol_deg):
            continue
        mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        dists = [math.hypot(mx - p[0], my - p[1]) for p in endpoints]
        k = int(np.argmin(dists))
        if dists[k] > near_r:
            continue
        endpoint_hits[k] += 1
        total_non_h += 1

    hit_eps = sum(1 for v in endpoint_hits if v > 0)
    return bool(total_non_h >= int(min_total_non_h) and hit_eps >= int(min_endpoint_hits))


def segment_to_abc(seg):
    x1, y1, x2, y2 = [float(v) for v in seg]
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    n = math.hypot(a, b)
    if n <= 1e-9:
        return None
    a /= n
    b /= n
    c /= n
    if a < 0 or (abs(a) < 1e-12 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)


def point_line_distance(pt, abc):
    x, y = float(pt[0]), float(pt[1])
    a, b, c = abc
    return abs(a * x + b * y + c)


def project_point_to_line(pt, abc):
    x, y = float(pt[0]), float(pt[1])
    a, b, c = abc
    d = a * x + b * y + c
    return (x - a * d, y - b * d)


def line_intersection_from_abc(abc1, abc2, tol=1e-9):
    a1, b1, c1 = abc1
    a2, b2, c2 = abc2
    det = a1 * b2 - a2 * b1
    if abs(det) <= tol:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return (float(x), float(y))


def circle_line_intersections(circle, abc):
    cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
    a, b, c = abc
    d = a * cx + b * cy + c
    if abs(d) > r + 1e-6:
        return []
    foot = (cx - a * d, cy - b * d)
    rr = max(0.0, r * r - d * d)
    t = math.sqrt(rr)
    dx, dy = -b, a
    p1 = (foot[0] + dx * t, foot[1] + dy * t)
    p2 = (foot[0] - dx * t, foot[1] - dy * t)
    if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) <= 1e-6:
        return [p1]
    return [p1, p2]


def detect_line_segments_raw(img, min_len_ratio=0.12):

    if img is None:
        return []

    h, w = img.shape[:2]
    min_hw = float(min(h, w))
    gray, bw = _gray_and_ink_mask(img)
    threshold = max(18, int(0.04 * min_hw))
    min_len = max(24, int(float(min_len_ratio) * min_hw))
    max_gap = max(8, int(0.03 * min_hw))

    lines = cv2.HoughLinesP(
        bw,
        1,
        np.pi / 180,
        threshold=threshold,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )
    if lines is None:
        edge = cv2.Canny(gray, 50, 140)
        lines = cv2.HoughLinesP(
            edge,
            1,
            np.pi / 180,
            threshold=max(14, int(0.03 * min_hw)),
            minLineLength=max(18, int(0.10 * min_hw)),
            maxLineGap=max(10, int(0.04 * min_hw)),
        )
        if lines is None:
            return []

    raw = []
    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in l]
        length = float(math.hypot(x2 - x1, y2 - y1))
        if length < min_len:
            continue
        seg = (x1, y1, x2, y2)
        abc = segment_to_abc(seg)
        if abc is None:
            continue
        raw.append(
            {
                "seg": seg,
                "abc": abc,
                "len": length,
                "ang": segment_angle_deg(seg),
            }
        )

    raw.sort(key=lambda t: -t["len"])
    return raw


def detect_line_segments(img, min_len_ratio=0.12):
    if img is None:
        return []
    h, w = img.shape[:2]
    min_hw = float(min(h, w))

    raw = detect_line_segments_raw(img, min_len_ratio=min_len_ratio)
    if not raw:
        return []

    dedup = []
    for it in raw:
        keep = True
        for jt in dedup:
            if angle_diff_deg(it["ang"], jt["ang"]) > 3.0:
                continue
            if abs(it["abc"][2] - jt["abc"][2]) > max(8.0, 0.025 * min_hw):
                continue
            keep = False
            break
        if keep:
            dedup.append(it)
    return dedup


def collinear_support_stats_on_segment(
    lines,
    reference_line,
    base_seg,
    min_hw,
    angle_tol_deg=3.0,
    offset_ratio=0.025,
    offset_floor_px=8.0,
    reach_left_t=0.12,
    reach_right_t=0.88,
):

    empty = {
        "coverage": 0.0,
        "center_supported": False,
        "center_gap": 1.0,
        "reaches_left": False,
        "reaches_right": False,
        "intervals": [],
        "segment_count": 0,
    }
    if not lines or reference_line is None or base_seg is None:
        return empty
    if not isinstance(reference_line, dict):
        return empty
    if "ang" not in reference_line or "abc" not in reference_line:
        return empty

    try:
        bx1, by1, bx2, by2 = [float(v) for v in base_seg[:4]]
    except Exception:
        return empty
    base = (bx1, by1, bx2, by2)
    offset_tol = max(float(offset_floor_px), float(offset_ratio) * float(min_hw))

    intervals = []
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        if "ang" not in ln or "abc" not in ln or "seg" not in ln:
            continue
        if angle_diff_deg(ln["ang"], reference_line["ang"]) > float(angle_tol_deg):
            continue
        if abs(float(ln["abc"][2]) - float(reference_line["abc"][2])) > offset_tol:
            continue
        try:
            sx1, sy1, sx2, sy2 = [float(v) for v in ln["seg"][:4]]
        except Exception:
            continue
        t1 = segment_projection_t(base, (sx1, sy1))
        t2 = segment_projection_t(base, (sx2, sy2))
        lo, hi = min(t1, t2), max(t1, t2)
        a, b = max(0.0, lo), min(1.0, hi)
        if b > a:
            intervals.append((a, b))

    if not intervals:
        return empty

    intervals.sort(key=lambda z: z[0])
    merged = []
    for a, b in intervals:
        if not merged or a > merged[-1][1]:
            merged.append([float(a), float(b)])
        else:
            merged[-1][1] = max(float(merged[-1][1]), float(b))

    merged_tuples = [(float(a), float(b)) for a, b in merged]
    coverage = float(sum(float(b) - float(a) for a, b in merged_tuples))
    center_supported = any(float(a) <= 0.5 <= float(b) for a, b in merged_tuples)
    if center_supported:
        center_gap = 0.0
    else:
        center_gap = float(min(min(abs(0.5 - float(a)), abs(0.5 - float(b))) for a, b in merged_tuples))

    reaches_left = bool(merged_tuples and float(merged_tuples[0][0]) <= float(reach_left_t))
    reaches_right = bool(merged_tuples and float(merged_tuples[-1][1]) >= float(reach_right_t))

    return {
        "coverage": coverage,
        "center_supported": bool(center_supported),
        "center_gap": float(center_gap),
        "reaches_left": bool(reaches_left),
        "reaches_right": bool(reaches_right),
        "intervals": merged_tuples,
        "segment_count": int(len(intervals)),
    }


def find_lines_through_point(lines, point, tol):
    x, y = float(point[0]), float(point[1])
    out = []
    for item in lines:
        d = point_line_distance((x, y), item["abc"])
        if d <= float(tol):
            out.append(item)
    return out


def segment_ink_ratio(img, p1, p2, thickness=2, trim_ratio=0.0):
    h, w = img.shape[:2]
    _, bw = _gray_and_ink_mask(img)
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx = x2 - x1
    dy = y2 - y1
    L = math.hypot(dx, dy)
    if L <= 1e-6:
        return 0.0
    t0 = max(0.0, float(trim_ratio))
    t1 = min(1.0, 1.0 - float(trim_ratio))
    if t1 <= t0:
        t0, t1 = 0.0, 1.0
    xa = x1 + dx * t0
    ya = y1 + dy * t0
    xb = x1 + dx * t1
    yb = y1 + dy * t1

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.line(
        mask,
        (int(round(xa)), int(round(ya))),
        (int(round(xb)), int(round(yb))),
        255,
        int(max(1, thickness)),
    )
    den = int((mask > 0).sum())
    if den <= 0:
        return 0.0
    num = int(((bw > 0) & (mask > 0)).sum())
    return float(num) / float(den)


def has_segment_between_points(img, p1, p2, ratio_th=0.22, thickness=2, trim_ratio=0.06):
    ratio = segment_ink_ratio(img, p1, p2, thickness=thickness, trim_ratio=trim_ratio)
    return bool(ratio >= float(ratio_th)), float(ratio)


def point_on_circle(pt, circle, tol):
    cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
    d = math.hypot(float(pt[0]) - cx, float(pt[1]) - cy)
    return abs(d - r) <= float(tol)


def split_points_by_circle(points, circle, on_tol):
    cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
    inside, on, outside = [], [], []
    for p in points:
        d = math.hypot(float(p[0]) - cx, float(p[1]) - cy)
        if abs(d - r) <= float(on_tol):
            on.append((float(p[0]), float(p[1])))
        elif d < (r - float(on_tol)):
            inside.append((float(p[0]), float(p[1])))
        else:
            outside.append((float(p[0]), float(p[1])))
    return inside, on, outside


def nearest_point(points, target):
    if not points:
        return None
    tx, ty = float(target[0]), float(target[1])
    return min(points, key=lambda p: math.hypot(float(p[0]) - tx, float(p[1]) - ty))


def collect_line_pair_intersections(lines, min_angle_deg=8.0):
    pts = []
    n = len(lines)
    for i in range(n):
        for j in range(i + 1, n):
            li = lines[i]
            lj = lines[j]
            if angle_diff_deg(li["ang"], lj["ang"]) < float(min_angle_deg):
                continue
            p = line_intersection_from_abc(li["abc"], lj["abc"])
            if p is None:
                continue
            pts.append((float(p[0]), float(p[1]), i, j))
    return pts


def estimate_external_point_from_line_pairs(
    lines, circle, prefer_right=True, min_angle_deg=8.0, image_shape=None
):

    if circle is None:
        return None
    cx, cy, r = [float(v) for v in circle]
    candidates = []
    h = w = None
    if image_shape is not None and len(image_shape) >= 2:
        h = float(image_shape[0])
        w = float(image_shape[1])
    for x, y, i, j in collect_line_pair_intersections(lines, min_angle_deg=min_angle_deg):
        d = math.hypot(x - cx, y - cy)
        if d <= r + max(8.0, 0.08 * r):
            continue
        if d >= 6.0 * r:
            continue
        if h is not None and w is not None:
            if x < (-0.10 * w) or x > (1.10 * w) or y < (-0.10 * h) or y > (1.10 * h):
                continue
        score = x if prefer_right else -x
        score -= 0.30 * abs(y - cy)

        score += 0.10 * (lines[i]["len"] + lines[j]["len"])
        candidates.append((score, (x, y), i, j))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    _, p, i, j = candidates[0]
    return p


def intersections_on_circle_from_lines(lines, circle, min_angle_deg=8.0, on_tol=None):
    if circle is None:
        return []
    cx, cy, r = [float(v) for v in circle]
    if on_tol is None:
        on_tol = max(10.0, 0.10 * r)
    pts = []
    for x, y, i, j in collect_line_pair_intersections(lines, min_angle_deg=min_angle_deg):
        d = math.hypot(x - cx, y - cy)
        if abs(d - r) <= float(on_tol):
            pts.append((x, y))
    return dedup_points(pts, tol=max(8.0, 0.08 * r))


def scale_px(min_hw, ratio, floor_px=0.0, ceil_px=None):

    out = float(min_hw) * float(ratio)
    out = max(float(floor_px), out)
    if ceil_px is not None:
        out = min(out, float(ceil_px))
    return float(out)


def scale_area(h, w, ratio, floor_px=0):

    out = int(round(float(ratio) * float(h) * float(w)))
    return int(max(int(floor_px), out))


def evaluate_plane_task(image_path, pid, judge_fn, require_ocr=True, task_type="plane"):

    if not callable(judge_fn):
        return {
            "passed": False,
            "reasoning": "judge_fn is not callable.",
            "criteria": {},
            "meta": {"pid": int(pid), "type": str(task_type)},
        }

    img = cv2.imread(str(image_path))
    if img is None:
        return {
            "passed": False,
            "reasoning": f"Failed to read image: {image_path}",
            "criteria": {},
            "meta": {"pid": int(pid), "type": str(task_type)},
        }

    try:
        raw = judge_fn(img)
    except Exception as e:
        return {
            "passed": False,
            "reasoning": f"Exception during judging: {str(e)}",
            "criteria": {},
            "meta": {"pid": int(pid), "type": str(task_type)},
        }

    if isinstance(raw, (list, tuple)):
        passed = bool(raw[0]) if len(raw) > 0 else False
        reason = str(raw[1]) if len(raw) > 1 and raw[1] else ""
        criteria = {"rule_check": passed}
    elif isinstance(raw, dict):
        passed = bool(raw.get("passed", raw.get("overall_pass", False)))
        reason = str(raw.get("reasoning", raw.get("reason", "")))
        criteria = raw.get("criteria", {"rule_check": passed})
        if not isinstance(criteria, dict):
            criteria = {"rule_check": passed}
    elif raw is None:
        passed = False
        reason = "Judge returned None."
        criteria = {"rule_check": False}
    else:
        passed = bool(raw)
        reason = ""
        criteria = {"rule_check": passed}

    meta = {
        "pid": int(pid),
        "type": str(task_type),
        "raw_reason": reason,
        "judge_return_type": type(raw).__name__,
    }
    return {
        "passed": bool(passed),
        "reasoning": str(reason),
        "criteria": criteria,
        "meta": meta,
    }


def point_on_segment_support(
    line_item,
    point,
    min_hw,
    dist_ratio=0.02,
    dist_floor_px=0.0,
    t_min=-0.45,
    t_max=1.35,
):

    if point is None:
        return False
    if not isinstance(line_item, dict):
        return False
    if "abc" not in line_item or "seg" not in line_item:
        return False
    tol = scale_px(min_hw, dist_ratio, floor_px=dist_floor_px)
    if point_line_distance(point, line_item["abc"]) > tol:
        return False
    t = segment_projection_t(line_item["seg"], point)
    return bool(float(t_min) <= t <= float(t_max))


def _signed_polygon_area(points):
    if points is None or len(points) < 3:
        return 0.0
    s = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += float(x1) * float(y2) - float(x2) * float(y1)
    return 0.5 * s


def _polygon_area(points):
    return abs(_signed_polygon_area(points))


def _sort_points_clockwise(points):
    if not points:
        return []
    cx = sum(float(p[0]) for p in points) / float(len(points))
    cy = sum(float(p[1]) for p in points) / float(len(points))
    ordered = sorted(points, key=lambda p: math.atan2(float(p[1]) - cy, float(p[0]) - cx))
    if _signed_polygon_area(ordered) < 0:
        ordered = list(reversed(ordered))
    return ordered


def extract_polygon_from_lines(
    lines,
    img_shape,
    sides,
    min_len_ratio,
    top_k=12,
    min_angle_sep_deg=12.0,
    margin_ratio=0.15,
    point_tol_ratio=0.04,
    point_tol_floor_px=0.0,
    min_area_ratio=0.01,
    min_area_floor_px=0.0,
    support_t=(-0.45, 1.35),
):

    if lines is None or img_shape is None:
        return None
    try:
        h, w = img_shape[:2]
    except Exception:
        return None
    if sides is None or int(sides) < 3:
        return None
    n_sides = int(sides)
    min_hw = float(min(h, w))

    long_th = scale_px(min_hw, float(min_len_ratio), floor_px=1.0)
    candidates = []
    for it in lines:
        if not isinstance(it, dict):
            continue
        if "abc" not in it or "seg" not in it or "len" not in it or "ang" not in it:
            continue
        try:
            if float(it["len"]) < long_th:
                continue
        except Exception:
            continue
        candidates.append(it)
    candidates.sort(key=lambda t: float(t["len"]), reverse=True)
    candidates = candidates[: max(int(top_k), n_sides)]
    if len(candidates) < n_sides:
        return None

    margin = scale_px(min_hw, margin_ratio, floor_px=0.0)
    point_tol = scale_px(min_hw, point_tol_ratio, floor_px=point_tol_floor_px)
    min_area = max(float(min_area_floor_px), float(min_area_ratio) * float(h) * float(w))
    best = None
    best_score = None
    t_min, t_max = float(support_t[0]), float(support_t[1])

    for combo in itertools.combinations(candidates, n_sides):
        raw_pts = []
        for i in range(n_sides):
            for j in range(i + 1, n_sides):
                li = combo[i]
                lj = combo[j]
                if angle_diff_deg(li["ang"], lj["ang"]) < float(min_angle_sep_deg):
                    continue
                p = line_intersection_from_abc(li["abc"], lj["abc"])
                if p is None:
                    continue
                x, y = float(p[0]), float(p[1])
                if x < -margin or x > (float(w) + margin) or y < -margin or y > (float(h) + margin):
                    continue
                if not point_on_segment_support(li, p, min_hw, t_min=t_min, t_max=t_max):
                    continue
                if not point_on_segment_support(lj, p, min_hw, t_min=t_min, t_max=t_max):
                    continue
                raw_pts.append((x, y))

        pts = dedup_points(raw_pts, tol=point_tol)
        if len(pts) != n_sides:
            continue

        good = True
        for ln in combo:
            cnt = sum(1 for p in pts if point_on_segment_support(ln, p, min_hw, t_min=t_min, t_max=t_max))
            if cnt != 2:
                good = False
                break
        if not good:
            continue

        poly = _sort_points_clockwise(pts)
        area = _polygon_area(poly)
        if area < min_area:
            continue

        score = sum(float(ln["len"]) for ln in combo)
        if best_score is None or score > best_score:
            best_score = score
            best = {
                "lines": list(combo),
                "vertices": poly,
                "score": float(score),
                "area": float(area),
            }
    return best


def extract_triangle_candidates_from_lines(
    lines,
    img_shape,
    min_len_ratio=0.18,
    top_k=12,
    min_angle_sep_deg=10.0,
    margin_ratio=0.22,
    point_tol_ratio=0.045,
    point_tol_floor_px=0.0,
    min_area_ratio=0.008,
    min_area_floor_px=0.0,
    support_t=(-0.75, 1.35),
):

    if lines is None or img_shape is None:
        return []
    try:
        h, w = img_shape[:2]
    except Exception:
        return []
    min_hw = float(min(h, w))
    long_th = scale_px(min_hw, float(min_len_ratio), floor_px=1.0)

    candidates = []
    for it in lines:
        if not isinstance(it, dict):
            continue
        if "abc" not in it or "seg" not in it or "len" not in it or "ang" not in it:
            continue
        try:
            if float(it["len"]) < long_th:
                continue
        except Exception:
            continue
        candidates.append(it)
    candidates.sort(key=lambda t: float(t["len"]), reverse=True)
    candidates = candidates[: max(3, int(top_k))]
    if len(candidates) < 3:
        return []

    margin = scale_px(min_hw, margin_ratio, floor_px=0.0)
    point_tol = scale_px(min_hw, point_tol_ratio, floor_px=point_tol_floor_px)
    min_area = max(float(min_area_floor_px), float(min_area_ratio) * float(h) * float(w))
    t_min, t_max = float(support_t[0]), float(support_t[1])
    seen = set()
    out = []

    for combo in itertools.combinations(candidates, 3):
        raw_pts = []
        for i in range(3):
            for j in range(i + 1, 3):
                li = combo[i]
                lj = combo[j]
                if angle_diff_deg(li["ang"], lj["ang"]) < float(min_angle_sep_deg):
                    continue
                p = line_intersection_from_abc(li["abc"], lj["abc"])
                if p is None:
                    continue
                x, y = float(p[0]), float(p[1])
                if x < -margin or x > (float(w) + margin) or y < -margin or y > (float(h) + margin):
                    continue
                if not point_on_segment_support(li, p, min_hw, t_min=t_min, t_max=t_max):
                    continue
                if not point_on_segment_support(lj, p, min_hw, t_min=t_min, t_max=t_max):
                    continue
                raw_pts.append((x, y))

        pts = dedup_points(raw_pts, tol=point_tol)
        if len(pts) != 3:
            continue

        good = True
        for ln in combo:
            cnt = sum(1 for p in pts if point_on_segment_support(ln, p, min_hw, t_min=t_min, t_max=t_max))
            if cnt != 2:
                good = False
                break
        if not good:
            continue

        poly = _sort_points_clockwise(pts)
        area = _polygon_area(poly)
        if area < min_area:
            continue

        key = tuple(sorted((round(float(x), 1), round(float(y), 1)) for x, y in poly))
        if key in seen:
            continue
        seen.add(key)
        score = sum(float(ln["len"]) for ln in combo)
        out.append(
            {
                "lines": list(combo),
                "vertices": poly,
                "score": float(score),
                "area": float(area),
            }
        )

    out.sort(key=lambda t: (-float(t["area"]), -float(t["score"])))
    return out


def find_support_line(
    lines,
    p1,
    p2,
    min_hw,
    ang_tol_deg=11.0,
    dist_ratio=0.07,
    dist_floor_px=0.0,
):

    if not lines or p1 is None or p2 is None:
        return None
    target_ang = segment_angle_deg((float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])))
    dist_tol = scale_px(min_hw, dist_ratio, floor_px=dist_floor_px)
    best = None
    best_score = None
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        if "abc" not in ln or "ang" not in ln or "len" not in ln:
            continue
        ad = angle_diff_deg(target_ang, ln["ang"])
        if ad > float(ang_tol_deg):
            continue
        d1 = point_line_distance(p1, ln["abc"])
        d2 = point_line_distance(p2, ln["abc"])
        if d1 > dist_tol or d2 > dist_tol:
            continue
        score = float(d1 + d2 + 0.25 * ad - 0.0008 * float(ln["len"]))
        if best_score is None or score < best_score:
            best_score = score
            best = ln
    return best


def has_support_line(
    lines,
    p1,
    p2,
    min_hw,
    ang_tol_deg=11.0,
    dist_ratio=0.07,
    dist_floor_px=0.0,
):

    return find_support_line(
        lines,
        p1,
        p2,
        min_hw=min_hw,
        ang_tol_deg=ang_tol_deg,
        dist_ratio=dist_ratio,
        dist_floor_px=dist_floor_px,
    ) is not None


def collect_intersection_anchors(
    lines,
    img_shape,
    min_angle_sep_deg=10.0,
    margin_ratio=0.12,
    point_tol_ratio=0.03,
    point_tol_floor_px=0.0,
    support_dist_ratio=0.02,
    support_dist_floor_px=0.0,
    support_t=(-0.5, 1.35),
):

    if not lines or img_shape is None:
        return []
    try:
        h, w = img_shape[:2]
    except Exception:
        return []
    min_hw = float(min(h, w))
    margin = scale_px(min_hw, margin_ratio, floor_px=0.0)
    point_tol = scale_px(min_hw, point_tol_ratio, floor_px=point_tol_floor_px)
    t_min, t_max = float(support_t[0]), float(support_t[1])
    pts = []

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            li = lines[i]
            lj = lines[j]
            if not isinstance(li, dict) or not isinstance(lj, dict):
                continue
            if "abc" not in li or "abc" not in lj or "ang" not in li or "ang" not in lj:
                continue
            if angle_diff_deg(li["ang"], lj["ang"]) < float(min_angle_sep_deg):
                continue
            p = line_intersection_from_abc(li["abc"], lj["abc"])
            if p is None:
                continue
            x, y = float(p[0]), float(p[1])
            if x < -margin or x > (float(w) + margin) or y < -margin or y > (float(h) + margin):
                continue
            if not point_on_segment_support(
                li,
                p,
                min_hw,
                dist_ratio=support_dist_ratio,
                dist_floor_px=support_dist_floor_px,
                t_min=t_min,
                t_max=t_max,
            ):
                continue
            if not point_on_segment_support(
                lj,
                p,
                min_hw,
                dist_ratio=support_dist_ratio,
                dist_floor_px=support_dist_floor_px,
                t_min=t_min,
                t_max=t_max,
            ):
                continue
            pts.append((x, y))

    return dedup_points(pts, tol=point_tol)


def snap_points_to_anchors(points_by_name, anchors, snap_tol_px):

    if not isinstance(points_by_name, dict):
        return {}
    out = {}
    for name, p in points_by_name.items():
        if p is None:
            out[name] = None
            continue
        base = (float(p[0]), float(p[1]))
        if not anchors:
            out[name] = base
            continue
        q = nearest_point(anchors, base)
        if q is None:
            out[name] = base
            continue
        if math.hypot(float(q[0]) - base[0], float(q[1]) - base[1]) <= float(snap_tol_px):
            out[name] = (float(q[0]), float(q[1]))
        else:
            out[name] = base
    return out


def pick_best_tokens_by_char(tokens, target_chars, min_conf=0.0, match_mode="contains"):

    if tokens is None:
        return {}
    if isinstance(target_chars, str):
        chars = list(_normalize_letter_whitelist(target_chars))
    else:
        chars = list(_normalize_letter_whitelist("".join(str(ch) for ch in target_chars)))
    out = {}
    mode = str(match_mode).strip().lower()
    if mode not in {"contains", "exact"}:
        raise ValueError("match_mode must be 'contains' or 'exact'.")

    for ch in chars:
        best = None
        best_conf = None
        for t in tokens:
            if not isinstance(t, dict):
                continue
            try:
                conf = float(t.get("conf", 0.0))
            except Exception:
                conf = 0.0
            if conf < float(min_conf):
                continue
            letters = str(t.get("letters", "")).upper()
            if not letters:
                letters = str(t.get("char", "")).upper()
            token_char = str(t.get("char", "")).upper()[:1]
            if mode == "contains":
                ok = (ch in letters) or (token_char == ch)
            else:
                ok = token_char == ch
            if not ok:
                continue
            if best_conf is None or conf > best_conf:
                best = t
                best_conf = conf
        if best is not None:
            out[ch] = best
    return out


def assign_labels_to_vertices_min_cost(tokens_by_char, vertices, target_labels):

    if not isinstance(tokens_by_char, dict):
        return None, None
    if vertices is None or target_labels is None:
        return None, None
    labels = [str(ch).upper()[:1] for ch in target_labels]
    if len(vertices) != len(labels):
        return None, None
    if any(ch not in tokens_by_char for ch in labels):
        return None, None

    dmat = {}
    for ch in labels:
        tok = tokens_by_char.get(ch)
        if not isinstance(tok, dict):
            return None, None
        dmat[ch] = [_point_to_token_edge_distance((float(v[0]), float(v[1])), tok) for v in vertices]
        if any(not np.isfinite(d) for d in dmat[ch]):
            return None, None

    n = len(labels)
    best_perm = None
    best_cost = None
    for perm in itertools.permutations(range(n), n):
        cost = 0.0
        for i, ch in enumerate(labels):
            cost += float(dmat[ch][perm[i]])
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm
    if best_perm is None:
        return None, None

    assign = {ch: int(best_perm[i]) for i, ch in enumerate(labels)}
    dists = {ch: float(dmat[ch][assign[ch]]) for ch in labels}
    return assign, dists


def match_labels_in_cycle(
    tokens,
    vertices,
    target_labels,
    max_dist,
    allow_reversed=True,
    anchor_label=None,
    min_conf=0.0,
    single_char_only=False,
):

    if vertices is None or target_labels is None:
        return False, [], []
    labels = [str(ch).upper()[:1] for ch in target_labels]
    n = len(labels)
    if len(vertices) != n:
        return False, [], []

    anchor = None if anchor_label is None else str(anchor_label).upper()[:1]
    if anchor is not None and anchor in labels and labels[0] != anchor:
        idx = labels.index(anchor)
        labels = labels[idx:] + labels[:idx]

    filtered = []
    for t in tokens if isinstance(tokens, list) else []:
        if not isinstance(t, dict):
            continue
        try:
            conf = float(t.get("conf", 0.0))
        except Exception:
            conf = 0.0
        if conf < float(min_conf):
            continue
        if bool(single_char_only):
            letters = str(t.get("letters", "")).upper()
            ch = str(t.get("char", "")).upper()
            if len(letters) != 1 or len(ch) != 1:
                continue
        filtered.append(t)

    orientations = [list(vertices)]
    if bool(allow_reversed):
        orientations.append(list(reversed(vertices)))

    best_hit = -1
    best_detected = []
    best_vertices = []

    for orient in orientations:
        for s in range(n):
            cyc = [orient[(s + i) % n] for i in range(n)]
            detected = []
            hit = 0
            for i in range(n):
                ch = labels[i]
                tok = select_token_near_point(filtered, expected_char=ch, point=cyc[i], max_dist=float(max_dist))
                if tok is None:
                    detected.append(None)
                else:
                    detected.append(ch)
                    hit += 1
            if anchor is not None and detected and detected[0] != anchor:
                continue
            if hit == n:
                return True, detected, cyc
            if hit > best_hit:
                best_hit = hit
                best_detected = detected
                best_vertices = cyc

    return False, best_detected, best_vertices


def compute_outside_ink_stats(
    img,
    allowed_lines,
    anchor_points=None,
    band_ratio=0.012,
    band_floor_px=0,
    anchor_radius_ratio=0.11,
    anchor_radius_floor_px=0,
    extra_allow_mask=None,
):

    if img is None:
        return 0.0, 0, 0
    h, w = img.shape[:2]
    min_hw = float(min(h, w))

    _, bw = _gray_and_ink_mask(img)
    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    ink = (bw > 0)
    total_ink = int(ink.sum())
    if total_ink <= 0:
        return 0.0, 0, 0

    allow = np.zeros((h, w), dtype=np.uint8)
    band = int(max(1, round(scale_px(min_hw, band_ratio, floor_px=band_floor_px))))

    for ln in allowed_lines if isinstance(allowed_lines, list) else []:
        if not isinstance(ln, dict):
            continue
        seg = ln.get("seg")
        if isinstance(seg, (list, tuple)) and len(seg) >= 4:
            x1, y1, x2, y2 = [int(round(float(v))) for v in seg[:4]]
            cv2.line(allow, (x1, y1), (x2, y2), 255, int(2 * band + 1))
            continue
        abc = ln.get("abc")
        if isinstance(abc, (list, tuple)) and len(abc) >= 3:
            a, b, c = [float(v) for v in abc[:3]]
            if abs(b) > 1e-9:
                x1f, x2f = 0.0, float(w - 1)
                y1f = (-a * x1f - c) / b
                y2f = (-a * x2f - c) / b
                x1, y1, x2, y2 = int(round(x1f)), int(round(y1f)), int(round(x2f)), int(round(y2f))
                cv2.line(allow, (x1, y1), (x2, y2), 255, int(2 * band + 1))

    radius = int(max(1, round(scale_px(min_hw, anchor_radius_ratio, floor_px=anchor_radius_floor_px))))
    for p in anchor_points if isinstance(anchor_points, list) else []:
        if p is None:
            continue
        px = int(round(float(p[0])))
        py = int(round(float(p[1])))
        cv2.circle(allow, (px, py), radius, 255, -1)

    if extra_allow_mask is not None:
        try:
            if extra_allow_mask.shape[:2] == allow.shape[:2]:
                ext = ((extra_allow_mask > 0).astype(np.uint8) * 255)
                allow = cv2.bitwise_or(allow, ext)
        except Exception:
            pass

    outside = int((ink & (allow == 0)).sum())
    ratio = float(outside) / float(max(1, total_ink))
    return ratio, outside, total_ink


def has_excess_outside_ink(
    img,
    allowed_lines,
    anchor_points=None,
    max_outside_ratio=0.06,
    max_outside_px_ratio=0.00004,
    max_outside_px_floor=0,
    extra_allow_mask=None,
):

    ratio, outside_px, total_ink = compute_outside_ink_stats(
        img,
        allowed_lines,
        anchor_points=anchor_points,
        extra_allow_mask=extra_allow_mask,
    )
    h, w = img.shape[:2]
    outside_px_th = max(int(max_outside_px_floor), int(float(max_outside_px_ratio) * float(h) * float(w)))
    violated = bool(outside_px > outside_px_th and ratio > float(max_outside_ratio))
    info = {
        "outside_ratio": float(ratio),
        "outside_px": int(outside_px),
        "outside_px_th": int(outside_px_th),
        "outside_ratio_th": float(max_outside_ratio),
        "total_ink": int(total_ink),
    }
    return violated, info


def _primitive_needed(inpt):

    pass
