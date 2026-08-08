"""
Shared constraint checking utilities for MathGen-Bench function evaluators.
Contains common OpenCV-based detection routines used across function_*.py scripts.
"""
import cv2
import numpy as np
import os
import re

# =========================
# Basic color and axis detection
# =========================

def extract_color_mask(img_hsv, h_range, s_min=50, v_min=50):
    """Extract a mask for a given HSV hue range."""
    lower = np.array([h_range[0], s_min, v_min])
    upper = np.array([h_range[1], 255, 255])
    return cv2.inRange(img_hsv, lower, upper)


def extract_blue_mask(img_hsv):
    """Extract blue pixels (hue 100-130)."""
    return extract_color_mask(img_hsv, (100, 130))


def extract_red_mask(img_hsv):
    """Extract red pixels (hue wraps around 0/180)."""
    mask_lo = extract_color_mask(img_hsv, (0, 10))
    mask_hi = extract_color_mask(img_hsv, (170, 180))
    return cv2.bitwise_or(mask_lo, mask_hi)


def detect_axes(img_gray):
    """Detect x-axis row and y-axis column positions using morphology."""
    _, binary = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    x_axis_y = float(np.argmax(np.sum(h_lines, axis=1)))
    y_axis_x = float(np.argmax(np.sum(v_lines, axis=0)))
    return x_axis_y, y_axis_x


# =========================
# Arrow direction detection (global single-headed-axis constraint)
# =========================

def check_single_headed_arrows(img_gray, x_axis_y, y_axis_x):
    """
    Check that coordinate axes use single-headed arrows (positive direction only).
    Double-headed arrows (arrows at both ends) = FAIL.

    Approach:
    1. Find where the x-axis line actually ends on left and right.
    2. Find where the y-axis line actually ends on top and bottom.
    3. At each endpoint, look at a region CENTERED on the endpoint.
       Count dark pixels in the perpendicular direction to the axis.
    4. An arrowhead will have significantly more dark pixels spreading
       perpendicular to the axis compared to the regular axis line width.
    5. We compare the perpendicular spread at the endpoint versus at a
       reference point safely in the middle of the axis (far from intersection).
    """
    h, w = img_gray.shape[:2]
    _, binary = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)

    x_ay = int(x_axis_y)
    y_ax = int(y_axis_x)

    # Scale detection window parameters by image size so the algorithm works
    # consistently across different resolutions (e.g. 1024px vs 2816px wide).
    # Use short-side as the reference dimension normalised to 768px baseline.
    _scale = min(h, w) / 768.0
    _half_spread = max(25, int(25 * _scale))
    _along_width = max(15, int(15 * _scale))
    _min_ref_spread = max(30, int(30 * _scale * _scale))   # scales with area
    _min_abs_spread = max(80, int(80 * _scale * _scale))

    # ---- Find actual endpoints of x-axis ----
    # Scan along the x-axis row with a thin strip
    strip_half = max(4, int(4 * _scale))
    x_strip = binary[max(0, x_ay - strip_half):min(h, x_ay + strip_half), :]
    col_has_axis = np.any(x_strip > 0, axis=0)
    axis_cols = np.where(col_has_axis)[0]

    if len(axis_cols) < 20:
        return {"is_single_headed": True, "detail": "x_axis_not_detected"}

    x_left_end = int(axis_cols[0])
    x_right_end = int(axis_cols[-1])

    # ---- Find actual endpoints of y-axis ----
    y_strip = binary[:, max(0, y_ax - strip_half):min(w, y_ax + strip_half)]
    row_has_axis = np.any(y_strip > 0, axis=1)
    axis_rows = np.where(row_has_axis)[0]

    if len(axis_rows) < 20:
        return {"is_single_headed": True, "detail": "y_axis_not_detected"}

    y_top_end = int(axis_rows[0])
    y_bot_end = int(axis_rows[-1])

    # ---- Measure arrowhead presence at each endpoint ----
    # For x-axis endpoints: check dark pixel spread VERTICALLY (perpendicular to x-axis)
    # For y-axis endpoints: check dark pixel spread HORIZONTALLY (perpendicular to y-axis)

    def _count_perpendicular_spread_x(col, half_spread=None, along_width=None):
        """
        At column `col` on the x-axis, count dark pixels in a vertical strip
        of height 2*half_spread centered on the axis, width `along_width`.
        Returns the count of dark pixels.
        """
        hs = half_spread if half_spread is not None else _half_spread
        aw = along_width if along_width is not None else _along_width
        c_start = max(0, col - aw // 2)
        c_end = min(w, col + aw // 2)
        r_start = max(0, x_ay - hs)
        r_end = min(h, x_ay + hs)
        region = binary[r_start:r_end, c_start:c_end]
        return int(np.sum(region > 0))

    def _count_perpendicular_spread_y(row, half_spread=None, along_width=None):
        """
        At row `row` on the y-axis, count dark pixels in a horizontal strip
        of width 2*half_spread centered on the axis, height `along_width`.
        """
        hs = half_spread if half_spread is not None else _half_spread
        aw = along_width if along_width is not None else _along_width
        r_start = max(0, row - aw // 2)
        r_end = min(h, row + aw // 2)
        c_start = max(0, y_ax - hs)
        c_end = min(w, y_ax + hs)
        region = binary[r_start:r_end, c_start:c_end]
        return int(np.sum(region > 0))

    # Measure at endpoints
    x_left_spread = _count_perpendicular_spread_x(x_left_end + 5)
    x_right_spread = _count_perpendicular_spread_x(x_right_end - 5)

    y_top_spread = _count_perpendicular_spread_y(y_top_end + 5)
    y_bot_spread = _count_perpendicular_spread_y(y_bot_end - 5)

    # Measure at reference points (25% and 75% along each axis, away from intersection)
    # For x-axis reference, pick a point between left end and y_axis, or between y_axis and right end
    x_ref_col = x_left_end + (y_ax - x_left_end) // 2  # midpoint on left half
    if x_ref_col <= x_left_end + 30:
        x_ref_col = y_ax + (x_right_end - y_ax) // 2  # use right half instead
    x_ref_spread = _count_perpendicular_spread_x(int(x_ref_col))

    y_ref_row = y_top_end + (x_ay - y_top_end) // 2  # midpoint on top half
    if y_ref_row <= y_top_end + 30:
        y_ref_row = x_ay + (y_bot_end - x_ay) // 2
    y_ref_spread = _count_perpendicular_spread_y(int(y_ref_row))

    # Arrowhead detection: endpoint spread should be significantly more than reference
    # An arrowhead typically has 1.5-4x the dark pixels of a plain axis line
    arrow_ratio_threshold = 1.4

    x_left_ratio = x_left_spread / max(1, x_ref_spread)
    x_right_ratio = x_right_spread / max(1, x_ref_spread)
    y_top_ratio = y_top_spread / max(1, y_ref_spread)
    y_bot_ratio = y_bot_spread / max(1, y_ref_spread)

    x_has_left_arrow = x_left_ratio > arrow_ratio_threshold
    x_has_right_arrow = x_right_ratio > arrow_ratio_threshold
    y_has_top_arrow = y_top_ratio > arrow_ratio_threshold
    y_has_bottom_arrow = y_bot_ratio > arrow_ratio_threshold

    # Reliability guard: if reference spread is very low, ratio is unreliable.
    # Also require a minimum absolute spread to declare an arrowhead.
    # Thresholds are scaled to image resolution.
    min_ref_spread = _min_ref_spread
    min_abs_spread = _min_abs_spread

    if x_ref_spread < min_ref_spread or x_left_spread < min_abs_spread:
        x_has_left_arrow = False
    if x_ref_spread < min_ref_spread or x_right_spread < min_abs_spread:
        x_has_right_arrow = False
    if y_ref_spread < min_ref_spread or y_top_spread < min_abs_spread:
        y_has_top_arrow = False
    if y_ref_spread < min_ref_spread or y_bot_spread < min_abs_spread:
        y_has_bottom_arrow = False

    # If x-axis doesn't extend much past the y-axis to the left,
    # skip left arrow check (e.g. domain [0,8] has no negative x-axis).
    x_left_extent = y_ax - x_left_end
    if x_left_extent < 40:  # axis barely extends left of origin
        x_has_left_arrow = False

    # Similarly if y-axis doesn't extend much below x-axis
    y_bot_extent = y_bot_end - x_ay
    if y_bot_extent < 40:
        y_has_bottom_arrow = False

    # Single-headed: negative ends (left, bottom) should NOT have arrowheads
    x_axis_ok = not x_has_left_arrow
    y_axis_ok = not y_has_bottom_arrow

    is_single_headed = x_axis_ok and y_axis_ok

    return {
        "is_single_headed": bool(is_single_headed),
        "x_left_spread": x_left_spread,
        "x_right_spread": x_right_spread,
        "x_ref_spread": x_ref_spread,
        "y_top_spread": y_top_spread,
        "y_bot_spread": y_bot_spread,
        "y_ref_spread": y_ref_spread,
        "x_left_ratio": round(x_left_ratio, 2),
        "x_right_ratio": round(x_right_ratio, 2),
        "y_top_ratio": round(y_top_ratio, 2),
        "y_bot_ratio": round(y_bot_ratio, 2),
        "x_has_left_arrow": bool(x_has_left_arrow),
        "y_has_bottom_arrow": bool(y_has_bottom_arrow),
    }


# =========================
# Tick spacing estimation
# =========================

def estimate_tick_spacing(img_gray, x_axis_y, y_axis_x):
    """
    Estimate pixel spacing between tick marks on both axes.
    """
    h, w = img_gray.shape[:2]
    _, binary = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
    x_ay = int(x_axis_y)
    y_ax = int(y_axis_x)

    # Vertical tick marks along x-axis
    tick_height = 15
    x_axis_strip = binary[
        max(0, x_ay - tick_height):min(h, x_ay + tick_height), :
    ]
    col_sums = np.sum(x_axis_strip > 0, axis=0)
    axis_baseline = np.median(col_sums)
    tick_threshold = axis_baseline + 3
    tick_cols = np.where(col_sums > tick_threshold)[0]

    x_tick_positions = []
    if len(tick_cols) > 0:
        groups = np.split(tick_cols, np.where(np.diff(tick_cols) > 5)[0] + 1)
        x_tick_positions = [int(np.mean(g)) for g in groups if len(g) > 0]
    x_tick_positions = [t for t in x_tick_positions if abs(t - y_ax) > 15]

    x_tick_spacing = None
    if len(x_tick_positions) >= 2:
        diffs = np.diff(sorted(x_tick_positions))
        x_tick_spacing = float(np.median(diffs))

    # Horizontal tick marks along y-axis
    y_axis_strip = binary[
        :, max(0, y_ax - tick_height):min(w, y_ax + tick_height)
    ]
    row_sums = np.sum(y_axis_strip > 0, axis=1)
    y_axis_baseline = np.median(row_sums)
    y_tick_threshold = y_axis_baseline + 3
    tick_rows = np.where(row_sums > y_tick_threshold)[0]

    y_tick_positions = []
    if len(tick_rows) > 0:
        groups = np.split(tick_rows, np.where(np.diff(tick_rows) > 5)[0] + 1)
        y_tick_positions = [int(np.mean(g)) for g in groups if len(g) > 0]
    y_tick_positions = [t for t in y_tick_positions if abs(t - x_ay) > 15]

    y_tick_spacing = None
    if len(y_tick_positions) >= 2:
        diffs = np.diff(sorted(y_tick_positions))
        y_tick_spacing = float(np.median(diffs))

    return x_tick_spacing, y_tick_spacing


# =========================
# Vertical Line Test (by definition of a function)
# =========================

def check_vertical_line_test(blue_mask, gap_threshold=15, violation_ratio_max=0.05):
    """
    Check that a curve passes the vertical line test:
    for each x-column, blue pixels should form at most one contiguous
    vertical cluster.  Multiple separated clusters = multiple y-values = not a function.

    gap_threshold:      min vertical pixel gap to count as separate clusters
    violation_ratio_max: max fraction of occupied columns allowed to violate
    """
    h, w = blue_mask.shape[:2]

    violation_count = 0
    total_columns = 0

    for col in range(w):
        blue_rows = np.where(blue_mask[:, col] > 0)[0]
        if len(blue_rows) < 2:
            continue
        total_columns += 1

        # Find gaps larger than threshold between blue pixel groups
        diffs = np.diff(blue_rows)
        if np.any(diffs > gap_threshold):
            violation_count += 1

    if total_columns == 0:
        return True, {"total_columns": 0, "violations": 0, "violation_ratio": 0.0}

    ratio = violation_count / total_columns
    ok = ratio <= violation_ratio_max

    return ok, {
        "total_columns": total_columns,
        "violations": violation_count,
        "violation_ratio": round(ratio, 4),
    }


# =========================
# OCR Tick Labels (check for duplicates, monotonicity)
# =========================

def _find_axis_endpoints(img_gray, x_axis_y, y_axis_x):
    """Find where each axis line starts and ends."""
    h, w = img_gray.shape[:2]
    _, binary = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    xa = int(round(y_axis_x))
    ya = int(round(x_axis_y))

    # Y-axis endpoints
    strip_half = 4
    y_strip = binary[:, max(0, xa - strip_half):min(w, xa + strip_half)]
    row_has = np.any(y_strip > 0, axis=1)
    axis_rows = np.where(row_has)[0]
    y_top = int(axis_rows[0]) if len(axis_rows) > 0 else 0
    y_bot = int(axis_rows[-1]) if len(axis_rows) > 0 else h - 1

    # X-axis endpoints
    x_strip = binary[max(0, ya - strip_half):min(h, ya + strip_half), :]
    col_has = np.any(x_strip > 0, axis=0)
    axis_cols = np.where(col_has)[0]
    x_left = int(axis_cols[0]) if len(axis_cols) > 0 else 0
    x_right = int(axis_cols[-1]) if len(axis_cols) > 0 else w - 1

    return y_top, y_bot, x_left, x_right


def ocr_axis_labels(img_bgr, x_axis_y, y_axis_x, axis="x", ocr_scale=6,
                    x_margin=None):
    """
    OCR tick labels along an axis.

    axis="x": reads labels below the x-axis, returns list of (pixel_x, value).
    axis="y": reads labels left of the y-axis, returns list of (pixel_y, value).

    x_margin: override for x-axis margin (pixels below axis to start ROI).
              If None, tries multiple margins and merges results.

    Returns (labels, error_string). error_string is None on success.
    """
    try:
        import pytesseract
    except ImportError:
        return [], "pytesseract not installed"

    h, w = img_bgr.shape[:2]
    ya = int(round(x_axis_y))
    xa = int(round(y_axis_x))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    y_top, y_bot, x_left, x_right = _find_axis_endpoints(img_gray, x_axis_y, y_axis_x)

    if axis == "x":
        # ROI: below x-axis, skip tick marks
        # Use caller-specified margin, or a moderate default
        margin = x_margin if x_margin is not None else 12
        label_size = int(0.12 * h)
        y0 = min(h - 1, ya + margin)
        y1 = min(h, ya + margin + label_size)
        if y1 <= y0:
            return [], "x-axis region too small"
        roi = img_bgr[y0:y1, :].copy()
        roi_gray = img_gray[y0:y1, :].copy()
        # Mask out y-axis line to avoid OCR confusion
        axis_half = 10
        roi[:, max(0, xa - axis_half):min(w, xa + axis_half + 1)] = 255
        roi_gray[:, max(0, xa - axis_half):min(w, xa + axis_half + 1)] = 255
        pos_offset = 0  # x positions are absolute
    else:  # y-axis
        # ROI: left of y-axis, constrained to plot area rows only
        margin = 5
        label_size = int(0.20 * w)
        x0 = max(0, xa - margin - label_size)
        x1 = max(0, xa - margin)
        if x1 <= x0:
            return [], "y-axis region too small"
        # Constrain rows to plot area (avoid function text at top/bottom)
        r0 = max(0, y_top - 10)
        r1 = min(h, y_bot + 10)
        roi = img_bgr[r0:r1, x0:x1].copy()
        roi_gray = img_gray[r0:r1, x0:x1].copy()
        # Mask out x-axis line
        axis_half = 10
        ya_local = ya - r0
        roi[max(0, ya_local - axis_half):min(roi.shape[0], ya_local + axis_half + 1), :] = 255
        roi_gray[max(0, ya_local - axis_half):min(roi.shape[0], ya_local + axis_half + 1)] = 255
        pos_offset = r0  # need to add back when converting y positions

    up = cv2.resize(roi_gray, None, fx=ocr_scale, fy=ocr_scale,
                    interpolation=cv2.INTER_CUBIC)

    def _run_ocr(bin_img):
        config = "--oem 3 --psm 11 -c tessedit_char_whitelist=-.0123456789"
        data = pytesseract.image_to_data(bin_img, config=config,
                                          output_type=pytesseract.Output.DICT)
        labels = []
        for i in range(len(data["text"])):
            txt = (data["text"][i] or "").strip()
            if txt == "":
                continue
            txt = txt.replace("\u2212", "-").replace("\u2014", "-").replace("\u2013", "-")
            txt = txt.strip(", ")
            if not re.fullmatch(r"-?\d+\.?\d*", txt):
                continue
            cx = data["left"][i] + data["width"][i] / 2.0
            cy = data["top"][i] + data["height"][i] / 2.0
            val = float(txt)
            if axis == "x":
                labels.append((cx / ocr_scale, val))
            else:
                labels.append(((cy / ocr_scale) + pos_offset, val))
        return labels

    # Merge results across multiple thresholds (union with dedup by position)
    all_labels = []
    for thresh in [60, 100, 140]:
        _, bin_img = cv2.threshold(up, thresh, 255, cv2.THRESH_BINARY_INV)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k, iterations=1)
        try:
            all_labels.extend(_run_ocr(bin_img))
        except Exception as e:
            return [], str(e)
        # Also try with dilation to thicken thin font strokes
        k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bin_dilated = cv2.dilate(bin_img, k_dilate, iterations=1)
        try:
            all_labels.extend(_run_ocr(bin_dilated))
        except Exception:
            pass

    # Also try tick-position-guided OCR as fallback
    all_labels.extend(
        _ocr_tick_labels(img_gray, x_axis_y, y_axis_x, axis)
    )

    # Try to read the origin label (often "0" at axis intersection)
    origin_label = _ocr_origin_label(img_gray, x_axis_y, y_axis_x, axis)
    if origin_label is not None:
        all_labels.append(origin_label)

    # Deduplicate by position proximity (within 30 pixels)
    merged = _deduplicate_labels(all_labels, proximity=30)

    # Sort by position along the axis
    merged.sort(key=lambda t: t[0])
    return merged, None


def _ocr_tick_labels(img_gray, x_axis_y, y_axis_x, axis="y", ocr_scale=6):
    """
    Fallback: find tick marks on an axis, then OCR a small window around each.
    """
    try:
        import pytesseract
    except ImportError:
        return []

    h, w = img_gray.shape[:2]
    xa = int(round(y_axis_x))
    ya = int(round(x_axis_y))
    _, binary = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    tick_half = 15

    # Find axis endpoints for filtering
    y_top, y_bot, x_left, x_right = _find_axis_endpoints(img_gray, x_axis_y, y_axis_x)
    endpoint_margin = 40  # ignore ticks near axis ends (arrowheads)

    if axis == "y":
        # Find tick marks along y-axis (horizontal protrusions)
        strip = binary[:, max(0, xa - tick_half):min(w, xa + tick_half)]
        sums = np.sum(strip > 0, axis=1)
        baseline = np.median(sums)
        tick_idx = np.where(sums > baseline + 3)[0]
        if len(tick_idx) == 0:
            return []
        groups = np.split(tick_idx, np.where(np.diff(tick_idx) > 5)[0] + 1)
        tick_positions = [int(np.mean(g)) for g in groups if len(g) > 0]
        # Filter: skip origin and ticks near axis endpoints
        tick_positions = [t for t in tick_positions
                          if abs(t - ya) > 12
                          and t > y_top + endpoint_margin
                          and t < y_bot - endpoint_margin]
    else:
        # Find tick marks along x-axis (vertical protrusions)
        strip = binary[max(0, ya - tick_half):min(h, ya + tick_half), :]
        sums = np.sum(strip > 0, axis=0)
        baseline = np.median(sums)
        tick_idx = np.where(sums > baseline + 3)[0]
        if len(tick_idx) == 0:
            return []
        groups = np.split(tick_idx, np.where(np.diff(tick_idx) > 5)[0] + 1)
        tick_positions = [int(np.mean(g)) for g in groups if len(g) > 0]
        # Filter: skip origin and ticks near axis endpoints
        tick_positions = [t for t in tick_positions
                          if abs(t - xa) > 12
                          and t > x_left + endpoint_margin
                          and t < x_right - endpoint_margin]

    labels = []
    for tp in tick_positions:
        if axis == "y":
            # Label is to the left of the y-axis
            r0 = max(0, tp - 25)
            r1 = min(h, tp + 25)
            c0 = max(0, xa - 100)
            c1 = max(0, xa - 5)
        else:
            # Label is below the x-axis
            r0 = min(h - 1, ya + 5)
            r1 = min(h, ya + 55)
            c0 = max(0, tp - 30)
            c1 = min(w, tp + 30)
        if c1 <= c0 or r1 <= r0:
            continue

        label_roi = img_gray[r0:r1, c0:c1]
        up_l = cv2.resize(label_roi, None, fx=ocr_scale, fy=ocr_scale,
                          interpolation=cv2.INTER_CUBIC)
        for thresh in [60, 80, 100]:
            _, b = cv2.threshold(up_l, thresh, 255, cv2.THRESH_BINARY_INV)
            config = "--oem 3 --psm 8 -c tessedit_char_whitelist=-.0123456789"
            try:
                txt = pytesseract.image_to_string(b, config=config).strip()
            except Exception:
                continue
            txt = txt.replace("\u2212", "-").replace("\u2014", "-").replace("\u2013", "-")
            # Strip trailing non-numeric artifacts (tick marks read as dashes)
            txt = re.sub(r"[.\-]+$", "", txt.strip(", "))
            # Re-check: allow leading dash (negative sign)
            if re.fullmatch(r"-?\d+\.?\d*", txt):
                labels.append((float(tp), float(txt)))
                break
    return labels


def _ocr_origin_label(img_gray, x_axis_y, y_axis_x, axis="x", ocr_scale=6):
    """
    Try to read the label at the origin (axis intersection).
    Returns (position, value) or None.
    """
    try:
        import pytesseract
    except ImportError:
        return None

    h, w = img_gray.shape[:2]
    xa = int(round(y_axis_x))
    ya = int(round(x_axis_y))

    # Origin label is typically below-left of the axis intersection
    r0 = min(h - 1, ya + 15)
    r1 = min(h, ya + 70)
    c0 = max(0, xa - 55)
    c1 = max(0, xa - 3)  # stop before y-axis to avoid interference
    if axis == "x":
        pos = float(xa)
    else:
        pos = float(ya)

    if c1 <= c0 or r1 <= r0:
        return None

    label_roi = img_gray[r0:r1, c0:c1].copy()

    up = cv2.resize(label_roi, None, fx=ocr_scale, fy=ocr_scale,
                    interpolation=cv2.INTER_CUBIC)
    for thresh in [60, 80, 100]:
        _, b = cv2.threshold(up, thresh, 255, cv2.THRESH_BINARY_INV)
        for psm in [8, 7, 10]:
            config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=-.0123456789"
            try:
                txt = pytesseract.image_to_string(b, config=config).strip()
            except Exception:
                continue
            txt = txt.replace("\u2212", "-").replace("\u2014", "-").replace("\u2013", "-")
            txt = re.sub(r"[.\-]+$", "", txt.strip(", "))
            if re.fullmatch(r"-?\d+\.?\d*", txt):
                return (pos, float(txt))
    return None


def _deduplicate_labels(labels, proximity=30):
    """Merge labels at similar positions, keeping the most common value."""
    if not labels:
        return []
    labels.sort(key=lambda t: t[0])
    groups = []
    current_group = [labels[0]]
    for lbl in labels[1:]:
        if abs(lbl[0] - current_group[-1][0]) < proximity:
            current_group.append(lbl)
        else:
            groups.append(current_group)
            current_group = [lbl]
    groups.append(current_group)

    merged = []
    for group in groups:
        # Pick most common value, average position
        from collections import Counter
        vals = Counter(v for _, v in group)
        best_val = vals.most_common(1)[0][0]
        positions = [p for p, v in group if v == best_val]
        avg_pos = sum(positions) / len(positions)
        merged.append((avg_pos, best_val))
    return merged


def check_tick_labels(labels, axis="x"):
    """
    Check that tick labels are:
    1. No duplicate values
    2. Strictly monotonic (increasing for x-axis, decreasing for y-axis)
    3. Roughly uniform value spacing (max step <= 3x median step)
    4. Roughly uniform pixel-position spacing (max gap <= 3x median gap)
    Returns (ok, detail_dict).
    """
    if len(labels) < 2:
        return False, {"reason": "too_few_labels", "count": len(labels)}

    values = [v for _, v in labels]
    positions = [p for p, _ in labels]

    has_duplicates = len(values) != len(set(values))

    # x-axis: values should increase left-to-right
    # y-axis: values should decrease top-to-bottom (high values at top)
    if axis == "y":
        is_monotonic = all(values[i] > values[i + 1] for i in range(len(values) - 1))
    else:
        is_monotonic = all(values[i] < values[i + 1] for i in range(len(values) - 1))

    # Check uniform value spacing
    uniform_value_spacing = True
    if is_monotonic and len(values) >= 3:
        diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
        median_diff = sorted(diffs)[len(diffs) // 2]
        if median_diff > 0:
            uniform_value_spacing = max(diffs) <= 3 * median_diff
        else:
            uniform_value_spacing = False

    # Check uniform pixel-per-unit spacing between adjacent labels.
    # This is stricter than raw pixel gap checks and supports unequal value steps.
    uniform_pos_spacing = True
    if len(positions) >= 3:
        pos_diffs = [abs(positions[i + 1] - positions[i]) for i in range(len(positions) - 1)]
        val_diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
        if any(vd <= 0 for vd in val_diffs):
            uniform_pos_spacing = False
        else:
            unit_gaps = [pd / vd for pd, vd in zip(pos_diffs, val_diffs)]
            min_gap = min(unit_gaps)
            max_gap = max(unit_gaps)
            # Allow modest perspective/anti-aliasing variation.
            uniform_pos_spacing = min_gap > 0 and (max_gap / min_gap) <= 1.35

    ok = (not has_duplicates and is_monotonic
          and uniform_value_spacing and uniform_pos_spacing)
    return ok, {
        "values": values,
        "has_duplicates": has_duplicates,
        "is_monotonic": is_monotonic,
        "uniform_value_spacing": uniform_value_spacing,
        "uniform_pos_spacing": uniform_pos_spacing,
    }


def _count_tick_marks(img_gray, x_axis_y, y_axis_x, axis="x"):
    """Count the number of distinct tick marks on an axis."""
    h, w = img_gray.shape[:2]
    _, binary = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
    x_ay = int(x_axis_y)
    y_ax = int(y_axis_x)
    tick_half = 15

    if axis == "x":
        strip = binary[max(0, x_ay - tick_half):min(h, x_ay + tick_half), :]
        sums = np.sum(strip > 0, axis=0)
        baseline = np.median(sums)
        idx = np.where(sums > baseline + 3)[0]
        if len(idx) == 0:
            return 0
        groups = np.split(idx, np.where(np.diff(idx) > 5)[0] + 1)
        positions = [int(np.mean(g)) for g in groups if len(g) > 0]
        # Exclude origin (near y-axis)
        positions = [t for t in positions if abs(t - y_ax) > 15]
    else:
        strip = binary[:, max(0, y_ax - tick_half):min(w, y_ax + tick_half)]
        sums = np.sum(strip > 0, axis=1)
        baseline = np.median(sums)
        idx = np.where(sums > baseline + 3)[0]
        if len(idx) == 0:
            return 0
        groups = np.split(idx, np.where(np.diff(idx) > 5)[0] + 1)
        positions = [int(np.mean(g)) for g in groups if len(g) > 0]
        positions = [t for t in positions if abs(t - x_ay) > 15]

    return len(positions)


def _tick_structure_ok(img_gray, x_axis_y, y_axis_x, axis="x",
                       img_hsv=None):
    """
    Structural tick mark validation: check that at least 3 tick marks exist
    with reasonably uniform spacing.

    Only looks at the label-side of the axis (below x-axis, left of y-axis)
    to avoid counting curve pixels that cross the axis.
    Optionally filters out colored pixels (blue/red curves) via img_hsv.
    Returns (ok, detail_dict).
    """
    h, w = img_gray.shape[:2]
    _, binary = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Filter out colored pixels (curves) — only keep achromatic (black/gray) features.
    if img_hsv is not None:
        sat = img_hsv[:, :, 1]
        colored_mask = sat > 40  # saturation > 40 → colored pixel
        binary[colored_mask] = 0

    x_ay = int(x_axis_y)
    y_ax = int(y_axis_x)

    if axis == "x":
        # Look BELOW x-axis only (where tick marks protrude toward labels).
        # Strip from 1-13px below axis: wide enough to catch tick marks in
        # various rendering styles, but still avoids most label text.
        tick_start = max(0, x_ay + 1)
        tick_end = min(h, x_ay + 13)
        if tick_end <= tick_start:
            return False, {"n_ticks": 0}
        strip = binary[tick_start:tick_end, :]
        sums = np.sum(strip > 0, axis=0)
    else:
        # Look LEFT of y-axis only, similar width.
        tick_start = max(0, y_ax - 13)
        tick_end = max(0, y_ax - 1)
        if tick_end <= tick_start:
            return False, {"n_ticks": 0}
        strip = binary[:, tick_start:tick_end]
        sums = np.sum(strip > 0, axis=1)

    baseline = np.median(sums)
    idx = np.where(sums > baseline + 2)[0]
    if len(idx) == 0:
        return False, {"n_ticks": 0}

    # Use larger gap (20px) for grouping to merge closely-spaced detections
    groups = np.split(idx, np.where(np.diff(idx) > 20)[0] + 1)
    positions = sorted(int(np.mean(g)) for g in groups if len(g) > 0)

    # Filter out positions near the axis intersection
    if axis == "x":
        positions = [t for t in positions if abs(t - y_ax) > 20]
    else:
        positions = [t for t in positions if abs(t - x_ay) > 20]

    # Filter out positions near axis endpoints (arrowheads and "x"/"y" labels)
    y_top, y_bot, x_left, x_right = _find_axis_endpoints(img_gray, x_axis_y, y_axis_x)
    endpoint_margin = 50
    if axis == "x":
        positions = [t for t in positions
                     if t > x_left + endpoint_margin and t < x_right - endpoint_margin]
    else:
        positions = [t for t in positions
                     if t > y_top + endpoint_margin and t < y_bot - endpoint_margin]

    n_ticks = len(positions)
    if n_ticks < 3:
        return False, {"n_ticks": n_ticks}

    diffs = np.diff(positions)
    median_d = float(np.median(diffs))

    # Robust uniformity: count how many spacings are within 2× of the median.
    # If at least half are consistent, the image has real tick structure.
    # This tolerates one or two gaps where a tick mark wasn't detected.
    if median_d > 0:
        consistent = sum(1 for d in diffs if 0.4 * median_d <= d <= 2.5 * median_d)
        ratio_ok = consistent >= len(diffs) / 2.0
    else:
        ratio_ok = False

    ok = median_d > 0 and ratio_ok
    return ok, {
        "n_ticks": n_ticks,
        "spacing_median": median_d,
        "consistent_spacings": consistent if median_d > 0 else 0,
        "total_spacings": len(diffs),
    }


def check_axis_labels_or_ticks(img_gray, img_bgr, x_axis_y, y_axis_x, axis="x"):
    """
    Combined check: try OCR tick labels with multiple x-axis margins; if OCR
    succeeds and labels pass validation, return True.  If OCR is unreliable,
    fall back to verifying that enough tick marks exist with uniform spacing.

    Logic:
      1. OCR labels valid           → PASS (strong positive evidence)
      2. OCR labels clearly wrong AND structural check also fails → FAIL
      3. OCR labels wrong/absent BUT structural check passes     → PASS
         (OCR may have misread small fonts; trust the tick structure)
      4. No labels found AND structural check fails              → FAIL

    Returns (ok, detail_dict).
    """
    detail = {}

    # --- Phase 1: try OCR with multiple x-axis margins ---
    # Different rendering styles place labels at different distances from the axis.
    best_labels = []
    margins_to_try = [6, 12, 20] if axis == "x" else [None]
    for m in margins_to_try:
        kwargs = {"x_margin": m} if m is not None else {}
        labels, ocr_err = ocr_axis_labels(
            img_bgr, x_axis_y, y_axis_x, axis=axis, **kwargs)
        if not ocr_err and len(labels) > len(best_labels):
            best_labels = labels

    if len(best_labels) >= 2:
        labels_ok, label_detail = check_tick_labels(best_labels, axis=axis)
        detail.update(label_detail)
        detail["method"] = "ocr"
        if labels_ok:
            return True, detail
    else:
        detail["method"] = "ocr_insufficient"

    # --- Phase 2: structural tick mark analysis ---
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    struct_ok, struct_detail = _tick_structure_ok(
        img_gray, x_axis_y, y_axis_x, axis=axis, img_hsv=img_hsv)
    detail.update(struct_detail)

    if struct_ok:
        detail["method"] = "tick_structure_fallback"
        return True, detail

    # Neither OCR nor structural check passed.
    return False, detail
