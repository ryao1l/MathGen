#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'plane'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluator_utils import print_report
import solid_common
import solid_common as _MODULE
import plane_common
import plane_common as _PLANE
from solid_common import *
from plane_common import *
PROMPT_ID = 11
PID = 14
TYPE = 'solid'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def evaluate(image_path: str):
    img = _MODULE.cv2.imread(image_path)
    if img is None:
        return {'passed': False, 'criteria': {'image_readable': False}, 'meta': {'case_id': PID}}
    gray, bw, ink_ratio = _MODULE._binarize(img)
    ys, xs = (bw > 0).nonzero()
    if xs.size == 0:
        return {'passed': False, 'criteria': {'image_readable': True, 'foreground_present': False}, 'meta': {'case_id': PID}}
    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_h = int(ys.max() - ys.min() + 1)
    aspect = float(bbox_w) / float(max(1, bbox_h))
    line_count = _MODULE._count_lines(bw)
    diverse, n_groups = _MODULE._has_line_diversity(bw)
    criteria = {'image_readable': True, 'foreground_present': 0.003 < ink_ratio < 0.2, 'wide_cuboid_aspect': aspect >= 1.45, 'line_structure_matches_prompt': line_count >= 8, '3d_line_angle_diversity': diverse}
    return {'passed': all((bool(v) for v in criteria.values())), 'criteria': criteria, 'meta': {'case_id': PID, 'ink_ratio': round(float(ink_ratio), 4), 'line_count': int(line_count), 'bbox_aspect': round(float(aspect), 3), 'angle_groups': int(n_groups)}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen solid case 11.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
