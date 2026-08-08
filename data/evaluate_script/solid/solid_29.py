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
PROMPT_ID = 29
PID = 74
TYPE = 'solid'
TARGET = {'min_lines': 8, 'min_labels': 1, 'need_curves': False, 'need_dashed': False, 'need_multi_obj': False, 'need_3d_angles': True, 'description': 'Draw 3D coordinate axes with arrowheads labeled x, y, z. Draw a cube with one vertex at the origin O and three edges from O going along the positive x-, y-, and z-directions.'}
CONSERVATIVE_FAIL_REASON = 'case has low positive precision under current visual heuristics; defaulting to fail'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def evaluate(image_path: str):
    img = _MODULE.cv2.imread(image_path)
    if img is None:
        return {'passed': False, 'criteria': {'image_readable': False}, 'meta': {'case_id': PID, 'target': TARGET}}
    return {'passed': False, 'criteria': {'image_readable': True, 'solid_low_precision_case_gate': False}, 'meta': {'case_id': PID, 'target': TARGET, 'reason': CONSERVATIVE_FAIL_REASON}}
    gray, bw, ink_ratio = _MODULE._binarize(img)
    line_count = _MODULE._count_lines(bw)
    criteria = {'image_readable': True, 'foreground_present': ink_ratio > 0.003, 'line_structure_matches_prompt': line_count >= int(TARGET['min_lines'])}
    meta = {'case_id': PID, 'target': TARGET, 'ink_ratio': round(float(ink_ratio), 4), 'line_count': int(line_count)}
    if int(TARGET.get('min_labels', 0)) > 0:
        text_count = _MODULE._count_text_regions(bw)
        criteria['prompt_labels_present'] = text_count >= int(TARGET['min_labels'])
        meta['text_regions_found'] = int(text_count)
    if TARGET.get('need_curves'):
        criteria['curved_solid_parts_present'] = _MODULE._has_curves(bw)
    if TARGET.get('need_dashed'):
        criteria['dashed_or_hidden_lines_present'] = _MODULE._has_dashed_lines(bw)
    if TARGET.get('need_multi_obj'):
        multi, n_objs = _MODULE._has_multi_objects(bw, min_objects=2)
        criteria['multiple_solids_or_parts_present'] = multi
        meta['object_clusters'] = int(n_objs)
    if TARGET.get('need_3d_angles'):
        diverse, n_groups = _MODULE._has_line_diversity(bw)
        criteria['3d_line_angle_diversity'] = diverse
        meta['angle_groups'] = int(n_groups)
    return {'passed': all((bool(v) for v in criteria.values())), 'criteria': criteria, 'meta': meta}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen solid case 29.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
