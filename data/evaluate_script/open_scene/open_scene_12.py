#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'plane'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluator_utils import print_report
from open_scene_common import *
import plane_common as _common
from plane_common import *
PROMPT_ID = 12
sys.path.insert(0, str(Path(__file__).resolve().parent))
CASE_ID = 14
SOURCE_TOPIC = 'set'
PROMPT = 'Draw a clean, realistic top-down view of a light gray table with two identical transparent glass bowls partially overlapping. Place exactly two green peas inside only the overlapping region. No peas outside the overlap.'
TARGET = {'source_topic': 'set', 'expected_count': 2, 'target_color': 'green', 'expected_circles': 2, 'expected_shapes': {}, 'relation': '', 'set_relation': 'pairwise_overlap_only', 'minute_hand': None, 'hour_hand': None, 'target_fraction': None, 'object': '', 'function_shape': '', 'solids': []}

def evaluate(image_path: str):
    return evaluate_real_set_case(image_path, CASE_ID, PROMPT, SOURCE_TOPIC, TARGET)

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen open_scene case 12.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
