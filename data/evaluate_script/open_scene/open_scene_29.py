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
PROMPT_ID = 29
sys.path.insert(0, str(Path(__file__).resolve().parent))
CASE_ID = 88
SOURCE_TOPIC = 'plane_geometry'
PROMPT = 'A strictly top-down, orthographic-style realistic photo of a clean dark desk mat. One thin white triangle outline contains one smaller thin white rectangular outline completely inside it, without touching. Clean minimal composition, no text or extra objects.'
TARGET = {'source_topic': 'plane_geometry', 'expected_count': None, 'target_color': 'dark', 'expected_circles': 0, 'expected_shapes': {'triangle': 1, 'line': 2}, 'relation': 'contains', 'set_relation': '', 'minute_hand': None, 'hour_hand': None, 'target_fraction': None, 'object': '', 'function_shape': '', 'solids': []}

def evaluate(image_path: str):
    return evaluate_real_set_case(image_path, CASE_ID, PROMPT, SOURCE_TOPIC, TARGET)

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen open_scene case 29.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
