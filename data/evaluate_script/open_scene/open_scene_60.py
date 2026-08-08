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
PROMPT_ID = 60
sys.path.insert(0, str(Path(__file__).resolve().parent))
CASE_ID = 178
SOURCE_TOPIC = 'function'
PROMPT = 'A strictly front-view, orthographic-style realistic photo of a plain white background. Show a single smooth spiral curve made of a single dark-gray line, starting from the center of the image and expanding outward in a clockwise direction for exactly two full rotations. The spiral spacing between each loop should be approximately equal. No other shapes, no grid lines, no text.'
TARGET = {'source_topic': 'function', 'expected_count': 2, 'target_color': 'gray', 'expected_circles': 0, 'expected_shapes': {}, 'relation': '', 'set_relation': '', 'minute_hand': None, 'hour_hand': None, 'target_fraction': None, 'object': '', 'function_shape': 'spiral', 'solids': []}

def evaluate(image_path: str):
    return evaluate_real_set_case(image_path, CASE_ID, PROMPT, SOURCE_TOPIC, TARGET)

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen open_scene case 60.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
