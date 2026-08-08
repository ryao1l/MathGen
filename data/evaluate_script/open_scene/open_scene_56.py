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
PROMPT_ID = 56
sys.path.insert(0, str(Path(__file__).resolve().parent))
CASE_ID = 169
SOURCE_TOPIC = 'function'
PROMPT = 'A strictly side-view, orthographic-style realistic photo of a plain white wall and flat horizontal floor. Show exactly three horizontal thin dark-gray shelves mounted on the wall at three different heights: one at the bottom third, one at the middle, and one at the top third of the image. All shelves have equal length and equal thickness. No objects on the shelves, no text, no other items.'
TARGET = {'source_topic': 'function', 'expected_count': 3, 'target_color': 'gray', 'expected_circles': 0, 'expected_shapes': {}, 'relation': '', 'set_relation': '', 'minute_hand': None, 'hour_hand': None, 'target_fraction': None, 'object': '', 'function_shape': 'shelves', 'solids': []}

def evaluate(image_path: str):
    return evaluate_real_set_case(image_path, CASE_ID, PROMPT, SOURCE_TOPIC, TARGET)

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen open_scene case 56.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
