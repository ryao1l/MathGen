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
PROMPT_ID = 17
PID = 23
TYPE = 'solid'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def evaluate(image_path: str):
    return _MODULE.evaluate_solid_feature_gate(image_path, PID, ink_min=0.035, ink_max=0.045, aspect_min=1.82, aspect_max=1.95, line_min=145, angle_groups_min=4, angle_groups_max=4)

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen solid case 17.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
