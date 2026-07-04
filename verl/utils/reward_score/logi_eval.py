# -*- coding: utf-8 -*-

import os
import re


def _debug_enabled():
    return os.getenv("VERL_LOGI_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def compute_score(data_source, solution_str, ground_truth):
    # 匹配 \boxed{A}, \boxed{{B}}, \boxed{(C)}, \boxed{{(D)}} 等情况
    matches = re.findall(r'\\boxed\{(?:\{\s*(?:\(\s*([A-Za-z])\s*\)|([A-Za-z]))\s*\}|\s*(?:\(\s*([A-Za-z])\s*\)|([A-Za-z]))\s*)\}(?!\})', solution_str)
    if _debug_enabled():
        print(f"Ground Truth: {ground_truth}")
        print(f"Solution String: {solution_str}")

    if matches:
        extracted_answer = [next(group for group in match if group).upper() for match in matches]
        if extracted_answer[-1] == ground_truth.upper():
            return 1.0
        else:
            return 0.0
    else:
        return 0.0
