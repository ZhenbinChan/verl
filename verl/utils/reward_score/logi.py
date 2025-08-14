# -*- coding: utf-8 -*-

import re

def compute_score(solution_str, ground_truth):
    # 匹配 \boxed{A} 或 \boxed{{B}} 等情况
    matches = re.findall(r'\\boxed\{\{?([A-Za-z])\}?\}', solution_str)
    # print(f"Ground Truth: {ground_truth}")
    # print(f"Solution String: {solution_str}")
    if matches:
        extracted_answer = [letter.upper() for letter in matches]
        # print(f"Extracted Answer: {extracted_answer[-1]}")
        if extracted_answer[-1] == ground_truth.upper():
            return 1.0, None
        else:
            return 0.0, None
    else:
        return 0.0, None
