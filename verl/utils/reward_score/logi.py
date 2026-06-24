# -*- coding: utf-8 -*-

import os
import re


def _debug_enabled():
    return os.getenv("VERL_LOGI_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def compute_score(solution_str, ground_truth):
    # 匹配 \boxed{A}, \boxed{{B}}, \boxed{(C)}, \boxed{{(D)}} 等情况
    matches = re.findall(r'\\boxed\{(?:\{\s*(?:\(\s*([A-Za-z])\s*\)|([A-Za-z]))\s*\}|\s*(?:\(\s*([A-Za-z])\s*\)|([A-Za-z]))\s*)\}(?!\})', solution_str)
    if _debug_enabled():
        print(f"Ground Truth: {ground_truth}")
        print(f"Solution String: {solution_str}")
    # if not matches:
    #     # 尝试抽取 Option(C) 中的 C
    #     matches = re.findall(r'Option\(\s*([A-Za-z])\s*\)', solution_str)
    # if not matches:
    #     # 尝试抽取 "The answer is C." 中的 C
    #     matches = re.findall(r'The answer is\s*([A-Za-z])\s*\.', solution_str)
    # if not matches:
    #     # 尝试抽取 "Answer: C" 中的 C
    #     matches = re.findall(r'Answer:\s*([A-Za-z])\s*', solution_str)
    # if not matches:
    #     # 尝试抽取 "C is the correct answer" 中的 C
    #     matches = re.findall(r'([A-Za-z])\s*is the correct answer', solution_str)
    # if not matches:
    #     # 尝试抽取 "The correct answer is C" 中的 C
    #     matches = re.findall(r'The correct answer is\s*([A-Za-z])\s*', solution_str)
    # if not matches:
    #     # 尝试抽取 Option C 中的 C
    #     matches = re.findall(r'Option\s*([A-Za-z])\s', solution_str)
    # if not matches:
    #     # 尝试抽取**Option (A)** 中的 A
    #     matches = re.findall(r'Option\s*\(\s*([A-Za-z])\s*\)', solution_str)


    if matches:
        extracted_answer = [next(group for group in match if group).upper() for match in matches]
        # print(f"Extracted Answer: {extracted_answer[-1]}")
        if extracted_answer[-1] == ground_truth.upper():
            return 1.0, None
        else:
            return 0.0, None
    else:
        return 0.0, None
