# -*- coding: utf-8 -*-

import re

def compute_score(solution_str, ground_truth):
    # 匹配 \boxed{A} 或 \boxed{{B}} 等情况
    matches = re.findall(r'\\boxed\{\{?([A-Za-z])\}?\}', solution_str)
    
    if not matches:
        # 尝试抽取 Option(C) 中的 C
        matches = re.findall(r'Option\(\s*([A-Za-z])\s*\)', solution_str)
    if not matches:
        # 尝试抽取 "The answer is C." 或 "The answer is Option C." 中的 C
        # (?:Option\s*)? 表示 "Option " 这个词是可选的
        # \b 或者 \.? 允许匹配句号
        matches = re.findall(r'The answer is\s*(?:Option\s*)?([A-Za-z])\b', solution_str)
    if not matches:
        # 尝试抽取 "Answer: C" 或 "Answer: Option C" 中的 C
        matches = re.findall(r'Answer:\s*(?:Option\s*)?([A-Za-z])\b', solution_str)
    if not matches:
        # 尝试抽取 "C is the correct answer" 中的 C
        matches = re.findall(r'([A-Za-z])\s*is the correct answer', solution_str)
    if not matches:
        # 尝试抽取 "The correct answer is C" 或 "The correct answer is Option C" 中的 C
        matches = re.findall(r'The correct answer is\s*(?:Option\s*)?([A-Za-z])\b', solution_str)
    if not matches:
        # 尝试抽取 Option C 中的 C（使用 \b 单词边界替代 \s，这样紧跟句号也能匹配）
        matches = re.findall(r'Option\s+([A-Za-z])\b', solution_str)
    if not matches:
        # 尝试抽取**Option (A)** 中的 A
        matches = re.findall(r'Option\s*\(\s*([A-Za-z])\s*\)', solution_str)

    if matches:
        extracted_answer = [letter.upper() for letter in matches]
        # print(f"Extracted Answer: {extracted_answer[-1]}")
        if extracted_answer[-1] == ground_truth.upper():
            return 1.0
        else:
            return 0.0
    else:
        return 0.0
    
def test_logiqa_reward():
    test_cases = [
        ("The answer is \\boxed{A}.", "A", 1.0),
        ("The correct answer is B", "B", 1.0),
        ("Selected Option(C)", "C", 1.0),
        ("Answer: D", "D", 1.0),
        ("Option A is the correct answer", "A", 1.0),
        ("Thinking process... The answer is Option B.", "B", 1.0),
        ("The answer is \\boxed{A}.", "B", 0.0),
        ("No answer given here.", "A", 0.0),
        ("Multiple answers: Option A, then Option B.", "B", 1.0), # Should pick the last one
    ]

    for solution, gt, expected in test_cases:
        score = compute_score(solution_str=solution, ground_truth=gt)
        print(f"Solution: {solution[:30]}... | GT: {gt} | Score: {score} | Expected: {expected}")
        assert score == expected, f"Expected {expected} but got {score} for solution: {solution}"