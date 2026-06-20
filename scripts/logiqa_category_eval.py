#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", str(question or "").lower().strip(" .:?？"))


def regex(pattern: str, text: str) -> bool:
    return re.search(pattern, text) is not None


def extract_tag(raw_prompt: str, tag: str) -> str:
    match = re.search(fr"<{tag}>(.*?)</{tag}>", str(raw_prompt or ""), re.DOTALL)
    return " ".join(match.group(1).strip().split()) if match else ""


def get_question_text(row: dict[str, Any]) -> str:
    raw_prompt = row.get("raw_prompt") or ""
    question = extract_tag(raw_prompt, "Question")
    if question:
        return question
    extra_info = row.get("extra_info") or {}
    if isinstance(extra_info, dict):
        return str(extra_info.get("query") or extra_info.get("question") or "")
    return ""


def get_context_text(row: dict[str, Any]) -> str:
    raw_prompt = row.get("raw_prompt") or ""
    context = extract_tag(raw_prompt, "Context")
    if context:
        return context
    extra_info = row.get("extra_info") or {}
    if isinstance(extra_info, dict):
        return str(extra_info.get("context") or "")
    return ""


def categorize_logiqa(row: dict[str, Any]) -> str:
    question = normalize_question(get_question_text(row))
    context = normalize_question(get_context_text(row))

    if regex(
        r"\b(arrange\w*|arrangement\w*|rank\w*|order\w*|sequence\w*|sit\w*|sitting|position\w*|floor\w*|valve\w*|team\w*|group\w*|committee|schedule\w*|week\w*|warehouse\w*|allocation|assigned|selected|which pair|how many|who (won|is|can|works)|where does|from left to right)\b",
        question,
    ):
        return "排列组合/约束求解类"
    if regex(
        r"\b(except|inconsistent|not belong|not part|does not belong|do not belong|doesn't fit|does not fit|not fit|not conform|does not conform|not match|does not agree|not agree|does not apply|not apply|not correct|incorrect|wrong|false|must be false|cannot be true|can\s*not|can't|least (helpful|important|likely|supportive)|all but|in addition to)\b",
        question,
    ):
        return "否定/例外/不符合类"
    if regex(
        r"\b(same|similar|parallel|analog\w*|resemblance|corresponds)\b.*\b(logic\w*|fallac\w*|mistake\w*|method\w*|argument\w*|reasoning|structure|inference\w*)\b|\b(logical (mistake|fallacy|loophole|method|structure)|fallac\w*|sophistr\w*|flaw\w*|loophole\w*|vulnerab\w*|shortcoming\w*|deficien\w*|error in .*reasoning|problem with .*argument)\b",
        question,
    ):
        return "类比/逻辑谬误/缺陷识别类"
    if regex(
        r"\b(technique\w*|strategy|method used|argument method|demonstration technique\w*|principle\w*|controversial focus|focus of .*argument|issues? discussed|complete .*argument|logically complete|complete the above|best completes?|corollary)\b",
        question,
    ):
        return "论证结构/技巧/补全类"
    if regex(
        r"\b(weaken\w*|undermin\w*|challeng\w*|refut\w*|rebut\w*|doubt\w*|question\w*|object\w*|oppos\w*|counterexample|against|shake\w*|critici\w*)\b",
        question,
    ):
        return "削弱/质疑/反驳类"
    if regex(
        r"\b(strengthen\w*|support\w*|justify\w*|confirm\w*|evidence|basis for|prove\w*|proof|favo[u]?r\w*|reinforce\w*)\b",
        question,
    ):
        return "加强/支持类"
    if regex(
        r"\b(assum\w*|hypoth\w*|presuppos\w*|premise\w*|necessary|sufficient|guarantee\w*|valid\w*|establish\w*|depends? on|rel(?:y|ies|ied|ying) on|implicit in|missing from)\b",
        question,
    ):
        return "假设/前提/必要条件类"
    if regex(
        r"\b(evaluat\w*|assess\w*|assessment|most important question|determine whether|to judge|need to know|accuracy of|validity of|rationality|relevant for evaluating|commenting on)\b",
        question,
    ):
        return "评价论证/关键信息类"
    if regex(
        r"\b(explain\w*|reason for|why|cause\w*|account for|resolve\w*|reconcile\w*|eliminat\w*.*inconsisten\w*|phenomenon|motivation|due to|because|situation|contradict\w*|anomal\w*|exception)\b",
        question,
    ):
        return "解释原因/解决矛盾类"
    if regex(
        r"\b(according to (the above )?definition|based on (the above )?definition|by definition|definition|belongs? to|category|concept|case belongs|which case|which behavior|which phenomenon|which are .*category|is .*category|such .* belongs to|embod\w*|reflect\w*|constitute\w*|within the scope|typical|example\w*)\b",
        question,
    ) or regex(r"\b(refers to|is called|means that|defined as|definition)\b", context[:800]):
        return "定义匹配/案例归类类"
    if regex(
        r"\b(main|mainly|summary|summari[sz]e|recap|gist|title|intend\w*|emphasi[sz]e|introduce\w*|states|talks about|point|view|meaning|mean|illustrat\w*|indicat\w*|express\w*|determination|assertion|understanding|say about|what does this text|what does this passage|this text mean|this passage mean|rephras\w*|restatement|closest to)\b",
        question,
    ):
        return "主旨/概括/语义转述类"
    if regex(r"\b(most likely|next paragraph|next is|will talk|attitude|author.*focus|likely to|unlikely|may be correct|possibly correct)\b", question):
        return "语境续写/作者态度/可能性类"
    if regex(r"\b(necessarily true|must be true|must also be true|which .* true|is true|also true|can be true|may be true|must\b|possible|impossible|reasonable inference)\b", question):
        return "必然/可能/真假判断类"
    if regex(
        r"\b(infer\w*|derived?|deduc\w*|drawn|conclud\w*|conclusion\w*|follows?|know from|learn\w*|based on|according to|can be launched|launch\w*|therefore|then$|so$|so\?|visible|shows?$|shows that|from this|can be seen|we can see|statement is correct|which is correct|correct statement|consistent with the original|can be obtained|it can be)\b",
        question,
    ):
        return "直接推断/结论类"
    return "其他混合类"


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return str(value)


def get_ground_truth(row: dict[str, Any]) -> str:
    reward_model = row.get("reward_model") or {}
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    if row.get("answer") is not None:
        return str(row["answer"])
    extra_info = row.get("extra_info") or {}
    if isinstance(extra_info, dict) and extra_info.get("answer") is not None:
        return str(extra_info["answer"])
    return ""


def response_list(value: Any) -> list[str]:
    value = json_safe(value)
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def score_response(response: str, ground_truth: str) -> float:
    matches = re.findall(r"\\boxed\{\{?([A-Za-z])\}?\}", response or "")
    if not matches:
        return 0.0
    return 1.0 if matches[-1].upper() == ground_truth.upper() else 0.0


def score_responses(responses: list[str], ground_truth: str, sample_agg: str) -> tuple[float, str]:
    if not responses:
        return 0.0, ""
    scores = [score_response(response, ground_truth) for response in responses]
    if sample_agg in {"best", "max", "best_of_n", "any", "any_correct"}:
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        return float(scores[best_idx]), responses[best_idx]
    if sample_agg == "first":
        return float(scores[0]), responses[0]
    if sample_agg == "mean":
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        return float(sum(scores) / len(scores)), responses[best_idx]
    raise ValueError(f"Unsupported --sample-agg: {sample_agg}")


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped["ALL"] = records
    for record in records:
        grouped[record["category"]].append(record)

    summary = []
    for category, items in grouped.items():
        total = len(items)
        correct = sum(1 for item in items if item["score"] > 0)
        summary.append(
            {
                "category": category,
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
            }
        )
    return sorted(summary, key=lambda row: (row["category"] != "ALL", -row["total"], row["category"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LogiQA generated responses by question category.")
    parser.add_argument("--input-path", required=True, help="Generated parquet containing a responses column.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-agg", choices=["best", "max", "best_of_n", "any", "any_correct", "first", "mean"], default="best")
    parser.add_argument("--response-key", default="responses")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_path)
    if args.response_key not in df.columns:
        raise KeyError(f"Missing response column {args.response_key!r} in {args.input_path}")

    records = []
    for idx, row in enumerate(df.to_dict("records")):
        row = json_safe(row)
        ground_truth = get_ground_truth(row)
        responses = response_list(row.get(args.response_key))
        score, selected_response = score_responses(responses, ground_truth, args.sample_agg)
        category = categorize_logiqa(row)
        records.append(
            {
                "row_index": idx,
                "sample_id": row.get("sample_id", f"logiqa_{idx}"),
                "split": (row.get("extra_info") or {}).get("split") if isinstance(row.get("extra_info"), dict) else None,
                "category": category,
                "question": get_question_text(row),
                "ground_truth": ground_truth,
                "score": score,
                "correct": bool(score > 0),
                "num_responses": len(responses),
                "selected_response": selected_response,
            }
        )

    summary = summarize(records)
    category_counts = Counter(record["category"] for record in records)
    metadata = {
        "input_path": str(args.input_path),
        "total": len(records),
        "sample_agg": args.sample_agg,
        "category_counts": dict(category_counts),
    }

    pd.DataFrame(records).to_parquet(output_dir / "per_sample_category_eval.parquet")
    pd.DataFrame(summary).to_csv(output_dir / "category_accuracy.csv", index=False)
    (output_dir / "category_accuracy.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved per-sample results to {output_dir / 'per_sample_category_eval.parquet'}", flush=True)
    print(f"Saved category summary to {output_dir / 'category_accuracy.csv'}", flush=True)


if __name__ == "__main__":
    main()
