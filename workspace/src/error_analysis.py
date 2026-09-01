from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from common import CHOICES, read_jsonl, write_json


TYPE_RULES = {
    "counting": ("몇 개", "몇개", "개입니까", "개인가", "수는"),
    "spatial": ("왼쪽", "오른쪽", "위에", "아래", "앞", "뒤", "가까", "위치"),
    "ocr_text": ("글자", "문구", "텍스트", "쓰여", "적혀", "라벨", "표시"),
    "color": ("색", "색깔"),
    "material": ("재질", "소재"),
    "action_state": ("하고", "중인", "행동", "상태"),
}


def classify_question(question: str) -> str:
    for category, keywords in TYPE_RULES.items():
        if any(keyword in question for keyword in keywords):
            return category
    return "object_or_general_visual"


def analyze(rows: list[dict]) -> dict:
    labeled = [row for row in rows if row.get("true_label") in CHOICES]
    by_type = defaultdict(lambda: {"total": 0, "wrong": 0})
    error_pairs = Counter()
    for row in labeled:
        category = classify_question(row["question"])
        by_type[category]["total"] += 1
        if row["true_label"] != row["predicted_label"]:
            by_type[category]["wrong"] += 1
            error_pairs[f"{row['true_label']}->{row['predicted_label']}"] += 1
    type_metrics = {
        category: {
            **counts,
            "error_rate": counts["wrong"] / counts["total"] if counts["total"] else 0,
        }
        for category, counts in by_type.items()
    }
    wrong = [row for row in labeled if row["true_label"] != row["predicted_label"]]
    return {
        "note": "Question types are keyword heuristics; manually audit high-confidence errors.",
        "by_type": dict(sorted(type_metrics.items(), key=lambda item: item[1]["error_rate"], reverse=True)),
        "error_pairs": dict(error_pairs.most_common()),
        "high_confidence_wrong": sorted(wrong, key=lambda row: row["confidence"], reverse=True)[:100],
        "low_confidence_correct": sorted(
            [row for row in labeled if row["true_label"] == row["predicted_label"]],
            key=lambda row: row["confidence"],
        )[:100],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze VQA errors and confidence")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(read_jsonl(args.predictions))
    output = args.output or args.predictions.with_suffix(".errors.json")
    write_json(output, report)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()