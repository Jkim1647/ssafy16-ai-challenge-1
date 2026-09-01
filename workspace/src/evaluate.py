from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from common import CHOICES, read_jsonl, write_json


def evaluate(rows: list[dict]) -> dict:
    labeled = [row for row in rows if row.get("true_label") in CHOICES]
    if not labeled:
        raise ValueError("Prediction file has no true_label values")
    confusion = {truth: {pred: 0 for pred in CHOICES} for truth in CHOICES}
    for row in labeled:
        confusion[row["true_label"]][row["predicted_label"]] += 1
    correct = sum(row["true_label"] == row["predicted_label"] for row in labeled)
    return {
        "samples": len(labeled),
        "accuracy": correct / len(labeled),
        "correct": correct,
        "label_distribution": dict(Counter(row["true_label"] for row in labeled)),
        "prediction_distribution": dict(Counter(row["predicted_label"] for row in labeled)),
        "confusion_matrix": confusion,
        "high_confidence_wrong": sum(
            row["true_label"] != row["predicted_label"] and row["confidence"] >= 0.8
            for row in labeled
        ),
        "low_confidence_correct": sum(
            row["true_label"] == row["predicted_label"] and row["confidence"] < 0.5
            for row in labeled
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate labeled prediction JSONL")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = evaluate(read_jsonl(args.predictions))
    output = args.output or args.predictions.with_suffix(".metrics.json")
    write_json(output, metrics)
    print(metrics)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()