from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


CHOICES = ("a", "b", "c", "d")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized(value: str) -> str:
    return " ".join(str(value).split())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    train_rows = read_csv(root / "train.csv")
    test_rows = read_csv(root / "test.csv")
    predictions = read_csv(args.submission.resolve())
    by_id = {row["id"]: row for row in test_rows}

    train_index: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        key = (digest(root / row["path"]), normalized(row["question"]))
        train_index[key].append(row)

    changes = []
    for prediction in predictions:
        test = by_id[prediction["id"]]
        key = (digest(root / test["path"]), normalized(test["question"]))
        matches = train_index.get(key, [])
        if not matches:
            continue
        answer_texts = {normalized(row[row["answer"]]) for row in matches}
        mapped = [choice for choice in CHOICES if normalized(test[choice]) in answer_texts]
        if len(answer_texts) == 1 and len(mapped) == 1 and prediction["answer"] != mapped[0]:
            changes.append({
                "id": prediction["id"],
                "old": prediction["answer"],
                "new": mapped[0],
                "train_ids": [row["id"] for row in matches],
            })
            prediction["answer"] = mapped[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "answer"])
        writer.writeheader()
        writer.writerows(predictions)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"changes": changes}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output} with {len(changes)} deterministic duplicate corrections")


if __name__ == "__main__":
    main()