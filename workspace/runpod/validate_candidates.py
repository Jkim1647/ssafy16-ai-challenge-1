from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pattern", default="submission_qwen35_qlora320_*.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sample = read(args.sample)
    sample_ids = [row["id"] for row in sample]
    baseline = {row["id"]: row["answer"] for row in read(args.baseline)}
    results = []
    for path in sorted(args.directory.glob(args.pattern)):
        rows = read(path)
        ids = [row.get("id") for row in rows]
        answers = [row.get("answer") for row in rows]
        valid = (
            len(rows) == 5074
            and list(rows[0]) == ["id", "answer"]
            and ids == sample_ids
            and all(answer in "abcd" for answer in answers)
        )
        results.append(
            {
                "file": path.name,
                "valid": valid,
                "rows": len(rows),
                "changed_vs_public_best": sum(
                    answer != baseline[item_id] for item_id, answer in zip(ids, answers)
                ),
            }
        )
    report = {"count": len(results), "all_valid": all(x["valid"] for x in results), "files": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"count": report["count"], "all_valid": report["all_valid"]}))


if __name__ == "__main__":
    main()