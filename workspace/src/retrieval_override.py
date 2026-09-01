from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

from common import CHOICES, project_root, read_csv, read_jsonl, workspace_root, write_jsonl
from inference import write_submission


def image_hash(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root() / candidate
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return " ".join(str(text).split())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optional exact-image + exact-question consistency override using train labels only"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--acknowledge-rules",
        action="store_true",
        help="Confirm that competition rules permit train-derived duplicate consistency mapping",
    )
    parser.add_argument("--submission", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_rules:
        raise RuntimeError(
            "Not applied. Review the competition rules, then pass --acknowledge-rules if this "
            "provided-train-data consistency mapping is permitted."
        )

    train_index = defaultdict(list)
    for row in read_csv("train.csv"):
        train_index[(image_hash(row["path"]), normalized(row["question"]))].append(row)

    predictions = read_jsonl(args.predictions)
    override_count = 0
    ambiguous_count = 0
    for row in predictions:
        key = (image_hash(row["image"]), normalized(row["question"]))
        matches = train_index.get(key, [])
        if not matches:
            continue
        answer_texts = {normalized(match[match["answer"]]) for match in matches}
        mapped_labels = [
            label for label in CHOICES if normalized(row[label]) in answer_texts
        ]
        if len(answer_texts) == 1 and len(mapped_labels) == 1:
            row["model_predicted_label"] = row["predicted_label"]
            row["predicted_label"] = mapped_labels[0]
            row["retrieval_override"] = True
            row["retrieval_train_ids"] = [match["id"] for match in matches]
            row["retrieval_answer_text"] = next(iter(answer_texts))
            override_count += 1
        else:
            row["retrieval_override"] = False
            row["retrieval_ambiguous"] = True
            ambiguous_count += 1

    output = workspace_root() / "outputs" / "predictions" / f"{args.name}.jsonl"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    write_jsonl(output, predictions)
    print(f"Overrides={override_count}; ambiguous_skipped={ambiguous_count}; saved={output}")
    if args.submission:
        submission = workspace_root() / "outputs" / "submissions" / f"submission_{args.name}.csv"
        write_submission(predictions, submission, overwrite=False)
        print(f"Saved: {submission}")


if __name__ == "__main__":
    main()