from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from common import (
    file_sha256,
    majority_vote,
    project_root,
    read_csv,
    stratified_group_split_by_image,
    stratified_split,
    workspace_root,
    write_json,
    write_jsonl,
)


def portable_row(row: dict[str, str]) -> dict[str, str]:
    return dict(row)


def prepare(seed: int, val_fraction: float) -> dict:
    train = read_csv("train.csv")
    train_rows, val_rows = stratified_split(train, val_fraction, seed)
    grouped_train, grouped_val, train_hashes = stratified_group_split_by_image(train, val_fraction, seed)

    dev_candidates = []
    for row in read_csv("dev.csv"):
        vote = majority_vote(row)
        if not vote["tie"] and vote["votes"] >= 4:
            prepared = {key: row[key] for key in ("id", "path", "question", "a", "b", "c", "d")}
            prepared.update(
                {
                    "answer": vote["label"],
                    "pseudo_votes": vote["votes"],
                    "valid_votes": vote["valid_votes"],
                    "source": "dev_vote_pseudo_label",
                }
            )
            dev_candidates.append(prepared)

    by_label = {label: [row for row in dev_candidates if row["answer"] == label] for label in "abcd"}
    per_label = min(map(len, by_label.values()))
    rng = random.Random(seed)
    balanced = []
    for label in "abcd":
        group = by_label[label]
        rng.shuffle(group)
        balanced.extend(group[:per_label])
    rng.shuffle(balanced)

    manifest_dir = workspace_root() / "data" / "manifests"
    write_jsonl(manifest_dir / "train_seed42.jsonl", map(portable_row, train_rows))
    write_jsonl(manifest_dir / "val_seed42.jsonl", map(portable_row, val_rows))
    write_jsonl(manifest_dir / "train_grouped_seed42.jsonl", map(portable_row, grouped_train))
    write_jsonl(manifest_dir / "val_grouped_seed42.jsonl", map(portable_row, grouped_val))
    write_jsonl(manifest_dir / "dev_high_conf_902.jsonl", map(portable_row, dev_candidates))
    write_jsonl(manifest_dir / "dev_high_conf_balanced_468.jsonl", map(portable_row, balanced))

    grouped_val_hashes = {train_hashes[row["id"]] for row in grouped_val}
    safe_candidates = [
        row
        for row in dev_candidates
        if file_sha256(project_root() / row["path"]) not in grouped_val_hashes
    ]
    safe_by_label = {label: [row for row in safe_candidates if row["answer"] == label] for label in "abcd"}
    safe_per_label = min(map(len, safe_by_label.values()))
    safe_balanced = []
    safe_rng = random.Random(seed)
    for label in "abcd":
        group = safe_by_label[label]
        safe_rng.shuffle(group)
        safe_balanced.extend(group[:safe_per_label])
    safe_rng.shuffle(safe_balanced)
    write_jsonl(manifest_dir / "dev_high_conf_safe.jsonl", map(portable_row, safe_candidates))
    write_jsonl(manifest_dir / "dev_high_conf_balanced_safe.jsonl", map(portable_row, safe_balanced))

    summary = {
        "seed": seed,
        "val_fraction": val_fraction,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_labels": dict(sorted(Counter(row["answer"] for row in train_rows).items())),
        "val_labels": dict(sorted(Counter(row["answer"] for row in val_rows).items())),
        "grouped_train_rows": len(grouped_train),
        "grouped_val_rows": len(grouped_val),
        "grouped_train_labels": dict(sorted(Counter(row["answer"] for row in grouped_train).items())),
        "grouped_val_labels": dict(sorted(Counter(row["answer"] for row in grouped_val).items())),
        "grouped_split_exact_hash_overlap": 0,
        "dev_high_conf_rows": len(dev_candidates),
        "dev_high_conf_labels": dict(sorted(Counter(row["answer"] for row in dev_candidates).items())),
        "dev_balanced_rows": len(balanced),
        "dev_balanced_labels": dict(sorted(Counter(row["answer"] for row in balanced).items())),
        "dev_high_conf_safe_rows": len(safe_candidates),
        "dev_high_conf_balanced_safe_rows": len(safe_balanced),
        "dev_high_conf_balanced_safe_labels": dict(
            sorted(Counter(row["answer"] for row in safe_balanced).items())
        ),
        "warning": "Dev labels are pseudo-labels from five predictions, not ground truth. Never use as validation.",
    }
    write_json(manifest_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic train/validation and dev pseudo-label manifests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    args = parser.parse_args()
    summary = prepare(args.seed, args.val_fraction)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()