from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from PIL import Image

from common import CHOICES, majority_vote, project_root, read_csv, workspace_root, write_json


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path, size: int = 8) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS).getdata())
    value = 0
    for row in range(size):
        offset = row * (size + 1)
        for col in range(size):
            value = (value << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return value


class BKTree:
    def __init__(self) -> None:
        self.root: tuple[int, list[tuple[str, str]], dict[int, object]] | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int, item: tuple[str, str]) -> None:
        if self.root is None:
            self.root = (value, [item], {})
            return
        node = self.root
        while True:
            current, items, children = node
            distance = self.distance(value, current)
            if distance == 0:
                items.append(item)
                return
            if distance not in children:
                children[distance] = (value, [item], {})
                return
            node = children[distance]  # type: ignore[assignment]

    def query(self, value: int, threshold: int) -> list[tuple[int, tuple[str, str]]]:
        if self.root is None:
            return []
        matches: list[tuple[int, tuple[str, str]]] = []
        stack = [self.root]
        while stack:
            current, items, children = stack.pop()
            distance = self.distance(value, current)
            if distance <= threshold:
                matches.extend((distance, item) for item in items)
            low, high = distance - threshold, distance + threshold
            stack.extend(child for edge, child in children.items() if low <= edge <= high)
        return matches


def analyze_images(rows_by_split: dict[str, list[dict[str, str]]], near_threshold: int) -> dict:
    root = project_root()
    records: list[dict] = []
    failures: list[dict] = []
    exact_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    near_pairs: list[dict] = []
    tree = BKTree()

    for split, rows in rows_by_split.items():
        for row in rows:
            path = root / row["path"]
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    mode, fmt = image.mode, image.format
                    image.verify()
                digest = sha256(path)
                perceptual = dhash(path)
                item = (split, row["id"])
                for distance, other in tree.query(perceptual, near_threshold):
                    other_split, other_id = other
                    if other_split != split:
                        near_pairs.append(
                            {
                                "left_split": other_split,
                                "left_id": other_id,
                                "right_split": split,
                                "right_id": row["id"],
                                "dhash_distance": distance,
                            }
                        )
                tree.add(perceptual, item)
                exact_groups[digest].append(item)
                records.append(
                    {
                        "split": split,
                        "id": row["id"],
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "format": fmt,
                        "bytes": path.stat().st_size,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - report every corrupt file
                failures.append({"split": split, "id": row["id"], "error": repr(exc)})

    exact_cross_split = []
    for digest, items in exact_groups.items():
        if len({split for split, _ in items}) > 1:
            exact_cross_split.append({"sha256": digest, "items": items})

    dimensions = Counter((record["width"], record["height"]) for record in records)
    return {
        "readable": len(records),
        "failures": failures,
        "modes": dict(Counter(record["mode"] for record in records)),
        "formats": dict(Counter(record["format"] for record in records)),
        "width": {
            "min": min(record["width"] for record in records),
            "median": median(record["width"] for record in records),
            "max": max(record["width"] for record in records),
        },
        "height": {
            "min": min(record["height"] for record in records),
            "median": median(record["height"] for record in records),
            "max": max(record["height"] for record in records),
        },
        "common_dimensions": [
            {"width": width, "height": height, "count": count}
            for (width, height), count in dimensions.most_common(20)
        ],
        "total_bytes": sum(record["bytes"] for record in records),
        "exact_cross_split_duplicates": exact_cross_split,
        "near_cross_split_pairs": near_pairs,
        "near_duplicate_threshold": near_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit competition CSVs and image files")
    parser.add_argument("--near-threshold", type=int, default=2, help="64-bit dHash distance")
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()

    rows_by_split = {name: read_csv(f"{name}.csv") for name in ("train", "dev", "test")}
    submission = read_csv("sample_submission.csv")
    train, dev, test = (rows_by_split[name] for name in ("train", "dev", "test"))

    dev_votes = [majority_vote(row) for row in dev]
    high_conf = [vote for vote in dev_votes if not vote["tie"] and vote["votes"] >= 4]
    report = {
        "project_root": str(project_root()),
        "rows": {name: len(rows) for name, rows in rows_by_split.items()},
        "columns": {name: list(rows[0]) for name, rows in rows_by_split.items()},
        "sample_submission": {
            "rows": len(submission),
            "columns": list(submission[0]),
            "ordered_id_match": [row["id"] for row in submission] == [row["id"] for row in test],
            "initial_answers_blank": all(not row["answer"].strip() for row in submission),
        },
        "label_distribution": dict(sorted(Counter(row["answer"] for row in train).items())),
        "invalid_train_labels": [row["id"] for row in train if row["answer"] not in CHOICES],
        "missing_image_paths": {
            name: [row["path"] for row in rows if not (project_root() / row["path"]).is_file()]
            for name, rows in rows_by_split.items()
        },
        "dev_votes": {
            "max_vote_distribution": dict(sorted(Counter(vote["votes"] for vote in dev_votes).items())),
            "tie_count": sum(vote["tie"] for vote in dev_votes),
            "rows_with_missing_votes": sum(vote["valid_votes"] < 5 for vote in dev_votes),
            "high_confidence_threshold": ">=4 valid votes for one unique label",
            "high_confidence_count": len(high_conf),
            "high_confidence_labels": dict(
                sorted(Counter(vote["label"] for vote in high_conf).items())
            ),
            "balanced_per_label": min(Counter(vote["label"] for vote in high_conf).values()),
        },
        "exact_question_overlap_rows": {
            "train_dev": sum(row["question"] in {r["question"] for r in train} for row in dev),
            "train_test": sum(row["question"] in {r["question"] for r in train} for row in test),
            "dev_test": sum(row["question"] in {r["question"] for r in dev} for row in test),
        },
    }
    if not args.skip_hashes:
        report["images"] = analyze_images(rows_by_split, args.near_threshold)

    output = workspace_root() / "reports" / "data_analysis.json"
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()