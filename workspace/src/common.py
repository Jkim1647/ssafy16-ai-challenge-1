from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

CHOICES = ("a", "b", "c", "d")
SYSTEM_PROMPT = (
    "You are a visual multiple-choice question answering assistant. "
    "Answer with exactly one lowercase letter: a, b, c, or d. "
    "Do not provide an explanation."
)


def project_root() -> Path:
    override = os.environ.get("SSAFY_DATA_ROOT")
    return Path(override).resolve() if override else Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_from_project(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root() / candidate


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with resolve_from_project(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_mc_prompt(row: dict[str, Any], language: str = "ko") -> str:
    suffix = {
        "ko": "정답은 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 답하세요.",
        "en": "Answer with exactly one lowercase letter from a, b, c, or d.",
    }[language]
    return (
        f"{row['question']}\n"
        f"(a) {row['a']}\n"
        f"(b) {row['b']}\n"
        f"(c) {row['c']}\n"
        f"(d) {row['d']}\n\n"
        f"{suffix}"
    )


def build_messages(
    row: dict[str, Any], image: Any, answer: str | None = None, language: str = "ko"
) -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_mc_prompt(row, language)},
            ],
        },
    ]
    if answer is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )
    return messages


def majority_vote(row: dict[str, str]) -> dict[str, Any]:
    votes = [row.get(f"answer{i}", "").strip().lower() for i in range(1, 6)]
    votes = [vote for vote in votes if vote in CHOICES]
    counts = Counter(votes)
    if not counts:
        return {"label": None, "votes": 0, "valid_votes": 0, "tie": False}
    top_count = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == top_count)
    return {
        "label": winners[0] if len(winners) == 1 else None,
        "votes": top_count,
        "valid_votes": len(votes),
        "tie": len(winners) > 1,
        "vote_counts": dict(sorted(counts.items())),
    }


def stratified_split(
    rows: Sequence[dict[str, Any]], val_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic stratified split matching sklearn's ceil(test_size * n)."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = str(row.get("answer", "")).lower()
        if label not in CHOICES:
            raise ValueError(f"Invalid answer label: {label!r}")
        groups[label].append(dict(row))

    target_val = math.ceil(len(rows) * val_fraction)
    exact = {label: len(group) * target_val / len(rows) for label, group in groups.items()}
    allocation = {label: math.floor(value) for label, value in exact.items()}
    remaining = target_val - sum(allocation.values())
    remainder_order = sorted(groups, key=lambda label: (exact[label] % 1, label), reverse=True)
    for label in remainder_order[:remaining]:
        allocation[label] += 1

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for label in sorted(groups):
        group = groups[label]
        rng.shuffle(group)
        val_count = allocation[label]
        val_rows.extend(group[:val_count])
        train_rows.extend(group[val_count:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stratified_group_split_by_image(
    rows: Sequence[dict[str, Any]], val_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Keep byte-identical images together while matching the stratified target exactly."""
    target_total = math.ceil(len(rows) * val_fraction)
    label_counts = Counter(str(row["answer"]).lower() for row in rows)
    exact = {label: count * target_total / len(rows) for label, count in label_counts.items()}
    target = {label: math.floor(value) for label, value in exact.items()}
    for label in sorted(exact, key=lambda x: (exact[x] % 1, x), reverse=True)[: target_total - sum(target.values())]:
        target[label] += 1

    hash_by_id: dict[str, str] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        digest = file_sha256(resolve_from_project(row["path"]))
        hash_by_id[row["id"]] = digest
        groups[digest].append(row)

    rng = random.Random(seed)
    multi_groups = [group for group in groups.values() if len(group) > 1]
    singleton_groups = [group for group in groups.values() if len(group) == 1]
    rng.shuffle(multi_groups)
    rng.shuffle(singleton_groups)

    val_groups: list[list[dict[str, Any]]] = []
    current = Counter()
    for group in multi_groups:
        contribution = Counter(row["answer"] for row in group)
        fits = all(current[label] + count <= target[label] for label, count in contribution.items())
        if fits and rng.random() < val_fraction:
            val_groups.append(group)
            current.update(contribution)

    singletons_by_label: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for group in singleton_groups:
        singletons_by_label[group[0]["answer"]].append(group)
    for label in CHOICES:
        needed = target[label] - current[label]
        if needed < 0 or len(singletons_by_label[label]) < needed:
            raise RuntimeError(f"Unable to satisfy grouped split target for label {label}")
        val_groups.extend(singletons_by_label[label][:needed])

    val_hashes = {file_sha256(resolve_from_project(group[0]["path"])) for group in val_groups}
    val_rows = [row for digest, group in groups.items() if digest in val_hashes for row in group]
    train_rows = [row for digest, group in groups.items() if digest not in val_hashes for row in group]
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    if len(val_rows) != target_total or Counter(row["answer"] for row in val_rows) != Counter(target):
        raise AssertionError("Grouped split failed to match the requested stratified size")
    return train_rows, val_rows, hash_by_id


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML before loading experiment configs") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)