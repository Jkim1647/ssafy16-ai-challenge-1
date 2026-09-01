from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


CHOICES = "abcd"


def read_jsonl(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        return {row["id"]: row for row in map(json.loads, filter(str.strip, handle))}


def probs(row: dict) -> list[float]:
    return [float(row[f"score_{label}"]) for label in CHOICES]


def label(values: list[float]) -> str:
    return CHOICES[max(range(4), key=values.__getitem__)]


def write_submission(path: Path, ids: list[str], answers: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "answer"])
        writer.writerows(zip(ids, answers))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dev", nargs=2, type=Path, required=True)
    parser.add_argument("--new-dev", nargs=3, type=Path, required=True)
    parser.add_argument("--base-test", nargs=2, type=Path, required=True)
    parser.add_argument("--new-test", nargs=3, type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="qwen35_multihr")
    args = parser.parse_args()

    bd = [read_jsonl(path) for path in args.base_dev]
    nd = [read_jsonl(path) for path in args.new_dev]
    bt = [read_jsonl(path) for path in args.base_test]
    nt = [read_jsonl(path) for path in args.new_test]
    dev_ids = list(bd[0])
    with args.sample.open(encoding="utf-8-sig", newline="") as handle:
        test_ids = [row["id"] for row in csv.DictReader(handle)]
    if any(set(x) != set(dev_ids) for x in bd + nd):
        raise ValueError("Dev prediction IDs differ")
    if any(set(x) != set(test_ids) for x in bt + nt):
        raise ValueError("Test prediction IDs differ")

    truth = [nd[0][item_id]["true_label"] for item_id in dev_ids]
    dev_base = [[sum(probs(x[item_id])[j] for x in bd) / 2 for j in range(4)] for item_id in dev_ids]
    test_base = [[sum(probs(x[item_id])[j] for x in bt) / 2 for j in range(4)] for item_id in test_ids]
    dev_new = [[[probs(x[item_id])[j] for j in range(4)] for x in nd] for item_id in dev_ids]
    test_new = [[[probs(x[item_id])[j] for j in range(4)] for x in nt] for item_id in test_ids]

    configs = []
    # Base weight 0.30..0.65; remaining mass is split across standard/896/1024.
    for base_tick in range(12, 27):
        wb = base_tick / 40
        remaining = 1.0 - wb
        for a in range(11):
            for b in range(11 - a):
                c = 10 - a - b
                wn = [remaining * a / 10, remaining * b / 10, remaining * c / 10]
                pred = []
                fold_correct = [0, 0, 0, 0]
                fold_total = [0, 0, 0, 0]
                for item_id, base, new, target in zip(dev_ids, dev_base, dev_new, truth):
                    mixed = [wb * base[j] + sum(wn[k] * new[k][j] for k in range(3)) for j in range(4)]
                    chosen = label(mixed)
                    pred.append(chosen)
                    fold = int(hashlib.md5(item_id.encode()).hexdigest(), 16) % 4
                    fold_total[fold] += 1
                    fold_correct[fold] += chosen == target
                accuracy = sum(x == y for x, y in zip(pred, truth)) / len(truth)
                folds = [x / y for x, y in zip(fold_correct, fold_total)]
                configs.append({"weights": [wb, *wn], "accuracy": accuracy, "fold_min": min(folds), "folds": folds})

    configs.sort(key=lambda row: (row["accuracy"], row["fold_min"]), reverse=True)
    best = configs[0]["accuracy"]
    chosen_configs = []
    seen_answers = set()
    for config in configs:
        if config["accuracy"] < best - 2 / len(dev_ids):
            break
        wb, *wn = config["weights"]
        answers = tuple(
            label([wb * base[j] + sum(wn[k] * new[k][j] for k in range(3)) for j in range(4)])
            for base, new in zip(test_base, test_new)
        )
        if answers in seen_answers:
            continue
        seen_answers.add(answers)
        chosen_configs.append((config, answers))
        if len(chosen_configs) == 16:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for rank, (config, answers) in enumerate(chosen_configs, 1):
        weights = config["weights"]
        name = f"submission_{args.prefix}_rank{rank:02d}.csv"
        write_submission(args.output_dir / name, test_ids, answers)
        candidates.append({"file": name, **config, "weights": weights})
    report = {"dev_samples": len(dev_ids), "grid_size": len(configs), "best_accuracy": best, "candidates": candidates, "top_grid": configs[:40]}
    report_path = args.output_dir / f"{args.prefix}_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_accuracy": best, "generated": len(candidates), "top": candidates[:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()