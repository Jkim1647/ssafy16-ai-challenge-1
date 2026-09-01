from __future__ import annotations

import argparse
from pathlib import Path

from common import CHOICES, read_jsonl, workspace_root, write_jsonl
from inference import write_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Probability-average compatible prediction JSONL files")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--submission", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    models = [read_jsonl(path) for path in args.inputs]
    id_maps = [{row["id"]: row for row in rows} for rows in models]
    ids = list(id_maps[0])
    if any(set(mapping) != set(ids) for mapping in id_maps[1:]):
        raise ValueError("All prediction files must contain the same IDs")
    ensemble = []
    for item_id in ids:
        base = dict(id_maps[0][item_id])
        scores = {
            label: sum(mapping[item_id][f"score_{label}"] for mapping in id_maps) / len(id_maps)
            for label in CHOICES
        }
        prediction = max(CHOICES, key=scores.get)
        base.update({f"score_{label}": score for label, score in scores.items()})
        base["predicted_label"] = prediction
        base["confidence"] = scores[prediction]
        if base.get("true_label"):
            base["correct"] = base["true_label"] == prediction
        ensemble.append(base)
    output = workspace_root() / "outputs" / "predictions" / f"{args.name}.jsonl"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {output}")
    write_jsonl(output, ensemble)
    print(f"Saved: {output}")
    if args.submission:
        submission = workspace_root() / "outputs" / "submissions" / f"submission_{args.name}.csv"
        write_submission(ensemble, submission, args.overwrite)
        print(f"Saved: {submission}")


if __name__ == "__main__":
    main()