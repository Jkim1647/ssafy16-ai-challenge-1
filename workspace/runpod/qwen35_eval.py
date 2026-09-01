from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "workspace"
sys.path.insert(0, str(WORKSPACE / "src"))

from common import CHOICES, build_messages, project_root, read_csv, read_jsonl, write_jsonl  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    return read_jsonl(path) if path.suffix.lower() == ".jsonl" else read_csv(path)


def choice_ids(tokenizer) -> list[int]:
    result = []
    for label in CHOICES:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Choice {label!r} is not one token: {ids}")
        result.append(ids[0])
    return result


def write_submission(rows: list[dict], output: Path) -> None:
    expected = read_csv("sample_submission.csv")
    expected_ids = [row["id"] for row in expected]
    by_id = {row["id"]: row["predicted_label"] for row in rows}
    if set(by_id) != set(expected_ids):
        raise ValueError("Submission IDs do not match sample_submission.csv")
    if any(label not in CHOICES for label in by_id.values()):
        raise ValueError("Submission contains an invalid label")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "answer"])
        writer.writeheader()
        writer.writerows({"id": item_id, "answer": by_id[item_id]} for item_id in expected_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--min-pixels", type=int, default=448 * 448)
    parser.add_argument("--max-pixels", type=int, default=672 * 672)
    parser.add_argument("--submission", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    model_id = args.model_id
    if args.adapter:
        model_id = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    processor_source = args.adapter if args.adapter else model_id
    processor = AutoProcessor.from_pretrained(
        processor_source, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )
    model_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "auto" if args.load_in_4bit else {"": 0},
        "attn_implementation": "sdpa",
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.eval()
    token_ids = torch.tensor(choice_ids(processor.tokenizer), device=model.device)
    processor.tokenizer.padding_side = "left"

    rows = load_rows(args.input.resolve())
    outputs = []
    model_name = model_id.rsplit("/", 1)[-1]
    for start in tqdm(
        range(0, len(rows), args.batch_size), desc=f"{model_name} non-thinking"
    ):
        batch_rows = rows[start : start + args.batch_size]
        conversations = []
        image_paths = []
        for row in batch_rows:
            image_path = Path(row["path"])
            if not image_path.is_absolute():
                image_path = project_root() / image_path
            image_paths.append(image_path)
            conversations.append(
                build_messages(row, Image.open(image_path).convert("RGB"), language="ko")
            )
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[:, -1, :]
            scores = logits.index_select(-1, token_ids).float().softmax(dim=-1).cpu()
        for row, image_path, probabilities in zip(batch_rows, image_paths, scores.tolist()):
            prediction = CHOICES[max(range(4), key=probabilities.__getitem__)]
            record = {
                "id": row["id"],
                "image": str(image_path),
                "true_label": row.get("answer"),
                "predicted_label": prediction,
                "confidence": max(probabilities),
                **{f"score_{label}": value for label, value in zip(CHOICES, probabilities)},
            }
            if record["true_label"] in CHOICES:
                record["correct"] = record["true_label"] == prediction
            outputs.append(record)

    prediction_path = WORKSPACE / "outputs" / "predictions" / f"{args.name}.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(prediction_path, outputs)
    summary = {"samples": len(outputs)}
    labeled = [row for row in outputs if row.get("true_label") in CHOICES]
    if labeled:
        summary["accuracy"] = sum(row["correct"] for row in labeled) / len(labeled)
    summary_path = prediction_path.with_suffix(".metrics.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    if args.submission:
        submission = WORKSPACE / "outputs" / "submissions" / f"submission_{args.name}.csv"
        write_submission(outputs, submission)
        print(f"Saved {submission}", flush=True)


if __name__ == "__main__":
    main()