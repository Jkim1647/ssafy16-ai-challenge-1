from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from common import CHOICES, build_messages, project_root, read_csv, read_jsonl, workspace_root, write_jsonl
from training_core import choice_token_ids


def load_rows(input_path: Path) -> list[dict[str, Any]]:
    if input_path.suffix.lower() == ".jsonl":
        return read_jsonl(input_path)
    return read_csv(input_path)


def extract_choice(text: str) -> str | None:
    match = re.search(r"(?<![a-z])[abcd](?![a-z])", text.strip().lower())
    return match.group(0) if match else None


def load_model(adapter_dir: Path, min_pixels: int, max_pixels: int):
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Qwen3-VL-4B inference")
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gib < 7:
        raise RuntimeError(f"Only {total_gib:.1f} GiB VRAM detected; use Kaggle/Colab for inference")
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    processor = AutoProcessor.from_pretrained(
        adapter_dir, min_pixels=min_pixels, max_pixels=max_pixels
    )
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
        dtype=dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, processor, torch


def predict(
    rows: list[dict[str, Any]],
    model: Any,
    processor: Any,
    torch: Any,
    method: str,
    prompt_language: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    from tqdm.auto import tqdm

    token_ids = choice_token_ids(processor.tokenizer)
    ordered_token_ids = torch.tensor(
        [token_ids[label] for label in CHOICES], device=model.device, dtype=torch.long
    )
    processor.tokenizer.padding_side = "left"
    outputs = []
    for start in tqdm(range(0, len(rows), batch_size), desc=f"Inference ({method})"):
        batch_rows = rows[start : start + batch_size]
        image_paths = []
        conversations = []
        for row in batch_rows:
            image_path = Path(row["path"])
            if not image_path.is_absolute():
                image_path = project_root() / image_path
            image_paths.append(image_path)
            conversations.append(build_messages(row, Image.open(image_path).convert("RGB"), language=prompt_language))
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            next_logits = model(**inputs, use_cache=False).logits[:, -1, :]
            choice_logits = next_logits.index_select(-1, ordered_token_ids).float()
            probabilities_batch = choice_logits.softmax(dim=-1).cpu().tolist()
            logit_predictions = [CHOICES[index] for index in choice_logits.argmax(dim=-1).cpu().tolist()]
        for row, image_path, probabilities, prediction in zip(
            batch_rows, image_paths, probabilities_batch, logit_predictions
        ):
            record = {
                "id": row["id"], "image": str(image_path), "question": row["question"],
                "a": row["a"], "b": row["b"], "c": row["c"], "d": row["d"],
                "true_label": row.get("answer"), "predicted_label": prediction,
                "confidence": max(probabilities),
                **{f"score_{label}": score for label, score in zip(CHOICES, probabilities)},
                "generated_text": None,
            }
            if record["true_label"]:
                record["correct"] = record["true_label"] == prediction
            outputs.append(record)
    return outputs


def write_submission(predictions: list[dict[str, Any]], output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing submission: {output}")
    expected = read_csv("sample_submission.csv")
    expected_ids = [row["id"] for row in expected]
    prediction_by_id = {row["id"]: row["predicted_label"] for row in predictions}
    if set(prediction_by_id) != set(expected_ids):
        missing = set(expected_ids) - set(prediction_by_id)
        extra = set(prediction_by_id) - set(expected_ids)
        raise ValueError(f"Submission ID mismatch: missing={len(missing)}, extra={len(extra)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "answer"])
        writer.writeheader()
        writer.writerows({"id": item_id, "answer": prediction_by_id[item_id]} for item_id in expected_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL adapter inference with direct choice logits")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--name", required=True, help="Unique prediction/submission stem")
    parser.add_argument("--method", choices=["logits", "generate"], default="logits")
    parser.add_argument("--prompt-language", choices=["ko", "en"], default="ko")
    parser.add_argument("--min-pixels", type=int, default=448 * 448)
    parser.add_argument("--max-pixels", type=int, default=672 * 672)
    parser.add_argument("--submission", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    rows = load_rows(args.input.resolve())
    rows = rows[args.shard_index :: args.num_shards]
    model, processor, torch = load_model(args.adapter.resolve(), args.min_pixels, args.max_pixels)
    predictions = predict(rows, model, processor, torch, args.method, args.prompt_language, args.batch_size)
    prediction_path = workspace_root() / "outputs" / "predictions" / f"{args.name}.jsonl"
    if prediction_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing predictions: {prediction_path}")
    write_jsonl(prediction_path, predictions)
    print(f"Saved predictions: {prediction_path}")
    if args.submission:
        submission_path = workspace_root() / "outputs" / "submissions" / f"submission_{args.name}.csv"
        write_submission(predictions, submission_path, args.overwrite)
        print(f"Saved submission: {submission_path}")


if __name__ == "__main__":
    main()