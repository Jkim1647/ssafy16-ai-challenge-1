from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "workspace"
sys.path.insert(0, str(WORKSPACE / "src"))

from common import CHOICES, build_messages, project_root, read_csv, read_jsonl, write_jsonl  # noqa: E402


def scores(row: dict) -> np.ndarray:
    return np.asarray([float(row[f"score_{label}"]) for label in CHOICES], dtype=np.float64)


def predicted(values: np.ndarray) -> str:
    return CHOICES[int(values.argmax())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-a", type=Path, required=True)
    parser.add_argument("--base-b", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-pixels", type=int, default=448 * 448)
    parser.add_argument("--max-pixels", type=int, default=672 * 672)
    args = parser.parse_args()

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    rows = read_jsonl(args.input) if args.input.suffix.lower() == ".jsonl" else read_csv(args.input)
    by_id = {row["id"]: row for row in rows}
    a = {row["id"]: row for row in read_jsonl(args.base_a)}
    b = {row["id"]: row for row in read_jsonl(args.base_b)}
    new_rows = read_jsonl(args.new)
    n = {row["id"]: row for row in new_rows}
    if set(by_id) != set(a) or set(by_id) != set(b) or set(by_id) != set(n):
        raise ValueError("Input and prediction ID sets differ")

    disagreement_ids = []
    for item_id in by_id:
        base = (scores(a[item_id]) + scores(b[item_id])) / 2
        if predicted(base) != predicted(scores(n[item_id])):
            disagreement_ids.append(item_id)
    print(f"Choice-rotation subset: {len(disagreement_ids)} disagreements", flush=True)

    model_id = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    processor = AutoProcessor.from_pretrained(
        args.adapter, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    token_ids = torch.tensor(
        [processor.tokenizer.encode(label, add_special_tokens=False)[0] for label in CHOICES],
        device=model.device,
    )

    output_by_id = {row["id"]: dict(row) for row in new_rows}
    for item_id in tqdm(disagreement_ids, desc="choice-rotation TTA"):
        row = by_id[item_id]
        image_path = Path(row["path"])
        if not image_path.is_absolute():
            image_path = project_root() / image_path
        image = Image.open(image_path).convert("RGB")
        log_sum = np.log(np.clip(scores(n[item_id]), 1e-12, 1.0))
        for rotation in (1, 2, 3):
            rotated = dict(row)
            for new_index, label_name in enumerate(CHOICES):
                rotated[label_name] = row[CHOICES[(new_index + rotation) % 4]]
            conversation = build_messages(rotated, image, language="ko")
            inputs = processor.apply_chat_template(
                [conversation], tokenize=True, add_generation_prompt=True, padding=True,
                return_dict=True, return_tensors="pt", enable_thinking=False,
            )
            inputs.pop("token_type_ids", None)
            inputs = inputs.to(model.device)
            with torch.inference_mode():
                logits = model(**inputs, use_cache=False).logits[:, -1, :]
                rotated_prob = logits.index_select(-1, token_ids).float().softmax(dim=-1)[0].cpu().numpy()
            mapped = np.asarray([rotated_prob[(index - rotation) % 4] for index in range(4)])
            log_sum += np.log(np.clip(mapped, 1e-12, 1.0))
        geometric = np.exp(log_sum / 4)
        geometric /= geometric.sum()
        record = output_by_id[item_id]
        record["predicted_label"] = predicted(geometric)
        record["confidence"] = float(geometric.max())
        for label_name, value in zip(CHOICES, geometric.tolist()):
            record[f"score_{label_name}"] = value
        if record.get("true_label") in CHOICES:
            record["correct"] = record["predicted_label"] == record["true_label"]

    output = [output_by_id[row["id"]] for row in new_rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, output)
    labeled = [row for row in output if row.get("true_label") in CHOICES]
    summary = {"samples": len(output), "tta_subset": len(disagreement_ids)}
    if labeled:
        summary["accuracy"] = sum(row["correct"] for row in labeled) / len(labeled)
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()