from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from common import CHOICES, build_messages, project_root, read_jsonl


class JsonlVQADataset:
    def __init__(self, manifest: str | Path) -> None:
        self.rows = read_jsonl(manifest)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        image_path = Path(row["path"])
        row["path"] = str(image_path if image_path.is_absolute() else project_root() / image_path)
        return row


def choice_token_ids(tokenizer: Any) -> dict[str, int]:
    result = {}
    for label in CHOICES:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise RuntimeError(
                f"Expected {label!r} to be one token, got {token_ids}. "
                "Run tokenizer inspection and adjust the answer surface before training."
            )
        result[label] = token_ids[0]
    if len(set(result.values())) != 4:
        raise RuntimeError(f"Choice token IDs are not unique: {result}")
    return result


@dataclass
class AnswerOnlyCollator:
    processor: Any
    prompt_language: str = "ko"
    enable_thinking: bool | None = None

    def __post_init__(self) -> None:
        self.label_token_ids = choice_token_ids(self.processor.tokenizer)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        texts, images, gold_ids = [], [], []
        for row in features:
            image = Image.open(row["path"]).convert("RGB")
            answer = str(row["answer"]).strip().lower()
            if answer not in CHOICES:
                raise ValueError(f"Invalid answer {answer!r} for {row.get('id')}")
            messages = build_messages(row, image, answer, self.prompt_language)
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": False,
            }
            if self.enable_thinking is not None:
                template_kwargs["enable_thinking"] = self.enable_thinking
            texts.append(self.processor.apply_chat_template(messages, **template_kwargs))
            images.append(image)
            gold_ids.append(self.label_token_ids[answer])

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        labels = torch.full_like(batch["input_ids"], -100)
        for index, gold_id in enumerate(gold_ids):
            positions = (batch["input_ids"][index] == gold_id).nonzero(as_tuple=False).flatten()
            if not len(positions):
                raise RuntimeError(f"Gold token {gold_id} missing from encoded sample")
            # The assistant answer is the final occurrence; choice letters in the prompt stay masked.
            labels[index, int(positions[-1])] = gold_id
        batch["labels"] = labels
        batch.pop("token_type_ids", None)
        return batch


def make_preprocess_logits(choice_ids: dict[str, int]):
    import torch

    ordered_ids = torch.tensor([choice_ids[label] for label in CHOICES], dtype=torch.long)

    def preprocess_logits_for_metrics(logits: Any, labels: Any) -> Any:
        if isinstance(logits, tuple):
            logits = logits[0]
        device_ids = ordered_ids.to(logits.device)
        answer_positions = (labels != -100).to(torch.int64).argmax(dim=1)
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        prediction_positions = (answer_positions - 1).clamp_min(0)
        return logits[batch_indices, prediction_positions][:, device_ids]

    return preprocess_logits_for_metrics


def make_compute_metrics(choice_ids: dict[str, int]):
    import numpy as np

    id_to_index = {token_id: index for index, token_id in enumerate(choice_ids.values())}

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        predictions = eval_prediction.predictions
        labels = eval_prediction.label_ids
        predicted = np.asarray(predictions).argmax(axis=-1)
        true_token_ids = []
        for row in labels:
            active = row[row != -100]
            if len(active) != 1:
                raise RuntimeError(f"Expected one supervised token, got {len(active)}")
            true_token_ids.append(int(active[0]))
        truth = np.asarray([id_to_index[token_id] for token_id in true_token_ids])
        return {"accuracy": float((predicted == truth).mean())}

    return compute_metrics