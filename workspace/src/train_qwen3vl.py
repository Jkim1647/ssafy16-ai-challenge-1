from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

from common import load_yaml, workspace_root, write_json
from training_core import (
    AnswerOnlyCollator,
    JsonlVQADataset,
    choice_token_ids,
    make_compute_metrics,
    make_preprocess_logits,
)


def check_training_environment(torch_module) -> None:
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Do not run Qwen3-VL-4B QLoRA on CPU.")
    properties = torch_module.cuda.get_device_properties(0)
    total_gib = properties.total_memory / 1024**3
    print(f"GPU: {properties.name} ({total_gib:.1f} GiB)")
    if total_gib < 14:
        raise RuntimeError(
            f"Detected only {total_gib:.1f} GiB VRAM. This pipeline requires about 16 GiB "
            "for batch=1 at the baseline resolution. Use Kaggle/Colab or a larger GPU."
        )


def train(config_path: Path, resume_from_checkpoint: str | None = None) -> Path:
    import numpy as np
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        AutoModelForImageTextToText,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        Qwen3VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    config = load_yaml(config_path)
    check_training_environment(torch)
    seed = int(config["seed"])
    set_seed(seed)
    np.random.seed(seed)

    bf16 = bool(torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"Compute dtype: {compute_dtype}; bf16_supported={bf16}")
    model_id = config["model_id"]
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=int(config["image"]["min_pixels"]),
        max_pixels=int(config["image"]["max_pixels"]),
    )
    token_report = {
        surface: processor.tokenizer.encode(surface, add_special_tokens=False)
        for surface in ["a", "b", "c", "d", " a", " b", " c", " d", "A", "B", "C", "D"]
    }
    print("Choice tokenization:", json.dumps(token_report, ensure_ascii=False))

    device_index = int(os.environ.get("LOCAL_RANK", "0"))
    model_class = (
        AutoModelForImageTextToText
        if config.get("model_family") == "qwen35"
        else Qwen3VLForConditionalGeneration
    )
    model = model_class.from_pretrained(
        model_id,
        quantization_config=quantization,
        dtype=compute_dtype,
        device_map={"": device_index},
        attn_implementation=config.get("attention", "sdpa"),
    )
    model.config.use_cache = False
    if config.get("model_family") == "qwen35":
        # PEFT's generic helper casts every non-quantized parameter to fp32.
        # On Qwen3.5-35B-A3B this needlessly consumes almost all of an 80 GiB
        # GPU before adapters are even attached.  Freeze the quantized base
        # directly; get_peft_model below will enable gradients for LoRA only.
        for parameter in model.parameters():
            parameter.requires_grad = False
    else:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    lora = config["lora"]
    init_adapter = config.get("init_adapter")
    if init_adapter:
        print(f"Continuing trainable adapter: {init_adapter}")
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(lora["r"]),
                lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                target_modules=list(lora["target_modules"]),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    model.print_trainable_parameters()

    train_dataset = JsonlVQADataset(config["data"]["train_manifest"])
    val_dataset = JsonlVQADataset(config["data"]["val_manifest"])
    collator = AnswerOnlyCollator(
        processor,
        config["prompt_language"],
        enable_thinking=config.get("enable_thinking"),
    )
    choice_ids = choice_token_ids(processor.tokenizer)

    output_dir = workspace_root() / "outputs" / "models" / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = config["training"]
    eval_enabled = str(train_cfg.get("eval_strategy", "epoch")).lower() != "no"
    max_steps = int(train_cfg.get("max_steps", -1))
    update_steps_per_epoch = math.ceil(
        len(train_dataset)
        / (int(train_cfg["batch_size"]) * int(train_cfg["gradient_accumulation"]))
    )
    planned_update_steps = (
        max_steps
        if max_steps > 0
        else math.ceil(update_steps_per_epoch * float(train_cfg["epochs"]))
    )
    warmup_steps = int(
        train_cfg.get(
            "warmup_steps",
            round(planned_update_steps * float(train_cfg.get("warmup_ratio", 0.0))),
        )
    )
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg["epochs"]),
        per_device_train_batch_size=int(train_cfg["batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        max_steps=max_steps,
        eval_strategy="epoch" if eval_enabled else "no",
        save_strategy="epoch" if eval_enabled else "no",
        logging_strategy="steps",
        logging_steps=int(train_cfg.get("logging_steps", 25)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        load_best_model_at_end=eval_enabled,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 2)),
        seed=seed,
        data_seed=seed,
        ddp_find_unused_parameters=False,
    )
    callbacks = []
    if float(train_cfg["epochs"]) > 1:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=1))
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=make_compute_metrics(choice_ids) if eval_enabled else None,
        preprocess_logits_for_metrics=make_preprocess_logits(choice_ids) if eval_enabled else None,
        callbacks=callbacks,
    )
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    best_dir = output_dir / "best_adapter"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(best_dir)
    trainer.save_state()
    shutil.copy2(config_path, output_dir / "experiment_config.yaml")
    write_json(output_dir / "tokenizer_choice_tokens.json", token_report)
    write_json(output_dir / "train_metrics.json", result.metrics)
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Saved adapter and processor: {best_dir}")
    return best_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen3-VL-4B with 4-bit QLoRA")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    train(args.config.resolve(), args.resume_from_checkpoint)


if __name__ == "__main__":
    main()