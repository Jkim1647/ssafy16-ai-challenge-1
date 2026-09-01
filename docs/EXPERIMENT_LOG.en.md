# Experiment log

This document is an English companion to the detailed [Korean experiment log](EXPERIMENT_LOG.ko.md). It records the decisions, validation results, public scores, failed experiments, and operational lessons required to understand the final pipeline.

## 1. Task and data

The task was four-choice visual question answering over recycling-related images. Each sample contained an image, a Korean question, and choices `a` through `d`. The prediction file required two columns, `id` and `answer`.

| Split | Rows | Notes |
| --- | ---: | --- |
| train | 5,073 | ground-truth answer available |
| test | 5,074 | used for submission |
| dev | 4,413 | five participant responses, not ground truth |

Every final CSV was checked for 5,074 rows, exact columns, valid labels, and sample-submission ID order.

## 2. Local Qwen3-VL-4B baseline

The first stable environment used an RTX 5060 Ti 16GB, Python 3.12, CUDA 12.8, 4-bit NF4 QLoRA, BF16, SDPA, and 448²–672² image pixels.

The model was trained with answer-only supervision: prompt and image tokens were masked, and loss was applied only to the one-letter assistant answer.

| Experiment | Train / validation | Train loss | Validation loss | Public LB |
| --- | --- | ---: | ---: | ---: |
| smoke test | 180 / 20 | 0.2166 | 0.1790 | 0.85777 |
| full baseline | 4,565 / 508 | 0.0993 | 0.0788 | **0.92116** |

The full baseline took about 79 minutes to train locally. Its 5,074 test answers were balanced across the four labels and had no parsing failures.

## 3. Pseudo-label fine-tuning

Among the dev responses, 902 samples had a unique four-vote consensus. Because their class distribution was highly skewed, 117 samples per class were selected, producing 468 balanced pseudo-labels.

One extra epoch at learning rate `1e-5` reduced validation loss from 0.0788 to 0.0736, but Public LB fell from 0.92116 to 0.91919. Lower loss did not imply better test accuracy, and participant consensus was not equivalent to ground truth.

## 4. Early choice-rotation implementation failure

An initial batched choice-rotation experiment used right padding while reading the final logit position. For shorter samples, `logits[:, -1, :]` pointed to padding rather than the true final token. Original and rotated predictions agreed on only 36.4%, and more than 1,856 test answers changed.

The candidate was discarded. The corrected implementation uses left padding or the actual final non-padding position and always maps rotated probabilities back to the original choice order.

## 5. Moving to A100 GPUs

The local 16GB GPU was sufficient for the 4B baseline but too slow for repeated full-data training and multiple 5,074-image inference passes. Experiments therefore moved to RunPod A100 instances.

The first A100 Qwen3-VL-8B baseline scored 0.92353. A separate Kaggle T4×2 Qwen3-VL-8B Stage 2 candidate scored 0.91761, showing that model size alone was not enough.

## 6. Qwen3.5-35B-A3B QLoRA

To meet the deadline, Qwen3.5-35B-A3B was trained with a 320-step 4-bit QLoRA schedule rather than the longer 571-step plan.

| Setting | Value |
| --- | --- |
| model | Qwen3.5-35B-A3B |
| quantization | NF4 4-bit |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| max steps | 320 |
| batch / accumulation | 1 / 8 |
| learning rate | `1e-4` |
| image pixels | 448²–672² |
| validation | 508 grouped samples |

Test inference stored the complete four-choice probability vector for every sample. This made later ensemble searches possible without rerunning the model.

## 7. Router versus soft ensemble

The base Qwen3-VL ensemble had dev accuracy 0.93307. Qwen3.5 and the base models disagreed on 33 dev samples and 394 test samples.

| Selective replacements | Dev accuracy | Rescue | Harm |
| ---: | ---: | ---: | ---: |
| 10 | 0.93701 | 6 | 4 |
| 15 | 0.93898 | 9 | 6 |
| 20 | 0.94094 | 12 | 8 |
| 30 | 0.93504 | 14 | 13 |
| 40 | 0.93504 | 15 | 14 |

Although several top-k rules improved dev, the submitted margin-top-15 candidate scored only 0.92668. Confidence and margin were not reliable enough to decide which model to trust on the test distribution.

The soft probability ensemble was far stronger. Qwen3.5 choice-rotation probabilities mixed 50:50 with the Qwen3-VL base reached **0.94245**.

## 8. Exact-match correction

Provided training samples were indexed by exact image hash and normalized question text. A correction was applied only when the answer text mapped unambiguously to one test choice and the competition rules permitted it. One verified final correction changed `test_0458.jpg` from `b` to `a`.

## 9. A100×2 high-resolution experiments

Two Qwen3.5 choice-rotation test jobs were run in parallel:

| Variant | Pixel range | Standalone dev | Best ensemble dev |
| --- | --- | ---: | ---: |
| hr896 | 672²–896² | 0.93898 | 0.94685 |
| hr1024 | 768²–1024² | 0.93898 | 0.94685 |

Each full test job completed 394 batches in about 17 minutes 36 seconds. Forty-eight direct high-resolution candidates were generated and validated.

A four-source multi-resolution grid searched 990 combinations and reached dev 0.94882. A calibrated temperature/probability grid searched 2,448 combinations and reached the same best dev score.

The Public Leaderboard contradicted the dev ranking:

| Candidate | Public LB |
| --- | ---: |
| standard rotation soft w0.50 | **0.94245** |
| hr896 soft w0.50 | 0.94048 |
| hr1024 soft w0.50 | 0.94087 |
| multi-resolution rank 1 | 0.94126 |
| calibrated high-resolution rank 1 | 0.94126 |

Global high resolution increased visual tokens and noise, while the large grid overfit the 508-sample development set. The standard-resolution anchor was therefore restored.

## 10. Conservative public-anchor rescue

The 0.94245 candidate was frozen as an anchor. High-resolution models were allowed to replace only a few answers when they strongly agreed.

| Candidate | Changed test rows | Dev accuracy | Public LB |
| --- | ---: | ---: | ---: |
| balanced top 3 | 3 | 0.94685 | 0.94245 |
| balanced top 5 | 5 | 0.94882 | 0.94205 |
| balanced top 8 | 8 | 0.94685 | 0.94205 |
| minimum-margin top 5 | 5 | 0.94882 | **0.94284** |

Even rules with identical dev accuracy produced different Public scores, underlining the uncertainty of a small validation set.

## 11. Fine ensemble-weight search

The final search stayed close to the proven standard candidate. The table reports the Qwen3.5 weight; the remaining mass belonged to the Qwen3-VL base.

| Qwen3.5 weight | Dev accuracy | Changes vs w0.50 | Public LB |
| ---: | ---: | ---: | ---: |
| 0.50 | 0.94488 | 0 | 0.94245 |
| 0.51 | 0.94488 | 5 | 0.94284 |
| 0.52 | 0.94488 | 10 | **0.94324** |
| 0.53 | **0.94685** | 12 | not submitted |
| 0.54 | 0.94488 | 20 | not submitted |

Weight 0.53 was not submitted because the daily limit of 20 submissions had already been exhausted. The final selected files were the w0.52 global ensemble and the more conservative minimum-margin top-5 candidate.

## 12. Final outcome

The best Public score was **0.94324**, an absolute gain of 0.02208 over the local 0.92116 baseline and 0.01971 over the A100 Qwen3-VL-8B baseline.

The main gain did not come from simply increasing training steps. It came from a larger but short-tuned model, correct choice-rotation TTA, saved choice probabilities, and conservative probability ensembling. Conversely, high resolution and complex routers looked stronger on dev but did not transfer publicly.

## 13. Lessons for the next competition

1. Save full choice probabilities, not only final labels.
2. Use pHash/SHA-grouped multi-fold out-of-fold validation.
3. Track rescue, harm, disagreement, and the minimum fold score.
4. Prefer models with diverse errors over many resolutions of one model.
5. Analyze question types such as counting, material, and color separately.
6. Reserve a fixed submission budget for final tuning.
7. Preserve the current best candidate before every risky experiment.

## 14. Repository status

The public repository contains reusable code, configuration files, and documentation only. Competition data, predictions, submissions, adapters, checkpoints, and cloud identifiers are excluded.