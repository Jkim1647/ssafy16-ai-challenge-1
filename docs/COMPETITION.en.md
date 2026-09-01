# Competition overview

This repository documents the experiments conducted for the **SSAFY 16th Cohort AI Challenge** on Kaggle. The task was to build a Visual Question Answering model that jointly understands a recycling-related image and a Korean question, then selects one of four choices.

- Official page: [SSAFY 16th Cohort AI Challenge](https://www.kaggle.com/competitions/ssafy-16-1-ai)
- Official description: [Overview](https://www.kaggle.com/competitions/ssafy-16-1-ai/overview)
- Dataset description: [Data](https://www.kaggle.com/competitions/ssafy-16-1-ai/data)
- Competition rules: [Rules](https://www.kaggle.com/competitions/ssafy-16-1-ai/rules)

> This page is a concise interpretation of the competition pages. The official Kaggle rules take precedence if any wording differs.

## Task definition

Each sample contains:

1. an image containing recyclable objects,
2. a Korean question about the image, and
3. four candidate answers labeled `a`, `b`, `c`, and `d`.

The expected output is a single answer label. This is therefore a **four-choice classification-style VQA task**, rather than open-ended text generation.

```text
Recycling image + Korean question + choices a/b/c/d
                              │
                              ▼
                       Multimodal VQA model
                              │
                              ▼
                        answer ∈ {a,b,c,d}
```

## Dataset structure

The competition page described approximately **1.87 GB across 14,567 files**. The original competition data is private and is not included in this repository.

| Item | Contents |
| --- | --- |
| `train.csv` + `train/` | image path, question, choices `a`–`d`, and answer |
| `test.csv` + `test/` | image path, question, and choices `a`–`d`; answer hidden |
| `dev.csv` + `dev/` | image, question, choices, and five participant responses; not ground truth |
| `sample_submission.csv` | required `id,answer` submission schema |

The participant responses in `dev.csv` have a different status from verified labels, so this project keeps them separate from ground-truth validation.

## Evaluation and submission

- Primary metric: **Accuracy**
- Submission format: CSV
- Required columns: `id`, `answer`
- Valid answers: one of `a`, `b`, `c`, or `d`
- The Public Leaderboard evaluates part of the test set; final placement is determined by the remaining Private test set

This makes validation design important: maximizing one small development split or the Public score does not guarantee transfer to the final data.

## Rules relevant to this implementation

| Area | Official rule summary | Project response |
| --- | --- | --- |
| Participation | SSAFY 16th-cohort learners; individual, one-person teams | single-participant pipeline |
| Training data | provided training data may be used | data stayed in local/RunPod environments |
| External data | publicly available external data allowed | public pretrained models only |
| Augmentation | allowed when based on publicly available resources | choice rotation and image preprocessing |
| Models | Hugging Face pretrained VQA models allowed | Qwen3-VL and Qwen3.5 families |
| Efficiency | fine-tuning, LoRA, and quantization allowed | 4-bit QLoRA |
| Prompting | prompt engineering allowed | one-token answer format |
| APIs | inference through API calls and responses prohibited | downloaded models ran directly on GPUs |
| Leakage | prior access to or leakage from test labels prohibited | inference used no hidden labels |
| Submission | maximum 20 submissions per day | candidates validated locally before upload |

## Practical challenges

- The model must align visual evidence with Korean language.
- Choice-position bias can change predictions when the same answers move between `a` and `d`.
- Duplicate or near-duplicate images across splits can inflate validation accuracy.
- A strong Public candidate is not guaranteed to be the strongest Private candidate.
- A limited submission budget makes local schema, ID-order, and candidate-difference checks essential.

See [METHODS.en.md](METHODS.en.md) and [EXPERIMENT_LOG.en.md](EXPERIMENT_LOG.en.md) for the implementation and experiment results.

