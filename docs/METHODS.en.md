# Methods summary

## Objective

Predict exactly one answer label from an image, a Korean question, and four choices. Because the metric is accuracy, the system directly scores the four choices instead of relying on free-form generation.

## Training

- Qwen3-VL-4B/8B and Qwen3.5-35B-A3B
- NF4 4-bit QLoRA
- answer-token-only supervision
- image-hash-grouped validation
- checkpoint selection by validation accuracy rather than training loss alone

## Inference

1. Compute next-token logits at the end of the chat template.
2. Select only the logits for `a`, `b`, `c`, and `d`.
3. Store four choice probabilities for evaluation and ensembling.
4. For rotated choices, map probabilities back to the original positions.
5. Compute a weighted average across models and choose the maximum.

## Validation principles

- Keep identical and near-identical images within the same fold.
- Inspect fold accuracy, disagreements, and rescue-versus-harm counts.
- Preserve the current best candidate because dev improvements may fail publicly.
- Count prediction changes before spending a leaderboard submission.
- Treat public-score-guided tuning as a possible source of overfitting.

## Best observed combination

An ensemble with roughly 48% Qwen3-VL-family probabilities and 52% Qwen3.5 rotation-TTA probabilities, followed by an exact-match correction, reached Public LB 0.94324.

## Next priorities

1. pHash-grouped 3-fold out-of-fold evaluation
2. question-type reliability routing
3. short hard-sample fine-tuning
4. models with lower error correlation rather than more resolutions of one model
5. reserve at least five submissions for final tuning