# SSAFY 16기 AI 챌린지 진행 정리

## 1. 과제 개요

재활용품 이미지와 질문, 네 개의 선택지를 입력받아 `a`, `b`, `c`, `d` 중 하나를 출력하는 VQA 모델을 만들었다. 평가 지표는 Accuracy이며, 학습 데이터 외에 `dev.csv`의 응답과 외부 사전학습 VQA 모델도 사용할 수 있었다.

데이터 규모는 다음과 같다.

- `train.csv`: 5,073개
- `test.csv`: 5,074개
- `dev.csv`: 각 문항에 교육생 응답 5개 포함
- 제출 형식: `id`, `answer` 두 열을 가진 CSV

## 2. 로컬 개발 환경

처음 만들어진 `.venv`는 Python 3.14여서 CUDA PyTorch 및 일부 학습 라이브러리와 호환 문제가 있었다. 별도의 `baseline` 가상환경을 만들고 VS Code와 Jupyter에서 같은 인터프리터를 사용하도록 통일했다.

| 항목 | 사용 구성 |
| --- | --- |
| GPU | NVIDIA RTX 5060 Ti 16GB |
| GPU 아키텍처 | Blackwell (`sm_120`) |
| Python | 3.12.10 (`baseline` 가상환경) |
| PyTorch | 2.7.0+cu128 |
| CUDA runtime | 12.8 |
| Transformers | 4.57.6 |
| 모델 | `Qwen/Qwen3-VL-4B-Instruct` |
| 학습 방식 | 4-bit NF4 QLoRA, BF16 |
| Attention | PyTorch SDPA |
| Optimizer | AdamW |
| Scheduler | Linear scheduler + 3% warmup |

RTX 5060 Ti는 비교적 새로운 GPU라 초기에는 CUDA 12.8을 지원하는 PyTorch 설치가 가장 오래 걸렸다. 모델을 `device_map="auto"`로 불러온 뒤에는 `model.to("cuda")`를 다시 호출하지 않았다.

## 3. 모델과 학습 설정

기본 모델은 Qwen3-VL-4B-Instruct를 사용했다. 이미지 속 작은 병이나 캔, 재질을 구분하는 문제가 있어 입력 해상도를 지나치게 낮추지 않았다.

| 파라미터 | 값 |
| --- | --- |
| 최소 이미지 픽셀 | `448 × 448` |
| 최대 이미지 픽셀 | `672 × 672` |
| 양자화 | 4-bit NF4, double quantization |
| 연산 dtype | BF16 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA 대상 | `q/k/v/o_proj`, `gate/up/down_proj` |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Epoch | 1 |
| Learning rate | `1e-4` |
| Gradient clipping | 1.0 |
| Seed | 42 |

프롬프트는 모델이 설명을 길게 생성하지 않도록 아래처럼 단순하게 구성했다.

> 이미지와 질문, 네 개의 선택지를 확인하세요. 정답은 반드시 a, b, c, d 중 하나의 소문자 한 글자만 출력하세요. 설명이나 다른 문자는 출력하지 마세요.

학습 시에는 `input_ids` 전체를 label로 사용하지 않았다. 이미지·질문·선택지·프롬프트 구간을 `-100`으로 마스킹하고 assistant가 출력하는 정답 한 글자만 loss에 반영했다. 이 부분을 적용하지 않으면 모델이 질문과 프롬프트까지 생성하도록 학습될 수 있다.

## 4. 실험 결과

### 4.1 소규모 200개 실험

전체 학습 전에 파이프라인이 정상적으로 동작하는지 확인하기 위해 200개만 사용했다.

- 학습 180개 / 검증 20개
- 학습 시간: 약 3분 4초
- Train loss: 0.2166
- Validation loss: 0.1790
- Public Leaderboard: **0.857770**

점수 자체는 낮았지만, QLoRA 학습부터 전체 추론과 제출 파일 생성까지 정상적으로 연결되는 것을 확인했다.

### 4.2 전체 train 학습

다음 실험에서는 `train.csv` 전체를 seed 42로 섞은 뒤 90:10으로 나눴다.

- 학습 4,565개 / 검증 508개
- 1 epoch 학습 시간: 1시간 19분 21초
- 검증 시간: 3분 23초
- Train loss: 0.0993
- Validation loss: 0.0788
- Public Leaderboard: **0.92116**

전체 test 5,074개 추론 결과의 답 분포는 아래와 같았다.

| 답 | 개수 |
| --- | ---: |
| a | 1,246 |
| b | 1,228 |
| c | 1,328 |
| d | 1,272 |

네 답의 분포가 크게 치우치지 않았고, 생성 결과를 파싱하지 못한 행도 없었다. 로컬에서 얻은 가장 안정적인 기준 모델은 `artifacts/qwen3vl4b_full`이며, 제출 파일은 `submission_qwen3vl4b_full.csv`이다.

### 4.3 dev.csv pseudo-label 추가 학습

`dev.csv`의 5개 응답 중 같은 답이 4개 이상인 902개 문항을 고신뢰 pseudo-label로 선택했다. 원본 분포는 `d`가 413개로 많이 치우쳐 있었기 때문에, 각 답에서 117개씩 뽑아 총 468개로 균형을 맞췄다.

- 추가 학습 데이터: 468개 (`a/b/c/d` 각 117개)
- Learning rate: `1e-5`
- Epoch: 1
- 추가 학습 시간: 8분 8초
- Validation loss: 0.0736
- Public Leaderboard: **0.91919**

Validation loss는 0.0788에서 0.0736으로 내려갔지만 실제 리더보드 점수는 0.92116에서 0.91919로 낮아졌다. 교육생 다수결 응답은 정답 label이 아니며, 균형 표집 과정에서 실제 데이터 분포도 달라질 수 있다. 이 실험을 통해 loss 감소만으로 test Accuracy 향상을 판단하면 안 된다는 점을 확인했다.

따라서 `qwen3vl4b_full_dev4`는 최종 모델로 사용하지 않았다.

### 4.4 선택지 순서 변경 앙상블 시도

선택지 위치 편향을 줄이기 위해 원래 순서와 순환 이동한 순서를 함께 추론하고 확률을 평균하는 방식을 시도했다. 첫 실행에서는 batch에 오른쪽 padding을 적용한 상태에서 마지막 위치의 logits를 읽는 문제가 발견됐다. 길이가 짧은 샘플은 실제 마지막 토큰이 아니라 padding 위치를 참조하게 되어 결과가 크게 왜곡됐다.

- 원본/회전 예측 일치율: 36.4%
- 기존 0.92116 제출 대비 변경: 1,856개 이상
- 답 분포도 크게 변화

이 결과로 생성된 아래 파일은 **제출용으로 사용하지 않는다**.

- `submission_qwen3vl4b_full_choice_ensemble.csv`
- `submission_qwen3vl4b_full_choice_conservative.csv`

배치 next-token scoring에서는 tokenizer의 padding을 왼쪽으로 두거나, 각 샘플의 `attention_mask`를 이용해 실제 마지막 토큰 위치를 골라야 한다. 이 실험은 아이디어보다 구현 검증이 먼저라는 점을 보여준 사례로 남겼다.

## 5. RunPod로 옮긴 이유와 주말 추가 작업

집에서 사용하는 노트북은 Qwen3-VL-4B 전체 학습과 5,074개 이미지 반복 추론을 여러 번 수행하기에 GPU 메모리와 처리 속도가 부족했다. 그래서 주말에는 RunPod GPU 인스턴스로 환경을 옮겨 로컬에서 검증한 학습·추론 파이프라인을 바탕으로 추가 실험을 진행했다.

로컬 RTX 5060 Ti에서는 한 번의 전체 학습에 약 80분, 전체 test 추론에 약 45분이 필요했다. RunPod를 사용하면서 더 큰 GPU 메모리, 더 큰 batch 또는 더 빠른 반복 실험이 가능해졌고, 로컬 최고 점수인 0.92116 이후의 개선 작업은 RunPod에서 이어갔다.

### RunPod 추가 실험 기록

RunPod에서는 NVIDIA A100으로 실험을 이어갔다. 먼저 Qwen3-VL 8B를 추론해 0.92353을 얻었고, 이후 Qwen3.5-35B-A3B를 4-bit QLoRA로 320 step 학습했다. 선택지 순서를 회전하는 choice rotation TTA를 올바르게 구현하고, 모델이 출력한 `a~d` 확률을 기존 결과와 soft ensemble하면서 점수가 크게 올랐다.

| 항목 | 기록 |
| --- | --- |
| GPU | NVIDIA A100 (VRAM 용량은 Discussion에 미기록) |
| 첫 RunPod 모델 | Qwen3-VL 8B |
| 최종 계열 모델 | Qwen3.5-35B-A3B |
| 학습 | 4-bit QLoRA, 320 step |
| 추론 개선 | choice rotation TTA + soft ensemble |
| 별도 해상도 실험 | 896px / 1024px TTA |
| 최종 앙상블 | 기존 최고 결과 유지 + ensemble weight `0.50 → 0.52` |
| 최종 Public Leaderboard | **0.94324** |
| 최종 순위 | **12위** |

RunPod에서의 점수 변화는 다음과 같다.

| 실험 | 핵심 변경 | Public LB |
| --- | --- | ---: |
| Qwen3-VL 8B | A100 추론 | 0.92353 |
| Qwen3.5-35B-A3B | 4-bit QLoRA 320 step + rotation TTA + soft ensemble | 0.94245 |
| 고해상도 TTA | 896px | 0.94048 |
| 고해상도 TTA | 1024px | 0.94087 |
| 최종 보정 | 0.94245 결과 고정 + ensemble weight 0.52 | **0.94324** |

고해상도 896/1024px 실험은 dev 결과가 좋아졌지만 Public에서는 0.94245보다 낮았다. 전체 이미지를 무조건 크게 처리하는 것이 항상 유리하지는 않았다. 이후 가장 좋았던 0.94245 결과를 기준으로 고정하고 앙상블 비중만 소폭 조정했으며, weight 0.52에서 최종 0.94324를 얻었다.

이 과정에서 최종 label만 저장하지 않고 각 문항의 `a~d` 확률을 함께 저장한 것이 도움이 됐다. 모델을 다시 추론하지 않고도 soft ensemble weight를 바꾸어 제출 파일을 만들 수 있었기 때문이다. 또한 큰 실험마다 기존 최고 제출과 결과를 따로 비교하고, 마지막 제출 횟수 일부를 weight 미세조정용으로 남겨두는 것이 중요했다.

RunPod 실험의 세부 learning rate, batch size, 정확한 A100 VRAM 및 실행 시간은 Kaggle Discussion에 기록되지 않아 이 문서에서도 추정하지 않았다.

## 6. 점수 변화 요약

| 실험 | 데이터 | 주요 변경 | Public LB | 판단 |
| --- | --- | --- | ---: | --- |
| 200개 파이프라인 확인 | train 180 / valid 20 | Qwen3-VL-4B QLoRA | 0.857770 | 동작 확인용 |
| 전체 train 학습 | train 4,565 / valid 508 | 1 epoch, LR 1e-4 | **0.92116** | 로컬 최고 기준 모델 |
| dev4 추가 학습 | dev 균형 표집 468개 추가 | 1 epoch, LR 1e-5 | 0.91919 | 점수 하락, 미사용 |
| RunPod 8B 기준 | 기존 데이터 | Qwen3-VL 8B, A100 추론 | 0.92353 | RunPod 기준점 |
| RunPod 35B-A3B | 기존 데이터 | QLoRA 320 step + rotation TTA + soft ensemble | 0.94245 | 큰 폭 상승 |
| 고해상도 TTA | 기존 데이터 | 896 / 1024px | 0.94048 / 0.94087 | dev 개선, Public 하락 |
| 최종 앙상블 | 기존 확률 결과 | weight 0.50 → 0.52 | **0.94324** | 최종 결과, 12위 |

## 7. 이번 실험에서 얻은 점

- 16GB GPU에서도 4-bit NF4 QLoRA를 사용하면 4B급 VLM을 학습할 수 있었다.
- 작은 물체가 중요한 이미지 문제에서는 입력 해상도가 성능과 VRAM 사용량에 직접 영향을 준다.
- 정답 한 글자만 학습하도록 prompt 영역을 label에서 마스킹하는 것이 중요했다.
- Validation loss가 감소해도 Accuracy와 리더보드 점수가 반드시 오르지는 않았다.
- pseudo-label은 합의 개수뿐 아니라 label 신뢰도와 실제 분포를 함께 확인해야 한다.
- 배치 추론에서는 padding 방향과 logits를 읽는 위치를 반드시 검증해야 한다.
- 처음부터 전체 학습을 반복하기보다 작은 데이터로 파이프라인을 확인한 뒤 전체 학습으로 넘어가는 편이 안전했다.
- 최종 label뿐 아니라 `a~d` 확률을 저장하면 재추론 없이 soft ensemble을 조절할 수 있다.
- 전체 이미지 해상도를 높여 dev가 좋아져도 Public 점수는 낮아질 수 있으므로 기존 최고 결과를 보존해야 한다.
- 제출 횟수 일부는 마지막 ensemble weight 미세조정용으로 남겨두는 편이 좋다.

## 8. 주요 산출물

- 로컬 전체 학습 노트북: `(260827)_baseline_desktop5060ti(2).ipynb`
- 로컬 최고 LoRA adapter: `artifacts/qwen3vl4b_full`
- 로컬 최고 제출 파일: `submission_qwen3vl4b_full.csv`
- dev 추가 학습 adapter: `artifacts/qwen3vl4b_full_dev4`
- dev 추가 학습 제출 파일: `submission_qwen3vl4b_full_dev4.csv`
- 선택지 앙상블 실험 코드: `run_choice_ensemble_fast.py`
- Kaggle 결과 공유 글: <https://www.kaggle.com/competitions/ssafy-16-1-ai/discussion/738274>

이후 RunPod 산출물과 후보 생성 스크립트를 로컬로 내려받아 재현 기록을 보완했다. 아래 9장부터는 기존 문서를 작성한 뒤 추가로 진행한 실험을 실제 summary와 제출 결과에 맞춰 상세히 정리한 내용이다.

## 9. 후속 실험 전체 흐름

후속 작업의 목표는 단일 모델의 학습 step을 무작정 늘리는 것이 아니라, 서로 다른 모델과 추론 설정이 틀리는 문항을 보완하도록 만드는 것이었다. 실제 진행 순서는 다음과 같다.

1. Qwen3-VL 4B 로컬 기준 모델을 만든다.
2. Kaggle T4×2와 RunPod A100에서 Qwen3-VL 8B 후보를 만든다.
3. 더 큰 `Qwen3.5-35B-A3B`를 4-bit QLoRA로 320 step 학습한다.
4. 선택지 순서를 회전한 뒤 원래 선택지 위치로 확률을 복원하는 choice-rotation TTA를 적용한다.
5. 각 모델의 최종 label이 아니라 `a`, `b`, `c`, `d` 확률을 섞는 soft ensemble을 수행한다.
6. 896px, 1024px 고해상도 추론과 calibration을 추가로 검증한다.
7. Public 결과를 확인한 뒤 기존 최고 후보 주변에서 weight를 0.01 단위로 미세 탐색한다.
8. 모든 제출 CSV에 대해 5,074행, `id/answer` 열, `a~d` label, sample submission ID 순서를 검사한다.

여기서 “모델을 합친다”는 것은 모델 파라미터나 checkpoint 자체를 병합한다는 뜻이 아니다. 같은 문항에 대해 모델별로 계산한 네 선택지의 확률을 가중 평균한 뒤 가장 높은 선택지를 답으로 고르는 **확률 앙상블**이다.

```text
Qwen3-VL 계열 확률 ─┐
                     ├─ 가중 평균 ─ argmax ─ 최종 a/b/c/d
Qwen3.5 계열 확률 ──┘
```

최종적으로 가장 좋은 Public 결과를 만든 비율은 Qwen3-VL 계열 약 48%, Qwen3.5 계열 약 52%였다. 서로 다른 모델이 같은 문제를 틀릴 수도 있으므로, 앙상블 자체가 성능 향상을 보장하는 것은 아니며 dev와 Public에서 모두 비교해야 했다.

## 10. Kaggle GPU 단계와 방향 전환

### 10.1 Qwen3-VL 8B Stage 2

Qwen3-VL-8B Stage 2를 Kaggle T4×2에서 실행했다. 12시간 세션 제한 안에 학습·추론을 끝내기 위해 주기적으로 진행률과 오류를 확인하고, 결과 CSV의 형식을 검증했다. 하지만 Public 결과는 **0.91761**로 기존 4B 전체 학습 결과 0.92116보다 낮았다.

이 결과로 모델 크기만 키우면 자동으로 점수가 오르지 않는다는 점을 다시 확인했다. 프롬프트, 해상도, adapter 상태, 추론 방식과 validation 분포가 함께 맞아야 했다.

### 10.2 Stage 3과 Stage 7

이후 다음 후보를 순서대로 준비했다.

- Qwen3-VL-4B exact 448~672px + dev 고신뢰 문항 자동 선택
- 기존 Qwen3-VL-8B adapter를 dev 문항으로 짧게 보정
- 4B와 8B가 다르게 답한 421개 문항만 고해상도로 재추론
- 전체 변경, confidence 0.55/0.70/0.85 기준의 hybrid 후보 생성

Stage 7은 adapter 및 cache 경로 문제 때문에 path-fix, no-cache path-fix 버전까지 갔다. Kaggle 세션 시간과 남은 대회 시간이 빠르게 줄어들었고, 이미 RunPod A100을 사용할 수 있게 되었기 때문에 최종 개선의 중심을 RunPod로 옮겼다. 이 시점부터 Kaggle #9는 보조 작업으로만 유지하다가, 최종적으로는 RunPod 결과에 집중했다.

## 11. RunPod A100 기반 Qwen3.5-35B-A3B

### 11.1 인프라와 실행 방식

초기에는 A100 1개를 사용했고, 마감 직전 고해상도 TTA는 A100 PCIe 2개가 있는 별도 Pod에서 병렬로 실행했다.

| 항목 | 기록 |
| --- | --- |
| 초기 Pod | `ssafy-qwen3vl-a100` |
| 고해상도 Pod | `ssafy-qwen35-a100x2` |
| GPU | A100 PCIe ×2 |
| 당시 표시 가격 | 시간당 약 `$2.78` |
| Network Volume | 100GB (작업 종료 후 데이터 보존) |
| 학습 모델 | `Qwen3.5-35B-A3B` |
| 양자화/학습 | 4-bit QLoRA, 320 step |
| 검증 세트 | 기존 90:10 split의 508문항 |
| test | 5,074문항 |

마감 시간 때문에 571 step 전체 계획 대신 320 step 마감형 학습을 선택했다. 학습 loss는 mini-batch 난이도와 이미지·질문 유형이 달라질 때 일시적으로 올라갈 수 있으므로 한 step의 loss가 아니라 이동 추세와 validation accuracy를 기준으로 판단했다.

### 11.2 Qwen3.5 학습 후 기본 후보

Qwen3.5-35B-A3B를 학습한 뒤 전체 test를 추론하고 기존 Qwen3-VL 계열 확률과 결합했다. 이때 다음 후보군을 생성했다.

- Qwen3.5 단독 soft 후보
- choice-rotation을 반영한 `rotgeo` soft 후보
- 모델 간 disagreement 문항만 교체하는 router 후보
- margin 기준 top-10/15/20/30/31/32/35/40 후보
- exact retrieval 보정을 적용한 대응 후보

기본 router 분석에서 Qwen3-VL 기준 dev accuracy는 0.93307이었다. dev에서 모델 간 disagreement는 33개, test에서는 394개였다. top-k 교체는 dev에서 다음처럼 움직였다.

| 교체 수 | Dev accuracy | Rescue | Harm |
| ---: | ---: | ---: | ---: |
| 10 | 0.93701 | 6 | 4 |
| 15 | 0.93898 | 9 | 6 |
| 20 | 0.94094 | 12 | 8 |
| 30 | 0.93504 | 14 | 13 |
| 32 | 0.93307 | 14 | 14 |
| 40 | 0.93504 | 15 | 14 |

dev에서는 일부 개선이 있었지만 Public에서 `submission_qwen35_qlora320_margin_top15_retrieval.csv`는 **0.92668**에 그쳤다. confidence나 margin만으로 답을 교체하는 규칙이 test 분포에서 안정적이지 않았다는 뜻이다. 이후 margin/router 계열은 최종 추천에서 제외했다.

반면 `submission_qwen35_qlora320_rotgeo_soft_w0p50_retrieval.csv`는 **0.94245**를 기록했다. hard label 교체보다 네 선택지 확률을 연속적으로 섞는 방식이 훨씬 안정적이었다.

## 12. Choice-rotation TTA와 exact retrieval

### 12.1 Choice-rotation TTA

선택지 순서가 `a, b, c, d`일 때와 순환 이동했을 때를 각각 추론했다. 회전된 출력은 반드시 원래 선택지 위치로 되돌린 뒤 확률을 평균했다. 로컬 초기 실험에서 발생했던 right-padding/마지막 logits 오류를 수정한 뒤 적용한 버전이다.

이 방식의 목적은 특정 선택지 위치를 선호하는 편향을 줄이는 것이다. 단순히 답 문자만 투표하지 않고, 복원된 `a~d` 확률을 평균했다.

### 12.2 Exact retrieval 보정

학습 이미지와 test 이미지가 사실상 동일하거나 확실하게 매칭되는 경우에는 모델 확률보다 deterministic retrieval 결과를 우선했다. 최종 후보 전체에 동일하게 적용한 확인된 보정은 다음과 같다.

- `test_0458.jpg`: 예측 `b`를 retrieval 결과 `a`로 변경

이 보정은 확률 앙상블의 weight 탐색과 별개로 마지막 단계에서 일관되게 적용했다.

## 13. A100×2 고해상도 실험

### 13.1 실행 설정

기본 rotgeo 결과를 더 개선하기 위해 A100 2개로 두 test 추론을 병렬 실행했다.

| 이름 | 최소 해상도 | 최대 해상도 | 단독 Dev | Soft ensemble 최고 Dev |
| --- | ---: | ---: | ---: | ---: |
| `hr896` | 672×672 | 896×896 | 0.93898 | 0.94685 |
| `hr1024` | 768×768 | 1024×1024 | 0.93898 | 0.94685 |

두 작업은 각각 394/394 batch를 처리했고 약 17분 36초에 완료됐다. 완료 뒤 48개의 단일 고해상도 weight 후보를 생성하고 형식을 검증했다.

### 13.2 4-model multi-resolution ensemble

다음 네 확률 소스를 함께 탐색했다.

1. Qwen3-VL `a100_full_ko`
2. Qwen3-VL `a100_dev_ko`
3. Qwen3.5 standard rotgeo
4. Qwen3.5 high-resolution (`hr896`, `hr1024`)

multi-resolution grid 990개 조합을 탐색한 결과 최고 dev accuracy는 **0.94882**였다. 상위 후보의 fold accuracy는 약 `0.93939 / 0.96324 / 0.93966 / 0.95161`로 fold 간 차이가 있었다. raw 후보 16개와 retrieval 적용 후보를 포함해 32개를 만들었다.

이어 확률의 온도와 sharpness를 조절하는 calibrated grid 2,448개를 탐색했다. 최고 dev accuracy는 역시 **0.94882**였고, raw 20개와 retrieval 적용본을 합쳐 40개를 만들었다.

### 13.3 Public 결과

dev만 보면 high-resolution/multi-resolution 계열이 가장 유망했지만 Public 결과는 반대였다.

| 제출 파일 | Public LB |
| --- | ---: |
| `submission_qwen35_qlora320_rotgeo_soft_w0p50_retrieval.csv` | **0.94245** |
| `submission_qwen35_hr896_soft_w0p50_retrieval.csv` | 0.94048 |
| `submission_qwen35_hr1024_soft_w0p50_retrieval.csv` | 0.94087 |
| `submission_qwen35_multihr_rank01_retrieval.csv` | 0.94126 |
| `submission_qwen35_multihr_rank02_retrieval.csv` | 0.94126 |
| `submission_qwen35_calhr_rank01_retrieval.csv` | 0.94126 |

고해상도는 작은 물체를 보는 데 도움을 줄 수 있지만, 전체 이미지에 일괄 적용하면 시각 token 수와 노이즈도 함께 늘어난다. 특히 508개 dev에서 탐색한 수많은 조합 중 최고를 고르는 과정 자체가 dev에 과적합될 수 있다. 이 결과를 보고 high-resolution 계열을 계속 확대하지 않고, Public에서 검증된 standard rotgeo 후보로 되돌아갔다.

## 14. Public anchor 보수 보정

Public 0.94245 후보를 anchor로 고정한 뒤, 고해상도 모델들이 강하게 합의하는 문항만 소수 교체하는 보수적 후보를 만들었다.

- Anchor dev accuracy: 0.94488
- `balanced top5`: dev 0.94882, rescue 3, harm 1, test 5개 변경
- `min_margin top5`: dev 0.94882, rescue 3, harm 1, test 5개 변경
- `balanced top3`: dev 0.94685, test 3개 변경
- `balanced top8`: dev 0.94685, test 8개 변경

Public 결과는 다음과 같았다.

| 후보 | 변경 수 | Public LB |
| --- | ---: | ---: |
| `public_anchor_balanced_top3_retrieval` | 3 | 0.94245 |
| `public_anchor_balanced_top5_retrieval` | 5 | 0.94205 |
| `public_anchor_balanced_top8_retrieval` | 8 | 0.94205 |
| `public_anchor_min_margin_top5_retrieval` | 5 | **0.94284** |

소수 교체 중 `min_margin top5`만 anchor보다 0.00039 올랐다. 같은 5개 변경이라도 어떤 기준으로 문항을 선택했는지가 중요했고, dev에서 같은 accuracy를 보인 규칙끼리도 Public 결과가 달랐다.

## 15. Standard ensemble weight 미세 탐색

큰 구조 변경보다 Public에서 안정적이었던 standard rotgeo 조합 주변을 0.01 단위로 탐색했다. 아래 weight는 Qwen3.5 쪽 비중이며 나머지는 Qwen3-VL 계열 비중이다.

| Qwen3.5 weight | Dev accuracy | w0.50 대비 test 변경 | Public LB |
| ---: | ---: | ---: | ---: |
| 0.46 | 0.94094 | 28 | 미제출 |
| 0.47 | 0.94094 | 23 | 미제출 |
| 0.48 | 0.94094 | 14 | 미제출 |
| 0.49 | 0.94094 | 10 | 미제출 |
| 0.50 | 0.94488 | 0 | 0.94245 |
| 0.51 | 0.94488 | 5 | 0.94284 |
| 0.52 | 0.94488 | 10 | **0.94324** |
| 0.53 | **0.94685** | 12 | 제출 한도 때문에 미제출 |
| 0.54 | 0.94488 | 20 | 미제출 |

최종 Public 최고는 `submission_qwen35_public_fine_w0p52_retrieval.csv`의 **0.94324**였다. w0.50, w0.51, w0.52에서 Public 점수가 연속으로 상승했고 w0.53은 dev도 더 높았지만, 일일 제출 한도 20회를 모두 사용해 확인하지 못했다.

## 16. 최종 제출 선택

Public leaderboard는 test의 약 50%만 사용하므로, 한 후보에만 의존하지 않고 서로 성격이 다른 두 파일을 최종 선택했다.

| 우선순위 | 최종 파일 | Public LB | 선택 이유 |
| ---: | --- | ---: | --- |
| 1 | `submission_qwen35_public_fine_w0p52_retrieval.csv` | **0.94324** | Public 최고, anchor 대비 10개만 변경 |
| 2 | `submission_qwen35_public_anchor_min_margin_top5_retrieval.csv` | 0.94284 | 5개만 바꾼 보수적 대안 |

두 후보는 동일한 방향의 복사본이 아니다. 첫 번째는 전체 확률 weight를 조정한 후보이고, 두 번째는 검증된 anchor에서 고신뢰 문항만 선택적으로 교체한 후보라 private 50%에서의 위험을 조금 분산할 수 있다.

대회 종료 직전 확인한 Public 상위권은 0.95270, 0.95151, 0.95072, 0.95033 수준이었다. 최종 Public 0.94324는 목표로 잡았던 0.96에는 못 미쳤지만, 로컬 기준 0.92116에서 **0.02208** 절대 상승했고 Qwen3-VL 8B 기준 0.92353에서는 **0.01971** 상승했다.

## 17. 제출 횟수 제한에서 얻은 교훈

일일 제출 한도가 20회라는 사실을 늦게 확인해 마지막 weight 탐색을 끝내지 못했다. 특히 w0.50 → w0.51 → w0.52로 Public이 계속 상승하는 구간에서 w0.53을 남겨둔 것이 가장 아쉬웠다.

다음 대회에서는 제출 예산을 아래처럼 미리 나누는 편이 좋다.

| 용도 | 권장 횟수 |
| --- | ---: |
| 모델·프롬프트의 큰 방향 확인 | 5 |
| 앙상블 weight 탐색 | 5 |
| OOF router/선택적 교체 | 5 |
| 최종 미세조정·비상용 | 5 |

제출 전에는 후보 간 변경 행 수와 disagreement 목록을 먼저 계산해야 한다. 거의 같은 파일을 반복 제출하기보다, 각 제출이 어떤 가설을 검증하는지 기록하고 대표 후보만 제출하는 것이 효율적이다.

## 18. Discussion에서 확인한 개선 방향

대회 종료 전 다른 참가자의 Discussion을 확인하면서 다음 방법이 특히 유망해 보였다.

- 유사 이미지를 pHash/SHA-256으로 묶은 뒤 group 단위로 validation을 분리한다.
- 3-fold OOF 예측으로 “어떤 모델을 믿을지” 학습하는 reliability router를 만든다.
- 전체 문항을 바꾸지 않고, OOF에서 rescue가 harm보다 확실히 큰 소수 문항만 교체한다.
- count, material, color 등 질문 유형별로 정확도와 최적 모델을 따로 분석한다.
- 특히 count 문제는 대상 확인 → 객체 분리 → 개수 계산 → 선택지 매핑의 여러 단계로 나눠 오류를 본다.
- 같은 모델의 해상도 변형만 늘리기보다, 모델 계열과 학습 방식이 다른 예측을 앙상블해 오류 상관을 낮춘다.
- hard sample을 별도로 모아 짧은 hard-focus fine-tuning을 수행하되 group OOF로 과적합을 확인한다.

이번 실험의 508개 단일 split에서는 고해상도 dev 최고 0.94882가 Public에서 재현되지 않았다. 다음에는 단일 validation 최고점보다 여러 fold의 평균과 최저 fold, rescue/harm 안정성을 우선해야 한다.

## 19. 재현용 코드와 산출물 위치

후속 작업에서 사용한 주요 로컬 경로는 다음과 같다.

### 19.1 후보 생성 코드

- `workspace/runpod/build_multi_hr_ensemble.py`: standard/896/1024 다중 해상도 확률 앙상블
- `workspace/runpod/build_calibrated_hr_ensemble.py`: 온도 및 확률 calibration 탐색
- `workspace/runpod/build_public_anchor_rescue.py`: Public anchor 기준 소수 문항 rescue 후보 생성
- `workspace/runpod/build_fine_standard_weights.py`: standard rotgeo weight 0.46~0.54 미세 탐색
- `workspace/runpod/make_discussion_chart.py`: Discussion용 점수 변화 그래프 생성

### 19.2 결과 폴더

- `workspace/outputs/submissions/qwen35_candidates`
- `workspace/outputs/submissions/qwen35_hr_candidates`
- `workspace/outputs/submissions/qwen35_multihr_candidates`
- `workspace/outputs/submissions/qwen35_calhr_candidates`
- `workspace/outputs/submissions/qwen35_public_anchor_candidates`
- `workspace/outputs/submissions/qwen35_public_fine_candidates`

각 후보 폴더에는 CSV, retrieval 적용본, validation JSON, grid summary와 bundle 파일이 함께 있다. 최종 제출에 사용한 대표 CSV는 `workspace/outputs/submissions` 최상위에도 복사했다.

### 19.3 Discussion 기록

- 결과 공유: <https://www.kaggle.com/competitions/ssafy-16-1-ai/discussion/738274>
- 제출 횟수 회고: <https://www.kaggle.com/competitions/ssafy-16-1-ai/discussion/738275>
- 로컬 초안: `workspace/outputs/discussion/discussion-post.md`
- 제출 제한 회고 초안: `workspace/outputs/discussion/submission-limit-retrospective.md`
- 점수 그래프: `workspace/outputs/discussion/public-score-chart.png`

## 20. 종료 상태

- Qwen3.5 기본 추론, 고해상도 TTA, multi-resolution/calibration 탐색과 최종 CSV 생성까지 완료했다.
- 모든 최종 CSV는 5,074행, `id/answer` 열, `a~d` label, sample submission ID 순서를 검증했다.
- A100 1개 Pod와 A100×2 Pod는 모두 중지해 추가 GPU 비용이 발생하지 않는 상태로 만들었다.
- Network Volume은 실수로 결과를 잃지 않도록 삭제하지 않고 보존했다.
- RunPod 모니터링 자동화는 작업 완료 후 종료했다.
- 로컬에 모델별 확률, 후보 CSV, summary와 Discussion 자료를 보존했다.

## 21. 최종 결론

가장 큰 성능 향상은 학습 step을 늘린 것 하나에서 나온 것이 아니라, 더 큰 Qwen3.5 모델을 짧게 보정하고 choice-rotation TTA와 Qwen3-VL 확률 앙상블을 결합하면서 나왔다. 반대로 고해상도와 복잡한 router는 dev 수치가 좋아도 Public에서 재현되지 않았다.

이번 실험에서 가장 재사용 가치가 높은 원칙은 다음 네 가지다.

1. 모델마다 최종 label뿐 아니라 선택지 전체 확률을 저장한다.
2. 유사 이미지가 fold 사이에 섞이지 않도록 group OOF를 사용한다.
3. 큰 변경보다 검증된 anchor 주변의 보수적 weight 탐색을 우선한다.
4. 제출 횟수를 실험 예산으로 보고 마지막 미세조정용 횟수를 반드시 남긴다.