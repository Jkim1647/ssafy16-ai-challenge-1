# 대회 개요

이 저장소는 Kaggle의 **SSAFY 16기 1회차 AI 챌린지**에서 진행한 실험을 재현 가능한 형태로 정리한 프로젝트입니다. 대회는 재활용품 이미지와 한국어 질문을 함께 이해하고, 네 개의 선택지 중 하나를 고르는 Visual Question Answering(VQA) 모델 개발을 목표로 했습니다.

- 공식 페이지: [SSAFY 16기 1회차 AI 챌린지](https://www.kaggle.com/competitions/ssafy-16-1-ai)
- 공식 설명: [Overview](https://www.kaggle.com/competitions/ssafy-16-1-ai/overview)
- 데이터 설명: [Data](https://www.kaggle.com/competitions/ssafy-16-1-ai/data)
- 대회 규칙: [Rules](https://www.kaggle.com/competitions/ssafy-16-1-ai/rules)

> 아래 내용은 대회 페이지를 이해하기 쉽게 요약한 것입니다. 원문 규정과 충돌할 경우 Kaggle에 게시된 공식 규칙이 우선합니다.

## 문제 정의

모델은 다음 입력을 받습니다.

1. 재활용품이 포함된 이미지
2. 이미지에 관한 한국어 질문
3. `a`, `b`, `c`, `d` 네 개의 선택지

출력은 정답 선택지를 나타내는 한 글자입니다. 즉, 자유형 문장을 생성하는 문제가 아니라 이미지와 질문을 함께 이해해 네 후보의 확률을 비교하는 **4지선다 분류형 VQA**입니다.

```text
재활용품 이미지 + 한국어 질문 + 선택지 a/b/c/d
                         │
                         ▼
                 멀티모달 VQA 모델
                         │
                         ▼
                   answer ∈ {a,b,c,d}
```

## 데이터 구성

대회 페이지 기준 전체 데이터는 약 **1.87 GB, 14,567개 파일**로 안내되었습니다. 원본 데이터는 비공개 대회 자료이므로 이 저장소에는 포함하지 않습니다.

| 항목 | 포함 내용 |
| --- | --- |
| `train.csv` + `train/` | 이미지 경로, 질문, 선택지 `a`~`d`, 정답 |
| `test.csv` + `test/` | 이미지 경로, 질문, 선택지 `a`~`d`; 정답 비공개 |
| `dev.csv` + `dev/` | 이미지·질문·선택지와 교육생 응답 1~5; ground truth는 아님 |
| `sample_submission.csv` | 제출해야 하는 `id,answer` 형식 |

`dev.csv`의 교육생 응답은 정답 label과 성격이 다르므로, 이 프로젝트에서는 ground-truth validation과 분리해 취급했습니다.

## 평가와 제출

- 핵심 평가지표: **Accuracy**
- 제출 파일: CSV
- 필수 열: `id`, `answer`
- 정답 값: `a`, `b`, `c`, `d` 중 하나
- Public Leaderboard는 test 일부로 계산되고, 최종 순위는 나머지 Private test 결과로 결정

따라서 작은 validation 또는 Public 점수만 최대화하면 최종 데이터에 일반화되지 않을 수 있습니다. 이 프로젝트에서 grouped split과 보수적인 앙상블 선택을 강조한 이유입니다.

## 프로젝트와 직접 관련된 규칙

| 구분 | 공식 규칙 요약 | 이 프로젝트의 대응 |
| --- | --- | --- |
| 참가 | SSAFY 16기 교육생, 개인전, 1인 1팀 | 단일 참가자 파이프라인 |
| 학습 데이터 | 제공 학습 데이터 사용 가능 | 원본 데이터는 로컬/RunPod에서만 사용 |
| 외부 데이터 | 공개적으로 사용 가능한 외부 데이터 허용 | 공개 pretrained model만 사용 |
| 증강 | 공개적으로 사용 가능한 범위에서 허용 | choice rotation과 이미지 전처리 실험 |
| 모델 | Hugging Face pretrained VQA 모델 허용 | Qwen3-VL, Qwen3.5 계열 사용 |
| 효율화 | Fine-tuning, LoRA, 양자화 허용 | 4-bit QLoRA 적용 |
| 프롬프트 | 성능 향상을 위한 prompt engineering 허용 | 정답 한 토큰 출력 형식 설계 |
| API | API 호출·응답을 이용한 추론 금지 | 다운로드한 모델을 GPU에서 직접 추론 |
| 누수 | test 사전 접근·유출 금지 | label을 사용하지 않는 추론만 수행 |
| 제출 | 하루 최대 20회 | 후보를 로컬 검증 후 선별하도록 개선 |

## 이 과제에서 중요했던 점

- 이미지와 자연어를 동시에 처리해야 합니다.
- 선택지 위치 편향 때문에 동일 의미라도 `a`~`d` 순서에 따라 결과가 달라질 수 있습니다.
- 동일하거나 유사한 이미지가 split 양쪽에 들어가면 validation이 과대평가될 수 있습니다.
- Public 점수가 높은 조합이 반드시 Private에서도 가장 좋다고 보장되지 않습니다.
- 하루 제출 횟수가 제한되므로 사전에 CSV 형식, ID 순서, 변경 행 수를 검증해야 합니다.

구체적인 해결 방법과 결과는 [METHODS.ko.md](METHODS.ko.md)와 [EXPERIMENT_LOG.ko.md](EXPERIMENT_LOG.ko.md)에서 확인할 수 있습니다.

