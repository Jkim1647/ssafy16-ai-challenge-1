# 방법론 요약

## 목표

이미지, 한국어 질문, 네 선택지로부터 정답 한 글자를 예측한다. Accuracy가 지표이므로 자유 생성보다 선택지별 점수를 직접 비교하는 구조를 사용했다.

## 학습

- Qwen3-VL-4B/8B와 Qwen3.5-35B-A3B 사용
- NF4 4-bit QLoRA
- 정답 한 token만 supervision
- 이미지 hash grouped validation
- 모델 선택은 loss보다 validation accuracy 우선

## 추론

1. chat template 끝의 next-token logits를 계산한다.
2. `a`, `b`, `c`, `d` token logits만 선택한다.
3. softmax 또는 log-softmax로 네 선택지 점수를 저장한다.
4. 선택지 순서를 회전한 추론은 확률을 원래 위치로 복원한다.
5. 모델별 확률을 가중 평균해 최종 답을 고른다.

## 검증 원칙

- 동일·유사 이미지가 train/validation에 나뉘지 않게 한다.
- 전체 accuracy뿐 아니라 fold별 accuracy, disagreement, rescue/harm을 함께 본다.
- dev가 좋아져도 Public에서 악화될 수 있으므로 기존 최고 후보를 보존한다.
- 후보를 제출하기 전에 기존 후보와 다른 행 수를 계산한다.
- Public 결과를 이용한 미세조정은 과적합 위험을 명시하고 보수적으로 수행한다.

## 가장 효과적이었던 조합

Qwen3-VL 계열 확률 약 48%와 Qwen3.5 rotation-TTA 확률 약 52%를 섞고 exact retrieval 보정을 적용한 후보가 Public 0.94324를 기록했다.

## 다음 개선 우선순위

1. pHash group 3-fold OOF
2. 질문 유형별 reliability router
3. hard-sample 중심의 짧은 추가 학습
4. 같은 모델의 해상도 변형보다 오류 상관이 낮은 다른 모델 계열 추가
5. 최종 제출 5회 이상 사전 확보