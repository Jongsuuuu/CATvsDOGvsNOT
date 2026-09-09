# CAT vs DOG vs NOT Image Classification
### 분류 난이도별 XAI & 정량 평가 지표 기반 성능 개선 전략 실험

> 동일한 3-클래스(cat / dog / not) 이미지 분류 문제라도 **분류 난이도(EASY / MID / HARD)에 따라 모델이 실패하는 방식이 다르며, 따라서 최적의 성능 개선 전략도 달라야 한다**는 가설을 XAI (Explainable AI) 기법과 정량 평가 지표로 검증한 프로젝트입니다.

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [데이터 출처 및 전처리](#3-데이터-출처-및-전처리)
4. [핵심 구현 및 최적화](#4-핵심-구현-및-최적화--before--after-난이도별-전략)
5. [기술 스택](#5-기술-스택)
6. [설치 및 실행 방법](#6-설치-및-실행-방법)
7. [폴더 구조](#7-폴더-구조)
8. [주요 트러블슈팅](#8-주요-트러블슈팅)
9. [결과 해석](#9-결과-해석)
10. [결과 및 그래프](#10-결과-및-그래프)
11. [최종 회고 및 성찰](#11-최종-회고-및-성찰)
12. [향후 발전 계획](#12-향후-발전-계획)

---

## 1. 프로젝트 개요

### 1.1 배경 및 가설

이미지 분류 프로젝트에서 흔히 "정확도를 올리려면 더 큰 모델, 더 센 증강을 쓰면 된다"는 식의 일괄적인 처방이 적용되곤 합니다. 이 프로젝트는 그 가정에 의문을 던지는 것에서 출발했습니다.

> **핵심 가설**: 같은 분류 문제라도 샘플의 "분류 난이도"에 따라 모델이 겪는 실패의 원인(과적합, 표현력 부족, 배경-전경 혼동, 클래스 불균형 등)이 서로 다르며, 따라서 정량적 평가지표와 XAI 시각화 근거에 기반해 **난이도별로 차별화된 개선 전략**을 적용해야 가장 효율적인 성능 향상을 얻을 수 있다.

이를 검증하기 위해 동일한 cat과 dog 데이터셋에 not 데이터셋을 **EASY / MID / HARD** 세 난이도로 나누고, 하나의 공통 베이스라인(`BEFORE`)에서 출발하여 각 난이도별 실패 패턴을 5가지 XAI 기법으로 진단한 뒤, 그 근거에 따라 서로 다른 개선 전략을 적용한 `AFTER` 버전을 만들어 비교했습니다.

### 1.2 실험 설계 요약

| 단계 | 내용 |
|---|---|
| **BEFORE** | EASY/MID/HARD 공통으로 ResNet18 단일 아키텍처 + 동일한 학습 레시피(Adam, 기본 증강)로 베이스라인 구축 |
| **진단** | Grad-CAM, Grad-CAM++, Score-CAM, Occlusion Sensitivity, LIME 5종 XAI + Confusion Matrix/F1로 난이도별 실패 원인 분석 |
| **AFTER** | 진단 결과에 근거해 EASY (정규화 완화) / MID (증강 강화) / HARD (백본 교체 + 최다 정규화 기법)로 차별화된 전략 적용 |
| **검증** | 동일한 XAI·정량 평가 파이프라인으로 BEFORE 대비 개선폭을 난이도별로 재측정 |

### 1.3 핵심 결과 미리보기

| 난이도 | BEFORE Accuracy | AFTER Accuracy (TTA) | 개선폭 |
|---|---|---|---|
| EASY | 0.9401 | 0.9583 | **+1.82%p** |
| MID | 0.8868 | 0.9111 | **+2.43%p** |
| HARD | 0.8298 | 0.8817 | **+5.19%p** |

가장 어려운 HARD 난이도에서 가장 정교한 전략을 적용했을 때 개선폭이 가장 크게 나타나, "난이도에 따라 처방이 달라야 한다"는 가설을 뒷받침하는 결과를 얻었습니다. 자세한 근거와 해석은 [4장](#4-핵심-구현-및-최적화--before--after-난이도별-전략)과 [9장](#9-결과-해석)에서 다룹니다.

---

## 2. 시스템 아키텍처

### 2.1 실행 환경

- **실행 플랫폼**: Google Colab (GPU 런타임)
- **영속 저장소**: Google Drive (`/content/drive/MyDrive/CATvsDOGvsNOT/`) — 데이터셋 압축파일, 학습된 가중치(`.pth`), 학습 히스토리(`.json`)를 모두 Drive에 저장하여 Colab 세션이 끊겨도 재학습 없이 이어서 진행 가능
- **노트북 구성**: `img_cls_BEFORE.ipynb` (베이스라인 학습 + 진단), `img_cls_AFTER.ipynb` (난이도별 개선 전략 학습 + 재진단)

### 2.2 파이프라인 흐름

<img width="3080" height="2003" alt="image" src="https://github.com/user-attachments/assets/a0e42c94-92b2-4b5d-b4f8-9b6e5ba0bfdc" />
<sub>※ 본 파이프라인 흐름 시각화에는 Claude 기반 이미지 생성 기능을 활용했습니다.</sub>

### 2.3 모델 진단(XAI) 아키텍처

`run_xai_pipeline` 내부에서는 hook 충돌을 방지하기 위해 **XAI 기법별로 독립된 모델 인스턴스**(`model_gc`, `model_gcpp`, `model_sc`)를 별도로 로드합니다. 하나의 모델에 Grad-CAM, Grad-CAM++, Score-CAM의 forward/backward hook을 동시에 등록하면 backward pass 과정에서 hook끼리 activation·gradient 버퍼를 덮어쓰는 간섭이 발생할 수 있기 때문입니다. (자세한 배경은 [8장 트러블슈팅](#8-주요-트러블슈팅) 참고)

---

## 3. 데이터 출처 및 전처리

### 3.1 데이터 출처 및 구조

데이터는 **Kaggle의 CIFAR-10에서 CAT과 DOG 클래스에 해당하는 이미지를, CIFAR-100에서 NOT 클래스에 해당하는 이미지를 수집하여 구성**하였습니다. 구성된 데이터셋을 Google Drive에 `CATvsDOGvsNOT/dataset.zip`으로 저장하였으며, Colab 환경에서 압축 해제 후 `find_dataset_root` 함수를 통해 데이터셋의 실제 루트 경로를 자동으로 탐색하도록 구성하였습니다.

```
dataset/
  easy/   → train/ test/  → cat/ dog/ not-easy/
  mid/    → train/ test/  → cat/ dog/ not-mid/
  hard/   → train/ test/  → cat/ dog/ not-hard/
```

### 3.2 난이도별 데이터 규모 및 클래스 불균형

테스트셋 Confusion Matrix의 support(정답 개수 총합) 기준으로 집계한 난이도별 클래스 분포는 다음과 같습니다.

| 난이도 | cat | dog | not-* | 총합 | not 비중 |
|---|---|---|---|---|---|
| EASY | 1,000 | 1,000 | 5,000 | 7,000 | 71% |
| MID | 1,000 | 1,000 | 2,700 | 4,700 | 57% |
| HARD | 1,000 | 1,000 | 2,200 | 4,200 | 52% |

난이도가 낮을수록(EASY) "not" 클래스 비중이 커지는 불균형 구조이고, 난이도가 높을수록(HARD) 상대적으로 균형 잡힌 구조입니다. 이는 4장에서 설명할 **클래스 가중치 도입**의 배경이 됩니다.

### 3.3 난이도 정의 — 실제 샘플 관찰 근거

`{난이도}_정답샘플*.png`, `{난이도}_오답샘플*.png` (CDN 결과 이미지) 및 학습 로그를 관찰한 결과, 세 난이도는 다음과 같은 시각적 특성 차이를 보였습니다.

- **EASY**: 피사체가 화면 중앙에 크고 선명하게 위치, 배경이 단순하거나 피사체와 뚜렷하게 구분됨. 정답 샘플의 클래스 확률이 대부분 95%+ 로 매우 확신에 찬 예측을 보임.
- **MID**: 피사체가 풀숲·덤불 등 자연/실외 배경에 일부 섞여 있어, 배경과 전경의 색상·질감이 유사한 "위장(camouflage)"형 오답이 다수 관찰됨. 오답 샘플의 Grad-CAM 히트맵이 피사체 경계를 넘어 배경으로 새어나가는(leakage) 경향이 확인됨.
- **HARD**: 극단적 클로즈업, 저조도, 초점 흐림, 부분 가림(occlusion) 등으로 피사체의 형태 정보 자체가 제한적. 오답 샘플에서는 활성화가 피사체가 아닌 질감/그림자 패턴에 집중되는 오귀인(misattribution)이 다수 관찰됨.

### 3.4 전처리 파이프라인

- **공통 전처리**: `Resize(224, 224)` → `ToTensor` → `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` (ImageNet 사전학습 백본과의 통계 호환성 확보)
- **Train/Val 분할**: `ImageFolder` 기반 클래스별 stratified 80/20 분할(`val_ratio=0.2`), `np.random.default_rng(seed=42)`로 시드 고정하여 재현성 확보
- **BEFORE 증강**: `RandomResizedCrop(224, scale=(0.8,1.0))` + `RandomHorizontalFlip` + `ColorJitter(0.2,0.2,0.2,0.05)` — 모든 난이도 동일 적용
- **AFTER 증강(`get_train_transform`, 난이도별 차등)**:

  | 난이도 | 추가 증강 | RandomErasing |
  |---|---|---|
  | EASY | `RandomGrayscale(p=0.05)` | p=0.15, scale=(0.02, 0.15) |
  | MID | `RandAugment(num_ops=2, magnitude=9)` | p=0.25, scale=(0.02, 0.20) |
  | HARD | `RandomVerticalFlip(p=0.1)` + `RandAugment(num_ops=2, magnitude=12)` | p=0.35, scale=(0.02, 0.25) |

  난이도가 높아질수록 증강 강도(magnitude, erasing 확률·면적)를 단계적으로 강화한 것이 핵심 설계입니다.
- **클래스 가중치(`compute_class_weights`)**: AFTER부터 도입. 학습 세트의 클래스별 샘플 수 역비율로 가중치를 산출해 Loss에 반영 — BEFORE에서 세 난이도 모두 cat 클래스의 precision/recall이 가장 낮았던 문제(3.2절의 불균형 구조 참고)에 대응.
- **TTA 전처리(AFTER, 추론 전용)**: 원본 / 수평 플립 / `Resize(256)+CenterCrop(224)` 3가지 변환에 대한 softmax 확률 평균.

---

## 4. 핵심 구현 및 최적화 — BEFORE → AFTER 난이도별 전략

이 프로젝트의 핵심은 "왜 이 난이도에는 이 전략을 선택했는가"입니다. 아래에서는 (1) 모든 난이도에 공통 적용된 개선, (2) EASY, (3) MID, (4) HARD 각각의 진단·전략·근거 논문·결과를 순서대로 설명합니다.

### 4.1 공통 개선 사항 (EASY / MID / HARD 공통 적용)

| 항목 | BEFORE | AFTER |
|---|---|---|
| Optimizer | Adam (lr=1e-4, wd=1e-4) | AdamW (wd=1e-2) |
| Scheduler | ReduceLROnPlateau | Linear Warmup + CosineAnnealing |
| Loss | CrossEntropyLoss | Label Smoothing CE + 클래스 가중치 |
| Gradient Clipping | 없음 | max_norm=1.0 |
| 평가 | 표준 평가만 | 표준 평가 + TTA(3-view) |

- **AdamW (weight decay 분리)** — Loshchilov & Hutter, *"Decoupled Weight Decay Regularization"*, arXiv:[1711.05101](https://arxiv.org/abs/1711.05101). 이 논문은 Adam의 L2 정규화 항이 실제로는 SGD에서와 같은 weight decay 효과를 내지 못한다는 점을 지적하고, 그래디언트 업데이트와 weight decay를 분리(decouple)한 AdamW를 제안합니다. BEFORE 학습 곡선(`BEFORE_CURVE.png`)에서 확인된 train/val loss 격차(과적합)를 줄이기 위해 weight_decay를 1e-4→1e-2로 100배 강화하며 AdamW로 전환했습니다.
- **Linear Warmup + Cosine Annealing** — Loshchilov & Hutter, *"SGDR: Stochastic Gradient Descent with Warm Restarts"*, arXiv:[1608.03983](https://arxiv.org/abs/1608.03983) (cosine annealing 스케줄) 및 Goyal et al., *"Accurate, Large Minibatch SGD"*, arXiv:[1706.02677](https://arxiv.org/abs/1706.02677) (학습 초반 linear warmup으로 불안정성 완화)의 조합을 `CosineWarmupScheduler` 클래스로 직접 구현했습니다. ReduceLROnPlateau는 loss가 정체될 때만 반응하는 반응형(reactive) 방식인 반면, cosine 스케줄은 처음부터 감소 궤적이 정해져 있어 MixUp/CutMix로 인해 매 epoch 손실 변동이 큰 AFTER 학습에 더 안정적으로 맞았습니다.
- **Label Smoothing** — Müller, Kornblith & Hinton, *"When Does Label Smoothing Help?"*, arXiv:[1906.02629](https://arxiv.org/abs/1906.02629). 정답 라벨을 `(1-ε) + ε/(K-1)`로 스무딩하여 모델이 과신(over-confidence)하지 않도록 유도합니다. BEFORE 학습 곡선에서 train_acc가 99%+로 수렴하는 반면 val_acc는 훨씬 낮은 지점에서 정체되는 현상(과적합)을 억제하기 위해 도입했으며, EASY=0.05, MID/HARD=0.10으로 난이도가 높을수록 더 강하게 적용했습니다.
- **클래스 가중치** — Cui et al., *"Class-Balanced Loss Based on Effective Number of Samples"*, arXiv:[1901.05555](https://arxiv.org/abs/1901.05555)의 문제의식(클래스별 유효 샘플 수가 다르면 단순 손실 합산이 다수 클래스에 편향된다는 지적)을 참고하되, 본 프로젝트는 구현 단순성을 위해 학습 세트 클래스별 역빈도(inverse frequency) 가중치를 `compute_class_weights`로 직접 산출해 Label Smoothing Loss에 반영했습니다.
- **Gradient Clipping** — Pascanu, Mikolov & Bengio, *"On the difficulty of training recurrent neural networks"*, arXiv:[1211.5063](https://arxiv.org/abs/1211.5063)에서 제안된 기법을 차용했습니다. 특히 MixUp/CutMix를 확률적으로 병행 적용하는 MID·HARD에서는 배치마다 손실의 스케일 변동이 커질 수 있어, 그래디언트 노름을 1.0으로 제한해 발산을 방지했습니다.
- **TTA (Test-Time Augmentation)** — Shanmugam et al., *"When and Why Test-Time Augmentation Works"* / *"Better Aggregation in Test-Time Augmentation"*, arXiv:[2011.11156](https://arxiv.org/abs/2011.11156). 이 연구는 TTA가 항상 이득이 아니라 일부 정답을 오답으로 바꾸기도 한다는 점을 실증적으로 보이며 신중한 aggregation을 강조합니다. 이에 따라 본 프로젝트는 과도한 변환 조합 대신 원본/수평 플립/리사이즈-크롭 **3가지 변환의 단순 평균**만 사용해 리스크를 제한했습니다.

### 4.2 EASY 전략 — "정규화 완화"

**진단(BEFORE)**: EASY는 BEFORE 단계에서 이미 Accuracy 0.9401로 세 난이도 중 가장 높았습니다. `EASY_GRAD.png`(클래스별 평균 Grad-CAM)에서 cat/dog 모두 피사체 중심에 활성화가 정확히 집중되어 있어, 모델이 "무엇을 봐야 하는지"는 이미 잘 학습된 상태였습니다. 그러나 학습 곡선(`BEFORE_CURVE.png`)을 보면 **불과 9 epoch만에 val loss 최적점을 찍은 뒤 이후 15 epoch 동안 개선 없이 정체**했고, train_acc는 0.9964까지 치솟은 반면 val_acc는 0.9523에 머물러 과적합 격차(gap)가 발생했습니다.

**전략**: 아키텍처는 ResNet18을 유지하되(문제가 표현력 부족이 아니었으므로),
- `Dropout` 0.5 → 0.3으로 완화
- `RandomGrayscale(p=0.05)`의 약한 색상 불변성 증강만 추가
- `RandomErasing`도 p=0.15로 가장 약하게 적용
- TTA 적용

**근거**: 데이터가 이미 쉬운 상황에서 과도한 정규화(무거운 Dropout, 강한 증강)는 오히려 불필요한 언더피팅과 학습 속도 저하를 유발한다는 것이 이 전략의 핵심 논리입니다. 실제로 AFTER 학습 곡선에서 EASY는 85 epoch까지 완만하게 학습되며 val_acc가 0.9549(best epoch)까지 점진적으로 개선되었고, train_acc(0.9858)와 val_acc(0.9530)의 최종 격차(3.3%p)도 BEFORE(4.4%p, 24 epoch 만에 조기 수렴)보다 완만한 형태를 보였습니다.

**결과**: Accuracy 0.9401 → 0.9583 (TTA, **+1.82%p**), F1-macro 0.8921 → 0.9217 (**+2.96%p**) — 세 난이도 중 개선폭은 가장 작았으나, 이는 애초에 성능 상한에 근접해 있었다는 진단과 일치하는 결과입니다.

### 4.3 MID 전략 — "배경 혼동에 대한 증강 강화"

**진단(BEFORE)**: `MID_정량적평가.png` confusion matrix에서 cat precision 0.7959, recall 0.7840으로 세 클래스 중 가장 취약했습니다. `MID_오답샘플1.png`를 보면 오답의 상당수가 **자연 배경(풀숲, 덤불 등)에 피사체가 일부 섞여 있는 케이스**였고, 해당 샘플들의 Grad-CAM 히트맵이 피사체 윤곽을 넘어 배경 텍스처로 새어나가는 모습이 관찰되었습니다. 즉, 표현력 부족보다는 **배경-전경 색상/질감 유사성으로 인한 혼동**이 주된 실패 원인으로 진단되었습니다.

**전략**: 아키텍처는 ResNet18을 유지(중간 난이도이므로 표현력보다 정규화·증강이 더 중요하다고 판단)하되,
- `RandAugment(num_ops=2, magnitude=9)` 도입
- `RandomErasing(p=0.25)`로 강화
- `MixUp(alpha=0.4)` 적용
- 클래스 가중치 적용

**근거**:
- **RandAugment** — Cubuk et al., arXiv:[1909.13719](https://arxiv.org/abs/1909.13719). 이 논문은 별도의 augmentation policy 탐색 없이도 무작위로 조합된 변환만으로 최신 자동 탐색 기법과 대등하거나 더 나은 일반화 성능을 낼 수 있음을 ImageNet 등에서 실증했습니다. "배경과 전경이 혼재"된 MID 데이터 특성상, 다양한 색상·기하 변환 조합을 무작위로 경험시키는 것이 특정 배경 패턴에 대한 과도한 의존을 줄이는 데 유리하다고 판단했습니다.
- **RandomErasing** — Zhong et al., arXiv:[1708.04896](https://arxiv.org/abs/1708.04896). 학습 이미지의 임의 사각 영역을 지워 다양한 가림 수준을 인위적으로 생성함으로써, 모델이 이미지의 특정 부분(예: 배경과 인접한 영역)에만 의존하지 않고 더 강건해지도록 유도한다고 설명합니다. MID의 위장/배경 혼입 패턴에 대한 강건성 확보를 위해 BEFORE 대비 강도(p=0.15→0.25)를 높였습니다.
- **MixUp** — Zhang et al., *"mixup: Beyond Empirical Risk Minimization"*, arXiv:[1710.09412](https://arxiv.org/abs/1710.09412). 두 이미지와 라벨을 선형 결합(`λx₁+(1-λ)x₂`)하여 학습시킴으로써 클래스 간 결정 경계를 더 매끄럽게(smoother) 만들어, 다양한 배경 조합에 대한 일반화력을 높인다고 제안합니다.

**결과**: Accuracy 0.8868 → 0.9111 (TTA, **+2.43%p**), F1-macro 0.8570 → 0.8916 (**+3.46%p**). 특히 cat recall이 0.7840(BEFORE) → 0.8500(AFTER, TTA)까지 개선되어, 배경 혼동으로 인한 실패를 겨냥한 전략이 실제로 해당 클래스의 재현율 개선으로 이어졌음을 확인했습니다.

### 4.4 HARD 전략 — "백본 교체 + 최대 강도 정규화"

**진단(BEFORE)**: HARD는 BEFORE 단계에서 Accuracy 0.8298로 세 난이도 중 가장 낮았고, 학습 곡선(`BEFORE_CURVE.png`)에서 **val loss가 7 epoch 이후 오히려 증가**하는 뚜렷한 과적합 패턴을 보였습니다(train_acc 0.9890 vs val_acc 0.8483, gap 14.1%p — 세 난이도 중 최대). `HARD_오답샘플1.png`에서는 극단적 클로즈업·저조도·흐림으로 피사체 정보 자체가 제한적이었고, Grad-CAM이 피사체가 아닌 배경 질감·그림자 패턴에 집중되는 오귀인이 다수 관찰되었습니다. 즉 HARD는 **① 얕은 ResNet18의 표현력 한계**와 **② 심각한 과적합**이라는 이중 문제를 겪고 있다고 진단했습니다.

**전략**: 이중 문제에 대응하기 위해 가장 많은 기법을 동시에 적용했습니다.
- 백본을 **ResNet18 → EfficientNet-B0**(timm, ImageNet pretrained)로 교체
- `RandAugment(num_ops=2, magnitude=12)` — 세 난이도 중 최대 강도
- `RandomVerticalFlip(p=0.1)` 추가
- `RandomErasing(p=0.35, scale=(0.02,0.25))` — 최대 강도
- **MixUp + CutMix 병행**(배치마다 50% 확률로 둘 중 하나 적용)
- Label Smoothing 0.10, 클래스 가중치, Gradient Clipping 1.0
- lr을 EASY/MID(1e-4) 대비 절반인 5e-5로 낮추고, patience를 20으로 늘려 더 완만하고 안정적인 수렴 유도

**근거**:
- **EfficientNet-B0** — Tan & Le, *"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"*, arXiv:[1905.11946](https://arxiv.org/abs/1905.11946). Depth·Width·Resolution을 compound scaling으로 균형 있게 확장하여, ResNet 계열 대비 적은 파라미터로도 더 높은 표현력을 달성한다고 제안합니다. 저품질·저정보 이미지가 많은 HARD에서는 더 정교한 특징 추출기가 필요하다고 판단해 채택했습니다. 반대로 EASY·MID는 이미 충분한 정보량 대비 ResNet18로 충분하다고 판단해 교체하지 않았는데, 이는 "무조건 큰 모델이 낫다"가 아니라 **정보 병목의 위치에 따라 처방을 달리해야 한다**는 이 프로젝트의 핵심 가설과 일치하는 설계입니다.
- **CutMix** — Yun et al., arXiv:[1905.04899](https://arxiv.org/abs/1905.04899). 이미지의 일부 영역을 다른 이미지의 패치로 대체하고 라벨도 면적 비율로 혼합하는 기법으로, 이미지 전체를 흐리게 섞는 MixUp과 달리 원본 픽셀 정보를 그대로 보존하면서도 국소적 특징에 대한 과도한 의존을 줄여 객체의 일부만 보여도 판별 가능하도록 유도한다고 설명합니다. 피사체 일부만 보이는 경우가 많은 HARD 데이터 특성과 부합한다고 판단해 MixUp과 병행 채택했습니다.
- **클래스 가중치(재확인)** — Cui et al., arXiv:[1901.05555](https://arxiv.org/abs/1901.05555)의 문제의식을 HARD에도 동일하게 적용: not-hard(2,200)가 cat/dog(각 1,000) 대비 많은 불균형 구조에 대응.
- **Gradient Clipping(재확인)** — Pascanu et al., arXiv:[1211.5063](https://arxiv.org/abs/1211.5063). CutMix+MixUp을 확률적으로 병행하면 배치 간 손실 스케일 변동이 가장 크게 나타나는 난이도가 HARD였기에, 그래디언트 폭주 방지가 특히 중요했습니다.

**결과**: Accuracy 0.8298 → 0.8817 (TTA, **+5.19%p, 세 난이도 중 최대 개선폭**), F1-macro 0.8124 → 0.8705 (**+5.81%p**). cat recall도 0.7240(BEFORE) → 0.8210(AFTER, TTA)까지 개선되었습니다. 이는 "난이도가 높을수록 더 정교하고 강한 전략이 필요하며, 그 효과도 크다"는 프로젝트의 핵심 가설을 가장 직접적으로 뒷받침하는 결과입니다.

### 4.5 XAI 5종 기법 구현 노트 (BEFORE·AFTER 공통, 진단에 사용)

| 기법 | 논문 | 특징 및 채택 이유 |
|---|---|---|
| **Grad-CAM** | Selvaraju et al., arXiv:[1610.02391](https://arxiv.org/abs/1610.02391) | 마지막 conv층 gradient를 전역 평균해 채널별 가중치 산출 후 activation과 가중합. 계산이 빠르고 안정적이어서 1차 스크리닝 용도로 사용 |
| **Grad-CAM++** | Chattopadhyay et al., arXiv:[1710.11063](https://arxiv.org/abs/1710.11063) | 픽셀별 고차 미분(2·3차)으로 산출한 alpha 가중치를 사용, 다중 객체·세밀한 국소 영역 포착에 강점 |
| **Score-CAM** | Wang et al., arXiv:[1910.01279](https://arxiv.org/abs/1910.01279) | gradient에 의존하지 않고 각 채널의 activation map으로 원본을 마스킹한 뒤 실제 softmax 점수 변화로 가중치 산출 → gradient saturation 문제에서 자유롭고 노이즈에 강건 |
| **Occlusion Sensitivity** | Zeiler & Fergus, arXiv:[1311.2901](https://arxiv.org/abs/1311.2901) | 슬라이딩 윈도우로 이미지 일부를 가리며 예측 확률 변화를 직접 관찰하는, 가장 모델-비의존적(model-agnostic)에 가까운 방식 |
| **LIME** | Ribeiro et al., arXiv:[1602.04938](https://arxiv.org/abs/1602.04938) | 슈퍼픽셀 단위로 이미지를 분할하고 국소적으로 선형 대리모델(surrogate model)을 학습해 중요 영역 도출 — 모델 구조에 완전히 독립적 |

5개 기법 각각의 노이즈를 상호 보완하기 위해 **Grad-CAM 30% + Grad-CAM++ 35% + Score-CAM 35%** 가중 평균 앙상블 CAM(`visualize_ensemble_cam`)도 함께 제공합니다.

---

## 5. 기술 스택

### Core (모델 · 학습)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Torchvision](https://img.shields.io/badge/Torchvision-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm%20(PyTorch%20Image%20Models)-EE4C2C?style=for-the-badge&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

### XAI & 해석가능성

![LIME](https://img.shields.io/badge/LIME-4B8BBE?style=for-the-badge&logoColor=white)
![Grad-CAM family](https://img.shields.io/badge/Grad--CAM%20%2F%20Grad--CAM%2B%2B%20%2F%20Score--CAM-654FF0?style=for-the-badge&logoColor=white)
![scikit-image](https://img.shields.io/badge/scikit--image-F7931E?style=for-the-badge&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

### 데이터 처리 & 시각화

![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-3776AB?style=for-the-badge&logoColor=white)

### 실행 환경 & 인프라

![Python](https://img.shields.io/badge/Python%203-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)
![NanumGothic](https://img.shields.io/badge/NanumGothic%20Font-2E7D32?style=for-the-badge&logoColor=white)

---

## 6. 설치 및 실행 방법

### 6.1 사전 준비물

- Google 계정 (Colab + Drive)
- GPU 런타임 (Colab 무료 T4 이상 권장 — HARD 학습이 최대 100 epoch까지 진행되므로 CPU 런타임은 비권장)
- Google Drive에 아래 구조로 사전 배치:

```
/content/drive/MyDrive/CATvsDOGvsNOT/
├── dataset.zip
├── img_cls_BEFORE.ipynb
├── img_cls_AFTER.ipynb
├── BEFORE/     (최초 실행 시 자동 생성됨, 비어 있어도 무방)
└── AFTER/      (최초 실행 시 자동 생성됨, 비어 있어도 무방)
```

### 6.2 실행 순서

1. `img_cls_BEFORE.ipynb`를 Colab으로 열고 **런타임 → 런타임 유형 변경 → GPU** 설정
2. 셀을 위에서부터 순서대로 실행(`런타임 → 모두 실행`)
   - CELL 1: `pip install lime scikit-image opencv-python-headless -q`
   - CELL 1*: 한국어 폰트(NanumGothic) 자동 설치
   - CELL 2: Google Drive 마운트 및 `dataset.zip` 자동 압축 해제
   - CELL 6: 난이도별(easy → mid → hard) 베이스라인 학습 — `BEFORE/best_model_{diff}.pth`가 이미 있으면 자동으로 로드만 수행
   - 이후 셀: XAI 진단 파이프라인 실행
3. `img_cls_AFTER.ipynb`를 열고 동일하게 실행
   - CELL 1: `pip install lime scikit-image opencv-python-headless timm -q` (timm 추가)
   - 나머지는 BEFORE와 동일한 흐름이나, 난이도별 차별화 전략(4장)이 자동 적용됨
4. 마지막 비교 셀(`results_after` vs `results_before`)에서 BEFORE 대비 AFTER 개선폭 자동 시각화

> **재실행 팁**: 하이퍼파라미터를 바꿔 재학습하고 싶다면 반드시 Drive의 해당 `best_model_{diff}.pth`(및 `_history.json`)를 먼저 삭제해야 합니다. 파일이 존재하면 두 노트북 모두 "저장된 모델 발견 → 재학습 없이 로드"로 분기합니다. (자세한 내용은 [8.6절](#8-주요-트러블슈팅) 참고)

### 6.3 로컬(비-Colab) 환경에서 실행 시 참고사항

- `from google.colab import drive` 및 `drive.mount(...)` 부분은 로컬 환경에 없는 API이므로, 로컬 데이터 경로로 직접 대체해야 합니다.
- 한국어 폰트 설치 셀은 Ubuntu 기반(Colab)의 `apt-get install fonts-nanum`을 가정하므로, 다른 OS에서는 해당 OS의 나눔고딕 설치 방법으로 대체해야 합니다.
- GPU가 없는 환경에서는 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`에 의해 자동으로 CPU로 폴백되지만, HARD 모델(EfficientNet-B0 + MixUp/CutMix, 최대 100 epoch)의 학습 시간이 매우 길어질 수 있습니다.

---

## 7. 폴더 구조

```
CATvsDOGvsNOT/
├── dataset.zip                        # easy/mid/hard 하위에 train/test/{cat,dog,not-*} 구조
├── img_cls_BEFORE.ipynb               # 베이스라인 학습 + XAI 진단 노트북
├── img_cls_AFTER.ipynb                # 난이도별 개선 전략 학습 + 재진단 노트북
│
├── BEFORE/                            # 베이스라인 모델 산출물 (Drive에 영속 저장)
│   ├── best_model_easy.pth
│   ├── best_model_easy_history.json   # {train_loss, val_loss, train_acc, val_acc}
│   ├── best_model_mid.pth
│   ├── best_model_mid_history.json
│   ├── best_model_hard.pth
│   └── best_model_hard_history.json
│
└── AFTER/                             # 개선 모델 산출물
    ├── best_model_easy.pth            # ResNet18 (dropout 0.3)
    ├── best_model_easy_history.json   # {..., lr}  ← AFTER는 lr 히스토리 추가 기록
    ├── best_model_mid.pth             # ResNet18 (dropout 0.4)
    ├── best_model_mid_history.json
    ├── best_model_hard.pth            # EfficientNet-B0 (timm)
    └── best_model_hard_history.json
```

> 아래 `assets/` (또는 `docs/images/`)는 본 저장소의 일부는 아니지만, [10장](#10-결과-및-그래프)에서 참조하는 결과 이미지들을 GitHub README에 실제로 삽입하려면 별도로 리포지토리에 추가하는 것을 권장합니다.

```
assets/
├── before/   # BEFORE_BASE.png, BEFORE_CURVE.png, BEFORE_VAL.png, {EASY,MID,HARD}_*.png ...
└── after/    # AFTER_BASE.png, AFTER_CURVE.png, AFTER_VAL.png, AFTER_LR.png, {EASY,MID,HARD}_*.png ...
```

---

## 8. 주요 트러블슈팅

BEFORE에서 AFTER로 개선해 나가는 과정에서 실제로 마주쳤을 가능성이 높은 문제들을, 노트북에 남아있는 수정 이력과 코드 구조를 근거로 단계적으로 정리했습니다.

### 8.1 `cam_to_overlay` 반환 타입 버그 — 히트맵이 새까맣게 보이는 문제

**증상**: CAM 오버레이 이미지를 `plt.imshow`로 그렸을 때 전체가 검게만 보임.
**원인**: `cam_to_overlay` 함수가 `uint8`([0,255]) 배열을 반환했는데, 호출부에서 `np.clip(img, 0, 1)`을 적용하면서 [0,1] 범위 밖의 모든 값이 손실되어 `imshow`가 이를 `float`로 잘못 해석. `uint8` 픽셀값(예: 128)이 `float` 컬러맵 기준([0,1])에서는 1.0을 초과해 잘려나가면서 대부분 픽셀이 검정으로 렌더링됨.
**해결**: `cam_to_overlay`가 항상 `(overlay / 255.0).astype(np.float32)`로 **float32 [0,1] 범위**를 반환하도록 수정. (BEFORE 노트북 CELL 0 마크다운의 "수정 사항"에 명시된 실제 변경 이력)

### 8.2 `visualize_mean_cam`의 정보 손실 — 순수 히트맵만으로는 해석 불가

**증상**: 클래스별 평균 활성화 맵이 원본 이미지 없이 `jet` 컬러맵 단독으로 표시되어, 활성화가 실제로 이미지의 어느 부분을 가리키는지 알 수 없음.
**원인**: 초기 구현이 클래스별 평균 CAM을 7×7 activation 해상도 그대로 `imshow(cmap='jet')`로만 시각화.
**해결**: 클래스별 "첫 번째 정답 샘플"을 대표 이미지로 저장해두었다가, 평균 CAM을 224×224로 업샘플링한 뒤 대표 이미지 위에 오버레이하도록 변경. (`EASY_GRAD.png` 등 결과물이 이 수정을 반영)

### 8.3 백본 교체 후 Grad-CAM 계열 hook이 깨지는 문제 (HARD → EfficientNet-B0)

**증상**: HARD에 EfficientNet-B0를 도입하면서 기존 `GradCAM`/`GradCAM++`/`Score-CAM` 클래스가 `model.layer4[-1].conv2`에 하드코딩되어 있어 `AttributeError: 'EfficientNet' object has no attribute 'layer4'` 발생 가능.
**원인**: ResNet 계열과 EfficientNet 계열은 마지막 conv 레이어의 속성명이 다름(ResNet: `layer4[-1].conv2`, EfficientNet(timm): `conv_head`).
**해결**: AFTER 노트북에서 `_get_last_conv(model, not_type)` 헬퍼 함수를 신설해 `BACKBONE_MAP`을 참조, 백본에 따라 올바른 레이어를 반환하도록 분기 처리. 이 덕분에 XAI 파이프라인 코드를 백본에 상관없이 재사용할 수 있게 됨.

### 8.4 timm 사전학습 가중치 다운로드 시 HuggingFace Hub 인증 경고

**증상**: `timm.create_model('tf_efficientnet_b0', pretrained=True, ...)` 최초 호출 시 `Warning: You are sending unauthenticated requests to the HF Hub.` 경고가 학습 로그에 출력됨.
**원인**: timm 최신 버전은 사전학습 가중치를 HuggingFace Hub에서 받아오는데, 비로그인 상태에서는 요청 속도 제한(rate limit)이 더 엄격하게 적용됨.
**해결/권장**: 대규모 반복 실험 시에는 `huggingface-cli login` 또는 `HF_TOKEN` 환경 변수 설정을 권장. 본 프로젝트 규모(3개 난이도, 1회성 다운로드)에서는 경고만 발생하고 학습에는 지장이 없었음.

### 8.5 MixUp/CutMix 적용 시 "Train Accuracy가 Val Accuracy보다 낮아 보이는" 착시

**증상**: AFTER의 MID·HARD 학습 곡선(`AFTER_CURVE.png`)에서 Val Accuracy 곡선이 Train Accuracy 곡선보다 계속 **위에** 그려지는, 일반적인 직관과 반대되는 현상이 관찰됨.
**원인 분석(코드 근거)**: `train_after` 함수에서 MixUp/CutMix가 적용된 배치는 `outputs = model(mixed)`로 혼합된 이미지에 대한 예측을 산출하지만, 정확도 집계는 `tr_correct += (preds == labels).sum()`으로 **원본(비혼합) 라벨** 기준으로 계산됩니다. 즉 이미지 절반이 다른 클래스와 섞인 상태에서 원래 라벨과 일치하는지를 채점하기 때문에, train_acc는 실제 모델의 표현 학습 진행도보다 **체계적으로 낮게 측정**됩니다. 반면 val_acc는 순수 원본 이미지에 대한 정확한 측정치입니다.
**결론**: 이는 버그가 아니라 MixUp/CutMix 사용 시 널리 알려진 지표 해석상의 함정입니다. 학습 진행도를 판단할 때는 train_acc보다 **train_loss의 하강 추세**나 val_acc를 기준으로 삼는 것이 더 정확합니다.

### 8.6 저장된 `.pth` 재사용 로직으로 인한 "하이퍼파라미터 변경이 반영되지 않는" 함정

**증상**: `TRAIN_CONFIG`의 학습률·label smoothing 값 등을 바꾸고 셀을 재실행해도 학습 로그가 전혀 출력되지 않고 곧바로 "저장된 모델 발견. 재학습 없이 로드합니다."만 출력됨.
**원인**: `train_after`/`train_baseline` 모두 `if os.path.exists(save_path): return load_model(...)` 형태로, Drive에 동일 경로의 `.pth`가 이미 존재하면 무조건 재학습을 건너뛰도록 설계됨(세션 재개·시간 절약 목적).
**해결**: 하이퍼파라미터를 바꿔 실험하려면 Drive에서 해당 `best_model_{diff}.pth`와 `best_model_{diff}_history.json`을 **먼저 수동으로 삭제**한 뒤 재실행해야 함.

### 8.7 Cosine 스케줄러 도입 후 Early Stopping 타이밍 변화

**증상**: BEFORE는 22~24 epoch 내 조기 종료되는 반면, AFTER는 63~100 epoch까지 학습이 지속됨(HARD는 patience=20 소진까지 100 epoch 풀로 진행).
**원인**: `ReduceLROnPlateau`는 val loss가 실제로 정체될 때 학습률을 낮춰 조기에 국소 최적점에 도달하는 경향이 있는 반면, `CosineWarmupScheduler`는 val loss와 무관하게 미리 정해진 궤적으로 학습률을 서서히 낮추기 때문에, val loss가 다소 진동(noisy plateau)하더라도 patience 카운터가 소진될 때까지 학습이 계속됨. `AFTER_VAL.png`에서 HARD의 val loss가 하락 후 좁은 범위에서 진동하는 패턴이 이를 뒷받침함.
**대응**: patience를 EASY/MID(15) 대비 HARD(20)로 더 여유 있게 설정하여, cosine 스케줄 특성상 발생하는 노이즈로 인한 조기 종료를 방지.

### 8.8 GPU 메모리/세션 제한 — Colab 무료 티어에서 HARD 학습 시간 초과 우려

**증상**: EfficientNet-B0 + MixUp/CutMix 조합으로 HARD 학습이 최대 100 epoch까지 진행되면서, Colab 무료 티어의 세션 제한 시간에 근접할 위험.
**대응**: [8.6절]에서 설명한 체크포인트 재사용 로직 덕분에, 세션이 끊기더라도 마지막으로 저장된 `best_model_hard.pth` 시점부터 손실 없이 이어갈 수 있는 구조(단, 이어서 학습을 계속하려는 것이 아니라 "로드 후 재사용"만 지원되므로, 학습 중간에 세션이 끊기면 해당 실행의 학습 자체는 처음부터 다시 시작해야 함에 유의).

---

## 9. 결과 해석

### 9.1 종합 성능 비교

| 난이도 | 지표 | BEFORE | AFTER (표준) | AFTER (TTA) | TTA 개선분 | 최종 개선폭(BEFORE→AFTER TTA) |
|---|---|---|---|---|---|---|
| EASY | Accuracy | 0.9401 | 0.9519 | 0.9583 | +0.64%p | **+1.82%p** |
| EASY | F1-macro | 0.8921 | 0.9087 | 0.9217 | +1.30%p | **+2.96%p** |
| MID | Accuracy | 0.8868 | 0.9045 | 0.9111 | +0.66%p | **+2.43%p** |
| MID | F1-macro | 0.8570 | 0.8825 | 0.8916 | +0.91%p | **+3.46%p** |
| HARD | Accuracy | 0.8298 | 0.8738 | 0.8817 | +0.79%p | **+5.19%p** |
| HARD | F1-macro | 0.8124 | 0.8618 | 0.8705 | +0.87%p | **+5.81%p** |

### 9.2 난이도가 높을수록 개선폭이 큰 이유

세 난이도 모두 AFTER가 BEFORE를 앞섰지만, 개선폭은 EASY(+1.82%p) < MID(+2.43%p) < HARD(+5.19%p) 순으로 뚜렷하게 커졌습니다. 이는 다음 두 가지로 해석할 수 있습니다.

1. **여유 공간(headroom)의 차이**: BEFORE 기준 EASY는 이미 0.94의 높은 정확도로 상한에 근접해 있었던 반면, HARD는 0.83으로 개선 여지가 컸습니다.
2. **문제 진단의 정합성**: HARD에서만 유일하게 백본을 교체(ResNet18→EfficientNet-B0)한 것은 "표현력 부족"이라는 진단에 따른 것이었고, 이 진단이 실제로 가장 큰 개선(+5.19%p)으로 이어졌다는 점에서, **정량·XAI 진단에 기반한 처방이 실제로 유효했다**는 근거가 됩니다.

### 9.3 클래스별 관점 — cat이 항상 가장 어려운 클래스였다

BEFORE 기준 세 난이도 모두에서 cat 클래스의 recall이 가장 낮았습니다(EASY 0.8150, MID 0.7840, HARD 0.7240 — 난이도가 높을수록 더 낮음). AFTER에서 클래스 가중치와 MixUp/CutMix를 도입한 결과:

- HARD cat recall: 0.7240(BEFORE) → 0.8050(AFTER 표준) → **0.8210(AFTER TTA)**
- MID cat recall: 0.7840(BEFORE) → 0.8330(AFTER 표준) → **0.8500(AFTER TTA)**
- EASY cat recall: 0.8150(BEFORE) → 0.8230(AFTER 표준) → **0.8460(AFTER TTA)**

세 난이도 모두 cat recall이 가장 크게 개선된 지표 중 하나로, 클래스 가중치 도입이 의도한 방향으로 작동했음을 확인할 수 있습니다. 다만 cat의 precision은 여전히 dog·not 클래스보다 낮은 경향이 있어(예: HARD cat precision 0.7994), cat과 dog 간의 혼동은 완전히 해소되지는 않았습니다.

### 9.4 TTA의 일관된 소폭 기여

TTA는 세 난이도 모두에서 표준 평가 대비 추가로 +0.64%p ~ +0.79%p의 정확도 개선을 제공했습니다. 개선폭 자체는 크지 않지만 방향이 일관되게 긍정적이었다는 점에서, [4.1절](#41-공통-개선-사항-easy--mid--hard-공통-적용)에서 인용한 Shanmugam et al.의 지적("TTA가 항상 이득은 아니다")과 달리 본 실험에서는 3-view의 보수적인 TTA 구성이 리스크 없이 안정적인 이득을 준 것으로 해석됩니다.

### 9.5 XAI 시각화 관점의 정직한 해석

`{EASY,MID,HARD}_GRAD.png`(클래스별 평균 Grad-CAM)를 BEFORE와 AFTER 사이에서 육안으로 비교했을 때, 세 난이도 모두 활성화가 이미 피사체 중심부에 집중되는 유사한 패턴을 보였으며, AFTER에서 정량적 성능이 개선된 것에 비해 **평균 활성화 맵 자체의 극적인 시각적 변화는 크지 않았습니다.** 이는 이번 개선이 "모델이 보는 위치"보다는 "결정 경계의 안정성·일반화"(정규화, 증강, 클래스 가중치) 쪽에 더 크게 기여했을 가능성을 시사하며, 개별 오답 샘플 단위([`{난이도}_오답샘플_앙상블.png`])에서의 변화를 더 세밀하게 추적하는 것이 향후 과제로 남습니다.

---

## 10. 결과 및 그래프

> 아래는 실제 이미지 삽입 위치와, 각 위치에 들어갈 CDN 결과 파일명을 정리한 표입니다. 실제 이미지 파일은 저장소의 `assets/before/`, `assets/after/` 등에 배치한 뒤 마크다운 이미지 문법(`![설명](경로)`)으로 교체해 주세요.

### 10.1 학습 안정성 (Learning Curves)

| 설명 | 파일명 |
|---|---|
| BEFORE 난이도별 Train/Val Loss·Accuracy 곡선 (2행×3열) | `before/BEFORE_CURVE.png` |
| AFTER 난이도별 Train/Val Loss·Accuracy 곡선 (2행×3열) | `after/AFTER_CURVE.png` |
| BEFORE 난이도별 Val Loss/Val Accuracy 중첩 비교 | `before/BEFORE_VAL.png` |
| AFTER 난이도별 Val Loss/Val Accuracy 중첩 비교 | `after/AFTER_VAL.png` |
| AFTER Cosine Warmup 학습률 스케줄 (EASY/MID/HARD) | `after/AFTER_LR.png` |

### 10.2 정량적 성능 비교

| 설명 | 파일명 |
|---|---|
| BEFORE 난이도별 Accuracy/F1-macro/F1-weighted 바차트 | `before/BEFORE_BASE.png` |
| BEFORE vs AFTER 난이도별 성능 비교 바차트(델타 포함) | `after/AFTER_BASE.png` |
| {EASY,MID,HARD} BEFORE Confusion Matrix + Classification Report | `before/{EASY,MID,HARD}_정량적평가.png` |
| {EASY,MID,HARD} AFTER 표준 평가 Confusion Matrix + Report | `after/{EASY,MID,HARD}_정량적평가.png` |
| {EASY,MID,HARD} AFTER TTA 평가 Confusion Matrix + Report | `after/{EASY,MID,HARD}_TTA평가.png` |

### 10.3 난이도별 XAI 정성 분석

| 설명 | 파일명 |
|---|---|
| {EASY,MID,HARD} 정답 샘플 5종 XAI 비교 (3세트씩) | `{before,after}/{EASY,MID,HARD}_정답샘플{1,2,3}.png` |
| {EASY,MID,HARD} 오답 샘플 5종 XAI 비교 (3세트씩, 핵심 분석) | `{before,after}/{EASY,MID,HARD}_오답샘플{1,2,3}.png` |
| {EASY,MID,HARD} 오답 샘플 앙상블 CAM 집중 분석 | `{before,after}/{EASY,MID,HARD}_오답샘플_앙상블.png` |
| {EASY,MID,HARD} 클래스별 평균 Grad-CAM 활성화 맵 | `{before,after}/{EASY,MID,HARD}_GRAD.png` |

---

## 11. 최종 회고 및 성찰

### 11.1 가설은 지지되었는가

이 프로젝트의 출발점이었던 가설 — "난이도에 따라 실패 원인이 다르므로 처방도 달라야 한다" — 은 결과로 뒷받침되었습니다. 특히 진단(BEFORE의 과적합·표현력 한계 분석) → 처방(HARD만 백본 교체 + 최대 강도 정규화) → 검증(HARD가 가장 큰 개선폭)으로 이어지는 흐름이 일관되게 나타난 점이 고무적이었습니다.

### 11.2 한계와 자기비판

- **Ablation 부재**: HARD의 개선이 "백본 교체" 때문인지 "증강·정규화 강화" 때문인지를 분리하는 ablation 실험을 수행하지 못했습니다. 현재 결과만으로는 두 요인의 기여도를 정량적으로 나눌 수 없습니다.
- **XAI의 정량적 근거 부족**: 5가지 XAI 기법을 적용했지만 결과 해석은 대부분 육안 관찰(qualitative)에 의존했습니다. [9.5절](#95-xai-시각화-관점의-정직한-해석)에서 밝혔듯, 평균 활성화 맵의 시각적 변화가 정량적 성능 개선만큼 뚜렷하지 않아, "XAI가 곧 성능 개선을 설명한다"고 과장할 수 없었습니다.
- **지표 해석의 함정 학습**: MixUp/CutMix 적용 시 train accuracy가 실제 학습 진행도를 과소평가한다는 점([8.5절](#8-주요-트러블슈팅))을 뒤늦게 발견했습니다. 이는 "지표가 이상하게 보인다고 반드시 버그는 아니다"라는 교훈을 주었습니다.
- **난이도 정의 자체의 사전 큐레이션 의존**: EASY/MID/HARD 구분이 데이터셋 제공 시점에 이미 정해져 있었고, 이 구분이 "정말 난이도를 잘 반영하는가"에 대한 자체 검증(예: 사람이 라벨링한 난이도와의 일치도 등)은 수행하지 못했습니다.

### 11.3 배운 점

- 성능이 낮은 부분에 자원을 집중하는 것이 항상 정답은 아니며, "왜 낮은가"에 대한 진단이 선행되어야 처방의 정합성을 확보할 수 있다는 점을 재확인했습니다.
- 동일한 코드베이스(XAI 파이프라인, 평가 함수)를 재사용하면서 아키텍처(ResNet18 ↔ EfficientNet)만 바꾸려면, 애초에 레이어 접근 로직을 추상화해두는 설계([8.3절](#8-주요-트러블슈팅))가 유지보수성에 크게 기여한다는 것을 체감했습니다.

---

## 12. 향후 발전 계획

1. **Ablation Study**: HARD에서 (a) 백본만 교체, (b) 증강만 강화, (c) 둘 다 적용의 3가지 조합을 분리 실험하여 각 요인의 순수 기여도를 정량화.
2. **다른 아키텍처와의 비교**: HARD에 ConvNeXt, Vision Transformer(ViT) 등 최신 백본을 추가로 비교하여 EfficientNet-B0가 최선의 선택이었는지 검증.
3. **Class-Balanced Loss 정식 도입**: 현재의 단순 역빈도 가중치 대신, Cui et al. (arXiv:1901.05555)이 제안한 "유효 샘플 수(effective number)" 공식 `(1-β)/(1-β^n)`을 직접 구현해 비교.
4. **XAI 정량 지표 도입**: Insertion/Deletion metric, Pointing Game 등 정량적 XAI 평가 지표를 추가하여, [11.2절](#11-최종-회고-및-성찰)에서 지적한 "육안 관찰 의존" 한계를 보완.
5. **오답 샘플 재검수**: 특히 HARD의 오답 샘플 중 일부가 실제로 라벨링 오류이거나 지나치게 모호한 이미지인지 사람이 직접 재검수하는 human-in-the-loop 절차 도입.
6. **경량화 및 배포**: 실제 서비스 적용을 가정해 ONNX 변환, 모바일向 양자화(quantization) 등을 통한 추론 속도·모델 크기 최적화 실험.

---

## 참고 문헌 (arXiv 링크 모음)

| 기법 | 논문 | 링크 |
|---|---|---|
| ResNet | Deep Residual Learning for Image Recognition | https://arxiv.org/abs/1512.03385 |
| EfficientNet | Rethinking Model Scaling for CNNs | https://arxiv.org/abs/1905.11946 |
| RandAugment | Practical automated data augmentation | https://arxiv.org/abs/1909.13719 |
| Random Erasing | Random Erasing Data Augmentation | https://arxiv.org/abs/1708.04896 |
| MixUp | Beyond Empirical Risk Minimization | https://arxiv.org/abs/1710.09412 |
| CutMix | Regularization Strategy with Localizable Features | https://arxiv.org/abs/1905.04899 |
| Label Smoothing | When Does Label Smoothing Help? | https://arxiv.org/abs/1906.02629 |
| Class-Balanced Loss | Based on Effective Number of Samples | https://arxiv.org/abs/1901.05555 |
| AdamW | Decoupled Weight Decay Regularization | https://arxiv.org/abs/1711.05101 |
| SGDR | Stochastic Gradient Descent with Warm Restarts | https://arxiv.org/abs/1608.03983 |
| LR Warmup | Accurate, Large Minibatch SGD | https://arxiv.org/abs/1706.02677 |
| Gradient Clipping | On the difficulty of training RNNs | https://arxiv.org/abs/1211.5063 |
| TTA | Better Aggregation in Test-Time Augmentation | https://arxiv.org/abs/2011.11156 |
| Grad-CAM | Visual Explanations from Deep Networks | https://arxiv.org/abs/1610.02391 |
| Grad-CAM++ | Generalized Gradient-based Visual Explanations | https://arxiv.org/abs/1710.11063 |
| Score-CAM | Score-Weighted Visual Explanations | https://arxiv.org/abs/1910.01279 |
| Occlusion Sensitivity | Visualizing and Understanding CNNs | https://arxiv.org/abs/1311.2901 |
| LIME | Why Should I Trust You? | https://arxiv.org/abs/1602.04938 |
