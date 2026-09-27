# CAT vs DOG vs NOT Image Classification 
### "어디서 틀리는가"를 진단한 개선 vs 모델·증강 일괄 확대

> 같은 데이터 · 학습 레시피 · 예산에서, **오류 방향을 진단하고 그 지점만 겨냥한 개선(Targeted)** 과 **집계 지표만 보고 모델·증강을 일괄로 키운 개선(Uniform)** 은 정말 다른 결과를 내는가?

> 이 질문에 답하기 위해 not 클래스가 개·고양이와 닮은 정도에 따라 **EASY · MID · HARD** 세 난이도를 만들고, 세 방법을 같은 조건(데이터 분할 · 학습 레시피 · 조기 종료 규칙)에서 비교했습니다.

- **Baseline** — 오류를 "실제 → 예측" 방향별로 분해해, 난이도와 함께 급증하는 방향을 진단
- **Targeted** — 진단된 방향만 겨냥해 처방
- **Uniform** — 전체 정확도만 보고 모델 · 증강을 일괄 확대

> 판정 기준은 전체 정확도가 아니라 **겨냥한 방향(동물 ↔ not)에서만 차이가 나는가**이며, 그 차이가 처방 때문인지 모델 때문인지는 요인 분해 · 시드 반복 · 학습률 검증으로 따로 확인했습니다.
---
📌
## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [데이터 출처 및 전처리](#3-데이터-출처-및-전처리)
4. [핵심 구현 및 최적화](#4-핵심-구현-및-최적화)
5. [기술 스택](#5-기술-스택)
6. [폴더 구조](#6-폴더-구조)
7. [개선 근거](#7-개선-근거)
8. [주요 트러블슈팅](#8-주요-트러블슈팅)
9. [결과 해석](#9-결과-해석)
10. [시각화](#10-시각화)
11. [최종 회고 및 성찰](#11-최종-회고-및-성찰)
12. [향후 발전 계획](#12-향후-발전-계획)
13. [참고 문헌](#13-참고-문헌)

---

## 1. 프로젝트 개요

> 💡 **3줄 요약 (전문 용어 없이)**
> 1. **무엇을 했나** — 고양이·개·그 외 사진을 구분하는 모델이 **어디서 어떻게 틀리는지** 먼저 분석하고, 그 부분만 골라 고친 방법과 "모델을 통째로 키우는" 흔한 방법을 똑같은 조건에서 비교했습니다.
> 2. **무엇을 알았나** — 쉬운·중간 난이도에서는 두 방법의 성능이 통계적으로 같았지만, 가장 어려운 문제(곰·늑대 등 개·고양이와 닮은 동물이 섞인 경우)에서는 분석해서 고친 방법이 확실히 더 좋았고, 모델 크기는 약 5분의 1이었습니다.
> 3. **그런데 왜였나** — 원인을 가리는 추가 실험을 해 보니, 이긴 이유는 "골라서 고친 학습 방식"이 아니라 **함께 고른 모델(EfficientNet-B0)** 에 있었습니다. 처음 세운 가설을 스스로 검증해 바로잡은 과정까지가 이 프로젝트의 핵심입니다.

### 1.1 문제의식

"cat / dog / 그 외(not)" 3-클래스 분류에서 not 클래스를 **cat·dog와 얼마나 닮았는지**에 따라 EASY·MID·HARD 세 난이도로 나누면, 난이도가 오를수록 accuracy·macro-F1이 고르게 떨어지는 것처럼 보입니다. 이때 흔한 대응은 "모델을 키우고 증강을 세게" 하는 것입니다. 하지만 오류를 **방향(실제 → 예측)** 으로 쪼개 보면 난이도와 함께 커지는 것은 **동물 ↔ not 경계뿐**이고, cat ↔ dog 혼동은 난이도와 무관합니다. 그렇다면 그 경계만 겨냥한 처방이 더 효율적이지 않을까요?

**실무적 의미** — "대상 클래스 + 그 외(reject) 클래스" 구조는 반려동물 앱, 콘텐츠 필터, 불량 검출처럼 **닮은 비대상을 걸러내야 하는 분류기** 전반에 나타납니다. 이런 시스템에서는 전체 정확도보다 "어떤 오류가 어디로 새는가"가 서비스 품질을 좌우하므로, 이 프로젝트는 ① 오류를 방향으로 진단하는 방법, ② 처방 효과를 교란 없이 검증하는 실험 설계, ③ 같은 성능을 더 적은 연산으로 얻는 선택지를 다룹니다.

### 1.2 실험 설계 (4개 노트북)

| 순서 | 노트북 | 역할 | 처방의 입력 |
|---|---|---|---|
| 1 | `01_baseline_diagnosis` | 통제된 출발점(B) 학습 + **validation 전용** 오류 진단 | — |
| 2 | `02_targeted_improvement` | 진단이 지목한 방향만 겨냥한 개선(**T**) | 방향별 오류율 · 확신도 · 일반화 격차 · XAI |
| 3 | `03_uniform_scaleup` | 집계 지표만 보고 일괄 확대(**U**) + **3-arm test 최종 비교** | accuracy · F1 · 클래스별 P/R만 |
| 4 | `04_ablation_seeds_robustness` | 01~03 결론의 약점 검증 (요인 분해 · 시드 · lr 공정성) | **사전 등록**된 가설 7개 |

#### 📖 먼저 알아둘 용어

| 용어 | 뜻 |
|---|---|
| **B / T / U** | 비교한 세 가지 방법(arm). **B** = 출발점(Baseline), **T** = 오류를 진단해 겨냥한 개선(Targeted), **U** = 모델·증강을 일괄로 키운 개선(Uniform) |
| **방향** | "실제 → 예측" 오류의 종류. 예: `cat→not` = 실제 고양이를 "그 외"로 틀린 경우. 3개 클래스라 총 6방향 |
| **겨냥 방향** | 진단 결과 난이도와 함께 급증해 T가 고치려고 한 방향(동물↔not). 여러 개를 묶어 **겨냥 방향군**이라고 부름 |
| **대조 방향** | 일부러 겨냥하지 않은 방향(cat↔dog). 여기서는 T와 U가 비슷해야 "차이가 겨냥한 곳에서만 났다"고 말할 수 있음 |
| **1,000장당 오류 건수** | test 이미지 1,000장 중 해당 방향(군)으로 틀린 장수. 낮을수록 좋음 |
| **95% CI** | 신뢰구간. 예: 차이가 −25.0 (−34.8, −15.0)이면 구간 전체가 0보다 작으므로 "우연이 아니라 실제로 줄었다"는 뜻 |
| **n.s.** | not significant. 통계적으로 의미 있는 차이가 없음 |
| **EASY / MID / HARD** | "그 외" 클래스가 개·고양이와 얼마나 닮았는지에 따른 세 난이도 (3.1절) |

### 1.3 한눈에 보는 결론 (TL;DR)

| 구분 | 결과 |
|---|---|
| EASY · MID | T와 U 모두 겨냥 방향 오류를 baseline 대비 크게 줄였지만(EASY 17.1 → 8.0 / 8.9, MID 67.9 → 49.8 / 50.6, test 1,000장당), **T와 U 사이에 유의한 차이 없음** |
| HARD | T가 겨냥 방향군 오류를 **122.6 → 97.6 (−25.0, 95% CI [−34.8, −15.0])** 로 U보다 유의하게 줄임. 대조 방향(cat↔dog)은 차이 없음 → **방향 특이적** |
| 견고성 (04) | HARD의 T < U는 **시드 3개 × 3개 = 9/9 조합에서 부호 일치**, 두 처방의 lr을 validation으로 각각 조정한 뒤에도 유지 |
| 원인 (04) | 2×2 요인 분해 결과 차이의 원인은 **백본(EfficientNet-B0, −23.8)** 이며, **진단 기반 증강 정책(−1.2, n.s.)** 과 **혼동 쌍 한정 혼합(−0.5, n.s.)** 은 기여가 확인되지 않음 |
| 효율 | HARD의 T는 U보다 **파라미터 5.5배 적고(4.34M vs 24.03M), 연산량 약 10배 적은(0.39 vs 4.09 GFLOPs)** 모델로 더 높은 macro-F1(0.865 vs 0.835) 달성. EASY·MID의 T는 ResNet18로 U(ResNet50)와 통계적으로 같은 성능을 **이미지당 연산량 약 56% 적게** 달성 |

> **한 줄 요약** — "진단이 겨냥한 방향에서 차이가 난다"는 현상은 HARD에서 재현 가능하게 확인됐지만, 그 차이를 만든 것은 겨냥한 증강이 아니라 **백본 선택(EfficientNet-B0)** 이었습니다. 이 프로젝트는 스스로 세운 인과 설명을 사전 등록 실험으로 반증한 기록이기도 합니다.

![targeted directions](experiments/final_comparison/figures_v2/targeted_directions.png)
<sub>겨냥 방향별 test 오류율 (회색 B · 파랑 T · 주황 U). EASY·MID에서는 T≈U, HARD의 cat→not·dog→not에서만 T가 유의하게 낮습니다.</sub>

### 1.4 이 프로젝트가 지킨 원칙

- **공정 비교** — 세 arm은 `SHARED_RECIPE`(옵티마이저·스케줄러·손실·예산·조기 종료·분할)를 SHA-256 지문으로 공유하며, 지문이 다르면 실행을 거부합니다. arm 간 차이는 `ARM_SPEC`(모델·증강)뿐입니다.
- **누수 차단** — train↔test 완전·준중복을 해시로 찾아 train 쪽에서 제거하고, 파생 사본은 그룹 단위로 묶어 train/val에 동시에 들어가지 않게 했습니다.
- **test 규율** — 진단·처방·모델 선택은 validation만 사용. test는 전 과정에서 **2회**(03 최종 비교, 04 사전 등록 검증)만 열렸고 모두 `test_access_log.jsonl`에 기록됩니다.
- **사전 등록** — 처방(02·03)과 가설·판정 규칙(04)을 학습 전에 지문으로 잠가, 결과를 본 뒤 바꾸는 사후 선택을 막았습니다.

---

## 2. 시스템 아키텍처

### 2.1 전체 파이프라인

```
                      ┌──────────────────────────────────────────────────────────┐
 dataset.zip ───────► │ C4 데이터 준비 (01에서 1회 생성, 02~04는 검증 후 재사용)      │
 (easy/mid/hard ×     │  MD5 · dHash64/256 → train↔test (준)중복 제거 → 그룹 층화 분할  │
  train/test ×        │  → experiments/_shared/{data_audit, splits}                 │
  cat/dog/not-*)      └──────────────────────────────┬───────────────────────────┘
                                                     │  split_signature
                    ┌────────────────────────────────▼───────────────────────────────┐
                    │ C5~C6 공통 학습 엔진 (SHARED_RECIPE 지문 · 재개 가능 · EASY→MID→HARD) │
                    └───────┬───────────────────────┬───────────────────────┬────────┘
                            │                       │                       │
               ┌────────────▼──────────┐ ┌──────────▼───────────┐ ┌─────────▼──────────┐
               │ 01 Baseline (B)        │ │ 02 Targeted (T)       │ │ 03 Uniform (U)      │
               │ ResNet18               │ │ 진단 근거 자동 검증     │ │ 집계 지표만 사용       │
               │ → val 오류 진단 + XAI   │─►│ → 처방 잠금 → 학습      │ │ → 처방 잠금 → 학습     │
               │   diagnosis.json       │ │   (쌍 지정 혼합·EMA 등) │ │   (ResNet50·RandAug)  │
               └────────────┬──────────┘ └──────────┬───────────┘ └─────────┬──────────┘
                            │     aggregate만 ──────────────────────────────►│
                            └───────────────────────┼───────────────────────┘
                                                    ▼
                    ┌───────────────────────────────────────────────────────────────┐
                    │ 03 최종 비교: test 1회 개방 → 방향별 오류 · paired bootstrap ·     │
                    │ McNemar · Holm · 가드(G1~G3) · 대조 방향 · 방향군 특이성            │
                    └───────────────────────────────┬───────────────────────────────┘
                                                    ▼
                    ┌───────────────────────────────────────────────────────────────┐
                    │ 04 확인 실험: 사전 등록 잠금 → 14개 run 학습 → val로 lr 선택 잠금 →   │
                    │ test vault(유일한 접근 지점) → 가설 H4~H10 판정                    │
                    └───────────────────────────────────────────────────────────────┘
```

### 2.2 모델 구조

모든 arm은 **torchvision `IMAGENET1K_V1` 사전학습 백본 + 동일한 분류 헤드**를 사용합니다.

```
Backbone (pretrained) ─► Linear(in_f, 256) ─► BatchNorm1d ─► ReLU ─► Dropout(0.5) ─► Linear(256, 3)
```

| 백본 | 사용처 | 파라미터(헤드 포함) | 백본 GFLOPs (224px) | ImageNet top-1 (torchvision V1) |
|---|---|---|---|---|
| ResNet18 | B 전체, T의 EASY·MID | 11.31M | 1.81 | 69.8% |
| ResNet50 | U 전체 | 24.03M | 4.09 | 76.1% |
| EfficientNet-B0 | T의 HARD | 4.34M | 0.39 | 77.7% |

### 2.3 arm별 개입 비교

| 개입 | B (01) | T (02) EASY | T (02) MID | T (02) HARD | U (03) 공통 |
|---|---|---|---|---|---|
| 백본 | ResNet18 | ResNet18 | ResNet18 | **EfficientNet-B0** | **ResNet50** |
| Random Erasing | — | p 0.5, 면적 0.02–0.4 | 동일 | 동일 | p 0.25, 면적 0.02–0.33 |
| CutMix | — | ✓ **혼동 쌍 한정** | ✓ 쌍 한정 | ✓ 쌍 한정 | ✓ 무작위 짝 |
| MixUp | — | — | ✓ **쌍 한정** | ✓ 쌍 한정 | ✓ 무작위 짝 |
| RandAugment | — | — | — | — | N=2, M=9 |
| EMA | — | 0.999 | 0.999 | 0.999 | 0.999 |
| 겨냥 방향 | — | cat→not, dog→not | + not→cat | + not→dog | (없음) |

> **혼동 쌍 한정**: 혼합 상대를 `cat↔not`, `dog↔not`으로만 제한합니다. 혼합 확률·α·예산은 U와 같으므로, MID·HARD에서 T와 U의 혼합 증강 차이는 "어느 경계를 규제하는가" 하나뿐입니다(EASY의 T는 CutMix만 사용). 단, 백본·RandAugment·Random Erasing 강도는 arm마다 다르므로 T−U 전체 차이는 04의 요인 분해로 따로 분리했습니다.

---

## 3. 데이터 출처 및 전처리

### 3.1 데이터 출처

Kaggle에 공개된 **CIFAR-10의 cat / dog 클래스**와 **CIFAR-100의 여러 클래스**를 조합한 3-클래스 데이터셋입니다. 모든 이미지는 **32×32 RGB**(총 95,400장, 읽기 실패 0장)입니다. <!-- TODO: Kaggle 데이터셋 URL -->

| 클래스 | 유사도 | 정의 | 예시 |
|---|---|---|---|
| **not-easy** | 낮음 | cat/dog와 시각적으로 완전히 다른 무생물·일상 사물 | 버스(bus), 자동차(car), 침대(bed), 컵(cup), 시계(clock) |
| **not-mid** | 중간 | 같은 생명체 범주이지만 외형이 확연히 달라 혼동 가능성이 상대적으로 낮음 | 곤충(spider 등), 물고기(flatfish, shark 등), 사람(man, woman 등) |
| **not-hard** | 높음 | 털·체형·자세에서 cat/dog와 공통점이 많아 혼동 가능성이 높음 | 곰(bear), 늑대(wolf), 호랑이(tiger), 사자(lion), 여우(fox) |

- cat·dog는 세 난이도에서 **동일한 이미지**(train 5,000장 · test 1,000장씩, MD5 기준 100% 일치)이므로, 난이도 간 차이는 오직 not 클래스에서 옵니다.
- not의 원본 규모는 EASY 25,000 / MID 13,500 / HARD 11,000장(train)으로, CIFAR-100 클래스당 train 500장 구성을 가정하면 각각 약 50 / 27 / 22개 클래스에 해당합니다(추정).

### 3.2 분할 결과 (seed 42, val 20%, 그룹 단위 층화)

| 난이도 | train | val | test | 제외 | 클래스 비율 (not : cat) |
|---|---|---|---|---|---|
| EASY | 27,799 | 6,950 | 7,000 | 251 | 4.96 : 1 |
| MID | 18,764 | 4,691 | 4,700 | 45 | 2.70 : 1 |
| HARD | 16,767 | 4,192 | 4,200 | 41 | 2.20 : 1 |

test는 원본 test 폴더를 **그대로** 사용합니다(cat 1,000 / dog 1,000 / not 5,000·2,700·2,200). 누수 제거는 test를 건드리지 않고 train 쪽에서만 수행합니다.

### 3.3 전처리 파이프라인

| 단계 | 처리 | 이유 |
|---|---|---|
| ① 구조 검증 | `{easy,mid,hard}/{train,test}/{cat,dog,not-*}` 폴더 구성이 다르면 즉시 중단 | ImageFolder 알파벳 순서(cat=0, dog=1, not=2)가 어긋나면 라벨이 조용히 뒤바뀜 |
| ② 해시·캐시 | 멀티프로세스로 원본 바이트 **MD5**, 흑백 **dHash 64/256bit** 계산, JPEG(q95) 캐시 zip을 Drive에 보관 | 한 번만 계산하고 세 arm이 같은 캐시를 사용 → 비교 조건 동일 |
| ③ train↔test 누수 | MD5 일치(완전중복) 또는 dHash64 해밍 ≤ 4 **그리고** dHash256 해밍 ≤ 32(준중복)인 train 파일 제외 | 재인코딩·리사이즈 사본은 MD5로 못 잡음. 두 해상도 동시 조건으로 오탐 감소 |
| ④ 그룹화 | 완전중복 · 준중복 · 파생 파일명(`_aug`, `_flip`, `(1)` 등)을 Union-Find로 한 그룹에 묶음 | 같은 원본의 사본이 train/val에 갈라지면 val이 낙관적으로 편향됨 |
| ⑤ 라벨 충돌 제외 | 한 그룹에 서로 다른 클래스가 섞이면 학습에서 제외 | 같은 이미지에 다른 정답 → 노이즈 라벨 |
| ⑥ 그룹 층화 분할 | 클래스별로 그룹 단위 셔플 후 20%를 val로 | 클래스 비율 유지 + 그룹 누수 방지. 분할은 01에서 **한 번만** 생성 |
| ⑦ 입력 변환 | 32×32 → 224×224, ImageNet 평균/표준편차 정규화(고정값) | ImageNet 사전학습 필터의 스케일에 맞춤. 정규화 통계에 val/test 정보가 들어가지 않음 |

**학습 증강(세 arm 공통 기반)**: `RandomResizedCrop(224, scale 0.7–1.0)` · `HorizontalFlip(0.5)` · `ColorJitter(0.3, 0.3, 0.3, 0.08)` → arm별 추가 증강(2.3절)
**평가**: `Resize(224, 224)` 단일 뷰가 주 지표이며, TTA(좌우반전 + center crop 평균)는 보조 지표로만 보고합니다.

### 3.4 누수 감사 결과

| 난이도 | train↔test 완전중복 | 준중복(추가) | 라벨 충돌 파일 | train 내부 다중 그룹 | 제외 합계 |
|---|---|---|---|---|---|
| EASY | 1 | 169 | 103 | 93 | 251 |
| MID | 6 | 34 | 4 | 86 | 45 |
| HARD | 1 | 38 | 0 | 81 | 41 |

MD5만 썼다면 EASY 누수의 **99%(170건 중 169건)** 를 놓쳤을 것입니다.

> ⚠️ 한계: EASY에서 103장이 하나의 그룹(not 98 · dog 4 · cat 1)으로 묶여 라벨 충돌로 판정됐습니다. 확인 결과 일몰·수평선·어두운 배경의 중앙 피사체처럼 **밝기 구조만 비슷한 서로 다른 이미지**였습니다. 32×32에서는 dHash의 정보량이 작아 충돌이 생기고(예: `f0f0f0f0f0f0f0f0` 한 값에 13장), Union-Find의 전이적 결합이 이를 키웠습니다. 제외된 train은 49장(EASY 학습 풀의 0.2% 미만)으로 결과에 영향을 줄 규모는 아니며, 개선안은 12장에 정리했습니다. 04에서 학습·분석 전마다 분할 무결성(경로·그룹·MD5 교차 없음)을 다시 검증했습니다.

---

## 4. 핵심 구현 및 최적화

### 4.1 통제된 비교를 코드로 강제

- **레시피 지문**: `SHARED_RECIPE`를 정렬된 JSON으로 직렬화해 SHA-256 앞 16자리를 지문으로 사용(`d82dd3e7935501e1`). 02·03·04는 01과 지문·분할 서명(`f0cb8ffb8fdfe2c5`)·엔진 버전이 다르면 `PipelineError`로 실행을 거부합니다.
- **근거 조건 자동 검증**: 02의 처방마다 "01 진단에서 이 수치가 성립해야 한다"는 조건을 람다로 붙이고(총 20개), 하나라도 실패하면 학습하지 않습니다. 03도 집계 지표 조건 5개를 같은 방식으로 검증합니다. 결과: **02·03 모두 전 조건 성립**.
- **처방 잠금**: 학습 시작 후 `targets`/`spec`이 바뀌면 거부(`prescription.json`). 04는 가설·판정 규칙·run 목록까지 잠급니다(`preregistration.json`, 잠금 2026-09-23, 수정 0건 → test 개방 2026-09-26).

### 4.2 재개 가능한 학습 엔진 (Colab 런타임 끊김 대응)

| 장치 | 구현 |
|---|---|
| 전체 상태 체크포인트 | 매 epoch 모델 · optimizer · scheduler · AMP scaler · EMA · patience · best · history · RNG(python/numpy/torch/cuda) 저장 |
| 원자적 저장 | 로컬 임시파일 → Drive 임시파일 → `os.replace` + 직전 epoch `.prev` 보관 → 쓰기 도중 끊겨도 기존 파일 무손상 |
| 결정적 재생 | `EpochSampler`(seed·epoch만으로 순서 결정) + 샘플별 시드 `(stage_seed, epoch, index)` → 끊긴 epoch를 같은 배치·같은 증강으로 재실행 |
| 상태 기계 | `running → training_finished → completed`, 완료 단계는 best 파일 SHA-256까지 일치해야 SKIP |
| 안전한 실패 | 설정 불일치 · 체크포인트 손상 · best만 있고 last 없음 → 조용히 이어 붙이지 않고 명확한 오류로 중단 |

→ 23개 run 중 15개 run에서 **16회 세션 재개**가 발생했고, 근사 재개(optimizer 상태 없이 재개)는 **0건**입니다.

### 4.3 GPU 효율화 (Tesla T4 16GB)

- **AMP(fp16) + `channels_last` + cuDNN benchmark + TF32** — 입력 크기가 고정이라 benchmark 이득이 큼
- **micro-batch 자동 탐색** — OOM 없이 올릴 수 있는 최대 micro-batch를 측정하고 부족분은 gradient accumulation으로 채워 **effective batch 64를 모든 run에서 동일하게 유지**(실측: 전 run micro 64 × accum 1). 학습 중 OOM 시 micro를 절반으로 줄이고 마지막 체크포인트에서 자동 재시작
- **평가 텐서 캐시** — val/test 이미지를 uint8 텐서로 한 번만 디코딩해 메모리에 두고, 정규화는 GPU에서 수행 → 매 epoch JPEG 디코딩 제거
- **GPU 혼동행렬 누적** — `torch.bincount(y*K + pred)`로 배치마다 `.cpu()` 동기화 없이 모든 방향 오류율을 매 epoch 계산

### 4.4 진단 지표 (01 → 02 처방의 입력)

| 지표 | 계산 | 판단 대상 |
|---|---|---|
| 방향별 오류율 | 6방향(cat→dog … not→dog), 분모 = 실제 클래스 수 | 난이도와 함께 커지는 경계 |
| 고확신 오류 비율 | 오류 중 예측 확률 ≥ 0.9 비율 | 애매한 경계형 오류인지, 체계적 오류인지 |
| 임계값 이동 최대 이득 | not의 log-확률에 τ ∈ [−1, 1] 가산 시 macro-F1 최대 개선 | 클래스 사전확률 문제인지(→ 재가중으로 해결 가능한지) |
| 일반화 격차 | **증강 없는** train 층화 3,000장 acc − val acc | MixUp 하에서 학습 곡선의 train acc는 왜곡되므로 별도 측정 |
| 흔들림 | warm-up 이후 epoch 간 \|Δ val acc\| 중앙값 | 학습 불안정 |
| best 시점 lr 비율 | best epoch의 lr ÷ 기본 lr | lr이 감쇠되기 전에 조기 종료됐는지 |

### 4.5 혼동 쌍 한정 혼합 (Targeted Mixing)

```python
def _pair_partner_index(y, pairs):              # pairs = [('cat','not'), ('dog','not')]
    allow = torch.zeros(K, K, dtype=torch.bool, device=y.device)
    for a, b in pairs:
        ia, ib = SHORT_CLASS.index(a), SHORT_CLASS.index(b)
        allow[ia, ib] = allow[ib, ia] = True
    m = allow[y][:, y].float()                   # (B, B) 허용 행렬
    empty = m.sum(1) == 0                         # 짝이 없는 샘플은 자기 자신과 짝 → 혼합 없음
    m[empty] = 0.0
    m[empty, torch.nonzero(empty, as_tuple=True)[0]] = 1.0
    return torch.multinomial(m, 1).squeeze(1)     # 허용된 상대 중 무작위 1개
```

배치 단위 `randperm` 대신 허용 행렬에서 상대를 샘플링하므로, **cat↔dog 경계에는 혼합 정규화가 전혀 작용하지 않습니다**(대조 방향 보존).

### 4.6 가중치 EMA

- `torch._foreach_lerp_`로 전 파라미터를 한 번에 갱신, BN running 통계도 같은 decay로 평균
- decay warm-up: `min(0.999, (1+t)/(10+t))` — 초기 무작위 가중치가 평균에 오래 남는 것을 방지
- **val 지표·best 선택은 EMA 가중치 기준**, 원 가중치 지표는 `*_raw`로 함께 기록해 EMA 효과를 분리 관찰
- 비정상(NaN/Inf) loss가 epoch step의 1%를 넘으면 그 epoch을 저장하지 않고 중단

### 4.7 평가 · 통계

| 항목 | 방법 |
|---|---|
| 주 지표 | 겨냥 방향 오류율 / 겨냥 방향군 오류(test 1,000장당 건수) |
| 신뢰구간 | 같은 이미지 재표본을 모든 arm이 공유하는 **paired bootstrap** (방향별 B=5,000, 전체 지표 B=2,000, 04 B=2,000) |
| 검정 | **McNemar**(연속성 보정, 불일치 < 25건이면 정확 이항검정) + **Holm** 다중 비교 보정 (α = 0.05) |
| 오류 이동 가드 | G1 source recall, G2 역방향 오류율, G3 macro-F1이 U보다 유의하게 나쁘지 않아야 "SUPPORTS" (한 방향 오류를 다른 곳으로 밀어낸 모델이 이기는 것을 차단) |
| 특이성 | 겨냥 방향군에서만 T < U이고 대조 방향군(cat↔dog)의 CI는 0을 포함해야 "SPECIFIC" |
| 시드 비교 (04) | 시드 × 이미지 2단계 bootstrap + 모든 T·U 시드 조합의 부호 일치도 |
| 보정 | ECE(15 bins), NLL |

### 4.8 XAI (정성적 보조 자료)

- **5종**: Grad-CAM · Grad-CAM++ · Score-CAM(상위 32채널, 배치) · Occlusion(배치) · LIME(300 샘플)
- **CAM 대상**: ResNet `layer4[-1]` 블록 출력, EfficientNet `features[-1]` / hook은 사용 직후 제거
- **샘플 선택(재현 가능)**: 같은 (실제 → 예측) 조합에서 ① 확신도 순위로 8개 구간 층화 → ② 특징 공간(대상 레이어 평균 풀링) **farthest-point** 선택. 선택 근거(확신도·마진·구간·최소 코사인 거리)를 CSV로 저장
- **비교 패널**: baseline이 틀린 **같은 val 샘플**을 B·T·U로 나란히 표시
- **초기 버전 대비 수정**: CAM 대상을 BN·잔차 합산 이전인 `layer4[-1].conv2`에서 블록 출력으로 교체, 모델 사본마다 누적되던 hook 제거, XAI 샘플을 test가 아닌 val 오분류에서만 선택(진단 단계의 test 유입 차단)
- XAI는 처방을 *고르는* 근거가 아니라 후보를 *지우는* 근거로만 사용했습니다(예: "CAM이 이미 동물 위에 있다" → 주의 영역 교정 처방 배제).

---

## 5. 기술 스택

### Core

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![torchvision](https://img.shields.io/badge/torchvision-F37626?style=flat&logo=pytorch&logoColor=white) ![CUDA AMP fp16](https://img.shields.io/badge/CUDA%20AMP%20fp16-76B900?style=flat&logo=nvidia&logoColor=white) ![ResNet18 / ResNet50](https://img.shields.io/badge/ResNet18%20%7C%20ResNet50-5C6BC0?style=flat) ![EfficientNet-B0](https://img.shields.io/badge/EfficientNet--B0-00897B?style=flat)

### 학습 기법

![MixUp / CutMix](https://img.shields.io/badge/MixUp%20%7C%20CutMix-8E24AA?style=flat) ![RandAugment](https://img.shields.io/badge/RandAugment-6D4C41?style=flat) ![Random Erasing](https://img.shields.io/badge/Random%20Erasing-455A64?style=flat) ![Weight EMA](https://img.shields.io/badge/Weight%20EMA-1565C0?style=flat) ![AdamW + Cosine](https://img.shields.io/badge/AdamW%20%2B%20Cosine-37474F?style=flat)

### 데이터 처리

![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white) ![Pillow](https://img.shields.io/badge/Pillow-3B7BBF?style=flat) ![multiprocessing](https://img.shields.io/badge/multiprocessing-546E7A?style=flat) ![MD5 + dHash LSH](https://img.shields.io/badge/MD5%20%2B%20dHash%20LSH-B71C1C?style=flat)

### 평가 · 통계 · XAI

![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white) ![Bootstrap / McNemar / Holm](https://img.shields.io/badge/Bootstrap%20%7C%20McNemar%20%7C%20Holm-2E7D32?style=flat) ![Grad-CAM / Score-CAM](https://img.shields.io/badge/Grad--CAM%20%7C%20Score--CAM-C62828?style=flat) ![LIME](https://img.shields.io/badge/LIME-AD1457?style=flat) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) ![scikit-image](https://img.shields.io/badge/scikit--image-E65100?style=flat) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)

### 인프라

![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white) ![NVIDIA Tesla T4](https://img.shields.io/badge/NVIDIA%20Tesla%20T4-76B900?style=flat&logo=nvidia&logoColor=white) ![Google Drive](https://img.shields.io/badge/Google%20Drive-0F9D58?style=flat&logo=googledrive&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) ![JSONL Logging](https://img.shields.io/badge/JSONL%20Logging-000000?style=flat&logo=json&logoColor=white)

<details>
<summary><b>세부 용도 보기</b></summary>

| 분류 | 기술 | 이 프로젝트에서의 용도 |
|---|---|---|
| Core | PyTorch · torchvision | 공통 학습 엔진, `IMAGENET1K_V1` 사전학습 백본, AMP(fp16) · `channels_last` · `torch._foreach_lerp_`(EMA) |
| Core | ResNet18 / ResNet50 / EfficientNet-B0 | B · T(EASY·MID) / U / T(HARD) 백본 |
| 학습 기법 | MixUp · CutMix · RandAugment · Random Erasing · EMA | arm별 개입 (혼동 쌍 한정 혼합은 직접 구현) |
| 데이터 처리 | NumPy · pandas · Pillow · multiprocessing | 해시 계산 · JPEG 캐시 · 분할 표 · 산출물 관리 |
| 데이터 처리 | MD5 + dHash(64/256bit) + 5-band LSH + Union-Find | train↔test 누수 감사, 그룹 단위 층화 분할 |
| 평가 · 통계 | scikit-learn · SciPy | `classification_report`, McNemar(χ²) · 정확 이항검정 |
| 평가 · 통계 | paired bootstrap · Holm | 방향별·가설별 신뢰구간과 다중 비교 보정 (직접 구현) |
| XAI | Grad-CAM · Grad-CAM++ · Score-CAM · Occlusion (직접 구현), LIME | 오분류 방향별 설명 맵, 3-arm 비교 패널 |
| 시각화 | Matplotlib (+NanumGothic) · OpenCV · scikit-image | 학습 곡선 · 혼동행렬 · CAM 오버레이 · LIME 경계 |
| 인프라 | Google Colab · Tesla T4 16GB · Google Drive | 23개 run 학습, 체크포인트·결과 영구 보관 |
| 인프라 | JSON / JSONL | 설정 지문 · 메타데이터 · 사전 등록 · test 접근 기록 |

</details>

---

## 6. 폴더 구조

```
CATvsDOGvsNOT/
├── dataset.zip                                  # 원본 데이터 (저장소 미포함)
├── 01_baseline_diagnosis.ipynb
├── 02_targeted_improvement.ipynb
├── 03_uniform_scaleup.ipynb
├── 04_ablation_seeds_robustness.ipynb
└── experiments/                                 # 노트북 실행 산출물 (checkpoints 제외)
    ├── _shared/
    │   ├── data_audit/                          # audit_summary.json, file_manifest.csv,
    │   │                                        # train_test_duplicates_*, label_conflicts_*, unreadable_*
    │   ├── splits/split_{easy,mid,hard}.csv     # 고정 분할 (01이 1회 생성)
    │   ├── dataset_cache_r256.zip               # 캐시 (124.7MB → 저장소 미포함 권장)
    │   └── test_access_log.jsonl                # test 접근 기록 (2회)
    ├── 01_baseline_diagnosis/
    │   ├── {easy,mid,hard}/                     # 단계별 공통 구조 ↓
    │   │   ├── metadata.json                    # 설정 지문 · 세션(재개) 이력 · best SHA-256 · val 지표
    │   │   ├── logs/                            # training_history.csv, train_log.txt
    │   │   ├── evaluation/{val,test}/           # predictions · confusion_matrix · direction_errors · metrics
    │   │   └── figures/                         # training_curve.png, confusion_matrix_{val,test}.png
    │   ├── diagnosis/                           # diagnosis.json, direction_table_val.csv,
    │   │   └── xai/{easy,mid,hard}/             # val_error_samples.csv, diagnosis_overview.png, XAI 패널
    │   └── arm_summary.json
    ├── 02_targeted_improvement/
    │   ├── diagnosis_evidence.json · prescription.json · val_target_check.csv · stability_report_val.csv
    │   ├── {easy,mid,hard}/  xai/{own, before_after}/
    │   └── arm_summary.json
    ├── 03_uniform_scaleup/
    │   ├── prescription.json · {easy,mid,hard}/ · xai/own/ · arm_summary.json
    ├── final_comparison/                        # 03의 3-arm test 비교
    │   ├── FINAL_REPORT.md · final_summary.json
    │   ├── aggregate_metrics_test.csv · target_direction_comparison_test.csv
    │   ├── control_direction_comparison_test.csv · family_specificity_test.csv
    │   ├── direction_errors_all_arms_test.csv · error_reduction_specificity.csv
    │   ├── budget_fairness_audit.csv · training_stability_val.csv
    │   ├── figures/ · figures_v2/ · xai_three_arms/
    └── 04_ablation_seeds/
        ├── ABLATION_REPORT.md · ablation_summary.json
        ├── preregistration.json · selection.json · val_overview.csv
        ├── analysis/                            # hypothesis_verdicts, factorial_*, seeds_*, pairing_*,
        │                                        # mix_weighting_*, lr_*, error_flow, efficiency, test_all_runs
        ├── figures/                             # factorial, seed_robustness, mechanism, error_flow, efficiency
        ├── runs/hard__{name}__s{seed}/hard/     # 새로 학습한 14개 run
        └── test_vault/                          # vault_state.json, predictions/*.npz
```

> ⚠️ `dataset_cache_r256.zip`(124.7MB)은 GitHub 단일 파일 한도(100MB)를 넘습니다. `.gitignore`에 추가하거나 Git LFS를 사용하세요. `checkpoints/`도 같은 이유로 제외했습니다.

### 재현 방법

1. Google Drive의 `MyDrive/CATvsDOGvsNOT/dataset.zip`에 데이터를 둡니다(다른 경로는 환경변수 `CDN_PROJECT_DIR`로 지정).
2. Colab(GPU 런타임)에서 **01 → 02 → 03 → 04** 순서로 "모두 실행"합니다. 03의 마지막 셀이 3-arm test 비교를 수행합니다.
3. 런타임이 끊기면 **같은 노트북을 처음부터 다시 실행**하면 됩니다. 완료된 단계는 SKIP, 끊긴 단계는 다음 epoch부터 이어집니다.
4. CPU 스모크 테스트: `CDN_SMOKE_TEST=1`(64px · 소규모 epoch · 사전학습 없음).
5. 환경: Colab 기본 PyTorch 환경을 사용하며, 누락 패키지(`lime`, `scikit-image`, `opencv-python-headless`, `tqdm`)는 노트북 첫 셀이 자동 설치합니다.

---

## 7. 개선 근거

### 7.1 공통 학습 레시피 (세 arm 동일)

| 결정 | 값 | 근거 |
|---|---|---|
| 사전학습 | torchvision `IMAGENET1K_V1` (세 백본 통일) | ImageNet 정확도와 전이 성능의 강한 상관(Kornblith+ CVPR'19). 구현체·정규화 통계를 통일해 timm 태그 혼용에 따른 교란 제거 |
| 옵티마이저 | AdamW, lr 2e-4, weight decay 1e-2 | 분리된 가중치 감쇠(Loshchilov & Hutter ICLR'19). wd는 PyTorch AdamW 기본값. lr은 사전학습 가중치 미세조정용으로 낮게 설정하고, 04에서 {1e-4, 4e-4}로 공정성 재검증 |
| 스케줄 | 선형 warm-up 3 epoch → cosine (최소 1%) | warm-up으로 초기 불안정 완화(Goyal+ 2017), cosine 감쇠(Loshchilov & Hutter ICLR'17) |
| 손실 | 클래스 가중 CE(`balanced`, train 분할만) + Label Smoothing 0.1 | not이 cat 대비 2.2–5.0배 많음 → 소수 클래스 보정. LS 0.1은 Szegedy+ CVPR'16 기본값 |
| 모델 선택 | val **macro-F1** 최고(동률 시 val loss), patience 10, min_delta 1e-4, 최대 100 epoch | 불균형 데이터에서 accuracy는 다수 클래스에 끌려감 → 클래스 평균 지표로 선택 |
| 배치 · 안정화 | effective batch 64, grad clip 1.0, AMP | T4 메모리 제약 내 최대 배치, 혼합 증강·fp16의 기울기 폭주 방지(Micikevicius+ ICLR'18) |
| 혼합 공통값 | 배치의 50%에 적용, MixUp α 0.4, CutMix α 1.0 | MixUp은 ImageNet 규모에서 α 0.1–0.4 권장(Zhang+ ICLR'18), CutMix α 1.0은 원 논문 기본값(Yun+ ICCV'19) |

### 7.2 02 Targeted — 진단 → 처방 → 근거

| 진단된 실패 유형 (01 val 근거 수치) | 처방 | 근거 문헌 · 설정 이유 |
|---|---|---|
| **A. 비전형·부분 장면에서 동물 → not**: 동물→not 오류 중 고확신(≥0.9) 비율 0.0 / 0.0 / 1.5% → 저확신 경계형 오류. XAI상 가림·근접·흐림 장면에서 실패 | **Random Erasing** p 0.5, 면적 0.02–0.4, 종횡비 0.3–3.3, 무작위 값 | Zhong+ AAAI'20 — 가림에 대한 강건성 향상. 설정값은 논문 §5.1.2 기본값. 공식 구현 `zhunzhong07/Random-Erasing` |
| **B. 과적합 + 부분만 보고 판단하는 능력 부족**: 일반화 격차 5.0 / 9.5 / 14.8%p | **CutMix — 혼동 쌍 한정** | Yun+ ICCV'19 — 국소 영역 단서로도 판단하도록 정규화. "혼동 쌍에서 표본을 뽑아 섞는다"는 착상은 CP-Mix(Yoon+ arXiv:2411.07621) |
| **C. 경계가 양쪽으로 무너짐**: not→cat 1.2 → 4.2%, not→dog 0.6 → 1.7% (EASY→MID) | **MixUp — 혼동 쌍 한정** (MID·HARD만) | Zhang+ ICLR'18 — 클래스 사이를 선형 보간해 결정 경계를 매끄럽게. EASY는 not→동물이 ≤ 2%p라 보류 |
| **D. 표현 한계**: 임계값 이동 이득 +0.0006(HARD) → 기준선 문제 아님, CAM은 이미 동물 위, cat recall 0.844 → 0.817 → 0.730 | **EfficientNet-B0** (HARD만) | Kornblith+ CVPR'19(더 좋은 ImageNet 모델이 전이도 잘함), Tan & Le ICML'19. ResNet50(76.1%)보다 top-1이 높으면서(77.7%) 5.3M·0.39 GFLOPs로 가벼움 |
| **E. 학습 불안정**: best 시점 lr 비율 0.97 / 0.93(MID·HARD) → lr 감쇠 전 조기 종료, val acc 흔들림 0.37 → 0.62 → 1.07pp | **가중치 EMA** decay 0.999 | Polyak & Juditsky 1992; Morales-Brotons+ TMLR'24(EMA는 노이즈를 평균해 lr 감쇠 의존을 줄임). 평균 창 1/(1−0.999) = 1,000 step ≈ 2.3(EASY)–3.8(HARD) epoch |

**처방을 난이도별로 누적한 이유** — EASY는 붕괴가 한쪽(동물→not)에 국한되어 A·B·E만, MID는 not→동물이 올라와 C를 추가, HARD는 네 방향 모두 단조 증가 + 임계값 무효로 D를 추가했습니다.

### 7.3 03 Uniform — 집계 지표 → 관행적 처방

| 집계 지표에서 읽은 것 | 관행적 해석 | 처방 | 근거 문헌 |
|---|---|---|---|
| accuracy 0.950 > 0.890 > 0.835 | 모델 용량 부족 | ResNet18 → **ResNet50** | He+ CVPR'16; Kornblith+ CVPR'19 |
| 격차 5.0 / 9.5 / 14.8%p (전 난이도 ≥ 3%p) | 과적합 → 데이터 다양성 | **RandAugment N=2, M=9** | Cubuk+ NeurIPS'20 (ResNet-50 ImageNet 설정) |
| 동일 | 표준 강한 정규화 | **MixUp + CutMix** (배치 50:50) | Wightman+ 2021(ResNet strikes back); Touvron+ ICML'21(DeiT) |
| 동일 | 가림 정규화 | **Random Erasing p 0.25** | DeiT 설정; Zhong+ AAAI'20 |
| best 시점 lr ≥ 50% (0.52 / 0.97 / 0.93) | 가중치 평균으로 안정화 | **EMA 0.999** | Wightman+ 2021; Morales-Brotons+ TMLR'24 |
| macro-F1 < accuracy | 클래스 불균형 | 클래스 가중치(이미 공통 레시피에 포함) | — |

큰 모델에는 강한 정규화를 함께 쓰는 것이 현대 ResNet 학습 레시피의 관행(ResNet strikes back)이므로, 모델 확대와 강한 증강을 **세 난이도에 똑같이**(난이도 분기 0건) 적용했습니다.

### 7.4 의도적으로 제외한 처방

| 제외 | 제외 근거 |
|---|---|
| 클래스 재가중 추가 · logit 보정(Menon+ ICLR'21) | 임계값 이동 최대 이득 +0.0007 / +0.0012 / +0.0006 → 사전확률(비율) 문제가 아님 |
| 주의 영역 교정 · 배경 제거 | XAI상 CAM이 이미 동물 위에 있음 |
| RandAugment (T에서) | 범용 다양성 증강으로 특정 오류 방향과 연결이 약함 → U의 관행 처방에만 사용 |
| 헤드 · Label Smoothing 변경 | 진단 근거 없음 → 비교 대상 개입만 남기기 위해 baseline과 동일 유지 |

### 7.5 평가 · 누수 감사 방법의 근거

| 방법 | 근거 |
|---|---|
| train↔test 준중복 감사 | CIFAR-10/100 자체에 test와 train의 준중복이 존재함이 보고됨(Barz & Denzler, *J. Imaging* 2020, ciFAIR) |
| dHash | Krawetz(HackerFactor, 2013)의 차분 해시. 리사이즈·재인코딩에 강함 |
| 5-band LSH 후보 탐색 | 64bit를 13bit 밴드 5개로 나누면 해밍 ≤ 4인 쌍은 비둘기집 원리로 반드시 한 밴드를 공유(Manku+ WWW'07의 테이블 분할 아이디어) → 전수 비교 없이 누락 0 |
| McNemar + paired bootstrap | 같은 test 이미지에서 두 분류기를 비교할 때 권장되는 짝지은 검정(Dietterich 1998; Efron & Tibshirani 1993) |
| Holm 보정 | 여러 방향·가설을 동시에 검정할 때의 1종 오류 통제(Holm 1979) |
| 시드 반복 | 학습 무작위성에 따른 분산을 평가 표본 분산과 분리(Bouthillier+ MLSys'21) |
| XAI를 보조 자료로만 사용 | 설명 맵이 모델·데이터와 무관하게 비슷해 보일 수 있음(Adebayo+ NeurIPS'18) |

### 7.6 참고 구현 (GitHub)

| 저장소 | 참고한 부분 |
|---|---|
| [pytorch/vision](https://github.com/pytorch/vision) | 사전학습 가중치 · `RandAugment` · `RandomErasing` · 백본 메타데이터(`_ops`) |
| [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | ResNet strikes back 레시피, EMA · MixUp/CutMix 스위칭 관행 |
| [clovaai/CutMix-PyTorch](https://github.com/clovaai/CutMix-PyTorch) | CutMix 박스 샘플링 · λ 재계산 |
| [facebookresearch/mixup-cifar10](https://github.com/facebookresearch/mixup-cifar10) | MixUp 손실 조합 방식 |
| [zhunzhong07/Random-Erasing](https://github.com/zhunzhong07/Random-Erasing) | Random Erasing 하이퍼파라미터 |
| [tensorflow/tpu (EfficientNet)](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) | EfficientNet 원 구현 |
| [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) | CAM 계열 대상 레이어 선택 관행 |
| [marcotcr/lime](https://github.com/marcotcr/lime) | LIME 이미지 설명 |

---

## 8. 주요 트러블슈팅

실험 과정에서 부딪힌 **핵심 기술 문제 5가지**입니다. 본문의 H4·H7·H9 등은 04에서 사전 등록한 검증 가설이며, 전체 목록과 판정은 9.3절에 있습니다. 각 항목의 "개선 전 → 개선 후" 수치는 모두 `experiments/` 산출물에서 계산한 실측값입니다. 변화율은 **상대 변화(%)** 를, 비율 지표의 절대 차이는 **%p**로 구분해 표기했습니다.

| # | 문제 | 해결 | 핵심 정량 개선 |
|---|---|---|---|
| ① | HARD의 표현 한계 (동물↔not 경계 붕괴) | 백본을 EfficientNet-B0로 교체 | 겨냥 방향군 오류 **약 19% 감소**, 연산량 **약 79% 감소** |
| ② | MID의 경계형 오류 + 과적합 | 혼동 쌍 한정 CutMix·MixUp + Random Erasing | 겨냥 방향군 오류 **약 27% 감소**, macro-F1 **약 2.9% 향상** |
| ③ | 학습 불안정 · 높은 lr에서 조기 종료 | 가중치 EMA | epoch 간 val 흔들림 **약 77–89% 감소** |
| ④ | 혼합 증강 후 not→동물 오류 역증가 | 혼합 배치 전용 손실 분리 (가중치 제거) | not→cat **약 31%**, not→dog **약 42% 감소** (트레이드오프 존재) |
| ⑤ | 고정 lr이 비교 대상(U)에 불리 | val 기반 lr 탐색 + 선택 잠금 | U의 macro-F1 **약 2.0% 향상**, 보정 후에도 결론 유지 |

---

### ① HARD의 표현 한계 — 정규화로는 풀리지 않던 동물↔not 경계 붕괴

**문제 (개선 전)**
01 진단에서 HARD만 동물↔not 네 방향이 모두 단조 증가했고(val cat→not **20.4%**), cat recall은 0.730으로 세 난이도·세 클래스 중 최저였습니다. 02에서 설계한 쌍 지정 증강 처방을 ResNet18에 그대로 적용해도(04의 `Tpol_R18`) test 겨냥 방향군 오류는 **121.0 / 1,000장**, cat→not은 **14.1%** 로 여전히 높았습니다.

**원인 분석**
- not 클래스의 log-확률에 τ ∈ [−1, 1]을 더해 판정 기준을 옮겨도 macro-F1 이득이 **+0.0006**에 그쳤습니다(`threshold_shift_gain`). 클래스 비율(사전확률) 문제가 아니라는 뜻입니다.
- XAI상 CAM은 이미 동물 위에 있었습니다. 즉 "어디를 보는가"가 아니라, 털 질감·체형이 비슷한 **다른 포유류(곰·늑대·여우 등)와 cat/dog를 가르는 표현력**이 부족한 상태였습니다. 이런 문제는 증강·정규화만으로는 해결되지 않습니다.

**해결**
- 공통 엔진의 `BACKBONES` 레지스트리에서 EfficientNet-B0를 선택하고, `cam_target_layer`를 백본별로 분기(`features[-1]`)해 XAI까지 같은 방식으로 적용했습니다. 헤드와 학습 레시피는 그대로 두어 백본 외의 변수는 고정했습니다.
- "더 큰 모델(ResNet50)"이 아니라 **ImageNet 표현 품질이 더 높은 작은 모델**(top-1 77.7%, 5.3M)을 골랐습니다(Kornblith+ CVPR'19).
- 04에서 같은 처방을 ResNet18 · ResNet50 · EfficientNet-B0에 적용해 백본 효과만 분리 검증했습니다(9.3절 H4: 처방은 고정하고 백본만 바꿨을 때의 효과).

**정성적 개선**
- XAI 3-arm 비교 패널에서 baseline이 cat→not으로 틀린 같은 val 샘플을 EfficientNet-B0 모델은 cat으로 바로잡았습니다(예: 확신도 0.84, 0.82).
- 개선이 대조 방향(cat↔dog)이 아닌 **겨냥 방향에 집중**됐습니다(대조 방향군 오류 변화 −3.4%로 사실상 동일).
- 더 작은 모델로 더 좋은 성능을 내, "성능을 올리려면 모델을 키운다"는 관행과 반대 방향의 해법을 얻었습니다.

**정량적 개선 (HARD test, 같은 처방 · 시드 42)**

| 지표 | 개선 전 (ResNet18) | 개선 후 (EfficientNet-B0) | 변화 |
|---|---|---|---|
| 겨냥 방향군 오류 (/1,000장) | 121.0 | **97.6** | **−19.3%** |
| cat→not 오류율 | 14.1% | 9.9% | −4.2%p (−29.8%) |
| dog→not 오류율 | 6.0% | 4.8% | −1.2%p (−20.0%) |
| macro-F1 | 0.8396 | **0.8648** | +2.5%p (**+3.0%**) |
| 파라미터 / 백본 GFLOPs | 11.31M / 1.81 | 4.34M / 0.39 | **−61.6% / −78.7%** |

> baseline(처방 없음) 대비로는 겨냥 방향군 오류 **−26.4%**(132.6 → 97.6), macro-F1 **+6.8%**(0.8100 → 0.8648), cat recall **+13.0%p**(69.9% → 82.9%)입니다. 같은 처방의 ResNet50 대비로도 **−20.7%**(123.1 → 97.6)이며, 백본 주효과는 −23.8 (95% CI −31.3 ~ −15.8)로 유의했습니다.

---

### ② MID의 경계형 오류와 과적합 — 혼동 쌍 한정 혼합 증강

**문제 (개선 전)**
MID baseline은 val에서 cat→not **12.1%**, dog→not **8.0%**, 일반화 격차 **9.5%p**였고, EASY → MID에서 not→cat이 1.2% → 4.2%로 올라 **경계가 양쪽으로 무너지기 시작**했습니다. 동물→not 오류 중 확신도 0.9 이상인 비율은 **0%** 로, 대부분 애매하게 틀리는 경계형 오류였습니다.

**원인 분석**
- 저확신 경계형 오류는 결정 경계 주변의 학습 신호가 부족하다는 뜻이고, XAI상 가림·근접·부분 장면에서 실패가 집중됐습니다.
- 관행적인 MixUp/CutMix는 **모든 클래스 쌍**을 섞습니다. 그러면 난이도와 무관했던 cat↔dog 경계까지 규제하게 되어 정규화의 효과가 문제 지점에 집중되지 않습니다.

**해결**
- Random Erasing(p 0.5, 면적 0.02–0.4) + CutMix + MixUp을 적용하되, `_pair_partner_index`에서 **허용 행렬**(cat↔not, dog↔not)로 혼합 상대를 샘플링했습니다(`torch.multinomial`). 짝이 없는 샘플은 자기 자신과 짝지어 혼합하지 않습니다.
- 혼합 확률(0.5)·α·예산은 U와 같게 두어 "어느 경계를 규제하는가"만 달라지게 했습니다.
- 처방은 01 진단 수치에 대한 근거 조건 20개를 자동 검증한 뒤(`evidence_all_supported: true`) `prescription.json`으로 잠갔습니다.

**정성적 개선**
- 개선이 **겨냥 방향에 집중**됐습니다. MID에서 T가 줄인 오류의 **95.5%** 가 겨냥 방향이었던 반면, U는 겨냥 방향 오류를 비슷하게 줄이면서 비겨냥 방향 오류가 22건 늘었습니다(`error_reduction_specificity.csv`).
- cat·dog recall이 함께 올라(80.9% → 85.7%, 83.2% → 90.2%) 두 동물 클래스의 경계가 모두 안정됐습니다.
- 다만 not→dog가 1.37% → 2.04%로 소폭 늘어, not→동물 방향의 부작용이 처음 관찰됐습니다(→ ④에서 원인 추적).

**정량적 개선 (MID test, 같은 ResNet18 백본)**

| 지표 | 개선 전 (baseline) | 개선 후 (Targeted) | 변화 |
|---|---|---|---|
| 겨냥 방향군 오류 (/1,000장) | 67.9 | **49.8** | **−26.6%** |
| cat→not 오류율 | 13.1% | 7.0% | −6.1%p (−46.6%) |
| dog→not 오류율 | 6.4% | 2.9% | −3.5%p (−54.7%) |
| macro-F1 | 0.8638 | **0.8886** | +2.5%p (**+2.9%**) |
| accuracy | 0.8894 | 0.9083 | +1.9%p (+2.1%) |
| 손실 격차 (val loss − 증강 없는 train loss) | 0.195 | 0.172 | −11.9% |

> 같은 처방을 HARD의 ResNet18에 적용했을 때(백본 고정)는 손실 격차 **−33.1%**(0.324 → 0.217), 정확도 격차 **−16.6%**(14.8%p → 12.4%p), macro-F1 **+3.7%**(0.8100 → 0.8396)였습니다. 이 수치에는 EMA 효과가 함께 포함되며, EMA만의 효과는 ③에서 분리했습니다. 또한 MID에서 T와 U의 전체 성능 차이는 통계적으로 유의하지 않았고(9.2절), 04의 H7(9.3절: 혼합 짝을 진단된 쌍으로 제한한 것과 무작위로 둔 것의 비교)에서 **혼동 쌍 한정 자체의 추가 효과는 확인되지 않았습니다**(HARD, 무작위 짝 대비 −0.5). 따라서 이 개선은 "쌍 한정"보다 **혼합 증강 · Random Erasing · EMA 묶음 전체의 효과**로 해석하는 것이 정확하며, 쌍 한정의 실질적 기여는 개선이 겨냥 방향에 집중된 특이성(95.5%) 쪽에서 관찰됩니다.

---

### ③ 학습 불안정과 높은 lr에서의 조기 종료 — 가중치 EMA

**문제 (개선 전)**
baseline의 MID·HARD는 best epoch 시점의 lr이 기본값의 **97% · 93%** 였습니다. 즉 lr이 감쇠되기도 전에 best가 나오고 10 epoch 뒤 조기 종료(25 · 30 epoch)됐습니다. epoch 간 val acc 흔들림은 난이도를 따라 0.37 → 0.62 → **1.07pp**로 커졌고, 동물→not 오류율은 epoch마다 **2.6–5.8pp**씩 요동쳐 best 모델이 "운 좋은 epoch"에 좌우될 위험이 있었습니다. 혼합 증강을 추가하자 원 가중치의 흔들림은 더 커졌습니다(U HARD 동물→not **9.51pp**).

**원인 분석**
- 높은 lr 구간의 SGD 노이즈에 더해, 혼합 증강은 배치마다 학습 목표를 크게 바꿉니다. 그 결과 val macro-F1이 patience(10) 안에 `min_delta`를 넘지 못해 학습이 일찍 끊겼습니다.

**해결**
- `ModelEMA` 구현: `torch._foreach_lerp_`로 전 파라미터를 한 번에 갱신, BN running 통계도 같은 decay로 평균, decay warm-up `min(0.999, (1+t)/(10+t))`.
- 역전파는 원 가중치로, **val 평가·best 선택은 EMA 가중치**로 수행하고, 원 가중치 지표는 `*_raw` 열로 매 epoch 함께 기록했습니다. 덕분에 **같은 run 안에서** EMA의 전후 효과를 직접 비교할 수 있습니다.
- decay 0.999의 평균 창은 약 1,000 step으로, 이 데이터에서 약 2.3(EASY) – 3.8(HARD) epoch에 해당합니다.

**정성적 개선**
- best epoch가 lr 감쇠 구간으로 이동해 스케줄 전체를 활용하게 됐습니다(baseline → Targeted 처방 전체 기준 MID best 15 → 51, HARD 20 → 44 epoch).
- 학습 곡선에서 EMA 지표(실선)가 원 가중치 지표(점선)보다 훨씬 매끄러워, 조기 종료와 best 선택이 우연에 덜 좌우됩니다(`training_curve.png`).

**정량적 개선 (같은 run의 원 가중치 vs EMA, T·U의 6개 단계, warm-up 이후)**

| 지표 | 개선 전 (원 가중치) | 개선 후 (EMA) | 변화 |
|---|---|---|---|
| epoch 간 val acc 흔들림 (중앙값) | 0.62–1.26pp | 0.07–0.17pp | **−77% ~ −89%** |
| epoch 간 동물→not 오류율 흔들림 (중앙값) | 1.75–9.51pp | 0.20–0.50pp | **−83% ~ −95%** |
| best val macro-F1 | 예: U HARD 0.8111 | 0.8378 | 단계별 **+0.2% ~ +3.3%** |

> 불안정이 가장 컸던 U HARD에서 효과도 가장 컸습니다(흔들림 −87% · −95%, best val macro-F1 +3.3%). T HARD는 흔들림 −77%, best val macro-F1 +0.6%(0.8682 → 0.8737)였습니다.

---

### ④ 혼합 증강 후 not→동물 오류가 오히려 증가 — 손실 가중치와 혼합의 상호작용

**문제 (개선 전)**
02의 val 확인에서 겨냥 방향인 not→cat이 HARD **+1.1%p**, MID +0.3%p로 오히려 악화됐습니다. test에서도 HARD not→cat이 baseline 5.23% → T **6.95%** 로 늘었고, 오류 흐름을 보면 T는 baseline의 not→cat 115건 중 61건을 고쳤지만 **110건을 새로 틀렸습니다**. ②의 MID not→dog 증가도 같은 현상이었습니다.

**원인 분석**
- 클래스 가중 CE(HARD 기준 cat·dog **1.40** vs not **0.64**)가 **혼합 라벨에도 그대로 적용**되고 있었습니다. 동물과 not을 섞은 이미지에서 동물 쪽 손실이 약 2.2배 크게 반영되므로, 결정 경계가 not 쪽으로 밀려 애매한 not 이미지를 cat으로 예측하게 됩니다.
- 쌍 지정 혼합은 모든 혼합이 동물↔not 쌍이므로 이 편향이 더 직접적으로 작용합니다.

**해결**
- `mixed_forward`에 혼합 배치 전용 손실 `criterion_mix`를 분리하고, 설정 `mix_loss='unweighted'`로 **혼합 배치에서만 클래스 가중치를 끄는** 옵션을 구현했습니다(일반 배치는 기존 가중치 유지).
- 04에서 이를 사전 등록 가설 H9(9.3절: 혼합 배치에서 가중치를 끄면 not→cat이 줄어드는가)로 검정하고, cat→not·dog→not·macro-F1이 유의하게 나빠지면 "TRADE-OFF"로 판정하는 가드를 미리 걸어 두었습니다.

**정성적 개선**
- 가설한 원인이 실험으로 확인됐습니다. 가중치를 끄자 not→동물 방향 오류가 baseline보다도 낮아졌습니다(not→cat 5.23% → 4.77%).
- 반대로 동물→not이 늘어, **손실 가중치와 혼합 증강은 독립된 요소가 아니며 한 방향의 개선이 반대 방향 오류로 이동한다**는 점을 수치로 확인했습니다. 03·04 판정에 둔 오류 이동 가드(G1~G3)가 왜 필요한지 보여 주는 실제 사례입니다.

**정량적 개선 (HARD test, T → T_mixunw, 시드 42)**

| 지표 | 개선 전 (가중 혼합 손실) | 개선 후 (혼합 배치 무가중) | 변화 |
|---|---|---|---|
| not→cat 오류율 | 6.95% | **4.77%** | −2.2%p (**−31.4%**) |
| not→dog 오류율 | 5.00% | **2.91%** | −2.1%p (**−41.8%**) |
| not recall | 88.0% | 92.3% | +4.3%p |
| 대조 방향군 오류 (/1,000장) | 26.9 | 23.1 | −14.2% |
| accuracy / macro-F1 | 0.8755 / 0.8648 | 0.8826 / 0.8701 | +0.8% / +0.6% |
| (부작용) cat→not 오류율 | 9.9% | 15.2% | +5.3%p |

> not→동물 방향은 유의하게 개선됐지만(not→cat 95% CI −3.01 ~ −1.37%p) 동물→not이 함께 악화되어 판정은 **TRADE-OFF**였고, 전체 정확도 향상(+0.8%)도 유의하지 않았습니다. 사전 등록한 규칙에 따라 기본값은 유지하고, 목적(예: not 오탐을 줄여야 하는 서비스)에 따라 고를 수 있는 옵션으로 남겼습니다.

---

### ⑤ 비교 공정성 — 고정 lr이 Uniform arm에 불리했던 문제

**문제 (개선 전)**
세 arm은 공통 레시피의 lr **2e-4**를 공유했습니다. 그런데 U HARD는 원 가중치 기준 동물→not 흔들림이 **9.51pp**로 모든 단계 중 가장 불안정했고, best epoch 26 / 36으로 일찍 끝났으며, val macro-F1은 **0.8378**에 머물렀습니다. 이 상태로는 "T가 이긴 것은 ResNet50에 lr이 맞지 않았기 때문"이라는 반론을 배제할 수 없었습니다.

**원인 분석**
- 적정 lr은 백본 크기와 정규화 강도(ResNet50 + RandAugment + 전 쌍 혼합)에 따라 달라지는데, 공정성을 위해 하나의 값으로 고정한 것이 오히려 특정 arm에 불리하게 작용할 수 있었습니다.

**해결**
- 04에서 T·U 각각 lr {1e-4, 4e-4}를 추가 학습하고, 선택 규칙("val macro-F1 최대, 동률 시 val loss 최소")을 **학습 전에 사전 등록**했습니다.
- validation만으로 lr을 고른 뒤 `selection.json`으로 잠그고, 그 다음에야 `open_test_vault()`에서 test를 열었습니다. 선택이 바뀌면 test 이후 분석을 거부합니다.

**정성적 개선**
- 각 처방이 "자기에게 가장 좋은 lr"을 쓴 상태에서 비교하게 되어, 결론이 하이퍼파라미터 선택의 부산물이 아니라는 근거가 생겼습니다.
- U는 lr 1e-4가 선택됐고(T는 기존 2e-4 유지), 원 가중치의 동물→not 흔들림도 9.51pp → 4.90pp(−48%)로 줄어 학습이 안정됐습니다.

**정량적 개선 (U HARD, lr 2e-4 → 1e-4)**

| 지표 | 개선 전 (lr 2e-4) | 개선 후 (lr 1e-4, val 선택) | 변화 |
|---|---|---|---|
| val macro-F1 (선택 기준) | 0.8378 | **0.8588** | +2.1%p (**+2.5%**) |
| test macro-F1 | 0.8354 | **0.8521** | +1.7%p (**+2.0%**) |
| test accuracy | 0.8481 | 0.8626 | +1.5%p (+1.7%) |
| 겨냥 방향군 오류 (/1,000장) | 122.6 | 114.0 | −7.0% |
| 대조 방향군 오류 (/1,000장) | 29.3 | 23.3 | −20.3% |

> 공정성 보정으로 T와 U의 겨냥 방향군 격차는 −25.0 → **−16.4**로 약 34% 줄었지만, 여전히 유의했습니다(95% CI −26.0 ~ −7.1, Holm p = 0.009). 즉 U의 불리함을 제거한 뒤에도 결론은 유지됐습니다.

---

## 9. 결과 해석

### 9.1 진단 (01, validation) — 집계 지표가 숨긴 것

| 방향 | EASY | MID | HARD | 추세 |
|---|---|---|---|---|
| cat → not | 8.8% | 12.1% | **20.4%** | ↑↑ 단조 증가 |
| dog → not | 3.2% | 8.0% | **13.4%** | ↑↑ 단조 증가 |
| not → cat | 1.2% | 4.2% | **5.5%** | ↑↑ 단조 증가 |
| not → dog | 0.6% | 1.7% | **5.4%** | ↑↑ 단조 증가 |
| cat → dog | 6.8% | 6.2% | 6.6% | 변화 없음 (대조) |
| dog → cat | 7.1% | 9.3% | 4.9% | 변화 없음 (대조) |
| val acc / macro-F1 | 0.950 / 0.911 | 0.890 / 0.865 | 0.835 / 0.819 | |

- 집계 지표는 "고르게" 떨어지는 것처럼 보이지만, 난이도와 함께 움직이는 것은 **동물 ↔ not 네 방향뿐**입니다. cat ↔ dog는 난이도와 무관합니다.
- not 쪽 판정 기준을 옮겨도 macro-F1 이득이 최대 +0.0012 → **클래스 비율이 아니라 특징이 겹치는 문제**입니다.
- 일반화 격차 5.0 → 9.5 → 14.8%p, MID·HARD는 lr이 거의 최대(0.97·0.93)일 때 best가 나오고 10 epoch 뒤 조기 종료 → 과적합 + 불안정.

### 9.2 최종 3-arm 비교 (03, test)

**전체 지표**

| 난이도 | B acc / macro-F1 | T acc / macro-F1 | U acc / macro-F1 | T − U macro-F1 (95% CI) |
|---|---|---|---|---|
| EASY | 0.9459 / 0.9012 | 0.9461 / 0.9054 | 0.9494 / 0.9086 | −0.0031 (−0.0122, 0.0063) |
| MID | 0.8894 / 0.8638 | 0.9083 / 0.8886 | 0.9019 / 0.8809 | +0.0077 (−0.0015, 0.0169) |
| HARD | 0.8305 / 0.8100 | **0.8755 / 0.8648** | 0.8481 / 0.8354 | **+0.0294 (0.0174, 0.0414)** |

**★ 방향군 특이성 (test 1,000장당 오류 건수)**

| 난이도 | 방향군 | B | T | U | T − U (95% CI) | 판정 |
|---|---|---|---|---|---|---|
| EASY | 겨냥 | 17.1 | 8.0 | 8.9 | −0.9 (−3.0, 1.3) | 차이 없음 |
| EASY | 대조 | 23.9 | 22.7 | 23.4 | −0.7 (−4.1, 2.9) | |
| MID | 겨냥 | 67.9 | 49.8 | 50.6 | −0.9 (−7.0, 5.3) | 차이 없음 |
| MID | 대조 | 34.9 | 30.2 | 32.8 | −2.6 (−7.2, 2.3) | |
| HARD | 겨냥 | 132.6 | **97.6** | 122.6 | **−25.0 (−34.8, −15.0)** | **SPECIFIC** |
| HARD | 대조 | 36.9 | 26.9 | 29.3 | −2.4 (−7.6, 3.1) | |

**HARD 방향별**: cat→not 21.7 / **9.9** / 14.3% (T−U −4.4pp, Holm p = 0.0006), dog→not 10.6 / **4.8** / 7.5% (−2.7pp, Holm p = 0.010) → 가드 통과 **SUPPORTS**. not→cat(−0.1pp)과 not→dog(−1.4pp, Holm p = 0.098)은 유의하지 않았습니다.

### 9.3 확인 실험 (04, 사전 등록 · HARD)

| 가설 | 검증 내용 | 추정치 (95% CI) | 판정 |
|---|---|---|---|
| H4 | 백본 효과: EfficientNet-B0 − ResNet50 (정책 평균) | **−23.8** (−31.3, −15.8) | ✅ SUPPORTED |
| H5 | 정책 효과: Targeted − Uniform (백본 평균) | −1.2 (−7.6, 4.8) | ❌ NOT SUPPORTED |
| H6 | 상호작용 (탐색) | −3.3 (−16.2, 9.0) | CI가 0 포함 |
| H7 | 혼동 쌍 한정 − 무작위 짝 | −0.5 (−8.3, 7.4) | ❌ NOT SUPPORTED |
| H8 | 시드 42·43·44 평균 T − U | **−18.3** (−28.4, −8.1) | ✅ SUPPORTED (9/9) |
| H9 | 혼합 배치 가중치 제거 → not→cat | −2.18pp (−3.01, −1.37) | ⚠️ TRADE-OFF |
| H10 | 각자 lr 조정 후 T* − U* | **−16.4** (−26.0, −7.1) | ✅ SUPPORTED |

(단위: 겨냥 방향군 오류 / test 1,000장, H9만 not→cat 오류율 pp. Holm 보정 α = 0.05)

**2×2 셀별 겨냥 방향군 오류** — R18: none 132.6 · uniform 116.4 · targeted 121.0 / R50: uniform 122.6 · targeted 123.1 / **EffB0: uniform 100.5 · targeted 97.6**. 같은 백본 안에서는 정책을 바꿔도 거의 같고, 백본을 EfficientNet-B0로 바꿀 때만 크게 떨어집니다.

### 9.4 종합 해석

1. **현상은 견고하다** — HARD에서 T의 겨냥 방향군 오류 감소는 대조 방향과 무관하게 나타났고(특이성), 시드 9/9 조합과 lr 조정 후에도 유지됐습니다.
2. **원인은 증강 정책이 아니라 백본이다** — 백본을 고정하면 진단 기반 정책과 관행 정책의 차이가 사라지고(H5), 쌍 한정 혼합도 무작위 짝과 같았습니다(H7). 쌍 한정이 효과가 없던 이유로는, 3-클래스에서 무작위 짝의 약 50%(HARD 49.9%)가 이미 동물↔not 쌍이고 cat↔dog 쌍은 11.4%에 불과해 **제한이 바꾸는 부분이 작았을** 가능성이 있습니다(추정).
3. **그래도 진단은 쓸모가 있었다** — ① 차이가 *어디서* 나타날지(동물↔not)와 나타나지 않을 곳(cat↔dog)을 미리 지목했고, ② 효과 없는 처방(재가중·logit 보정)을 근거 있게 배제했으며, ③ 진단 D(표현 한계)는 "더 큰 모델"이 아닌 **"더 좋은 표현의 작은 모델"** 을 고르는 근거가 됐습니다. "모델을 키운다"는 관행(ResNet50)은 HARD에서 ResNet18 + 강한 증강(116.4)보다도 나아지지 않았습니다(122.6, 점추정 기준). 다만 집계 지표만 보는 설계자도 EfficientNet-B0를 고를 수 있으므로, **이 백본 선택이 진단에서만 나올 수 있었다고는 이 실험으로 증명되지 않습니다**.
4. **EASY·MID에서는 둘 다 충분했다** — 두 arm 모두 baseline 대비 겨냥 방향군 오류를 약 25–53% 줄였고, 병목이 표현이 아닌 과적합·불안정이었기 때문에 어떤 정규화 묶음이든 비슷하게 작동한 것으로 해석됩니다.

### 9.5 부가 관찰

| 관찰 | 수치 | 해석 |
|---|---|---|
| EMA 안정화 | HARD val acc 흔들림(중앙값) B 1.07pp → T 0.14 / U 0.17pp. 원 가중치 기준 동물→not 흔들림은 U 9.51pp vs EMA 0.50pp | EMA가 epoch 간 요동을 대부분 흡수 → best 선택이 우연에 덜 좌우됨 |
| 효율 (HARD) | T(EffB0) 4.34M · 0.39 GFLOPs vs U(R50) 24.03M · 4.09 GFLOPs | 더 적은 연산으로 더 높은 성능. 단, T는 더 오래 학습(54 vs 36 epoch)해 학습 시간은 145.5 vs 110.8분 |
| 효율 (EASY·MID) | T(R18) 1.81 GFLOPs vs U(R50) 4.09 GFLOPs, EASY 학습 시간 237.9 vs 366.7분 | 통계적으로 같은 성능을 이미지당 연산량 약 56% 적게, EASY 학습 시간은 약 35% 짧게 달성 |
| 보정(ECE) 악화 | HARD ECE B 0.033 · T 0.190 · U 0.079 (T 시드 43·44: 0.169·0.164) | 정확도는 올랐지만 확률의 신뢰도는 나빠짐. 무작위 짝(T_randpair)은 0.104 → 쌍 한정 혼합과의 관련 가능성(단일 run) |
| 오류 이동 | HARD T: cat→not 217건 중 130건 수정 · 25건 신규, not→cat 61건 수정 · 110건 신규 | 개선의 상당 부분은 동물→not에서, 대가는 not→cat에서 발생 (트러블슈팅 ④) |
| TTA (보조) | HARD macro-F1 B 0.823 · T 0.872 · U 0.839 | HARD의 순위가 단일 뷰와 동일 → 핵심 결론이 평가 프로토콜에 의존하지 않음 |

### 9.6 해석 시 유보

- 01~03은 단일 시드이며, 시드 반복(04)은 HARD의 T·U에만 수행했습니다. 2×2 분해와 절제는 시드 42 단일 run입니다.
- 설계자는 원본 프로젝트의 과거 test 결과를 본 적이 있어 **완전한 맹검은 아닙니다**. test는 03·04에서 총 2회 열렸습니다.
- arm마다 파라미터·연산량이 다릅니다. 동일하게 맞춘 것은 최대 epoch · step 정의 · 데이터 · 조기 종료 규칙 · 레시피입니다.
- T−B·U−B에는 EMA 효과가 포함되며, T−U 비교에서는 양쪽 모두 EMA를 쓰므로 상쇄됩니다.
- 02의 근거 조건 임계값(예: 고확신 비율 < 5%, 격차 ≥ 2%p)은 01의 val 수치를 본 설계자가 정했습니다. 따라서 "근거 조건 20개 성립"은 독립 검정이 아니라 **처방 논리를 수치로 문서화하고 사후 변경을 막는 장치**입니다.
- 32×32 원본을 JPEG(q95)로 다시 저장한 캐시를 사용해 한 번의 추가 손실 압축이 들어갔습니다. 세 arm에 동일하게 적용되어 비교의 공정성에는 영향이 없습니다.

---

## 10. 시각화

> 저장소에 그림을 올린 뒤 아래 경로가 그대로 렌더링됩니다.

**① 진단: 집계 지표 vs 방향별 오류 vs 일반화 격차 (val)**
집계 지표는 고르게 하락하지만, 굵은 선(동물↔not 네 방향)만 단조 증가하고 cat↔dog는 평평합니다.

![diagnosis overview](experiments/01_baseline_diagnosis/diagnosis/diagnosis_overview.png)

**② 최종 비교: 전체 macro-F1 vs 겨냥 방향 오류율 (test)**
왼쪽의 전체 지표 차이보다 오른쪽 겨냥 방향에서 차이가 더 선명하게, 그리고 HARD에서만 나타납니다.

![core result](experiments/final_comparison/figures/core_result_aggregate_vs_target.png)
![targeted directions](experiments/final_comparison/figures_v2/targeted_directions.png)

**③ T − U 포레스트 플롯 (겨냥 방향별 95% CI, 파랑 = Holm p < 0.05)**

![forest](experiments/final_comparison/figures_v2/target_T_minus_U_forest.png)

**④ 방향별 변화 히트맵 (baseline 대비 pp, 테두리 = 겨냥 방향)**

![delta heatmap](experiments/final_comparison/figures/direction_delta_heatmap.png)

**⑤ HARD 2×2 요인 분해 — 차이를 만든 것은 백본**

![factorial](experiments/04_ablation_seeds/figures/factorial_hard.png)

**⑥ 시드 강건성 (seed 42·43·44 + 시드×이미지 pooled CI)**

![seeds](experiments/04_ablation_seeds/figures/seed_robustness.png)

**⑦ 오류 흐름 (baseline 대비 수정 ↓ / 신규 오류 ↑)**

![error flow](experiments/04_ablation_seeds/figures/error_flow_hard.png)

**⑧ XAI 3-arm 비교 — baseline이 cat→not으로 틀린 같은 val 샘플의 Grad-CAM**

![xai three arms](experiments/final_comparison/xai_three_arms/hard/cat_to_not.png)

| 그 밖의 그림 | 경로 |
|---|---|
| 단계별 학습 곡선(손실 · F1 · lr · 방향별 오류율) | `experiments/{arm}/{easy,mid,hard}/figures/training_curve.png` |
| 혼동행렬(val/test, 건수·행 정규화) | `experiments/{arm}/{easy,mid,hard}/figures/confusion_matrix_{val,test}.png` |
| XAI 5종 패널(방향별 오류 + 클래스별 정답) | `experiments/01_baseline_diagnosis/diagnosis/xai/`, `02·03/xai/own/` |
| 02 전후 비교 Grad-CAM | `experiments/02_targeted_improvement/xai/before_after/` |
| 기전 절제 · 효율 산점도 | `experiments/04_ablation_seeds/figures/{mechanism_directions_hard, efficiency_hard}.png` |

> ℹ️ 03에서 만든 일부 그림은 제목의 `−`(U+2212)가 NanumGothic에 없어 네모로 표시됩니다. 04에서 ASCII `-`로 바꿔 `figures_v2/`로 다시 생성했으므로 README에는 v2를 우선 사용했습니다.

---

## 11. 최종 회고 및 성찰

**잘한 점**
- **스스로의 가설을 반증할 수 있는 구조를 먼저 만들었습니다.** 03의 결과만 보고 "진단 기반 개선이 이겼다"고 결론 낼 수 있었지만, 사전 등록한 2×2 분해가 원인이 백본임을 보여 줬습니다. 원하는 결론이 아니었기에 오히려 가장 신뢰할 수 있는 결과입니다.
- **집계 지표를 방향으로 분해한 것 자체가 핵심 인사이트였습니다.** "HARD는 어렵다"가 아니라 "동물↔not 경계가 무너진다"로 문제를 다시 정의하자, 효과 없는 처방을 근거 있게 버리고 차이가 날 위치를 미리 지목할 수 있었습니다.
- **재현성 인프라가 실험 규모를 가능하게 했습니다.** 23개 run · 58.0 GPU시간 · 16회 세션 재개를 "노트북을 다시 실행하는 것"만으로 이어 갈 수 있었던 것은 결정적 재개 엔진 덕분입니다.

**아쉬운 점**
- **처방을 묶어서 넣었습니다.** HARD에 백본과 증강을 동시에 바꾼 탓에 04라는 별도 실험이 필요했습니다. 처음부터 요인 설계로 계획했다면 비용이 줄었을 것입니다.
- **보정을 주 지표에 넣지 않았습니다.** T의 ECE가 U의 2배 이상이라는 사실을 사후 분석에서야 확인했습니다. 실제 배포라면 정확도만큼 중요한 지표입니다.
- **32×32 이미지를 224로 올려 학습했습니다.** 사전학습 가중치를 살리기 위한 선택이었지만 입력 픽셀 수가 49배로 늘어 학습 비용이 크게 증가했습니다.
- **공통 엔진을 노트북 4개에 복제했습니다.** 셀 단위 독립 실행을 위해 C1~C7을 그대로 복사했고, 그 때문에 `ENGINE_VERSION` 검사로 불일치를 막아야 했습니다. 패키지·테스트·버전 고정(`requirements.txt`)으로 분리했다면 유지보수와 재현이 더 쉬웠을 것입니다.
- **완전한 맹검이 아니었습니다.** 원본 프로젝트의 test 결과를 본 상태에서 처방을 설계했고, 04에서 같은 test를 다시 열었습니다. 새 test 세트로 확인하기 전까지는 확인적 결론이 아닌 강한 근거로 봐야 합니다.

**배운 점**
- "어디서 틀리는가"를 아는 것은 **무엇을 할지**보다 **무엇을 하지 않을지, 그리고 결과를 어디서 확인할지**를 정하는 데 더 큰 힘을 발휘했습니다.
- 한 방향의 오류 감소는 다른 방향으로의 이동일 수 있습니다. 가드 지표 없이 겨냥 지표만 보는 평가는 위험합니다.

---

## 12. 향후 발전 계획

| 우선순위 | 계획 | 기대 효과 |
|---|---|---|
| 1 | 새로운 test 세트(예: ciFAIR-10/100처럼 중복을 정리한 CIFAR 대체 test)로 HARD 결론 재확인 | 설계자의 비맹검·test 재사용 문제 해소 |
| 2 | EASY·MID 시드 반복(P5 그룹 활성화)과 2×2 분해의 다중 시드화 | "차이 없음" 결론과 분해 결과의 시드 강건성 확인 |
| 3 | 연산량을 맞춘 백본 비교(EfficientNet-B0/B1, RegNet, ConvNeXt-T 등) | "좋은 표현"의 효과가 EfficientNet 고유인지 일반적인지 구분 |
| 4 | 저해상도 학습(64–128px, CIFAR용 stem 수정)과 224px 비교 | 학습 비용 절감 및 32×32 데이터에 맞는 설정 탐색 |
| 5 | Temperature scaling(val로 적합) 적용 후 ECE 재평가 | T에서 나빠진 보정(ECE 0.190) 복구 |
| 6 | not→cat 트레이드오프 대응: 혼합 배치용 균형 샘플링 · 손실 가중치 스케줄 · CP-Mix식 온라인 혼동 추정 | 동물→not 개선을 유지하면서 not→동물 증가 억제 |
| 7 | 준중복 탐지 개선: 후보 쌍을 SSIM·임베딩으로 재검증, 전이적 결합 대신 complete-linkage | 저해상도 해시 오탐(103장 거대 그룹) 제거 |
| 8 | XAI 정량화(deletion/insertion 곡선 등)와 진단 → 처방 자동화 도구화 | 정성적 관찰을 재현 가능한 지표로 전환 |
| 9 | 공통 엔진(C1~C7)을 `cdn_engine/` 패키지로 분리, pytest 스모크 테스트(`CDN_SMOKE_TEST`) CI 연동, `requirements.txt` 버전 고정 | 노트북 간 코드 중복 제거, 환경 차이로 인한 재현 실패 방지 |

---

## 13. 참고 문헌

**데이터**
- Krizhevsky, A. *Learning Multiple Layers of Features from Tiny Images*. Tech. Report, Univ. of Toronto, 2009. (CIFAR-10/100)
- Barz, B., Denzler, J. *Do We Train on Test Data? Purging CIFAR of Near-Duplicates*. Journal of Imaging, 2020. (ciFAIR)

**모델 · 전이학습**
- He, K. et al. *Deep Residual Learning for Image Recognition*. CVPR 2016.
- Tan, M., Le, Q. V. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML 2019. arXiv:1905.11946
- Kornblith, S., Shlens, J., Le, Q. V. *Do Better ImageNet Models Transfer Better?* CVPR 2019. arXiv:1805.08974

**증강 · 정규화**
- Zhong, Z. et al. *Random Erasing Data Augmentation*. AAAI 2020. arXiv:1708.04896
- Yun, S. et al. *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*. ICCV 2019. arXiv:1905.04899
- Zhang, H. et al. *mixup: Beyond Empirical Risk Minimization*. ICLR 2018. arXiv:1710.09412
- Yoon, Y. et al. *Mix from Failure: Confusion-Pairing Mixup for Long-Tailed Recognition*. arXiv:2411.07621, 2024.
- Cubuk, E. D. et al. *RandAugment: Practical Automated Data Augmentation with a Reduced Search Space*. NeurIPS 2020. arXiv:1909.13719
- Szegedy, C. et al. *Rethinking the Inception Architecture for Computer Vision*. CVPR 2016. (Label Smoothing)
- Menon, A. K. et al. *Long-tail Learning via Logit Adjustment*. ICLR 2021.

**학습 레시피 · 최적화**
- Wightman, R., Touvron, H., Jégou, H. *ResNet Strikes Back: An Improved Training Procedure in timm*. arXiv:2110.00476, 2021.
- Touvron, H. et al. *Training Data-Efficient Image Transformers & Distillation through Attention*. ICML 2021. (DeiT)
- Loshchilov, I., Hutter, F. *Decoupled Weight Decay Regularization*. ICLR 2019. (AdamW)
- Loshchilov, I., Hutter, F. *SGDR: Stochastic Gradient Descent with Warm Restarts*. ICLR 2017.
- Goyal, P. et al. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. arXiv:1706.02677, 2017.
- Micikevicius, P. et al. *Mixed Precision Training*. ICLR 2018.
- Polyak, B. T., Juditsky, A. B. *Acceleration of Stochastic Approximation by Averaging*. SIAM J. Control Optim., 1992.
- Morales-Brotons, D., Vogels, T., Hendrikx, H. *Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits*. TMLR 2024. arXiv:2411.18704

**XAI**
- Selvaraju, R. R. et al. *Grad-CAM*. ICCV 2017.
- Chattopadhay, A. et al. *Grad-CAM++*. WACV 2018.
- Wang, H. et al. *Score-CAM*. CVPR Workshops 2020.
- Zeiler, M. D., Fergus, R. *Visualizing and Understanding Convolutional Networks*. ECCV 2014. (Occlusion)
- Ribeiro, M. T., Singh, S., Guestrin, C. *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. KDD 2016. (LIME)
- Adebayo, J. et al. *Sanity Checks for Saliency Maps*. NeurIPS 2018.

**평가 · 통계 · 중복 탐지**
- McNemar, Q. *Note on the Sampling Error of the Difference between Correlated Proportions or Percentages*. Psychometrika, 1947.
- Dietterich, T. G. *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms*. Neural Computation, 1998.
- Holm, S. *A Simple Sequentially Rejective Multiple Test Procedure*. Scandinavian Journal of Statistics, 1979.
- Efron, B., Tibshirani, R. J. *An Introduction to the Bootstrap*. Chapman & Hall, 1993.
- Bouthillier, X. et al. *Accounting for Variance in Machine Learning Benchmarks*. MLSys 2021.
- Guo, C. et al. *On Calibration of Modern Neural Networks*. ICML 2017. (ECE)
- Manku, G. S., Jain, A., Das Sarma, A. *Detecting Near-Duplicates for Web Crawling*. WWW 2007.
- Krawetz, N. *Kind of Like That* (dHash). HackerFactor Blog, 2013.

**GitHub**
- pytorch/vision — https://github.com/pytorch/vision
- huggingface/pytorch-image-models — https://github.com/huggingface/pytorch-image-models
- clovaai/CutMix-PyTorch — https://github.com/clovaai/CutMix-PyTorch
- facebookresearch/mixup-cifar10 — https://github.com/facebookresearch/mixup-cifar10
- zhunzhong07/Random-Erasing — https://github.com/zhunzhong07/Random-Erasing
- tensorflow/tpu (EfficientNet) — https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
- jacobgil/pytorch-grad-cam — https://github.com/jacobgil/pytorch-grad-cam
- marcotcr/lime — https://github.com/marcotcr/lime
