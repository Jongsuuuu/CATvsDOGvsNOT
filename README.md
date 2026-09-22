# CAT / DOG / NOT Image Classification
### 진단 기반 개선(Targeted) vs 일괄 확대(Uniform)

> **"어디서, 어떻게 틀리는지"를 먼저 진단하고 그 지점만 겨냥한 개선은, 집계 지표만 보고 모델과 증강을 일괄로 키운 개선과 다른 결과를 내는가?**

> 본 프로젝트는 <strong>오류 진단에 기반한 선택적 개선(Targeted)</strong>과 <strong>모델·증강을 일괄적으로 확대하는 개선(Uniform)</strong>을 동일한 조건에서 비교합니다.

> 통제된 baseline을 포함해 학습 레시피 · 데이터 분할 · 조기 종료 규칙을 동일하게 맞춘 3-arm 실험을 구성하고, 사전에 등록한 가설을 같은 test 이미지에 대한 짝지은 통계 검정으로 검증합니다. 이를 통해 단순한 성능 향상 여부가 아니라 '진단이 겨냥한 오분류 방향에서 실제로 차이가 나타나는가'를 확인합니다.



---

## 📌 목차

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

### 1.1 한 줄 요약

고양이(cat) · 개(dog) · 그 밖의 모든 것(not)을 구분하는 3-클래스 분류 문제를, not 클래스의 **의미적 난이도(EASY / MID / HARD)** 별로 세 번 학습하면서
**① 통제된 baseline(01) → ② 오류 방향 진단 기반 개선(02) → ③ 집계 지표 기반 일괄 확대(03)** 를 같은 조건에서 비교했습니다.

### 1.2 문제의식

실무에서 모델 성능이 낮으면 흔히 accuracy · F1 같은 **집계 지표**만 보고 "모델을 키우고, 증강을 세게 넣는" 처방을 내립니다.
하지만 집계 지표는 *어디서* 틀리는지 알려주지 않습니다. 01의 진단 결과, 난이도가 올라갈 때 나빠지는 곳은 **동물 ↔ not 경계 하나뿐**이었고 cat ↔ dog 혼동은 난이도와 무관했습니다.

| 오분류 방향 (val) | EASY | MID | HARD | 추세 |
|---|---|---|---|---|
| cat → not | 8.8% | 12.1% | **20.4%** | 단조 증가 |
| dog → not | 3.2% | 8.0% | **13.4%** | 단조 증가 |
| not → cat | 1.2% | 4.2% | 5.5% | 단조 증가 |
| not → dog | 0.6% | 1.7% | 5.4% | 단조 증가 |
| cat → dog | 6.8% | 6.2% | 6.6% | 변화 없음 → **대조 방향** |
| dog → cat | 7.1% | 9.3% | 4.9% | 단조 아님 → **대조 방향** |

→ 그렇다면 "진단이 지목한 경계만 겨냥한 처방"은 "전반적으로 키운 처방"과 **겨냥한 방향에서** 다른 결과를 내야 합니다. 이것이 이 프로젝트의 검증 대상입니다.

### 1.3 사전 등록 가설

| 가설 | 내용 | 결과 |
|---|---|---|
| **H1** | 겨냥 방향(동물 ↔ not)에서 02(T)의 오분류율이 03(U)보다 낮다 (T − U < 0, Holm 보정 후 유의) | **HARD에서만 지지** (cat→not −4.4pp, dog→not −2.7pp) |
| **H2** | 대조 방향(cat ↔ dog)에서는 T와 U의 차이가 유의하지 않다 (차이가 겨냥 방향에 **특이적**) | **지지** (6개 대조 비교 모두 p > 0.13, HARD 방향군 판정 `SPECIFIC`) |
| **H3** | 전체 accuracy · macro-F1에서는 03이 02와 비슷하거나 더 나을 수 있다 | EASY·MID는 부합(유의차 없음), **HARD는 02가 전체 지표에서도 유의하게 우세** |

### 1.4 핵심 결과 (test, 단일 시드)

| 난이도 | 01 Baseline acc / macro-F1 | 02 Targeted acc / macro-F1 | 03 Uniform acc / macro-F1 | 겨냥 방향 판정 |
|---|---|---|---|---|
| EASY | 0.9459 / 0.9012 | 0.9461 / 0.9054 | **0.9494 / 0.9086** | NOT SUPPORTED (T ≈ U) |
| MID | 0.8894 / 0.8638 | **0.9083 / 0.8886** | 0.9019 / 0.8809 | NOT SUPPORTED (T ≈ U) |
| HARD | 0.8305 / 0.8100 | **0.8755 / 0.8648** | 0.8481 / 0.8354 | **SPECIFIC — 겨냥 방향군에서만 T < U** |

- HARD에서 02(EfficientNet-B0, **4.34M** 파라미터)는 03(ResNet50, **24.03M** 파라미터)보다 파라미터가 약 5.5배 적지만,
  cat→not 오류율 **9.9%**(03: 14.3%), dog→not 오류율 **4.8%**(03: 7.5%)로 더 낮았습니다 (Holm 보정 p = 0.0006 / 0.0101, 가드 3종 통과).
- HARD의 전체 오류 차이(02: 523건 vs 03: 638건, **−115건**) 중 **−105건(약 91%)이 겨냥 방향군**에서 발생했습니다. 즉, 집계 지표의 격차는 겨냥 방향의 격차가 드러난 것입니다.
- EASY · MID에서는 두 방식이 동물 → not 오류를 **거의 같은 만큼** 줄였습니다. 진단이 처방을 바꾼 효과는 **난이도가 가장 높은 조건에서만** 통계적으로 확인됐습니다.

### 1.5 이 프로젝트가 지키는 원칙

- **진단은 validation으로만.** test는 03의 마지막 셀에서 세 arm을 **단 한 번** 평가하며, 접근 기록이 `experiments/_shared/test_access_log.jsonl` 에 남습니다.
- **처방은 사전 등록.** 02는 01의 validation 산출물로 **근거 조건 20개**를, 03은 **집계 지표 근거 조건 5개**를 자동 검증하고, 하나라도 성립하지 않으면 학습을 거부합니다. 학습 시작 후 처방이 바뀌면 실행을 거부합니다(처방 잠금).
- **공정성은 코드가 강제.** 공통 레시피의 지문(`d82dd3e7935501e1`) · 분할 서명(`f0cb8ffb8fdfe2c5`) · 엔진 버전 · best 모델 SHA-256이 다르면 02·03이 실행을 거부합니다.
- **개선되지 않았으면 그대로 보고.** EASY · MID의 "NOT SUPPORTED" 판정과 HARD에서 H3가 예상과 다르게 나온 사실을 숨기지 않습니다.

### 1.6 핵심 주장과 근거 위치

README의 핵심 주장은 모두 아래 파일(`experiments/`) 또는 코드 셀에서 확인할 수 있습니다.

| 주장 | 근거 파일 · 코드 |
|---|---|
| 데이터는 95,400장의 32×32 JPEG이며 CIFAR-10/100 계열로 추정 | `_shared/data_audit/file_manifest.csv`, `audit_summary.json` |
| 난이도와 함께 커지는 오류는 동물 ↔ not뿐, cat ↔ dog는 무관 | `01_*/diagnosis/diagnosis.json › direction_trend`, `direction_table_val.csv` |
| 원인은 클래스 비율이 아니라 특징 겹침 (재가중 무효) | `02_*/diagnosis_evidence.json › threshold_shift_max_gain, hiconf_error_share` |
| 02의 처방은 01 진단에서 자동 검증된 근거 조건으로 도출 (20/20) | `02_*/prescription.json › evidence, locked, excluded` / 02 노트북 「2 — 진단 근거 수치 계산」 · 「3 — 진단 → 처방」 |
| 03의 처방은 집계 지표만으로 도출, 난이도 분기 0건 (5/5) | `03_*/prescription.json › inputs_used, branches_by_difficulty` / 03 노트북 「1 — 선행 조건 · 이 arm 의 입력」 · 「2 — 관행적 처방」 |
| 02와 03의 차이는 혼합 쌍 선택 (cat↔not, dog↔not 한정) | C6 `_pair_partner_index`, `mixed_forward` / `02_*/*/metadata.json › model_spec.mix_pairs` |
| 세 arm의 레시피 · 분할 · 예산 규칙이 동일 | `final_comparison/budget_fairness_audit.csv`, 각 단계 `metadata.json` |
| test는 진단 · 처방 · 모델 선택에 쓰지 않고 1회만 평가 | `_shared/test_access_log.jsonl` (1줄, 모든 arm 완료 이후) |
| HARD에서 겨냥 방향 T < U, 대조 방향 차이 없음 (`SPECIFIC`) | `target_direction_comparison_test.csv`, `control_direction_comparison_test.csv`, `family_specificity_test.csv` |
| HARD 전체 오류 차이의 약 91%가 겨냥 방향군 | `direction_errors_all_arms_test.csv` (9.6 ③의 계산) |
| EASY에서 오류 이동(not → cat 증가)을 가드가 검출 | `target_direction_comparison_test.csv › guard_ok`, `error_reduction_specificity.csv` |
| EMA가 흔들림을 줄이고 선택 모델을 개선 | `training_stability_val.csv`, 각 단계 `training_history.csv › *_raw` |
| 통계 방법 (bootstrap · McNemar · Holm · 가드 · 판정 규칙) | 03 노트북 최종 비교의 `# ── 통계` 셀 |

---

## 2. 시스템 아키텍처

### 2.1 실험 파이프라인

```
[dataset.zip]  easy|mid|hard / train|test / cat, dog, not-*
   │
   ▼
[C4] 데이터 준비 · 누수 감사 · 고정 분할   (01이 1회 생성, 02·03은 서명 검증 후 재사용)
   ├─ MD5 + dHash(64/256bit) → train↔test 완전중복·준중복 제거 (train 쪽만)
   ├─ Union-Find 그룹 → 그룹 단위 층화 train/val 분할 (val 20%)
   └─ splits/split_{easy,mid,hard}.csv  (분할 서명 f0cb8ffb8fdfe2c5)
   │
   ▼
[01] baseline_diagnosis : ResNet18 학습 (EASY → MID → HARD) + validation 진단
   ├─ diagnosis.json › aggregate            ──► 02, 03 모두 사용 (03은 이것만)
   ├─ diagnosis.json › directions / trend   ──► 02만 사용
   ├─ 일반화 격차 · 고확신 오류 · 임계값 실험 ──► 02만 사용 (03은 acc_gap · 집계 열만)
   └─ XAI 5종 (Grad-CAM/++, Score-CAM, Occlusion, LIME) ──► 02의 후보 제거용 보조 근거
   │
   ├──► [02] targeted_improvement
   │       ├─ 근거 조건 20개 자동 검증 (20/20 ✓)
   │       ├─ 실패 유형 A–E → 난이도별 누적 처방
   │       │     EASY : Random Erasing + 쌍 지정 CutMix + EMA
   │       │     MID  : EASY + 쌍 지정 MixUp
   │       │     HARD : MID + EfficientNet-B0
   │       └─ prescription.json (처방 잠금)
   │
   └──► [03] uniform_scaleup
           ├─ 근거 조건 5개 자동 검증 (5/5 ✓), 입력은 집계 지표뿐
           ├─ 난이도 분기 0건 — 세 난이도 동일 처방
           │     ResNet50 + RandAugment(N2, M9) + 무작위 MixUp·CutMix + RE 0.25 + EMA
           └─ prescription.json (처방 잠금)
                   │
                   ▼
[Final] 03의 마지막 셀 : test 단 1회 개방
   ├─ 예산 · 공정성 감사 (레시피 지문 · 분할 · 예산 · 모델 해시)
   ├─ 세 arm test 평가 (동일 코드 · 동일 전처리)
   ├─ 겨냥 방향 T−U : paired bootstrap 95% CI (B=5000) · McNemar · Holm 보정 · 가드 3종
   ├─ 대조 방향(cat↔dog) · 방향군 특이성(1,000장당) · 오류 감소 특이성 · 학습 안정성
   └─ FINAL_REPORT.md · final_summary.json · figures/ · xai_three_arms/
```

### 2.2 노트북 내부 구조 (세 노트북 공통 셀 C1–C7 + X1)

각 노트북은 공통 셀(C1–C7, X1)만으로 독립 실행되며, arm마다 달라지는 것은 `ARM_SPEC`(모델 · 증강 개입)뿐입니다.

- **02 · 03**: 공통 셀이 **완전히 동일**합니다 (엔진 build 2.1).
- **01**: 먼저 학습을 마친 **엔진 build 2.0** 입니다. C2 · C4 · C6 · C7에 build 2.1의 추가 기능(가중치 EMA, 쌍 지정 혼합, 비정상 loss 감시, dict 형식 Random Erasing, 안정성 요약)이 없습니다.
  추가 기능은 모두 `spec` 에서 켤 때만 동작하도록 만들어, 학습 조건 호환 키(`ENGINE_VERSION = 'cdn-engine-2.0'`)와 공통 레시피 지문이 01과 같게 유지됩니다(8.5 참조).

| 셀 | 역할 | 핵심 요소 |
|---|---|---|
| **C1** | 환경 준비 | 패키지 자동 설치, NanumGothic 폰트, Google Drive 마운트 |
| **C2** | 공통 학습 레시피 | `SHARED_RECIPE` (세 arm 동일, 지문으로 대조), 경로 · 디바이스 · AMP · TF32 |
| **C3** | 공통 유틸리티 | 원자적 저장, SHA-256 지문, RNG 스냅샷/복원, 설정 diff |
| **C4** | 데이터 · 누수 감사 · 분할 | 리사이즈 캐시, MD5/dHash, 준중복 탐색, Union-Find 그룹 분할, 결정적 Dataset |
| **C5** | 모델 · 지표 · 평가 | torchvision 백본 3종, GPU 혼동행렬 누적, 방향별 오분류율, 평가 산출물 저장 |
| **C6** | 재개 가능한 학습 엔진 | EpochSampler, OOM 자동 대응, 정확한 재개 / (build 2.1) EMA, 쌍 지정 혼합, 비정상 loss 감시 |
| **C7** | arm 간 공정성 검증 | 선행 arm 완료 · 레시피 · 분할 · 엔진 · 모델 해시 대조, 안정성 요약 |
| **X1** | XAI 도구 | Grad-CAM / Grad-CAM++ / Score-CAM / Occlusion / LIME, 확신도 층화 + 특징 다양성 샘플 선택 |

### 2.3 arm 간 데이터 흐름 계약 (누가 무엇을 읽는가)

| 산출물 | 01이 생성 | 02가 읽음 | 03이 읽음 |
|---|---|---|---|
| `diagnosis.json › aggregate` | ✓ | ✓ | ✓ **(유일하게 읽는 진단 섹션)** |
| `diagnosis.json › directions / direction_trend / primary_direction` | ✓ | ✓ | ✗ |
| `training_history.csv` | ✓ | `val_acc`, 방향별 `val_rate_*`, `lr` (흔들림 · best 시점 lr 계산) | `epoch, train_loss, val_loss, val_acc, val_macro_f1, lr, is_best` 열만 |
| `generalization_gap.json` | ✓ | ✓ | `acc_gap` 만 |
| val `predictions.csv` (확률) | ✓ | ✓ (고확신 오류 · 임계값 실험) | ✗ |
| XAI 패널 | ✓ | 후보 제거용 보조 근거 | ✗ (학습 후 분석 전용) |

03의 처방 구간은 방향별 정보(`directions`, `val_rate_*`, predictions)를 **코드 수준에서 읽지 않습니다** (`prescription.json › inputs_used`, `branches_by_difficulty = 0`).

---

## 3. 데이터 출처 및 전처리

### 3.1 데이터 출처

- 과제용으로 제공된 `dataset.zip` 을 사용했습니다. 구조는 `{easy|mid|hard}/{train|test}/{cat, dog, not-easy|not-mid|not-hard}/` 입니다.
- 전체 **95,400장이 모두 32 × 32 JPEG** 입니다(`_shared/data_audit/file_manifest.csv` 의 width · height 기준). 학습 시 224 × 224로 확대합니다.
- **출처 추정 (추정임을 명시)**: cat · dog는 클래스당 6,000장(train 5,000 + test 1,000)으로 **CIFAR-10** 과 정확히 일치하고, not은 600장 단위(30,000 / 16,200 / 13,200)라 **CIFAR-100** 계열로 추정됩니다. 크기 · 수량 · 내용에 근거한 판단이며 공식 출처 확인은 아닙니다.
- 난이도는 not 클래스가 cat · dog와 얼마나 **의미적으로 가까운가**로 결정됩니다.

| 난이도 | not 클래스의 내용 | 전체 파일 | train (cat / dog / not) | test (cat / dog / not) |
|---|---|---|---|---|
| **EASY** | 사물 · 풍경 · 식물 · 탈것 | 42,000 | 5,000 / 5,000 / 25,000 | 1,000 / 1,000 / 5,000 |
| **MID** | 곤충 · 갑각류 · 파충류 · 어류 · 사람 | 28,200 | 5,000 / 5,000 / 13,500 | 1,000 / 1,000 / 2,700 |
| **HARD** | 사자 · 표범 · 너구리 · 햄스터 · 토끼 등 **다른 포유류** | 25,200 | 5,000 / 5,000 / 11,000 | 1,000 / 1,000 / 2,200 |

> HARD는 "고양이인가 아닌가"가 아니라 **"고양이인가, 고양이와 비슷한 다른 털 짐승인가"** 를 32픽셀 사진으로 가려야 하는 문제입니다.

### 3.2 데이터 누수 감사 (`_shared/data_audit/audit_summary.json`)

CIFAR 계열은 train과 test 사이에 중복이 존재한다고 알려져 있습니다(Barz & Denzler, 2020: CIFAR-10 test의 3.3%, CIFAR-100 test의 10%). 그래서 학습 전에 전체 파일을 감사했습니다.

1. 모든 파일의 **MD5(원본 바이트)** 와 **dHash 64bit / 256bit** 를 계산합니다.
2. **완전 중복**: MD5가 같은 train ↔ test 쌍.
3. **준중복**: dHash64 해밍 거리 ≤ 4 **그리고** dHash256 해밍 거리 ≤ 32 (리사이즈 · 재인코딩 사본).
   후보 탐색은 64bit를 13bit 밴드 5개로 나눈 버킷 방식입니다 (해밍 ≤ 4인 쌍은 비둘기집 원리로 반드시 한 밴드를 공유).
4. 겹치는 이미지는 **train 쪽에서만 제외**하고 test는 건드리지 않습니다. 같은 원본에서 파생된 그룹의 일부만 test와 겹쳐도 **그룹 전체를 제외**합니다.
5. 같은 이미지가 서로 다른 클래스에 있으면(**라벨 충돌**) 학습에서 제외합니다.

| 난이도 | train↔test 완전중복 | 준중복 | 라벨 충돌 파일 | 제외 (test_overlap / label_conflict / overlap_group) | 읽기 실패 |
|---|---|---|---|---|---|
| EASY | 1 | 169 | 103 | 170 / 49 / 32 | 0 |
| MID | 6 | 34 | 4 | 40 / 4 / 1 | 0 |
| HARD | 1 | 38 | 0 | 39 / 0 / 2 | 0 |

### 3.3 그룹 단위 층화 분할

- 완전 중복 · 준중복 · 파생 파일명(`_aug`, `_flip`, `_rot`, `(1)` 등 정규식)을 **Union-Find** 로 한 그룹으로 묶고,
  같은 그룹이 train과 val에 **동시에 들어가지 않도록** 클래스별 그룹 단위로 val 20%를 뽑습니다 (seed 42 + 1000 × class).
- 분할은 01이 **한 번만** 생성해 `experiments/_shared/splits/` 에 저장하고, 02 · 03은 서명(`f0cb8ffb8fdfe2c5`)을 검증한 뒤 재사용합니다.
- 분할 직후 "같은 그룹이 두 분할에 걸쳐 있으면" 즉시 오류로 중단합니다.

| 난이도 | train | val | test |
|---|---|---|---|
| EASY | 27,799 (3,995 / 3,992 / 19,812) | 6,950 (999 / 998 / 4,953) | 7,000 |
| MID | 18,764 (3,996 / 3,994 / 10,774) | 4,691 (999 / 999 / 2,693) | 4,700 |
| HARD | 16,767 (3,996 / 3,994 / 8,777) | 4,192 (999 / 999 / 2,194) | 4,200 |

### 3.4 전처리 · 증강 파이프라인

| 단계 | 내용 |
|---|---|
| 캐시 | 원본을 1회 디코딩해 RGB JPEG(q95)로 재저장하면서 MD5 · dHash를 함께 계산. 짧은 변이 256px을 넘을 때만 축소하므로 이 데이터(32×32)는 크기가 그대로 유지됩니다. 결과는 무압축 zip(`dataset_cache_r256.zip`)으로 Drive에 보관하고 런타임마다 로컬로 한 번에 복원 (세 arm이 같은 캐시 사용) |
| 공통 train 증강 (세 arm 동일) | `RandomResizedCrop(224, scale=(0.7, 1.0))` → `RandomHorizontalFlip(0.5)` → `ColorJitter(0.3, 0.3, 0.3, 0.08)` |
| arm별 증강 | 03만 `RandAugment(num_ops=2, magnitude=9)` / 02 · 03 모두 정규화 이후 `RandomErasing` (설정은 arm별 상이, 7장 참조) |
| 정규화 | ImageNet 고정 통계 (mean 0.485/0.456/0.406, std 0.229/0.224/0.225) |
| 평가 (주 지표) | `Resize((224, 224))` 단일 뷰, uint8 텐서로 캐시 후 GPU에서 정규화 |
| 평가 (보조 지표, TTA) | 원본 · 좌우 반전 · `Resize(256) + CenterCrop(224)` 3개 뷰의 확률 평균 (세 arm 동일 적용, 결론에는 사용하지 않음) |
| 클래스 가중치 | **train 분할만**으로 계산한 balanced 가중치 → val/test 정보가 학습에 들어가지 않음 |

---

## 4. 핵심 구현 및 최적화

### 4.1 공통 학습 레시피 (`SHARED_RECIPE`, 세 arm 동일)

| 항목 | 값 |
|---|---|
| 사전학습 가중치 | torchvision `IMAGENET1K_V1` (구현체 · 정규화 통계 통일, timm 태그 혼용 제거) |
| 입력 크기 | 224 |
| Optimizer | AdamW, lr 2e-4, weight decay 1e-2 |
| Scheduler | linear warm-up 3 epoch + cosine (epoch 단위), 최소 lr 비율 0.01 |
| 배치 | effective batch 64 (micro-batch는 GPU 메모리에 맞춰 자동, 부족분은 gradient accumulation) |
| 예산 | 최대 100 epoch, patience 10, min_delta 1e-4 |
| 손실 | CrossEntropy (train 기준 balanced class weight) + label smoothing 0.1 |
| 안정화 | gradient clipping 1.0, AMP(fp16) + GradScaler, channels_last |
| 혼합 공통값 | 적용 확률 0.5, MixUp α = 0.4, CutMix α = 1.0 (둘 다 켜면 혼합 배치 안에서 50:50 선택) |
| 헤드 | `Linear(in, 256) → BN → ReLU → Dropout(0.5) → Linear(256, 3)` (세 arm 동일) |
| best 선택 기준 | **val macro-F1** (동률이면 val loss) |

### 4.2 arm별 개입 (`ARM_SPEC`)

| 항목 | 01 Baseline | 02 Targeted | 03 Uniform |
|---|---|---|---|
| 백본 | ResNet18 ×3 | ResNet18 / ResNet18 / **EfficientNet-B0** | **ResNet50** ×3 |
| 파라미터 (헤드 포함) | 11.31M | 11.31M / 11.31M / **4.34M** | **24.03M** |
| Random Erasing | — | p 0.5, 면적 0.02–0.4, 종횡비 0.3–3.3, random 값 | p 0.25, 면적 0.02–0.33, 종횡비 0.3–3.3, random 값 |
| CutMix | — | EASY · MID · HARD, **혼동 쌍 한정** | 세 난이도, 무작위 쌍 |
| MixUp | — | MID · HARD, **혼동 쌍 한정** | 세 난이도, 무작위 쌍 |
| 혼합 대상 쌍 | — | **cat ↔ not, dog ↔ not 만** (cat ↔ dog는 섞지 않음) | 모든 쌍 (cat ↔ dog 포함) |
| RandAugment | — | — | N = 2, M = 9 |
| 가중치 EMA | — | decay 0.999 | decay 0.999 (동일 → T − U에서 상쇄) |

### 4.3 쌍 지정 혼합 (Confusion-Pair Mixing) — 02와 03을 가르는 핵심 장치

```python
def _pair_partner_index(y, pairs):
    # pairs = [('cat','not'), ('dog','not')] → 허용 행렬 allow[K,K] (대칭)
    m = allow[y][:, y].float()          # (B, B): 배치 안에서 i가 j와 섞일 수 있는가
    empty = m.sum(1) == 0
    m[empty] = 0.0
    m[empty, empty.nonzero()] = 1.0     # 짝이 없으면 자기 자신과 짝 → 혼합 없음
    return torch.multinomial(m, 1).squeeze(1)
```

- 혼합의 효과는 **섞인 두 클래스 사이의 결정 경계를 부드럽게 하는 것**입니다(Zhang et al., 2018 §2). 따라서 "어느 쌍을 섞는가"가 곧 "어느 경계를 규제하는가"입니다.
- 02는 짝 선택만 바꾸고 혼합 확률 · α · 예산은 03과 같습니다. 그래서 cat ↔ dog 경계는 02의 혼합을 **전혀 받지 않으며**, 대조 방향이 "겨냥하지 않은 방향"의 기준선으로 실제로 기능합니다.
- CutMix의 라벨 비율 λ는 잘린 영역의 **실제 면적**으로 다시 계산합니다 (경계 클리핑 반영).

### 4.4 가중치 EMA (`ModelEMA`)

- 평가 · best 저장은 EMA 모델로 하고, 역전파에는 관여하지 않습니다.
- decay warm-up `min(0.999, (1 + t) / (10 + t))`, `torch._foreach_lerp_` 로 한 번에 갱신.
- BN running 통계도 같은 decay로 평균합니다(Morales-Brotons et al. §3.4가 따르는 Cai et al. 2021 방식). 정수 버퍼(`num_batches_tracked`)는 복사합니다.
- 원 가중치 지표도 `*_raw` 열로 매 epoch 기록해, EMA의 안정화 효과를 수치로 비교할 수 있게 했습니다.

### 4.5 정확한 재개 (Exact Resume) 엔진

Colab 런타임은 수시로 끊깁니다. 실제로 9개 학습 단계 중 **8개가 한 번 이상 중단 후 재개**됐습니다(`metadata.json › sessions`). "끊기지 않은 것처럼" 이어가기 위해 다음을 구현했습니다.

| 장치 | 내용 |
|---|---|
| 체크포인트 내용 | model · optimizer · scheduler · GradScaler · EMA · patience · best 상태 · history · Python/NumPy/Torch/CUDA RNG 전부 |
| 이중 보관 | `last_checkpoint.pth` + `.prev.pth`, `best_model.pth` + `.prev.pth` (쓰기 도중 손상 대비) |
| 원자적 저장 | 로컬 임시파일 → Drive 임시파일 복사 → `os.replace` (Drive 쓰기 중 끊겨도 기존 파일 보존) |
| 배치 순서 고정 | `EpochSampler`: `(seed, epoch)` 만으로 순열 결정 |
| 증강 난수 고정 | `SplitDataset`: `(base_seed, epoch, index)` 로 샘플마다 재시드 → 워커 수 · 재시작과 무관 |
| DataLoader seed 소비 | `iter(loader)` 직후에 epoch 단위 재시드 (연속 실행과 재개 실행의 전역 RNG 소비 시점 차이 제거) |
| 설정 지문 | 체크포인트의 지문이 현재 코드와 다르면 diff를 출력하고 **중단** (서로 다른 실험이 섞이는 것 방지) |
| 상태 머신 | `metadata.status`: `running → training_finished → completed`, 완료 + 해시 일치 + 지문 일치면 SKIP |
| 근사 재개 차단 | optimizer · EMA 상태가 없으면 기본적으로 거부 (`ALLOW_WEIGHTS_ONLY_RESUME=False`), 허용 시 보고서에 표시 |

검증 근거: 설계근거서 §7에 따르면 합성 데이터 반복 실험에서 중단 후 재개 결과가 끊지 않은 실행과 **비트 단위로 일치**했고(학습 기록 · best 가중치), 엔진 build 2.0과 2.1에서 01 결과가 비트 단위로 동일했습니다. `experiments` 에서는 최종 감사표(`budget_fairness_audit.csv`)의 `approx_resume` 이 9개 단계 모두 `False` 입니다.

### 4.6 학습 안정성 · 자원 대응

- **비정상 loss 감시**: 한 epoch에서 NaN/Inf loss가 1%를 넘으면 그 epoch을 저장하지 않고 중단합니다. GPU 동기화는 epoch당 1회뿐입니다(카운터를 GPU 텐서로 누적). 02 · 03의 6개 단계에서 실제 발생 건수는 **0**입니다(`training_stability_val.csv › nonfinite_steps`, 01은 build 2.0이라 이 감시 기능이 없음).
- **micro-batch 자동 탐색**: 가짜 입력으로 forward/backward/step을 2회 돌려 최대 메모리가 80% 미만인 가장 큰 micro-batch를 찾고, 나머지는 gradient accumulation으로 채웁니다. 학습 중 OOM이 나면 micro-batch를 절반으로 줄이고 마지막 체크포인트에서 해당 epoch을 다시 수행합니다(effective batch 64 유지).
- **평가 최적화**: val/test 이미지를 uint8 텐서로 **한 번만** 디코딩해 메모리에 캐시하고, 정규화는 GPU에서 합니다. 혼동행렬은 GPU에서 `bincount` 로 누적해 배치마다 `.cpu()` 를 호출하지 않습니다.
- **처리량**: `cudnn.benchmark`, TF32, channels_last, AMP, `persistent_workers`, `prefetch_factor=4`.

### 4.7 근거 조건 자동 검증 · 처방 잠금

- 02는 01의 validation 산출물로 **근거 조건 20개**(EASY 6 · MID 6 · HARD 8)를 다시 계산하고, 하나라도 성립하지 않으면 `ON_UNSUPPORTED_EVIDENCE='raise'` 로 학습을 거부합니다. 실제 결과는 **20/20 성립**.
- 03은 집계 지표만으로 **근거 조건 5개**를 검증합니다. 실제 결과는 **5/5 성립**.
- `prescription.json` 에 처방의 지문을 잠그고, 학습이 시작된 뒤 `targets` 나 `spec` 이 바뀌면 실행을 거부합니다 (사후 선택 방지).
- 02는 `diagnosis.json` 이 **현재 baseline 모델에서 만들어졌는지** SHA-256으로 확인합니다.

### 4.8 최종 비교의 통계 설계

| 도구 | 내용 |
|---|---|
| 겨냥 방향 T − U | 같은 test 이미지에서 짝지은 비교. paired bootstrap 95% CI (B = 5,000), McNemar(연속성 보정, 불일치 < 25면 정확 이항검정), **9개 겨냥 방향에 Holm 보정** |
| 가드 3종 | G1 source 클래스 recall, G2 역방향 오분류율, G3 전체 macro-F1 — 겨냥 방향이 줄었어도 이들이 유의하게 나빠지면 `TRADE-OFF`(오류 이동)로 판정 |
| 판정 규칙 | `SUPPORTS`: CI 상한 < 0 · Holm p < 0.05 · 가드 통과 / `CONTRADICTS`: CI 하한 > 0 · 유의 / 그 외 `NOT SUPPORTED` |
| 대조 방향 | cat → dog, dog → cat 에서 T − U가 0 근처여야 "특이적" |
| 방향군 특이성 | 겨냥 방향군 · 대조 방향군의 오류 건수(1,000장당)를 각각 집계해 T − U 비교. 겨냥군 CI 상한 < 0 이고 대조군 CI가 0을 포함하면 `SPECIFIC` |
| 전체 지표 | acc · macro-F1 bootstrap CI (B = 2,000), 전체 정오답 McNemar |
| 예산 감사 | 레시피 지문 · 분할 · 최대 epoch · patience · 배치 · train/val 규모가 arm 간 다르면 즉시 중단 |

### 4.9 XAI 도구 개선

- **validation 오분류 샘플에만** 적용 (진단 단계에서 test를 보지 않기 위해).
- ResNet은 `layer4[-1]` **블록 출력**, EfficientNet은 `features[-1]` 을 대상 레이어로 사용 (BN · 잔차 합산 이후).
- hook을 사용 후 제거, Score-CAM(상위 32채널) · Occlusion을 배치로 계산, LIME은 샘플 300개.
- **샘플 선택 규칙 (무작위 아님, 재실행해도 동일)**: 같은 (실제 → 예측) 조합의 후보를 예측 확률 순위로 N개 구간으로 나누고(확신도 층화), 각 구간에서 이미 뽑힌 샘플과 특징 공간 코사인 거리가 가장 먼 샘플을 고릅니다(farthest-point). 선택 근거는 패널마다 `*_samples.csv` 로 저장됩니다.
- **XAI의 지위**: 방향당 8장으로 표본이 적으므로, 처방을 *고르는* 근거가 아니라 후보를 *지우는* 보조 근거로만 사용합니다.

---

## 5. 기술 스택

| 분류 | 사용 기술 |
|---|---|
| 언어 · 환경 | Python 3, Jupyter Notebook, Google Colab, Google Drive |
| 하드웨어 | NVIDIA **Tesla T4** (9개 학습 단계 전부) |
| 딥러닝 | PyTorch (AMP, GradScaler, channels_last, `torch._foreach_lerp_`), torchvision (`resnet18`, `resnet50`, `efficientnet_b0`, `IMAGENET1K_V1`, `transforms.RandAugment`, `RandomErasing`) |
| 데이터 처리 | NumPy, pandas, Pillow, `multiprocessing` (해시 · 캐시 병렬 생성), `hashlib` (MD5 · SHA-256) |
| 통계 | SciPy (`chi2`, `binomtest`), scikit-learn (`classification_report`), 자체 구현 paired bootstrap · McNemar · Holm |
| XAI | 자체 구현 Grad-CAM / Grad-CAM++ / Score-CAM / Occlusion, `lime`, `scikit-image`, `opencv-python-headless` |
| 시각화 | Matplotlib (NanumGothic) |
| 기타 | tqdm |

> 총 학습 시간(9개 단계 합계, 평가 · XAI 제외): 약 **1,644분 ≈ 27.4시간** (Tesla T4).

---

## 6. 폴더 구조

### 6.1 저장소 구성 (권장)

```
CATvsDOGvsNOT/
├── README.md
├── notebooks/
│   ├── 01_baseline_diagnosis.ipynb        # baseline 학습 + validation 진단 + XAI
│   ├── 02_targeted_improvement.ipynb      # 진단 → 근거 조건 검증 → 겨냥 처방 학습
│   └── 03_uniform_scaleup.ipynb           # 집계 지표 → 일괄 확대 학습 + 최종 3-arm test 비교
├── docs/
│   └── 02_03_설계근거.pdf                  # 02·03 학습 설계 근거서 (17쪽)
└── experiments/                            # 세 노트북 실행 결과 (checkpoint 제외)
```

### 6.2 `experiments/` 상세

```
experiments/
├── _shared/
│   ├── data_audit/
│   │   ├── audit_summary.json               # 난이도별 중복·충돌·제외·분할 규모
│   │   ├── file_manifest.csv                # 전체 파일 MD5 · dHash64 · dHash256
│   │   ├── train_test_duplicates_{easy,mid,hard}.csv
│   │   ├── label_conflicts_{easy,mid,hard}.csv
│   │   └── unreadable_{easy,mid,hard}.csv
│   ├── splits/split_{easy,mid,hard}.csv     # 고정 분할 (01이 1회 생성)
│   ├── dataset_cache_r256.zip               # 리사이즈 캐시 (⚠ 약 125MB, 아래 주의 참고)
│   └── test_access_log.jsonl                # test 접근 기록 (1회)
│
├── 01_baseline_diagnosis/
│   ├── arm_summary.json
│   ├── diagnosis/
│   │   ├── diagnosis.json                   # aggregate / directions / trend / gap / training
│   │   ├── diagnosis_overview.png
│   │   ├── direction_table_val.csv
│   │   ├── val_error_samples.csv
│   │   └── xai/{easy,mid,hard}/             # 6개 오류 방향 + 3개 정답 클래스 × 5종 XAI 패널 (+ xai_index.csv)
│   └── {easy,mid,hard}/
│       ├── evaluation/val/                  # predictions · 혼동행렬 · 방향별 오류 · metrics · generalization_gap
│       ├── evaluation/test/                 # 최종 비교에서 생성 (+ metrics_tta_secondary.json)
│       ├── figures/                         # training_curve.png, confusion_matrix_{val,test}.png
│       ├── logs/                            # training_history.csv, train_log.txt
│       └── metadata.json                    # 상태 · 지문 · 세션 · best SHA-256 · 예산
│
├── 02_targeted_improvement/
│   ├── arm_summary.json
│   ├── diagnosis_evidence.json              # 01에서 다시 계산한 근거 수치
│   ├── prescription.json                    # 잠긴 처방 · 근거 조건 20개 판정 · 문헌
│   ├── val_target_check.csv                 # (개발용) 겨냥 방향 val 변화
│   ├── stability_report_val.csv
│   ├── xai/own/{easy,mid,hard}/             # 02 모델 자신의 오분류 XAI (xai/xai_index.csv)
│   ├── xai/before_after/{easy,mid,hard}/    # 같은 샘플의 baseline vs targeted Grad-CAM (+ xai_index.csv)
│   └── {easy,mid,hard}/ …                   # 01과 동일 구조
│
├── 03_uniform_scaleup/
│   ├── arm_summary.json
│   ├── prescription.json                    # 입력 = 집계 지표뿐, 난이도 분기 0건
│   ├── xai/own/{easy,mid,hard}/             # 03 모델 자신의 오분류 XAI (학습 후 분석 전용)
│   ├── xai/xai_index.csv
│   └── {easy,mid,hard}/ …
│
└── final_comparison/
    ├── FINAL_REPORT.md                      # 최종 보고서 (자동 생성)
    ├── final_summary.json                   # 판정 · 모델 해시
    ├── aggregate_metrics_test.csv
    ├── target_direction_comparison_test.csv # ★ 겨냥 방향 T−U · 가드 · 판정
    ├── control_direction_comparison_test.csv
    ├── family_specificity_test.csv          # ★ 방향군 특이성
    ├── direction_errors_all_arms_test.csv
    ├── error_reduction_specificity.csv
    ├── budget_fairness_audit.csv
    ├── training_stability_val.csv
    ├── figures/                             # 핵심 그림 4종
    └── xai_three_arms/{easy,mid,hard}/      # 같은 샘플의 B · T · U Grad-CAM 비교
```

> ⚠ **GitHub 업로드 주의**
> - `experiments/_shared/dataset_cache_r256.zip`(약 125MB)은 GitHub의 **파일당 100MB 제한**을 넘습니다. `.gitignore` 에 추가하거나 Git LFS로 관리하세요. 이 파일은 `dataset.zip` 으로부터 재생성됩니다.
> - `checkpoints/` 폴더(`last_checkpoint.pth`, `best_model.pth`)는 저장소에 포함하지 않았습니다. 최종 비교에 쓰인 best 모델의 SHA-256은 `final_comparison/final_summary.json › model_sha256` 에 기록되어 있습니다.
> - 원본 `dataset.zip` 은 포함하지 않습니다 (CIFAR 계열 추정 데이터의 재배포 조건 확인 필요).

권장 `.gitignore`:

```gitignore
dataset.zip
experiments/_shared/dataset_cache_r*.zip
experiments/**/checkpoints/
experiments/**/_archive/
_local/
```

### 6.3 실행 방법

```text
1. Google Drive에 MyDrive/CATvsDOGvsNOT/dataset.zip 을 둡니다.
   (다른 위치라면 환경 변수 CDN_PROJECT_DIR, CDN_LOCAL_ROOT 로 지정)
2. Colab GPU 런타임에서 01 → 02 → 03 순서로 "모두 실행" 합니다.
   - 01 : 캐시·누수 감사·분할 생성 → baseline 학습(EASY→MID→HARD) → 진단 → XAI
   - 02 : 01 완료·레시피·분할·해시 대조 → 근거 조건 20개 검증 → 처방 잠금 → 학습 → XAI
   - 03 : 집계 지표 근거 5개 검증 → 학습 → (마지막 셀) 세 arm test 1회 평가 → FINAL_REPORT.md
3. 런타임이 끊기면 같은 노트북을 처음부터 다시 실행합니다.
   완료된 단계는 SKIP, 끊긴 단계는 마지막 체크포인트의 다음 epoch부터 이어갑니다.
4. 특정 단계를 처음부터 다시 하려면 FORCE_RESTART_STAGES = {'mid'} 처럼 지정합니다
   (기존 폴더는 _archive/ 로 이동).
5. CPU 스모크 테스트: CDN_SMOKE_TEST=1 (입력 64px, 사전학습 없음, 최대 6 epoch)
```

---

## 7. 개선 근거

> 원칙: **(1)** CIFAR 계열 데이터에서 여러 논문이 반복해 개선을 보고한 기법만 고르고, **(2)** 01의 진단 수치가 그 기법의 작동 조건과 맞을 때만 적용하며, **(3)** EMA · gradient clipping · 비정상 loss 감시 · 정확한 재개로 학습 안정성 위험을 줄입니다.
> 어떤 기법도 특정 데이터에서의 개선을 수학적으로 보장하지 못하며, 판정은 03의 최종 test 비교에서 내립니다.

### 7.1 01이 보여준 진단 신호 (validation 전용)

| 신호 | EASY | MID | HARD | 해석 |
|---|---|---|---|---|
| 동물 → not 오류 중 확률 ≥ 0.9 비율 | 0.0% | 0.0% | 1.5% | 확신 없이 애매하게 틀림 → **경계형 오류** |
| not → cat 오류 중 확률 ≥ 0.9 비율 | 27.6% | 11.6% | 19.2% | 반대 방향에는 **과신 오류 꼬리** |
| not 판정 기준(logit)만 옮겼을 때 macro-F1 최대 이득 | +0.07%p | +0.12%p | +0.06%p | 사실상 0 → **비율 문제가 아니라 특징 겹침** |
| 일반화 격차 (증강 없는 train acc − val acc) | 5.0%p | 9.5%p | 14.8%p | 난이도와 함께 과적합 증가 |
| best epoch / 실행 epoch | 51 / 61 | 15 / 25 | 20 / 30 | MID · HARD는 일찍 멈춤 |
| best 시점 lr ÷ 기본 lr | 0.52 | 0.97 | 0.93 | MID · HARD는 **lr이 거의 최대일 때** 멈춤 |
| epoch 간 \|Δ val acc\| 중앙값 | 0.37pp | 0.62pp | 1.07pp | HARD가 가장 흔들림 |
| val cat recall | 0.844 | 0.817 | **0.730** | HARD cat이 전체 최저 |

- **임계값 실험**: not의 log-확률에 τ ∈ [−1, 1](0.1 간격)를 더해 보면, cat → not이 줄어드는 만큼 not → cat이 늘어 macro-F1이 개선되지 않았습니다. → 클래스 재가중 · logit 보정(Menon et al., ICLR 2021 계열)은 **제외**.
- **XAI 관찰**
  1. 히트맵은 대체로 **동물 위**에 있습니다 → 주의 영역 교정 · 배경 제거 증강은 **제외**.
  2. cat → not 오류는 일부만 보임 · 초근접 · 흐림 · 어두움 · 가림 같은 **비전형 장면**에 몰려 있습니다.
  3. HARD의 not → cat은 호랑이 줄무늬 근접, 여우, 고양이와 비슷한 얼굴의 **포유류**에서 일어납니다 (CNN의 질감 편향, Geirhos et al., ICLR 2019와 일치하는 해석).

### 7.2 진단 → 실패 유형 A–E → 처방

| 유형 | 내용 | 근거 신호 | 처방 | 적용 난이도 |
|---|---|---|---|---|
| **A** | 비전형 · 부분 장면의 동물을 not으로 판단 | 동물 → not 저확신, XAI 관찰 2 | Random Erasing | EASY · MID · HARD |
| **B** | 부분만 보고 판단하는 능력 부족 + 과적합 | 격차 ≥ 2%p, 동물 → not 최다 | CutMix (혼동 쌍 한정) | EASY · MID · HARD |
| **C** | 동물 ↔ not 경계가 양쪽으로 무너짐 | not → 동물 상승(MID부터), 과신 꼬리, 시소 현상 | MixUp (혼동 쌍 한정) | MID · HARD |
| **D** | 특징 표현의 한계 (세밀 구분) | 임계값 이동 무효, CAM은 동물 위, HARD의 포유류 not | EfficientNet-B0 | HARD |
| **E** | 학습 불안정, 높은 lr에서 조기 종료 | \|Δacc\|, best 시점 lr 비율 | 가중치 EMA | EASY · MID · HARD |

### 7.3 02의 처방별 근거 (논문 · 확인 위치 · 설정)

#### ① Random Erasing — 유형 A

- **왜 맞나**: 01의 cat → not 오류는 일부만 보이거나 가려진 사진에 몰려 있고, 확신 없이 틀립니다(고확신 0%). "불완전한 증거로도 동물로 판단하는 능력"을 키우는 처방입니다.
- **논문**: Zhong et al., *Random Erasing Data Augmentation*, AAAI 2020 ([arXiv:1708.04896](https://arxiv.org/abs/1708.04896))
  - §1: 학습 데이터에 가림이 적으면 부분적으로 가려진 물체를 인식하지 못한다는 문제의식
  - §5.1.2: 기본 설정 p = 0.5, s_l = 0.02, s_h = 0.4, r_1 = 0.3 / Table 1: CIFAR-10/100의 여러 ResNet · WRN에서 오류율 감소
  - Figure 4: p, s_h의 넓은 범위에서 baseline보다 좋음 → 설정에 둔감 / Table 2: 무작위 값 채우기(RE-R)가 가장 좋음
  - Table 4: random crop · flip과 상호 보완적 / Figure 5: 가림 강건성
- **공식 구현**: [github.com/zhunzhong07/Random-Erasing](https://github.com/zhunzhong07/Random-Erasing)
- **설정**: 논문 기본값 그대로 `p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value='random'` (정규화 이후 적용 → 표준정규 무작위 값, timm 방식과 동일)

#### ② CutMix (혼동 쌍 한정) — 유형 B

- **왜 맞나**: 세 난이도 모두 최다 오류가 동물 → not이고 일반화 격차가 5.0–14.8%p입니다. "고양이 조각 + not 조각" 합성 이미지로 동물과 not의 경계를 직접 학습시키고, 강한 정규화로 과적합을 줄입니다.
- **논문**: Yun et al., *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*, ICCV 2019 ([arXiv:1905.04899](https://arxiv.org/abs/1905.04899))
  - §1: 붙여 넣은 조각이 부분 장면으로부터 객체를 식별하도록 요구
  - §3.1: λ ~ Beta(α, α), 모든 실험에서 α = 1, 라벨은 면적 비율로 혼합
  - §3.2, Figure 2: lr 감쇠 이후 baseline은 과적합으로 검증 오류가 오르지만 CutMix는 꾸준히 감소
  - §4.1.2, Table 5 · 6: CIFAR-100에서 PyramidNet-200, ResNet-110 모두 오류율 감소
  - §4.1.3, Figure 3: 시험한 모든 α에서 baseline보다 좋고 α = 1이 최고 / §4.4: 가림 강건성
- **공식 구현**: [github.com/clovaai/CutMix-PyTorch](https://github.com/clovaai/CutMix-PyTorch)
- **설정**: α = 1.0, 배치 단위 적용 확률 0.5 (공통 레시피)

#### ③ 혼동 쌍 한정 혼합 — 02와 03을 가르는 설계

- **착상 출처**: Yoon et al., *Mix from Failure: Confusion-Pairing Mixup for Long-Tailed Recognition*, 2024 ([arXiv:2411.07621](https://arxiv.org/abs/2411.07621))
  - 초록: 모델의 혼동 분포를 추정해 **혼동 쌍에서 표본을 뽑아 실시간으로 섞는** 방법(CP-Mix). 모델이 자주 혼동하는 클래스 쌍을 구분하도록 약점을 겨냥해 학습
  - Figure 2 · 3: 표준 Mixup이 소수 클래스 경계를 왜곡할 수 있고, 유사 클래스 쌍의 혼합이 경계를 개선함을 토이 예제와 혼동행렬로 제시
- **보조 근거**: Zhang et al. (2018) §2 — 같은 라벨끼리만 보간하면 mixup의 이득이 나타나지 않았다 → 이득의 원천은 **서로 다른 클래스 사이의 보간**이며, 쌍 선택이 작동 지점을 결정
- **차이점 (정직하게)**: CP-Mix는 불균형 데이터용 라벨 혼합식 보정과 혼동 쌍의 온라인 추정을 포함합니다. 이 프로젝트는 "혼동 쌍에서 짝을 고른다"는 착상만 차용하고, 쌍은 01 진단으로 **사전 등록**하며, 라벨 혼합식은 표준(면적 · λ 비례)을 유지합니다. 즉 **공개 기법의 단순화된 변형**입니다.
- **설계 개정 이유**: 초기 설계의 02는 03과 같은 방식(모든 쌍 무작위)으로 혼합을 썼습니다. 이 경우 03이 02의 개입을 거의 포함하는 **상위집합**이 되어, 두 arm의 차이가 겨냥 방향에 나타나는지 검증할 수 없었습니다 → 8.1 참조.

#### ④ MixUp (혼동 쌍 한정) — 유형 C, MID · HARD

- **왜 MID부터**: EASY에서는 not → 동물이 1.2% / 0.6%로 미미하지만, MID에서 not → cat 4.2%, not → dog 1.7%로 **반대 방향도 함께 커집니다**. HARD에서는 네 방향이 모두 커지고 not → cat에 과신 오류(19.2%)가 있으며, epoch마다 cat → not과 not → cat이 **시소처럼** 오갑니다.
- **논문**: Zhang, Cisse, Dauphin, Lopez-Paz, *mixup: Beyond Empirical Risk Minimization*, ICLR 2018 ([arXiv:1710.09412](https://arxiv.org/abs/1710.09412))
  - §2, Figure 1(b): 결정 경계가 클래스 사이에서 선형적으로 전환 → 불확실성 추정이 매끄러워짐
  - Figure 2: 학습 샘플 사이 지점에서 예측 실패가 적고 입력 기울기가 작음
  - §3.2, Figure 3(a): PreAct ResNet-18 CIFAR-10 5.6% → 4.2%, CIFAR-100 25.6% → 21.1%
  - §3.1: ImageNet에서 α ∈ [0.1, 0.4]일 때 개선, α가 크면 과소적합
- **공식 구현**: [github.com/facebookresearch/mixup-cifar10](https://github.com/facebookresearch/mixup-cifar10)
- **설정**: α = 0.4 (과소적합 위험을 피하는 보수적 값), CutMix와 혼합 배치 안에서 50:50 (RSB 부록 A의 방식)

#### ⑤ EfficientNet-B0 백본 — 유형 D, HARD만

- **왜 HARD만**: 임계값을 옮겨도 개선이 없고(특징 겹침), 히트맵은 이미 동물 위에 있으며(주의 문제 아님), not이 비슷한 포유류라 세밀한 구분 능력이 필요합니다. **표현력의 문제**이므로 백본 교체가 정당화됩니다. 핵심은 "더 큰 모델"이 아니라 **"더 좋은 특징을 가진, 오히려 더 작은 모델"** 입니다.

| torchvision `IMAGENET1K_V1` | ImageNet top-1 | 파라미터 | GFLOPs |
|---|---|---|---|
| ResNet18 (01 전체, 02 EASY · MID) | 69.8% | 11.7M | 1.81 |
| **EfficientNet-B0 (02 HARD)** | **77.7%** | **5.3M** | **0.39** |
| ResNet50 (03 전체) | 76.1% | 25.6M | 4.09 |

*(수치 출처: torchvision 가중치 메타데이터 `weights.meta`, [pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html))*

- **논문**
  - Kornblith, Shlens, Le, *Do Better ImageNet Models Transfer Better?*, CVPR 2019 ([arXiv:1805.08974](https://arxiv.org/abs/1805.08974)) — 16개 구조 × 12개 데이터셋(CIFAR-10/100, Oxford-IIIT Pets 등 세밀 분류 포함)에서 미세조정 시 ImageNet 정확도와 전이 정확도의 상관 r = 0.96
  - Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019 ([arXiv:1905.11946](https://arxiv.org/abs/1905.11946)) — 8개 전이 데이터셋 중 5개(CIFAR-100, Flowers 등)에서 훨씬 적은 파라미터로 최고 성능
- **구현**: torchvision (`efficientnet_b0`, 기본 stochastic depth 0.2 포함), 헤드는 baseline과 같은 구조
- **위험**: 백본과 증강을 함께 바꾸므로 HARD에서 둘의 기여를 분리할 수 없습니다 (11장 · 12장).

#### ⑥ 가중치 EMA — 유형 E

- **왜 맞나**: 01에서 epoch마다 동물 → not 비율이 2.6–5.8pp씩 흔들리고, MID · HARD는 lr이 최대의 93–97%일 때 best를 찍고 멈췄습니다. cosine 감쇠의 효과를 받기 전에 조기 종료된 것입니다.
- **논문**: Morales-Brotons, Vogels, Hendrikx, *Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits*, TMLR 2024 ([arXiv:2411.18704](https://arxiv.org/abs/2411.18704))
  - 초록 · §3.2: 평균화가 SGD 잡음을 줄여 **lr 감쇠를 덜 필요로 하며**, 일종의 암묵적 정규화로 작용
  - §3.4: BN 통계가 EMA decay 선택의 제한 요인 → BN 재계산 없는 온라인 EMA에서는 빠른 decay 권장
  - §4.1 / 부록 F: decay warm-up `min(α, (t+1)/(t+10))`
  - 일반화 · 예측 일관성(churn) · 보정(calibration) · 노이즈 라벨 강건성 개선 보고
- **고전적 근거**: Polyak & Juditsky, *Acceleration of Stochastic Approximation by Averaging*, SIAM J. Control Optim. 1992
- **설정**: 매 step decay 0.999 (창 약 1,000 step ≈ 2.3–3.8 epoch), BN running 통계도 같은 decay로 평균
- **공정성**: EMA는 일반적 안정화 기법이라 03에도 똑같이 넣었습니다 → T − U 비교에서 EMA 효과는 상쇄됩니다.

#### 공통 제외 항목과 이유

| 제외한 처방 | 이유 |
|---|---|
| 클래스 재가중 · logit 보정 | 임계값 이동 이득 ≤ 0.12%p → 사전확률 문제가 아님 |
| 주의 영역 교정 · 배경 제거 증강 | CAM이 이미 동물 위에 있음 |
| RandAugment | 특정 오류 방향과 연결이 약한 범용 증강 → 03의 관행 처방에만 |
| 헤드 · Label Smoothing 변경 | 진단 근거 없음 → baseline과 동일 유지 |
| ResNet50 확대 | "키우기"는 03의 역할 |

### 7.4 02 근거 조건 20개 (실제 01 결과로 자동 검증, 20/20 ✓)

| 난이도 | 근거 조건 | 값 |
|---|---|---|
| EASY | cat → not이 최다 오분류 방향 | cat→not 8.8%, dog→cat 7.1%, cat→dog 6.8% |
| EASY | 동물 → not 고확신 비율 < 5% (경계형 오류) | 0.0% |
| EASY | 일반화 격차 ≥ 2%p | 5.0%p |
| EASY | not → 동물 ≤ 2% → MixUp 보류 | not→cat 1.2%, not→dog 0.6% |
| EASY | baseline best epoch ≤ 예산의 60% (혼합 증강의 느린 수렴 감당 가능) | 51 / 100 |
| EASY | 동물 → not 흔들림 ≥ 1pp (EMA) | 2.60pp |
| MID | cat→not · dog→not이 EASY → MID에서 증가 | 8.8→12.1, 3.2→8.0 |
| MID | not→cat · not→dog도 증가 (MixUp 추가 근거) | 1.2→4.2, 0.6→1.7 |
| MID | cat ↔ dog는 늘지 않음 (쌍 제한 근거) | cat→dog 6.8→6.2 |
| MID | 격차 ≥ 5%p이고 EASY보다 큼 | 9.5%p (EASY 5.0%p) |
| MID | 임계값 이동 이득 < 0.005 | +0.0012 |
| MID | best 시점 lr ≥ 50% (EMA) | 0.97 |
| HARD | 동물 ↔ not 네 방향 단조 증가 | 네 방향 모두 ↑ |
| HARD | cat ↔ dog 단조 증가 아님 (쌍 제한 근거) | 둘 다 × |
| HARD | 임계값 이동 이득 < 0.005 | +0.0006 |
| HARD | 격차가 세 난이도 중 최대 | 5.0 / 9.5 / 14.8 |
| HARD | val acc 흔들림이 최대 | 0.37 / 0.62 / 1.07 |
| HARD | not → cat 고확신 ≥ 10% (과신 경계) | 19.2% |
| HARD | cat recall이 난이도 · 클래스 중 최저 (표현 한계) | 0.844 / 0.817 / 0.730 |
| HARD | best 시점 lr ≥ 50% (EMA) | 0.93 |

### 7.5 03의 처방 근거 — 집계 지표만 보는 "일반적인 개선"

03은 XAI와 방향별 혼동을 보지 않는 실무자의 표준적 판단을 재현합니다. 처방은 현대 ResNet 표준 학습 레시피(ResNet strikes back, DeiT 계열)를 **세 난이도에 똑같이** 적용합니다.

| 집계 지표에서 읽은 것 | 관행적 해석 → 처방 | 근거 |
|---|---|---|
| accuracy 0.950 → 0.890 → 0.835 | 용량 부족 → **ResNet18 → ResNet50** | Kornblith+ 2019 (r = 0.96); He+ 2016; RSB §3 "Regularization" |
| 격차 5–15%p, train acc 98–100% | 과적합 → **RandAugment N = 2, M = 9** | Cubuk+ 2020 Figure 2 (N = 2, ResNet-50의 최적 M = 9, torchvision 기본값과 동일); RSB Table 2 |
| 동일 | 표준 강한 정규화 → **MixUp + CutMix** (배치별 50:50, 무작위 쌍) | RSB §3 "Data-Augmentation", 부록 A; Zhang+ 2018; Yun+ 2019 |
| 동일 | 가림 정규화 → **Random Erasing p = 0.25** | RSB Table 2의 DeiT 설정(erasing 0.25); Zhong+ 2020 |
| val 곡선이 흔들리고 lr이 높을 때 조기 종료 | 평균화 안정 → **EMA 0.999** | RSB Table 11 (Procedure B, EMA On); Morales-Brotons+ 2024 |
| macro-F1 < accuracy | 소수 클래스 보정 → 클래스 가중치 | 이미 공통 레시피에 포함 (세 arm 동일) |

- **핵심 문헌**: Wightman, Touvron, Jégou, *ResNet strikes back: An improved training procedure in timm*, 2021 ([arXiv:2110.00476](https://arxiv.org/abs/2110.00476)) — §1: 정확도 = f(구조, 학습 절차, 측정 잡음)이므로 baseline에도 최신 "재료"를 넣어야 공정 / §3: RRC + flip 위에 RandAugment · Mixup · CutMix / Table 2: DeiT(RA 9/0.5, Mixup 0.8, CutMix 1.0, Erasing 0.25) 등 표준 설정 비교
- **참고 구현**: [github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) (timm, RSB 레시피), [github.com/facebookresearch/deit](https://github.com/facebookresearch/deit), torchvision [`RandAugment`](https://pytorch.org/vision/stable/generated/torchvision.transforms.RandAugment.html)
- **근거 조건 5개 (5/5 ✓)**: accuracy 하락(0.950 > 0.890 > 0.835), 모든 난이도에서 macro-F1 < accuracy, 모든 난이도에서 격차 ≥ 3%p(5.0 / 9.5 / 14.8), 모든 난이도에서 조기 종료(61 / 25 / 30 epoch), 적어도 한 난이도에서 best 시점 lr ≥ 50%(0.52 / 0.97 / 0.93)

---

## 8. 주요 트러블슈팅

> 이 프로젝트에서 가장 위험했던 문제는 "학습이 안 되는 것"이 아니라 **"결론이 틀려도 알아챌 수 없는 것"** 이었습니다. 그래서 실험의 타당성을 무너뜨릴 수 있었던 문제 6개를 핵심으로 정리했습니다.
> 각 항목은 **상황 → 왜 치명적인가 → 원인 → 해결(코드 위치) → 검증 근거(`experiments` 파일 · 수치) → 교훈** 순서입니다.

| # | 분류 | 문제 | 해결의 핵심 | 검증 근거 |
|---|---|---|---|---|
| 8.1 | 실험 설계 | 03이 02의 상위집합이 되어 가설 검증 불가 | 혼동 쌍 한정 혼합 + 대조 방향 + 방향군 특이성 | HARD `SPECIFIC`, 대조 6개 모두 n.s. |
| 8.2 | 평가 지표 | 겨냥 방향 오류율만 보면 "오류 이동"도 개선처럼 보임 | 가드 3종 + 짝지은 검정 + Holm + 판정 규칙 | EASY cat→not 가드 G2 실패 검출 |
| 8.3 | 진단 → 처방 | 집계 지표가 가리키는 "재가중"을 처방할 뻔함 | 임계값 이동 실험 · 고확신 분석 · 근거 조건 코드화 · 처방 잠금 | 임계값 이득 ≤ +0.0012, 근거 20/20 |
| 8.4 | 재현성 | Colab 단절 후 재개가 "다른 실험"이 됨 | 정확한 재개 엔진 (RNG · 샘플러 · 상태 전부 복원) | 9단계 중 8단계 재개, `approx_resume` 전부 False |
| 8.5 | 공정성 | 엔진을 고치면 01 baseline을 다시 학습해야 함 | 호환 키와 빌드 분리, 신기능은 spec에서만 활성 | 9개 단계 레시피 지문 동일 |
| 8.6 | 데이터 | CIFAR 계열의 train↔test 준중복 누수 | MD5 + 이중 dHash 감사, Union-Find 그룹 분할 | EASY 251장 · MID 45장 · HARD 41장 제외 |

### 8.1 [실험 설계] "03이 02의 상위집합" — 결과가 어떻게 나와도 주장을 검증할 수 없는 설계

- **상황**: 초기 설계의 02는 Random Erasing · CutMix · MixUp을 03과 **같은 방식(배치 내 모든 쌍 무작위 혼합)** 으로 사용했습니다.
- **왜 치명적인가**: 03도 같은 메커니즘으로 동물 ↔ not 경계를 규제하므로, 03이 02의 개입을 거의 포함하는 **상위집합**이 됩니다. 이 경우 T ≈ U가 나와도, T < U가 나와도 "진단 정보가 처방을 바꿨기 때문"이라고 말할 수 없습니다. 또 cat ↔ dog 경계도 두 arm 모두에서 규제되어, "겨냥하지 않은 방향"이라는 기준선이 사라집니다.
- **원인**: 혼합의 효과는 **섞인 두 클래스 사이의 경계를 부드럽게 하는 것**입니다(Zhang et al., 2018 §2, Figure 1b). 그런데 무작위 쌍 혼합은 "어느 경계를 규제할지"를 선택하지 않습니다. 진단이 처방의 *종류*만 바꾸고 *작용 지점*은 바꾸지 못하는 구조였습니다.
- **해결**
  - 엔진 build 2.1에 `_pair_partner_index` (C6)를 추가했습니다. `mix_pairs = [('cat','not'), ('dog','not')]` 로 허용 행렬을 만들고, 배치 안에서 허용된 상대 중에서만 짝을 뽑습니다. 허용된 상대가 없으면 자기 자신과 짝지어 **혼합하지 않습니다**. cat ↔ dog, cat ↔ cat은 섞이지 않습니다.
  - 혼합 확률(0.5) · α(MixUp 0.4 / CutMix 1.0) · 예산은 `SHARED_RECIPE` 에 두어 03과 같게 유지 → 두 arm의 차이는 **"어느 경계를 규제하는가" 하나**로 좁혀집니다.
  - 대조 방향(`control_directions = ['cat->dog', 'dog->cat']`)과 **방향군 특이성 지표**를 새로 사전 등록했습니다 (가설 H2).
  - 착상은 CP-Mix(Yoon et al., 2024)의 "혼동 쌍에서 표본을 뽑아 섞는다"에서 가져왔고, 쌍은 01 진단으로 사전 등록했습니다.
- **검증 근거**
  - `02_targeted_improvement/prescription.json › mix_pairs, control_directions` 와 `02_*/{easy,mid,hard}/metadata.json › model_spec.mix_pairs` 에 쌍 제한이 기록되어 있고, 03의 spec에는 `mix_pairs` 가 없습니다.
  - `control_direction_comparison_test.csv`: 6개 대조 비교의 T − U가 모두 유의하지 않습니다 (p = 0.131–0.901).
  - `family_specificity_test.csv`: HARD에서 겨냥 방향군 T − U = −25.0건/1,000장 (CI −34.8 ~ −15.0), 대조 방향군 −2.4건 (CI −7.6 ~ 3.1) → `SPECIFIC`.
- **교훈**: "효과가 있었나"를 물으려면 **처방이 닿지 않아야 할 곳**을 함께 설계해야 합니다. 설계 개정이 없었다면 HARD의 결과도 "EfficientNet이 더 좋았다"는 말 이상이 될 수 없었습니다.

### 8.2 [평가 지표] 겨냥 방향 오분류율의 함정 — "오류를 옮긴 모델"도 이긴 것처럼 보인다

- **상황**: 핵심 지표는 겨냥 방향의 오분류율(예: 실제 cat 중 not으로 예측된 비율)입니다.
- **왜 치명적인가**: 이 지표는 오류를 **다른 방향으로 밀어내도** 좋아집니다. 극단적으로 cat을 전부 dog로 예측하면 cat → not은 0%가 됩니다(03 노트북 통계 셀의 주석에 이 예시를 명시). 동물 ↔ not 경계를 한쪽으로 미는 규제는 반대 방향(not → 동물) 오류를 늘릴 수 있으므로, 이 프로젝트의 처방에서는 실제로 일어날 수 있는 문제였습니다.
- **해결** (03 마지막 셀 `# ── 통계`)
  - **가드 3종**을 모든 겨냥 방향에 붙였습니다: G1 source 클래스 recall, G2 역방향(예: not → cat) 오분류율, G3 전체 macro-F1. 각각 T − U의 paired bootstrap CI로 "U보다 유의하게 나쁘지 않은가"를 봅니다.
  - **판정 규칙을 코드로 고정**: CI 상한 < 0 · Holm p < 0.05 · 가드 통과 → `SUPPORTS` / 겨냥 방향은 좋아졌지만 가드 실패 → `TRADE-OFF (오류 이동 의심)` / 반대로 유의 → `CONTRADICTS` / 그 외 → `NOT SUPPORTED`.
  - 같은 이미지에서의 **짝지은 비교**(McNemar, 불일치 < 25면 정확 이항검정)와 9개 겨냥 방향 전체에 **Holm 보정**을 적용했습니다.
  - 가드와 별도로, baseline 대비 **겨냥 방향 vs 나머지 방향**의 오류 건수 변화(`error_reduction_specificity.csv`)도 기록했습니다.
- **검증 근거**: 가드가 실제로 오류 이동을 잡아냈습니다.
  - `target_direction_comparison_test.csv`: EASY cat → not의 G2(역방향 not → cat, T − U) CI가 **(0.1, 0.88)pp** 로 0보다 커서 `guard_ok = False`.
  - `direction_errors_all_arms_test.csv`: 02 EASY의 not → cat이 baseline 61건 → **112건**(03: 88건).
  - `error_reduction_specificity.csv`: 02 EASY는 겨냥 방향 오류가 64건 줄었지만 **나머지 방향 오류가 62건 늘었습니다**.
  - 겨냥 방향 차이 자체가 유의하지 않아 최종 판정은 `NOT SUPPORTED` 이지만, 가드가 없었다면 이 부작용은 보고서에 드러나지 않았습니다.
- **교훈**: 특정 방향을 겨냥한 개선일수록, **그 개선이 오류를 어디로 옮겼는지**를 같은 수준의 엄밀함으로 봐야 합니다.

### 8.3 [진단 → 처방] "not이라고 너무 자주 말한다"는 착시 — 잘못된 처방을 고를 뻔한 문제

- **상황**: 01의 최다 오류는 세 난이도 모두 cat → not이었고(8.8% / 12.1% / 20.4%), not 클래스가 train 분할의 52–71%(HARD 8,777 / 16,767 ~ EASY 19,812 / 27,799)를 차지합니다. 집계 지표만 보면 "모델이 다수 클래스(not)로 쏠린다 → 클래스 재가중이나 logit 보정" 이 가장 자연스러운 처방입니다.
- **왜 치명적인가**: 원인이 사전확률(클래스 비율)이 아니라 **특징 겹침**이라면, 재가중은 cat → not을 줄이는 만큼 not → cat을 늘릴 뿐입니다. 처방이 진단과 어긋나면 실험 전체가 "진단 기반"이라는 이름만 남습니다. 또 설계자가 결과를 보며 처방을 바꾸면 사후 선택(HARKing)이 됩니다.
- **해결** (02 노트북 「2 — 진단 근거 수치 계산」 · 「3 — 진단 → 처방」)
  - **임계값 이동 실험** `threshold_shift_gain`: 01의 val 예측에서 not의 log-확률에 τ ∈ [−1.0, 1.0](0.1 간격)를 더해 macro-F1의 최대 이득을 계산했습니다. 이득이 0이면 "판정 기준을 옮기는 처방"(재가중 · logit 보정, Menon et al. ICLR 2021 계열)은 효과가 없습니다.
  - **고확신 오류 비율**: 동물 → not 오류 중 확률 ≥ 0.9인 비율로 "애매하게 틀리는 경계형 오류"인지 확인했습니다.
  - **근거 조건 20개를 코드로 작성**해 01 산출물에서 자동 계산하고, 하나라도 성립하지 않으면 학습을 거부합니다(`ON_UNSUPPORTED_EVIDENCE='raise'`).
  - **처방 잠금**: `prescription.json` 에 targets · spec의 지문을 저장하고, 학습이 시작된 뒤 바뀌면 실행을 거부합니다. 02는 `diagnosis.json` 이 현재 baseline 모델에서 만들어졌는지 SHA-256으로 확인합니다.
- **검증 근거**
  - `02_targeted_improvement/diagnosis_evidence.json › threshold_shift_max_gain`: **+0.0007 / +0.0012 / +0.0006** (EASY / MID / HARD) → 사실상 0.
  - 같은 파일 `hiconf_error_share`: cat → not · dog → not의 고확신 비율 0.0% / 0.0% / 1.5%, not → cat은 27.6% / 11.6% / 19.2%.
  - `prescription.json › excluded` 에 "class re-weighting / logit adjustment (threshold shift gain < 0.005)"가 제외 사유와 함께 기록되어 있고, `evidence_all_supported = true` (20/20), `locked_fingerprint = 27b52e0d813007f9`.
  - 03도 집계 지표 근거 5개를 같은 방식으로 검증하고 잠갔습니다(`03_uniform_scaleup/prescription.json`, 5/5).
- **교훈**: 진단은 "무엇을 할까"만큼 **"무엇을 하지 말아야 하는가"** 를 정해 줍니다. 그리고 그 판단은 사람의 기억이 아니라 파일로 남아야 검증할 수 있습니다.

### 8.4 [재현성] Colab 런타임 단절 — "이어서 학습"이 "다른 실험"이 되는 문제

- **상황**: 한 난이도 학습이 1–6시간(Tesla T4) 걸렸고, 9개 학습 단계 중 **8개가 한 번 이상 중단 후 재개**됐습니다.

  | 단계 | 재개 시작 epoch | 단계 | 재개 시작 epoch | 단계 | 재개 시작 epoch |
  |---|---|---|---|---|---|
  | 01 EASY | 57 | 02 EASY | 24 | 03 EASY | 49 |
  | 01 MID | (재개 없음) | 02 MID | 27 | 03 MID | 11 |
  | 01 HARD | 26 | 02 HARD | 23 | 03 HARD | 3 |

  *(출처: 각 단계 `metadata.json › sessions`)*
- **왜 치명적인가**: `best_model.pth` 만 불러와 이어가면 optimizer 모멘텀 · scheduler 위치 · patience · EMA · RNG가 사라집니다. 재개 전후가 **서로 다른 학습 경로**가 되고, arm마다 재개 횟수와 시점이 다르므로 "같은 레시피"라는 전제가 깨집니다.
- **원인 분석**: 상태 저장만으로는 부족했습니다. 재개 후 배치 순서가 연속 실행과 달라지는 원인이 두 가지 더 있었습니다.
  1. PyTorch `DataLoader` 는 **처음 `iter()` 될 때** 전역 RNG에서 base seed를 한 번 뽑습니다. 이 시점이 연속 실행에서는 epoch 1, 재개 실행에서는 재개 epoch이라 이후 난수 흐름이 달라집니다.
  2. 워커 수 · `persistent_workers` 설정에 따라 증강 난수의 소비 순서가 달라집니다.
- **해결** (C6 학습 엔진)
  - `EpochSampler`: 배치 순서를 `(seed, epoch)` 만의 함수로 만듦.
  - `SplitDataset`: `(base_seed, epoch, index)` 로 샘플마다 재시드 → 워커 수와 무관하게 같은 epoch에 같은 증강.
  - `iter(train_loader)` **직후**에 epoch 단위 재시드 (코드 주석에 이유 명시).
  - 체크포인트에 model · optimizer · scheduler · GradScaler · EMA · patience · best 상태 · history · Python/NumPy/Torch/CUDA RNG를 모두 저장, `last` / `.prev` 이중 보관, "로컬 임시파일 → Drive 임시파일 → `os.replace`" 원자적 저장.
  - 체크포인트의 설정 지문이 현재 코드와 다르면 diff를 출력하고 중단, `best` 만 있고 `last` 가 없으면 "정확한 재개 불가"로 중단, optimizer · EMA 상태가 없으면 근사 재개를 기본 거부(`ALLOW_WEIGHTS_ONLY_RESUME=False`).
- **검증 근거**
  - 설계근거서 §7: 합성 데이터 반복 실험에서 중단 후 재개 결과가 끊지 않은 실행과 비트 단위로 일치(학습 기록 · best 가중치).
  - `final_comparison/budget_fairness_audit.csv › approx_resume`: 9개 단계 모두 `False`.
  - 각 단계의 `metadata.json` 에 `status = completed` 와 `best_sha256` 이 기록되고, 최종 비교 전에 `check_prerequisite_arm` 이 이 해시를 다시 대조합니다.
- **교훈**: 재현성은 시드 한 줄이 아니라 **샘플러 · 증강 · DataLoader 내부 상태 · 체크포인트 형식**까지 포함한 시스템 문제였습니다.

### 8.5 [공정성] 엔진 업그레이드 vs baseline 재사용 — 01을 다시 학습하지 않고 공정성을 지키는 방법

- **상황**: 01(ResNet18 baseline, 3단계 합계 약 420분)을 마친 뒤, 02 · 03을 위해 엔진에 가중치 EMA · 쌍 지정 혼합 · 비정상 loss 감시 · dict 형식 Random Erasing(종횡비 · 채움값 지정)을 추가해야 했습니다.
- **왜 치명적인가**: 엔진 코드가 바뀌면 "01과 02 · 03이 같은 학습 조건인가"를 보장할 수 없습니다. 반대로 공정성을 위해 01을 다시 학습하면 GPU 시간이 크게 들고, 이미 validation으로 끝낸 **진단이 무효**가 됩니다(02 · 03의 처방이 모두 01 진단에 의존).
- **해결** (C2 · C4 · C6 · C7)
  - **호환 키와 빌드를 분리**: `ENGINE_VERSION = 'cdn-engine-2.0'`(학습 조건 호환 키)은 유지하고, `ENGINE_BUILD = '2.1'` 을 따로 기록.
  - **신기능은 spec에서 켤 때만 동작**: `ema_decay`, `mix_pairs` 가 spec에 없으면 build 2.0과 같은 경로로 학습합니다. Random Erasing은 dict 형식을 새로 받되 **구형식 튜플 `(p, scale)` 도 그대로 지원**합니다.
  - arm마다 같아야 하는 조건은 `SHARED_RECIPE` 한 곳에 모으고, 그 SHA-256 지문을 모든 단계의 `metadata.json` 에 기록. 02 · 03은 `check_prerequisite_arm` 으로 선행 arm의 완료 상태 · 엔진 키 · 레시피 지문 · 분할 서명 · best 모델 해시를 대조하고, 다르면 **어느 항목이 어떻게 다른지 diff를 출력하고 중단**합니다.
  - 최종 비교 직전에 레시피 지문 · 분할 · 최대 epoch · patience · 배치 · train/val 규모가 arm 간 하나라도 다르면 `PipelineError` 로 중단.
- **검증 근거**
  - `01_*/{easy,mid,hard}/metadata.json` 에는 `engine_build` 가 없고(build 2.0), 02 · 03은 `engine_build = "2.1"` 입니다. 세 arm 모두 `engine = cdn-engine-2.0`.
  - `budget_fairness_audit.csv`: 9개 단계의 `recipe_fp = d82dd3e7935501e1`, `split = f0cb8ffb8fdfe2c5` 로 모두 동일.
  - 설계근거서 §7: build 2.0과 2.1에서 01 결과가 비트 단위로 동일, 설정 지문 동일 → 01 재사용 가능.
- **교훈**: 실험 코드를 개선하는 것과 이전 결과를 유효하게 유지하는 것은 충돌하기 쉽습니다. **"무엇이 같아야 하는가"를 데이터(지문)로 정의**해 두면 둘을 함께 얻을 수 있습니다.

### 8.6 [데이터] train ↔ test 준중복과 라벨 충돌 — 보이지 않는 누수

- **상황**: 데이터가 CIFAR-10/100 계열로 추정되며, CIFAR는 test에 train의 준중복이 섞여 있다고 알려져 있습니다(Barz & Denzler, 2020: CIFAR-10 test의 3.3%, CIFAR-100 test의 10%).
- **왜 치명적인가**: 준중복이 train에 남으면 test 성능이 부풀려지고, 이 프로젝트처럼 **방향별 오분류(수십 건 단위)** 를 비교할 때는 몇십 장의 누수만으로도 결론이 흔들릴 수 있습니다. 또 같은 원본의 파생 사본이 train과 val에 나뉘면, val로 한 진단과 best 선택이 낙관적으로 편향됩니다.
- **해결** (C4 `build_splits`)
  - MD5로 완전 중복, **dHash64 해밍 ≤ 4 그리고 dHash256 해밍 ≤ 32** 의 이중 조건으로 준중복을 판정.
  - 전수 쌍 비교(O(N²))를 피하려고 64bit를 13bit 밴드 5개로 나눠 같은 밴드 값을 가진 후보끼리만 비교 (해밍 ≤ 4인 쌍은 비둘기집 원리로 반드시 한 밴드를 공유).
  - 겹치는 이미지는 **train 쪽에서만 제외**하고 test는 건드리지 않음. 그룹의 일부만 test와 겹쳐도 **그룹 전체 제외**. 서로 다른 클래스에 같은 이미지가 있으면(라벨 충돌) 학습에서 제외.
  - 완전 중복 · 준중복 · 파생 파일명(`_aug`, `_flip`, `(1)` 등)을 **Union-Find** 로 묶어 그룹 단위 층화 분할. 분할 직후 같은 그룹이 train과 val에 걸쳐 있으면 즉시 중단.
- **검증 근거** (`_shared/data_audit/audit_summary.json`)

  | 난이도 | 완전중복 | 준중복 | 라벨 충돌 파일 | 제외 합계 |
  |---|---|---|---|---|
  | EASY | 1 | 169 | 103 | **251장** |
  | MID | 6 | 34 | 4 | **45장** |
  | HARD | 1 | 38 | 0 | **41장** |

  쌍 목록은 `train_test_duplicates_*.csv`, `label_conflicts_*.csv` 에 파일 단위로 남아 있습니다.
- **교훈**: 공개 데이터 계열이라도 "분할은 이미 되어 있다"를 믿지 않고, **감사 결과를 파일로 남기는 것**이 결론의 신뢰도를 지킵니다.

### 8.7 그 밖에 해결한 이슈

| 이슈 | 해결 | 근거 |
|---|---|---|
| **Drive I/O · CPU 증강 병목**: 95,400개의 작은 파일을 Drive에서 읽고, 224 해상도에서 CPU 증강을 수행 | 캐시를 무압축 zip 하나로 Drive에 보관 → 로컬로 한 번에 복원, val/test는 uint8 텐서로 메모리 캐시 후 GPU 정규화, 혼동행렬은 GPU `bincount` 누적, `persistent_workers` · `prefetch_factor=4` · AMP · channels_last. 데이터 · epoch을 줄여 시간을 맞추는 것은 비교 조건을 바꾸므로 금지(`TIME_BUDGET_MIN` 초과는 경고만) | 평균 epoch: 01 EASY 4분 31초, 02 EASY 4분 19초, 03 EASY 5분 05초 (`train_log.txt`) — 설계근거서 §7의 "03은 EASY에서 5분을 넘을 수 있다" 예상과 일치 |
| **XAI가 틀린 것을 보여줌**: 원본 코드가 `layer4[-1].conv2`(BN · 잔차 합산 이전)에 hook, 모델 사본 4개에 hook 누적, test 오분류 샘플 사용 | 대상 레이어를 ResNet `layer4[-1]` 블록 출력 · EfficientNet `features[-1]` 로 수정, hook 사용 후 제거, **validation 샘플에만** 적용, Score-CAM · Occlusion 배치 계산, 확신도 층화 + farthest-point 샘플 선택 | X1 셀 설명, `*_samples.csv` (확신도 · 마진 · 구간 · 최소 거리 기록) |
| **MixUp이 왜곡하는 train accuracy**: 혼합 배치의 `train_acc` 로는 과적합을 판단할 수 없음 | 증강 없는 train 층화 부분집합 3,000장으로 **일반화 격차**를 따로 측정 | `evaluation/val/generalization_gap.json` |
| **GPU 메모리 차이 · OOM** | micro-batch 자동 탐색(최대 메모리 80% 미만) + gradient accumulation으로 effective batch 64 유지, OOM 시 micro-batch를 절반으로 줄여 같은 epoch 재수행 | `metadata.json › micro_batch = 64, accum_steps = 1` (T4에서는 누적 없이 학습) |

---

## 9. 결과 해석

> 모든 수치는 **test, 단일 시드(42), 단일 실행**입니다. 통계 검정은 평가 표본의 변동만 다루며 학습 변동(시드 간 분산)은 다루지 않습니다.

### 9.1 예산 · 공정성 감사 (`budget_fairness_audit.csv`)

| arm | 난이도 | 백본 | 파라미터 | 실행 / best epoch | optimizer steps | 본 이미지 수 | 학습 시간 |
|---|---|---|---|---|---|---|---|
| 01 | EASY | ResNet18 | 11.31M | 61 / 51 | 26,474 | 1,694,336 | 276.3분 |
| 01 | MID | ResNet18 | 11.31M | 25 / 15 | 7,325 | 468,800 | 68.5분 |
| 01 | HARD | ResNet18 | 11.31M | 30 / 20 | 7,830 | 501,120 | 75.7분 |
| 02 | EASY | ResNet18 | 11.31M | 55 / 45 | 23,870 | 1,527,680 | 237.9분 |
| 02 | MID | ResNet18 | 11.31M | 61 / 51 | 17,873 | 1,143,872 | 180.1분 |
| 02 | HARD | EfficientNet-B0 | 4.34M | 54 / 44 | 14,094 | 902,016 | 145.5분 |
| 03 | EASY | ResNet50 | 24.03M | 72 / 62 | 31,248 | 1,999,872 | 366.7분 |
| 03 | MID | ResNet50 | 24.03M | 50 / 40 | 14,650 | 937,600 | 182.3분 |
| 03 | HARD | ResNet50 | 24.03M | 36 / 26 | 9,396 | 601,344 | 110.8분 |

- 레시피 지문 · 분할 · 최대 epoch · patience · 배치 · train/val 규모는 **모든 arm에서 동일**하며, 9개 단계 모두 조기 종료로 끝났습니다.
- 동일하게 맞추지 **않은** 것: 실제 실행 epoch, 파라미터 수, FLOPs. 특히 HARD에서 02는 03보다 **약 1.5배 많은 이미지**를 보고 멈췄습니다(902K vs 601K). 같은 조기 종료 규칙에서 나온 결과지만, 해석할 때 고려해야 합니다.

### 9.2 전체 지표 (test, bootstrap 95% CI)

| 난이도 | arm | accuracy | macro-F1 | weighted-F1 |
|---|---|---|---|---|
| EASY | 01 Baseline | 0.9459 (0.9404–0.9511) | 0.9012 (0.8915–0.9110) | 0.9456 |
| EASY | 02 Targeted | 0.9461 (0.9409–0.9511) | 0.9054 (0.8960–0.9145) | 0.9469 |
| EASY | 03 Uniform | **0.9494** (0.9443–0.9544) | **0.9086** (0.8987–0.9177) | 0.9499 |
| EASY | T − U | −0.0033 (−0.0083, 0.0019) | −0.0031 (−0.0122, 0.0063) | — |
| MID | 01 Baseline | 0.8894 (0.8806–0.8985) | 0.8638 (0.8533–0.8745) | 0.8894 |
| MID | 02 Targeted | **0.9083** (0.9004–0.9168) | **0.8886** (0.8792–0.8991) | 0.9092 |
| MID | 03 Uniform | 0.9019 (0.8938–0.9109) | 0.8809 (0.8709–0.8914) | 0.9030 |
| MID | T − U | +0.0064 (−0.0013, 0.0138) | +0.0077 (−0.0015, 0.0169) | — |
| HARD | 01 Baseline | 0.8305 (0.8188–0.8419) | 0.8100 (0.7971–0.8227) | 0.8288 |
| HARD | 02 Targeted | **0.8755** (0.8652–0.8855) | **0.8648** (0.8533–0.8758) | 0.8761 |
| HARD | 03 Uniform | 0.8481 (0.8376–0.8595) | 0.8354 (0.8239–0.8477) | 0.8484 |
| HARD | T − U | **+0.0274 (0.0162, 0.0383)** | **+0.0294 (0.0174, 0.0414)** | — |

전체 정오답 McNemar p(T vs U): EASY 0.218, MID 0.119, **HARD 9.5 × 10⁻⁷**.

### 9.3 ★ 겨냥 방향 (핵심 결과)

| 난이도 | 겨냥 방향 | Baseline | 02 T | 03 U | T − U (95% CI) | Holm p | 가드 | 판정 |
|---|---|---|---|---|---|---|---|---|
| EASY | cat → not | 9.5% | 4.7% | 5.0% | −0.3pp (−1.6, 1.0) | 1.000 | ✗ (G2) | NOT SUPPORTED |
| EASY | dog → not | 2.5% | 0.9% | 1.2% | −0.3pp (−1.1, 0.5) | 1.000 | ✓ | NOT SUPPORTED |
| MID | cat → not | 13.1% | 7.0% | 6.9% | +0.1pp (−1.4, 1.5) | 1.000 | ✓ | NOT SUPPORTED |
| MID | dog → not | 6.4% | 2.9% | 2.8% | +0.1pp (−0.7, 0.9) | 1.000 | ✓ | NOT SUPPORTED |
| MID | not → cat | 4.59% | 5.00% | 5.22% | −0.2pp (−1.1, 0.7) | 1.000 | ✓ | NOT SUPPORTED |
| **HARD** | **cat → not** | 21.7% | **9.9%** | 14.3% | **−4.4pp (−6.5, −2.4)** | **0.0006** | ✓ | **SUPPORTS** |
| **HARD** | **dog → not** | 10.6% | **4.8%** | 7.5% | **−2.7pp (−4.2, −1.1)** | **0.0101** | ✓ | **SUPPORTS** |
| HARD | not → cat | 5.23% | 6.95% | 7.09% | −0.1pp (−1.3, 1.0) | 1.000 | ✓ | NOT SUPPORTED |
| HARD | not → dog | 5.41% | 5.00% | 6.41% | −1.4pp (−2.5, −0.3) | 0.098 | ✓ | NOT SUPPORTED |

- **HARD cat → not**: 같은 test 고양이 1,000장 중 02만 피한 오류 80건, 03만 피한 오류 36건.
- **HARD not → dog**: 보정 전 CI는 0을 포함하지 않지만, Holm 보정 후 p = 0.098로 유의 수준에 못 미칩니다. 사전 등록한 판정 규칙대로 NOT SUPPORTED로 보고합니다.

### 9.4 대조 방향 (cat ↔ dog) — H2

| 난이도 | 대조 방향 | Baseline | 02 T | 03 U | T − U (95% CI) | p |
|---|---|---|---|---|---|---|
| EASY | cat → dog | 7.8% | 9.8% | 9.4% | +0.4pp (−1.3, 2.2) | 0.737 |
| EASY | dog → cat | 8.9% | 6.1% | 7.0% | −0.9pp (−2.6, 0.8) | 0.349 |
| MID | cat → dog | 6.0% | 7.3% | 8.7% | −1.4pp (−3.1, 0.3) | 0.131 |
| MID | dog → cat | 10.4% | 6.9% | 6.7% | +0.2pp (−1.4, 1.8) | 0.901 |
| HARD | cat → dog | 8.4% | 7.2% | 7.7% | −0.5pp (−2.2, 1.2) | 0.649 |
| HARD | dog → cat | 7.1% | 4.1% | 4.6% | −0.5pp (−1.9, 0.9) | 0.596 |

→ 6개 대조 비교 모두 유의한 차이가 없습니다. 02는 cat ↔ dog를 섞지 않았는데도 03과 비슷한 수준을 유지했습니다.

### 9.5 ★ 방향군 특이성 (test 1,000장당 오류 건수)

| 난이도 | 방향군 | Baseline | 02 T | 03 U | T − U (95% CI) | 판정 |
|---|---|---|---|---|---|---|
| EASY | 겨냥 (cat→not, dog→not) | 17.1 | 8.0 | 8.9 | −0.9 (−3.0, 1.3) | NOT SUPPORTED |
| EASY | 대조 (cat↔dog) | 23.9 | 22.7 | 23.4 | −0.7 (−4.1, 2.9) | |
| MID | 겨냥 (cat→not, dog→not, not→cat) | 67.9 | 49.8 | 50.6 | −0.9 (−7.0, 5.3) | NOT SUPPORTED |
| MID | 대조 | 34.9 | 30.2 | 32.8 | −2.6 (−7.2, 2.3) | |
| **HARD** | **겨냥 (동물↔not 네 방향)** | 132.6 | **97.6** | 122.6 | **−25.0 (−34.8, −15.0)** | **SPECIFIC** |
| HARD | 대조 | 36.9 | 26.9 | 29.3 | −2.4 (−7.6, 3.1) | |

### 9.6 해석

**① 진단의 가치는 난이도에 비례해 나타났습니다.**
EASY · MID에서는 02와 03이 동물 → not 오류를 **거의 같은 만큼** 줄였습니다(MID cat → not: 13.1% → 7.0% vs 6.9%). not이 사물 · 풍경(EASY)이나 곤충 · 어류 · 사람(MID)처럼 cat · dog와 의미적으로 먼 경우에는, 어떤 쌍을 섞든 일반적인 정규화만으로 동물 ↔ not 경계가 충분히 개선된다는 뜻으로 해석됩니다. 반면 not이 다른 포유류인 HARD에서는 경계 자체가 어려워 "어느 경계에 규제를 집중하는가"와 "어떤 특징을 쓰는가"가 차이를 만들었습니다.

**② HARD의 차이는 겨냥 방향에 특이적입니다.**
겨냥 방향군에서는 1,000장당 25.0건 적었지만 대조 방향군에서는 2.4건 차이로 유의하지 않았습니다. baseline 대비 줄어든 오류 중 겨냥 방향이 차지하는 비율도 02가 **77.8%**(−147 / −189), 03이 56.8%(−42 / −74)로, 02의 개선이 진단이 지목한 곳에 더 몰려 있었습니다.

**③ HARD에서 H3는 예상과 다르게 나왔고, 그 이유가 곧 프로젝트 주장을 뒷받침합니다.**
H3는 "전체 지표에서는 03이 비슷하거나 더 나을 수 있다"였지만, HARD에서는 02가 전체 accuracy에서도 +2.74pp 유의하게 앞섰습니다. 그러나 오류 건수를 분해하면 전체 차이 −115건(523 vs 638) 중 **−105건이 겨냥 방향군, −10건이 대조 방향군**입니다. 즉, 집계 지표의 격차는 새로운 종류의 개선이 아니라 **겨냥 방향의 격차가 집계에 드러난 것**입니다. 파라미터가 5.5배 적은 모델이 이 결과를 냈다는 점은, HARD의 문제가 "용량"이 아니라 "표현과 경계"였다는 진단과 일치합니다.

**④ 반대 방향(not → 동물)은 어느 쪽도 고치지 못했습니다.**
HARD not → cat은 두 arm 모두 baseline보다 **오히려 늘었습니다**(5.23% → 6.95% / 7.09%). MID not → cat도 두 arm 모두 소폭 증가했습니다. 동물 → not을 강하게 줄이는 방향의 규제가 경계를 동물 쪽으로 밀어낸 것으로 보이며, 02의 쌍 지정 MixUp도 이 방향에는 효과를 보이지 못했습니다. 설계근거서가 지적한 "not → cat 과신 오류 꼬리"는 이번 처방으로 해결되지 않은 과제입니다.

**⑤ EASY에서는 오류 이동의 신호가 있었습니다.**
EASY cat → not의 가드 G2(역방향 not → cat, T − U CI 0.1–0.88pp)가 실패했습니다. 02 EASY는 not → cat을 61건 → 112건으로 늘렸고, 겨냥하지 않은 방향의 오류가 baseline 대비 **+62건** 늘었습니다. 반대 방향 오류가 거의 없던 EASY에 cat ↔ not 혼합을 넣은 것이 경계를 동물 쪽으로 옮긴 것으로 보입니다. 겨냥 방향 차이 자체가 유의하지 않아 최종 판정은 NOT SUPPORTED이지만, "진단이 약한 곳에 넣은 처방에는 부작용이 따를 수 있다"는 점을 보여줍니다.

**⑥ MID에서 02의 전체 지표 우위는 겨냥 방향에서 오지 않았습니다.**
MID 전체 오류 차이 −30건(431 vs 461) 중 겨냥 방향군은 −4건뿐이고, 나머지는 대조 방향(−12건)과 not → dog(−14건)에서 나왔습니다. 전체 지표 차이도 유의하지 않으므로, MID의 우위를 진단의 효과로 해석하지 않습니다.

**⑦ EMA는 학습 곡선의 흔들림을 크게 줄였습니다.** (val, warm-up 이후 epoch 간 변화의 중앙값)

| arm | 난이도 | \|Δ val acc\| EMA / raw | \|Δ 동물→not\| EMA / raw |
|---|---|---|---|
| 02 | EASY / MID / HARD | 0.13 / 0.13 / 0.14pp ← raw 0.72 / 0.68 / 0.62pp | 0.40 / 0.30 / 0.30pp ← raw 2.30 / 2.60 / 2.25pp |
| 03 | EASY / MID / HARD | 0.07 / 0.15 / 0.17pp ← raw 0.65 / 0.80 / 1.26pp | 0.20 / 0.40 / 0.50pp ← raw 1.75 / 3.85 / 9.51pp |

baseline의 |Δ val acc|는 0.37 / 0.62 / 1.07pp였습니다. 02 · 03의 6개 단계에서 비정상 loss는 0건이었습니다.

EMA는 흔들림만 줄인 것이 아니라 **선택된 모델 자체를 바꿨습니다.** 모든 EMA 단계에서 best 시점의 EMA val macro-F1이 같은 학습의 원 가중치(raw)가 한 번이라도 기록한 최고값보다 높았습니다 (`training_history.csv › val_macro_f1` vs `val_macro_f1_raw`).

| 단계 | best epoch | best 시점 lr ÷ 기본 lr | EMA val macro-F1 (best) | raw val macro-F1 최고값 |
|---|---|---|---|---|
| 01 MID / HARD (EMA 없음) | 15 / 20 | 0.97 / 0.93 | — (raw 0.8655 / 0.8187) | — |
| 02 EASY / MID / HARD | 45 / 51 / 44 | 0.62 / 0.52 / 0.64 | 0.9117 / 0.8849 / 0.8737 | 0.9036 / 0.8796 / 0.8682 |
| 03 EASY / MID / HARD | 62 / 40 / 26 | 0.35 / 0.70 / 0.88 | 0.9133 / 0.8862 / 0.8378 | 0.9116 / 0.8717 / 0.8111 |

- MID · HARD baseline이 lr 최대 부근(0.97 / 0.93)에서 멈추던 문제는 02에서 해소되어, best 시점 lr 비율이 0.52 / 0.64로 내려갔습니다. 모든 단계의 best epoch은 62 이하로, "EMA의 최고점은 예산의 3/4 이전"이라는 설계 가정(Morales-Brotons et al. 부록 B)과 어긋나지 않습니다.
- 반면 **03 HARD는 lr 0.88 시점(epoch 26)에서 멈췄고, 원 가중치의 val macro-F1 최고값(0.8111)은 baseline(0.8187)보다도 낮았습니다.** 증강 없는 train 정확도도 0.958로 9개 단계 중 가장 낮아(`generalization_gap.json › train_eval_acc`), ResNet50 + RandAugment + 무작위 혼합의 강한 정규화가 HARD에서는 과소적합 쪽으로 작용했을 가능성이 있습니다. 03 HARD의 개선(val 0.8378)은 상당 부분 EMA에서 나왔습니다. 단, 단일 시드이므로 이는 가설 수준의 해석입니다.

**⑧ 보조 지표(TTA)도 같은 순서입니다.** TTA acc: EASY 0.9496 / 0.9477 / 0.9516, MID 0.8977 / 0.9138 / 0.9043, HARD 0.8407 / **0.8814** / 0.8512 (B / T / U).

### 9.7 가설 판정 요약

| 가설 | 판정 | 비고 |
|---|---|---|
| H1 (겨냥 방향 T < U) | **HARD에서 부분 지지** | 9개 겨냥 방향 중 2개(HARD cat→not, dog→not) 유의, 가드 통과 |
| H2 (대조 방향 T ≈ U) | **지지** | 6개 대조 방향 모두 유의차 없음, HARD `SPECIFIC` |
| H3 (전체 지표 T ≤ U 가능) | EASY · MID 부합, **HARD 불부합** | HARD의 전체 격차 −115건 중 −105건이 겨냥 방향군 |

**결론**: 프로젝트 주장("진단 기반 개선은 겨냥한 오분류 방향에서 일괄 확대와 다른 결과를 낸다")은 **HARD 조건에서 지지**되고, **EASY · MID 조건에서는 지지되지 않았습니다.** 단일 시드 결과이므로, 이 결론은 다중 시드 반복으로 확인되기 전까지 잠정적입니다.

---

## 10. 시각화

> 아래 경로는 저장소 루트 기준입니다. 한글 폰트 환경 차이를 고려해 그림 제목은 영문 위주로 작성했습니다.

### 10.1 01 진단 개요 — 집계 지표는 "고르게" 떨어지지만, 방향으로 쪼개면 특정 경계만 움직인다

![diagnosis overview](experiments/01_baseline_diagnosis/diagnosis/diagnosis_overview.png)

왼쪽부터 집계 지표(acc · macro-F1), 6개 방향별 val 오분류율(단조 증가 방향은 굵은 선), 일반화 격차입니다. 동물 ↔ not 네 선만 EASY → HARD로 올라가고 cat ↔ dog 두 선은 평평합니다.

### 10.2 핵심 결과 — 전체 지표 vs 겨냥 방향

![core result](experiments/final_comparison/figures/core_result_aggregate_vs_target.png)

왼쪽은 test macro-F1(95% CI), 오른쪽은 9개 겨냥 방향의 오분류율과 T − U · Holm p · 판정입니다. HARD cat → not · dog → not에서 파란 막대(02)가 주황 막대(03)보다 확연히 낮습니다.

### 10.3 T − U 포레스트 플롯

![forest](experiments/final_comparison/figures/target_T_minus_U_forest.png)

겨냥 방향별 T − U와 95% bootstrap CI입니다. 점선(0)의 왼쪽이 02가 우세한 쪽이며, CI 전체가 0보다 왼쪽인 항목은 파란색으로 표시됩니다(HARD not → dog는 보정 전 CI만 음수, Holm 보정 후에는 유의하지 않음).

### 10.4 baseline 대비 방향별 변화 히트맵

![heatmap](experiments/final_comparison/figures/direction_delta_heatmap.png)

왼쪽 T − B, 오른쪽 U − B (pp, 굵은 테두리 = 겨냥 방향). HARD cat → not에서 02는 −11.8pp, 03은 −7.4pp입니다. not → cat 행은 두 arm 모두 붉은색(증가)입니다.

### 10.5 난이도별 6방향 오분류율

![direction rates](experiments/final_comparison/figures/direction_rates_by_difficulty.png)

★ 표시가 겨냥 방향입니다. 대조 방향(cat → dog, dog → cat)에서 파란 막대와 주황 막대가 비슷한지 확인할 수 있습니다.

### 10.6 그 밖의 그림

| 그림 | 경로 | 내용 |
|---|---|---|
| 학습 곡선 | `experiments/{arm}/{easy,mid,hard}/figures/training_curve.png` | loss, val acc · macro-F1(EMA vs raw), lr, 방향별 val 오분류율 (4패널, best epoch 점선) |
| 혼동행렬 | `experiments/{arm}/{easy,mid,hard}/figures/confusion_matrix_{val,test}.png` | 건수 · 행 정규화 |
| 01 XAI | `experiments/01_baseline_diagnosis/diagnosis/xai/{easy,mid,hard}/` | 6개 오류 방향 + 3개 정답 클래스 × 5종(Grad-CAM, Grad-CAM++, Score-CAM, Occlusion, LIME) + 예측 확률 |
| 02 XAI | `experiments/02_targeted_improvement/xai/{own,before_after}/` | 02 자신의 오분류 / baseline이 틀린 같은 샘플의 전후 비교 |
| 03 XAI | `experiments/03_uniform_scaleup/xai/own/{easy,mid,hard}/` | 03 자신의 val 오분류 · 정답 샘플 (학습 후 분석 전용, 처방에는 미사용) |
| 3-arm XAI | `experiments/final_comparison/xai_three_arms/{easy,mid,hard}/` | baseline이 겨냥 방향으로 틀린 같은 val 샘플을 B · T · U로 나란히 비교 |

---

## 11. 최종 회고 및 성찰

### 11.1 잘한 점

- **"무엇이 좋아졌나"보다 "어디가 좋아졌나"를 물었습니다.** 방향별 분해가 없었다면 HARD의 +2.7pp accuracy 차이를 "EfficientNet이 더 좋은 모델"로만 설명했을 것입니다. 오류 건수 분해 덕분에 그 차이의 91%가 진단이 지목한 경계에서 나왔다는 것을 보일 수 있었습니다.
- **반증 가능한 형태로 설계를 고쳤습니다.** 초기 설계(02도 무작위 혼합)는 결과가 어떻게 나오든 주장을 검증할 수 없는 구조였습니다. 쌍 지정 혼합과 대조 방향 · 방향군 특이성을 도입해 "틀릴 수 있는 가설"로 바꾼 것이 이 프로젝트에서 가장 중요한 결정이었습니다.
- **공정성과 맹검을 사람의 주의가 아니라 코드로 강제했습니다.** 레시피 지문 · 분할 서명 · 모델 해시 대조, 근거 조건 자동 검증, 처방 잠금, test 1회 접근 기록은 "결과를 보고 설정을 바꾸고 싶은 유혹"을 원천적으로 막았습니다.
- **가드가 실제로 작동했습니다.** "오류를 옮겨도 이긴 것처럼 보이는" 지표의 함정을 미리 막아 둔 덕분에, EASY에서 02가 not → cat 오류를 늘린 부작용(61 → 112건)이 자동으로 드러났습니다.
- **불리한 결과를 그대로 보고했습니다.** EASY · MID의 NOT SUPPORTED, EASY의 가드 실패, HARD H3의 불부합, not → cat 악화를 모두 남겼습니다.

### 11.2 아쉬운 점과 한계

- **단일 시드**: bootstrap과 McNemar는 평가 표본의 변동만 다룹니다. RSB(§4.2)도 강조하듯 시드 간 분산이 이 정도 차이를 만들 수 있으므로, 결론은 잠정적입니다.
- **HARD의 복합 개입**: 02 HARD는 백본(EfficientNet-B0)과 증강(쌍 지정 혼합)을 함께 바꿔, 개선이 어느 쪽에서 왔는지 분리할 수 없습니다. 가장 강한 결과가 나온 조건에서 가장 큰 교란이 있다는 점이 가장 아쉽습니다.
- **계산량 불일치**: epoch · 조기 종료 규칙 · 레시피는 같지만 파라미터 · FLOPs · 실제로 본 이미지 수는 다릅니다. HARD에서 02는 03보다 약 1.5배 많은 이미지를 봤습니다.
- **03은 관행 전체가 아니라 한 경로입니다.** ResNet50 + RSB 계열 레시피는 "일괄 확대"의 대표적 예시일 뿐, 다른 관행(예: 더 큰 EfficientNet, ViT)을 대표하지 않습니다.
- **완전한 맹검이 아닙니다.** 처방은 이번 01의 validation으로만 도출했지만, 설계자는 원본 프로젝트의 과거 test 결과를 본 적이 있습니다.
- **val 재사용**: 같은 val로 best 선택과 진단을 함께 했으므로 val 수치는 낙관적입니다. 최종 판정을 test로만 한 이유입니다.
- **반대 방향 미해결**: not → cat 과신 오류 꼬리를 진단해 놓고도 이를 줄이는 처방은 찾지 못했습니다.

### 11.3 배운 점

- 집계 지표는 **요약**이지 **진단**이 아닙니다. 같은 accuracy 향상도 어디서 왔는지에 따라 전혀 다른 의미를 가집니다.
- "처방이 효과가 있었는가"를 보려면 **처방이 닿지 않아야 할 곳(대조 방향)** 을 함께 설계해야 합니다.
- 진단의 가치는 문제의 난이도에 따라 달랐습니다. 쉬운 문제에서는 일반적 정규화로 충분했고, 어려운 문제에서 진단이 차이를 만들었습니다. "언제 진단이 필요한가"도 진단의 대상이라는 것을 배웠습니다.
- 재현성은 난수 시드 한 줄이 아니라 **샘플러 · 증강 · DataLoader 내부 상태 · 체크포인트 형식**까지 포함한 시스템 문제였습니다.

---

## 12. 향후 발전 계획

| 우선순위 | 계획 | 목적 |
|---|---|---|
| ★★★ | **다중 시드 반복** (최소 3개 시드) | 시드 간 분산을 포함한 효과 크기 추정, HARD 결론의 견고성 확인 |
| ★★★ | **HARD 분해 실험**: ① EfficientNet-B0 + 무작위 혼합, ② ResNet18 + 쌍 지정 혼합, ③ ResNet50 + 쌍 지정 혼합 | 백본 효과와 쌍 지정 혼합 효과의 분리 (2 × 2 요인 설계) |
| ★★☆ | **계산량 정합 비교** | 본 이미지 수 또는 FLOPs를 맞춘 조건에서 재비교 |
| ★★☆ | **not → 동물 방향 전용 처방** | 과신 오류 꼬리 대응 (예: 비대칭 쌍 가중치, 과신 샘플 재가중, 보정 기법) 및 EASY의 오류 이동 억제 |
| ★★☆ | **CP-Mix 완전 구현과 비교** | 혼동 쌍의 온라인 추정 + 라벨 혼합식 보정을 넣은 원 기법과 단순화 변형의 비교 |
| ★★☆ | **03 HARD의 정규화 강도 점검** | 원 가중치 val F1(0.8111)이 baseline보다 낮았던 원인 확인 — RandAugment 강도 · 혼합 확률 스윕 (validation 기준) |
| ★☆☆ | **새 test 세트로 맹검 평가** | 설계자의 과거 test 노출 문제 제거 |
| ★☆☆ | **32 × 32 네이티브 설정 실험** | 224 확대 없이 CIFAR용 구조(WRN, PreAct-ResNet)로 같은 비교를 수행해 해상도 의존성 확인 |
| ★☆☆ | **코드 모듈화** | 공통 셀 C1–C7을 `src/` 패키지로 분리, CLI 실행, `CDN_SMOKE_TEST` 기반 CI 스모크 테스트 |
| ★☆☆ | **그림 폰트 수정** | 일부 그림 제목의 유니코드 마이너스(U+2212, 예: "T−U")가 NanumGothic에서 □로 표시되는 문제 → ASCII 하이픈으로 교체하거나 DejaVu Sans 대체 폰트 지정 |

---

## 13. 참고 문헌

### 13.1 처방 · 실험 설계

1. Zhong, Z., Zheng, L., Kang, G., Li, S., Yang, Y. **Random Erasing Data Augmentation.** AAAI 2020. [arXiv:1708.04896](https://arxiv.org/abs/1708.04896) · [GitHub](https://github.com/zhunzhong07/Random-Erasing)
   — 사용 위치: §1, §5.1.2, Table 1 · 2 · 4, Figure 4 · 5
2. Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., Yoo, Y. **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features.** ICCV 2019. [arXiv:1905.04899](https://arxiv.org/abs/1905.04899) · [GitHub](https://github.com/clovaai/CutMix-PyTorch)
   — 사용 위치: §1, §3.1, §3.2(Figure 2), §4.1.2(Table 5 · 6), §4.1.3(Figure 3), §4.4
3. Zhang, H., Cisse, M., Dauphin, Y. N., Lopez-Paz, D. **mixup: Beyond Empirical Risk Minimization.** ICLR 2018. [arXiv:1710.09412](https://arxiv.org/abs/1710.09412) · [GitHub](https://github.com/facebookresearch/mixup-cifar10)
   — 사용 위치: §2(Figure 1b, Figure 2), §3.1, §3.2(Figure 3a)
4. Yoon, Y., Hong, S., Joo, H., Qin, Y., Jeong, H., Lee, J. **Mix from Failure: Confusion-Pairing Mixup for Long-Tailed Recognition.** 2024. [arXiv:2411.07621](https://arxiv.org/abs/2411.07621)
   — 사용 위치: 초록, Figure 2 · 3 (혼동 쌍에서 표본을 뽑아 섞는 착상)
5. Kornblith, S., Shlens, J., Le, Q. V. **Do Better ImageNet Models Transfer Better?** CVPR 2019. [arXiv:1805.08974](https://arxiv.org/abs/1805.08974)
   — 사용 위치: 초록, Figure 1 · 2
6. Tan, M., Le, Q. V. **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.** ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
   — 사용 위치: 초록, 전이 학습 실험
7. Morales-Brotons, D., Vogels, T., Hendrikx, H. **Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits.** TMLR 2024. [arXiv:2411.18704](https://arxiv.org/abs/2411.18704)
   — 사용 위치: §3.2, §3.4, §4.1, 부록 F
8. Polyak, B. T., Juditsky, A. B. **Acceleration of Stochastic Approximation by Averaging.** SIAM Journal on Control and Optimization, 30(4), 1992.
9. Cai, Z., Ravichandran, A., Maji, S., Fowlkes, C., Tu, Z., Soatto, S. **Exponential Moving Average Normalization for Self-supervised and Semi-supervised Learning.** CVPR 2021. [arXiv:2101.08482](https://arxiv.org/abs/2101.08482)
   — EMA에서 BN 통계도 함께 평균하는 방식
10. Wightman, R., Touvron, H., Jégou, H. **ResNet strikes back: An improved training procedure in timm.** 2021. [arXiv:2110.00476](https://arxiv.org/abs/2110.00476) · [GitHub (timm)](https://github.com/huggingface/pytorch-image-models)
    — 사용 위치: §1, §3, §4.2, §4.4, Table 2 · 11, 부록 A
11. Cubuk, E. D., Zoph, B., Shlens, J., Le, Q. V. **RandAugment: Practical Automated Data Augmentation with a Reduced Search Space.** NeurIPS 2020 / CVPRW 2020. [arXiv:1909.13719](https://arxiv.org/abs/1909.13719)
    — 사용 위치: Figure 2 (N = 2, ResNet-50 최적 M = 9), 모델 크기와 최적 강도의 관계
12. Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., Jégou, H. **Training data-efficient image transformers & distillation through attention (DeiT).** ICML 2021. [arXiv:2012.12877](https://arxiv.org/abs/2012.12877) · [GitHub](https://github.com/facebookresearch/deit)
13. He, K., Zhang, X., Ren, S., Sun, J. **Deep Residual Learning for Image Recognition.** CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

### 13.2 진단 · 해석 · 데이터

14. Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., Kumar, S. **Long-tail Learning via Logit Adjustment.** ICLR 2021. [arXiv:2007.07314](https://arxiv.org/abs/2007.07314)
    — 제외한 처방 계열 (임계값 실험으로 배제)
15. Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., Brendel, W. **ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness.** ICLR 2019. [arXiv:1811.12231](https://arxiv.org/abs/1811.12231)
    — HARD 질감 오류의 해석 (가설)
16. Barz, B., Denzler, J. **Do We Train on Test Data? Purging CIFAR of Near-Duplicates.** Journal of Imaging, 2020. [arXiv:1902.00423](https://arxiv.org/abs/1902.00423)
    — 중복 감사의 근거 (CIFAR-10 3.3%, CIFAR-100 10%)
17. Krizhevsky, A. **Learning Multiple Layers of Features from Tiny Images.** Technical Report, 2009. (CIFAR-10/100)

### 13.3 XAI

18. Selvaraju, R. R. et al. **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.** ICCV 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
19. Chattopadhay, A. et al. **Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks.** WACV 2018. [arXiv:1710.11063](https://arxiv.org/abs/1710.11063)
20. Wang, H. et al. **Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks.** CVPRW 2020. [arXiv:1910.01279](https://arxiv.org/abs/1910.01279)
21. Zeiler, M. D., Fergus, R. **Visualizing and Understanding Convolutional Networks.** ECCV 2014. [arXiv:1311.2901](https://arxiv.org/abs/1311.2901) — Occlusion
22. Ribeiro, M. T., Singh, S., Guestrin, C. **"Why Should I Trust You?": Explaining the Predictions of Any Classifier.** KDD 2016. [arXiv:1602.04938](https://arxiv.org/abs/1602.04938) · [GitHub (lime)](https://github.com/marcotcr/lime)

### 13.4 통계

23. McNemar, Q. **Note on the sampling error of the difference between correlated proportions or percentages.** Psychometrika, 12(2), 1947.
24. Holm, S. **A Simple Sequentially Rejective Multiple Test Procedure.** Scandinavian Journal of Statistics, 6(2), 1979.
25. Efron, B., Tibshirani, R. J. **An Introduction to the Bootstrap.** Chapman & Hall, 1993.

### 13.5 소프트웨어

- PyTorch · torchvision — [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch), [github.com/pytorch/vision](https://github.com/pytorch/vision), [torchvision 모델 · 가중치 문서](https://pytorch.org/vision/stable/models.html)
- timm (pytorch-image-models) — [github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- LIME — [github.com/marcotcr/lime](https://github.com/marcotcr/lime)
- SciPy · scikit-learn · pandas · NumPy · Matplotlib

---

<sub>실험 기록: 엔진 호환 키 `cdn-engine-2.0` (build 2.1) · 레시피 지문 `d82dd3e7935501e1` · 분할 서명 `f0cb8ffb8fdfe2c5` · seed 42 · test 접근 1회 (2026-09-20) · 최종 보고서 `experiments/final_comparison/FINAL_REPORT.md`</sub>
