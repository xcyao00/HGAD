# HGAD Continual Learning

이 프로젝트는 HGAD (Hierarchical Gaussian Mixture Normalizing Flow Modeling for Unified Anomaly Detection)를 continual learning 환경에서 사용할 수 있도록 확장한 버전입니다.

## Continual Learning 개요

continual learning은 새로운 task들을 순차적으로 학습하면서 이전 task들의 지식을 유지하는 학습 방법입니다. 이 구현에서는 MVTecAD 데이터셋의 15개 클래스를 3개씩 5개의 task로 나누어 순차적으로 학습합니다.

### Task 분할

- **Task 0**: `['bottle', 'cable', 'capsule']`
- **Task 1**: `['carpet', 'grid', 'hazelnut']`
- **Task 2**: `['leather', 'metal_nut', 'pill']`
- **Task 3**: `['screw', 'tile', 'toothbrush']`
- **Task 4**: `['transistor', 'wood', 'zipper']`

## 파일 구조

```
continual_main.py        # Continual learning을 위한 메인 스크립트
continual_train.py       # Continual learning 학습 로직
run_continual_learning.py # 실행 예제 스크립트
models/model.py          # 확장된 HGAD 모델 (expand_for_new_classes 메서드 포함)
```

## 사용 방법

### 1. 기본 실행

```bash
python continual_main.py
```

### 2. 매개변수 설정

```bash
python continual_main.py \
    --dataset mvtec \
    --img_size 512 \
    --batch_size 4 \
    --meta_epochs 10 \
    --sub_epochs 4 \
    --lr 2e-4 \
    --gpu 0 \
    --seed 42 \
    --num_tasks 5 \
    --classes_per_task 3 \
    --output_dir ./continual_outputs
```

### 3. 편리한 실행 스크립트 사용

```bash
# 전체 continual learning 실행
python run_continual_learning.py

# 빠른 테스트 실행 (2개 task만)
python run_continual_learning.py --quick
```

## 주요 매개변수

- `--num_tasks`: continual learning task 수 (기본값: 5)
- `--classes_per_task`: 각 task당 클래스 수 (기본값: 3)
- `--meta_epochs`: 메타 에포크 수 (기본값: 25)
- `--sub_epochs`: 서브 에포크 수 (기본값: 8)
- `--img_size`: 이미지 크기 (기본값: 1024)
- `--batch_size`: 배치 크기 (기본값: 8)
- `--lr`: 학습률 (기본값: 2e-4)

## 출력 파일

학습 과정에서 다음과 같은 파일들이 생성됩니다:

### 로그 파일
- `logs/training_log_task_{task_id}.csv`: 각 task의 학습 로그
- `logs/task_{task_id}/`: 각 task의 모델 체크포인트

### 결과 파일
- `results/evaluation_results_task_{task_id}.csv`: 각 task의 평가 결과
- `results/unified_evaluation_results.csv`: 모든 task의 통합 평가 결과
- `results/continual_results_after_task_{task_id}.csv`: 각 task 후 전체 결과
- `results/final_continual_summary.csv`: 최종 요약 결과

## 평가 결과 저장 시스템

### 1. Task별 평가 결과 (`evaluation_results_task_{task_id}.csv`)
- 각 task의 학습 중 평가 결과를 저장
- 해당 task의 클래스들에 맞는 헤더 구조 사용
- 헤더 호환성 검사를 통해 파일 무결성 보장

### 2. 통합 평가 결과 (`unified_evaluation_results.csv`)
- 모든 task의 평가 결과를 하나의 파일에 저장
- 모든 클래스에 대한 column을 포함하여 헤더 고정
- 평가되지 않은 클래스는 -1로 표시
- Continual learning 전체 진행 상황을 한 눈에 파악 가능

### 3. 헤더 호환성 검사
- 기존 파일의 헤더와 현재 저장하려는 데이터의 헤더 구조 비교
- 불일치시 경고 메시지 출력 및 파일 재생성
- 데이터 무결성 및 분석 시 오류 방지

## Continual Learning 특징

### 1. 모델 확장
- 새로운 task가 시작될 때 모델의 클래스 관련 파라미터들이 자동으로 확장됩니다.
- `expand_for_new_classes()` 메서드를 통해 다음 파라미터들이 확장됩니다:
  - `mus`: 메인 클래스 센터
  - `mu_deltas`: 델타 클래스 센터
  - `phi_intras`: intra-class 가중치
  - `phi_inters`: inter-class 가중치

### 2. 글로벌 클래스 매핑
- 모든 클래스에 대해 글로벌 ID가 할당됩니다.
- 각 task에서 사용되는 클래스들은 글로벌 ID를 통해 매핑됩니다.

### 3. 평가 방식
- 각 task 학습 후 해당 task에 대한 평가
- 모든 이전 task들에 대한 평가 (catastrophic forgetting 측정)
- Task별 평균 성능 및 전체 평균 성능 계산

## 예제 실행 결과

```
Starting Continual Learning Training
Tasks: [['bottle', 'cable', 'capsule'], ['carpet', 'grid', 'hazelnut'], ...]

=== Training Task 0: ['bottle', 'cable', 'capsule'] ===
...
Task 0 Average Image AUC: 0.850, Average Pixel AUC: 0.920

=== Evaluating all tasks after Task 0 ===
Results after Task 0:
bottle: Image AUC: 0.860, Pixel AUC: 0.930
cable: Image AUC: 0.840, Pixel AUC: 0.910
capsule: Image AUC: 0.850, Pixel AUC: 0.920
Overall Average Image AUC: 0.850
Overall Average Pixel AUC: 0.920

=== Training Task 1: ['carpet', 'grid', 'hazelnut'] ===
...
```

## 주의사항

1. **메모리 사용량**: continual learning은 모든 클래스에 대한 파라미터를 유지하므로 메모리 사용량이 높을 수 있습니다.

2. **데이터 경로**: MVTecAD 데이터셋이 `/Volume/VAD/Data/MVTecAD`에 있어야 합니다.

3. **GPU 메모리**: 큰 이미지 크기나 배치 크기를 사용할 때 GPU 메모리 부족에 주의하세요.

## 문제 해결

### 메모리 부족 시
- `--img_size`를 줄여보세요 (예: 512 또는 256)
- `--batch_size`를 줄여보세요 (예: 2 또는 4)

### 빠른 테스트를 위해
- `--meta_epochs`와 `--sub_epochs`를 줄여보세요
- `--num_tasks`를 줄여서 일부 task만 실행해보세요

### 데이터 경로 오류 시
- `continual_main.py`의 `data_path` 설정을 확인하세요
- MVTecAD 데이터셋이 올바른 경로에 있는지 확인하세요

## 성능 분석

생성된 CSV 파일들을 통해 다음을 분석할 수 있습니다:

1. **각 task의 학습 진행상황** (`training_log_task_{task_id}.csv`)
2. **각 task별 성능** (`evaluation_results_task_{task_id}.csv`)
3. **Catastrophic forgetting 분석** (`continual_results_after_task_{task_id}.csv`)
4. **전체 성능 요약** (`final_continual_summary.csv`)

## 확장 가능성

이 구현은 다음과 같이 확장 가능합니다:

1. **다른 데이터셋 지원**: `continual_train.py`의 `create_task_dataset` 함수 수정
2. **다른 task 분할 방식**: `CONTINUAL_TASKS` 변수 수정
3. **Regularization 기법 추가**: catastrophic forgetting을 줄이기 위한 방법들 추가
4. **Memory replay 기법**: 이전 task의 일부 데이터를 저장하여 재학습

이 continual learning 구현을 통해 HGAD 모델의 지속적 학습 능력을 평가하고 개선할 수 있습니다. 