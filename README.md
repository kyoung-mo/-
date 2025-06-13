# 최종 프로젝트 – CNN 성능 개선 실험 전략

## 프로젝트 목표

- **200 클래스, 64x64 RGB 이미지(클래스당 500장) 데이터셋**
- 기본 제공 ipynb 기반 CNN 분류 모델의 **정확도 최대화**  
- 데이터 전처리, 증강, 모델 구조 개선, 과적합 방지, 최적화, QAT 및 TensorRT 적용 등  
- 실험별 결과와 원인 분석, 개선 방법을 **결과 보고서에 체계적으로 작성**

---

## 실험 단계별 추천 플로우

### 1. 데이터 전처리 및 증강
- **정규화 / 표준화:**  
  - `transforms.ToTensor()` (0~1로 정규화)  
  - `transforms.Normalize(mean, std)` (채널별 평균·표준편차 적용)
- **데이터 증강:**  
  - `RandomHorizontalFlip`, `RandomCrop`, `ColorJitter`, `Cutout` 등  
  - 추가 실험: `AutoAugment`, `Mixup`, `CutMix` 등 최신 증강 기법  
- **테스트셋**에는 증강 적용 ❌ (평가 왜곡 주의)

### 2. CNN 모델 구조 개선
- **기본 CNN 구조 → 다양한 아키텍처 실험**  
  - Conv 레이어/채널/커널/스트라이드/필터수 등 수정  
  - `BatchNorm2d`, `Dropout` 레이어 추가  
  - `Residual Block`, `Squeeze-and-Excitation` 등 고급 블록 추가  
  - `Activation`: ReLU, LeakyReLU, ReLU6 등 바꿔서 실험  
  - `AvgPooling`/`GlobalAvgPooling` 실험 (특히 QAT에서 성능 도움)

### 3. 과적합 방지 기법 적용
- **BatchNorm**: 학습 안정화 및 일반화 성능 향상
- **Dropout**: 뉴런 랜덤 제거, 과적합 방지 (적용 위치/비율 실험)
- **Label Smoothing**: CrossEntropyLoss에서 `label_smoothing` 사용
- **EarlyStopping**: 성능 정체 시 조기 중단
- **Weight Decay(L2 정규화)**: Optimizer에 적용

### 4. 최적화/하이퍼파라미터 튜닝
- **Optimizer**: SGD(+momentum), Adam, AdamW 등 성능 비교
- **Learning Rate, Scheduler**: StepLR, ReduceLROnPlateau, CosineAnnealing 등  
- **Batch Size/Epoch**: 여러 값 실험  
- **실험별 로그, 그래프, 모델 저장** (훈련/검증 acc/loss, confusion matrix 등)

### 5. 성능 평가 및 시각화
- **훈련/검증 정확도 및 loss 그래프**
- **Confusion matrix, misclassified sample 시각화**
- **각 실험별 개선 전후 수치, 그래프 보고서에 포함**

### 6. QAT 및 TensorRT 적용 (Jetson Orin Nano 평가)
- **PTQ → QAT → TensorRT 변환**
- **Jetson Orin Nano에서 추론 후**  
  - **정확도, 속도, 전력, 메모리** 측정
- **최종 성능 보고서에 정리**

---

## 실습 진행 팁

- **한 번에 여러 변수를 바꾸지 말고**, 한 번에 하나씩 실험하여 개선 효과를 명확히 파악
- **각 실험마다 코드/결과/그래프/느낀점**을 별도로 기록 (보고서 활용)
- **모델 구조/파라미터/전처리/증강 기법/하이퍼파라미터/실험 환경** 등 로그 남기기
- **최고 성능 조합을 찾는 과정** 자체가 보고서에서 매우 중요

---

## 예시 코드 스니펫

```python
# 데이터 증강 예시 (torchvision)
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(64, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
