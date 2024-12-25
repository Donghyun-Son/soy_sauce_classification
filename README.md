# 간장 숙성 기간 분류기

이미지를 통해 간장의 숙성 기간을 분류하는 딥러닝 모델입니다.

## 환경 설정

### 필수 라이브러리 설치

1. PyTorch 설치:
   
   a. conda 사용 시 (권장):
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```
   
   b. pip 사용 시:
      1. requirements.txt 파일의 pytorch 관련 주석을 해제
      ```bash
      # --extra-index-url https://download.pytorch.org/whl/cu118
      # torch
      # torchvision
      # torchaudio
      ```

      2. pip install 명령어 실행
      
      ```bash
      pip install -r requirements.txt
      ```

2. 기타 필요 라이브러리 설치:
   ```bash
   pip install -r requirements.txt
   ```

> Note: PyTorch는 CUDA 11.8 버전을 기준으로 설치됩니다. 다른 CUDA 버전을 사용하는 경우 설치 명령어의 cu118 부분을 적절히 수정해주세요.

## 프로젝트 구조

```
soy_sauce_classification/
├── data/
│   └── soy_sauce/
│       ├── day0/        # 0일차 숙성 이미지
│       ├── day30/       # 30일차 숙성 이미지
│       ├── day60/       # 60일차 숙성 이미지
│       ├── day60/       # 90일차 숙성 이미지
│       ├── train/       # 학습용 데이터 (data_separator.py 실행 시 자동 생성)
│       └── val/         # 검증용 데이터 (data_separator.py 실행 시 자동 생성)
├── data_separator.py    # 데이터셋 분리
├── dataset.py          # 데이터 로더
├── train.py           # 모델 학습
├── test.py            # 모델 평가
├── inference.py       # 단일 이미지 추론
└── requirements.txt    # 의존성 패키지 설치 목록
```

## 모델 구조

- 베이스 모델: EfficientNetV2-S
- 전이학습 적용
- 출력: 숙성 기간별 분류 (day0, day30, day60, ...)

## 데이터 준비

1. 데이터셋 구조:
   - 파일명 형식: `ganjang_d{숙성일수}_*.jpg`
   - 예: `ganjang_d60_e20_9_6.jpg`

2. 데이터 분리:
```bash
python data_separator.py data/soy_sauce
```
- 실행 후 train/val 폴더로 자동 분리됨

## 모델 학습

```bash
python train.py \
    --data_dir data/soy_sauce \
    --model_dir models \
    --batch_size 16 \
    --epochs 5 \
    --lr 0.0001
```

주요 인자:
- `--data_dir`: 데이터셋 경로
- `--model_dir`: 모델 저장 경로
- `--batch_size`: 배치 크기
- `--epochs`: 학습 에폭 수
- `--lr`: 학습률

## 모델 테스트

```bash
python test.py \
    --model_path models/best.pth \
    --data_dir data/soy_sauce \
    --batch_size 16
```

결과:
- 클래스별 정확도
- Confusion Matrix (confusion_matrix.png로 저장)
- 전체 정확도

## 단일 이미지 추론

```bash
python inference.py \
    --model_path models/best.pth \
    --image_path path/to/image.jpg
```
