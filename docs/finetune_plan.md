# 1차 Fine-tuning 실행 계획 (2026-04-28)

> 분석 방법: sequential-thinking MCP 9-step 체인  
> 수치 근거 없는 주장 제외 / 추정치는 (추정) 명시

---

## 현재 상태

| 항목 | 내용 |
|------|------|
| 분류 대상 | 초인종 / 노크 / 화재경보 (3종) |
| 모델 | YAMNet Fine-tuning (TensorFlow + Keras) |
| 목표 정확도 | 90% 이상 |
| 데이터 | train 1,954 / val 434 / test 410 (파일 단위 분할) |
| 클립 현황 | doorbell 436 / knock 714 / fire_alarm 1,648 |
| 신뢰도 임계값 | 70% |
| 직접 녹음 | 미완료 (초인종 90 / 화재경보 240 / 노크 180클립 예정) |

---

## 항목 1. 모델 아키텍처

### YAMNet frozen 2단계 전략

데이터 2,798클립 규모에서 전체 언프리징 시 catastrophic forgetting + 과적합 위험.  
YAMNet이 이미 AudioSet에서 doorbell/knock/fire_alarm을 학습했으므로 frozen이 기본 전략.

| 단계 | YAMNet | Dense 레이어 | 목적 |
|------|--------|-------------|------|
| 1단계 | 완전 frozen | 학습 | 빠른 수렴, 과적합 방지 |
| 2단계 | 상위 conv 2~3레이어 unfreeze | 미세조정 | 한국 도메인 적응 |

**Dense 레이어:** YAMNet embedding(1024-dim) → Dropout(0.3) → Dense(3, softmax)

```python
# train/model.py
import tensorflow as tf
import tensorflow_hub as hub
import os

os.environ['TFHUB_CACHE_DIR'] = './tfhub_cache'

def build_classifier():
    inputs = tf.keras.Input(shape=(1024,), name='embedding')
    x = tf.keras.layers.Dropout(0.3)(inputs)
    outputs = tf.keras.layers.Dense(3, activation='softmax', name='output')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

def extract_embeddings(waveforms):
    # waveforms: (batch, 48000) float32 [-1, 1]
    embeddings_batch = tf.vectorized_map(
        lambda x: yamnet(x)[1],  # [1] = embeddings (time_frames, 1024)
        waveforms
    )
    return tf.reduce_mean(embeddings_batch, axis=1)  # (batch, 1024)
```

---

## 항목 2. 학습 파라미터

| 파라미터 | 1단계 | 2단계 | 근거 |
|---------|-------|-------|------|
| optimizer | Adam | Adam | 소규모 데이터 + 빠른 수렴 |
| learning rate | 1e-3 | 1e-4 (1/10) | 언프리징 시 사전학습 파괴 방지 |
| batch size | 32 | 32 | MacBook M4 메모리 충분 |
| max epoch | 30 | 20 | EarlyStopping으로 10~20 수렴 예상 |
| EarlyStopping patience | 5 | 3 | |
| monitor | val_loss | val_loss | 불균형 데이터에서 accuracy misleading |

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5,
        restore_best_weights=True, min_delta=1e-4
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'checkpoints/best_model.h5', monitor='val_loss', save_best_only=True
    ),
]
```

---

## 항목 3. class_weight 및 sample_weight

### class_weight 실제 수치 (train 기준: doorbell 306 / knock 497 / fire_alarm 1,151)

```
class_weight[doorbell]   = 1,954 / (3 × 306)   = 2.129
class_weight[knock]      = 1,954 / (3 × 497)   = 1.311
class_weight[fire_alarm] = 1,954 / (3 × 1,151) = 0.566
```

불균형 비율: doorbell : knock : fire_alarm = **3.76 : 2.32 : 1.0** (fire_alarm 기준)

### sample_weight 통합 (class_weight × AI Hub 1.5배 결합)

```python
# train/train.py
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

LABEL_MAP = {'doorbell': 0, 'knock': 1, 'fire_alarm': 2}
KOREAN_WEIGHT = 1.5  # AI Hub 클립 강조

def compute_sample_weights(df):
    y = np.array([LABEL_MAP[l] for l in df['label']])
    cw = compute_class_weight('balanced', classes=np.unique(y), y=y)
    cw_dict = dict(enumerate(cw))  # {0:2.129, 1:1.311, 2:0.566}

    return np.array([
        cw_dict[LABEL_MAP[row['label']]] *
        (KOREAN_WEIGHT if row['source'] == 'aihub' else 1.0)
        for _, row in df.iterrows()
    ])

# model.fit에서 sample_weight 사용 시 class_weight=None
```

> 주의: tf.data.Dataset 사용 시 (x, y, weight) 튜플 구조로 전달.  
> class_weight와 sample_weight 동시 사용 불가 → 위 방식으로 통합.

---

## 항목 4. Augmentation 구현 계획

### 오프라인 vs 온라인 분리 전략

| 기법 | 방식 | 라이브러리 | 적용 대상 |
|------|------|-----------|----------|
| time-stretching ×0.85/1.15 | 오프라인 | librosa.effects.time_stretch | 전체 |
| noise-mix SNR 10/20dB | 오프라인 | numpy | 초인종·노크 |
| noise-mix SNR 25/35dB | 오프라인 | numpy | 화재경보 |
| volume scaling -6dB | 오프라인 | numpy | 전체 |
| pitch-shift ±2semitone | 오프라인 | librosa.effects.pitch_shift | AI Hub 클립만 |
| SpecAugment freq/time mask | 온라인 (on-the-fly) | TensorFlow | embedding 레벨 |

- 서양 클립: 원본 포함 **×8** / 한국 클립(AI Hub): 원본 포함 **×10**
- 총 예상: ~2,798 × 8~10 = **22,000~28,000클립** (추정)

```python
# train/augment.py
import librosa
import numpy as np

SR, N_SAMPLES = 16000, 48000

def time_stretch(wav, rate):
    out = librosa.effects.time_stretch(wav.astype(np.float32), rate=rate)
    return librosa.util.fix_length(out, size=N_SAMPLES)

def mix_noise(signal, noise, snr_db):
    sig_rms = np.sqrt(np.mean(signal**2)) + 1e-9
    noise_rms = np.sqrt(np.mean(noise**2)) + 1e-9
    target_rms = sig_rms / (10 ** (snr_db / 20))
    return np.clip(signal + noise * (target_rms / noise_rms), -1.0, 1.0)

def volume_scale(wav, db=-6.0):
    return wav * (10 ** (db / 20))  # × 0.501

def pitch_shift(wav, n_steps):  # AI Hub 클립만
    return librosa.effects.pitch_shift(wav.astype(np.float32), sr=SR, n_steps=n_steps)
```

> **사전 필요 파일 (blocking):** `data/noise/corridor_noise.wav`, `data/noise/tv_noise.wav`  
> → Freesound CC0 라이선스로 확보 필요

---

## 항목 5. 학습 데이터 로딩 파이프라인

**tf.data.Dataset 선택** (TF 네이티브, MacBook M4 Metal prefetch 활용)

```python
# train/dataset.py
import tensorflow as tf
import pandas as pd
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE
LABEL_MAP = {'doorbell': 0, 'knock': 1, 'fire_alarm': 2}

def load_waveform(path):
    raw = tf.io.read_file(path)
    wav, _ = tf.audio.decode_wav(raw, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)           # (48000,)
    return tf.cast(wav, tf.float32) / 32768.0  # int16 → [-1, 1] float32

def build_dataset(csv_path, sample_weights, batch_size=32, shuffle=True):
    df = pd.read_csv(csv_path)
    paths = df['path'].values
    labels = np.array([LABEL_MAP[l] for l in df['label']])

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, sample_weights))
    ds = ds.map(
        lambda p, l, w: (load_waveform(p), l, w),
        num_parallel_calls=AUTOTUNE
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(AUTOTUNE)
```

> YAMNet은 단일 waveform 입력 설계 → `tf.vectorized_map`으로 배치 처리

---

## 항목 6. 검증 전략

### 직접 녹음 우선순위 조정 기준 (test set F1 기준)

| 조건 | 상황 판단 | 조치 |
|------|-----------|------|
| doorbell F1 < 0.75 | 도메인 불일치 심각 | 초인종 직접 녹음 **즉시** 착수 |
| doorbell F1 0.75~0.85 | 개선 여지 있음 | 초인종 직접 녹음 예정대로 착수 |
| doorbell F1 ≥ 0.85 | 서양 데이터로 충분 | 보류 가능, 화재경보 집중 |
| knock F1 < 0.80 | 노크 데이터 부족 | 노크 직접 녹음 착수 |
| fire_alarm F1 < 0.85 | AI Hub 다양성 부족 | 화재경보 직접 녹음 착수 |
| macro F1 ≥ 0.90 | 목표 조기 달성 | 직접 녹음 최소화 가능 |

**초인종↔노크 혼동 허용 기준:** `confusion[doorbell→knock] + confusion[knock→doorbell] ≤ test 전체의 10%`

```python
# train/evaluate.py
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

LABELS = ['doorbell', 'knock', 'fire_alarm']

def full_evaluate(classifier, test_ds, save_dir='results/'):
    y_true, y_pred = [], []
    for waveforms, labels, _ in test_ds:
        emb = extract_embeddings(waveforms)
        preds = classifier(emb, training=False)
        y_pred.extend(tf.argmax(preds, 1).numpy())
        y_true.extend(labels.numpy())

    print(classification_report(y_true, y_pred, target_names=LABELS))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=LABELS, yticklabels=LABELS, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=150)
    plt.close()
    return cm
```

---

## 항목 7. 학습 환경 및 파일 구조

### 환경 결정: MacBook M4 로컬 학습 (t2.micro 학습 불가)

| | MacBook M4 | t2.micro |
|--|-----------|----------|
| RAM | 16GB+ | 1GB (TF 로드만 ~500MB → 학습 불가) |
| 연산 | Metal GPU | CPU only |
| **용도** | **학습 전담** | **추론 API 서버 전담** |

### 파일 구조

```
ddingdong-pretest/
├── train/
│   ├── dataset.py       # tf.data 파이프라인
│   ├── model.py         # YAMNet + Dense
│   ├── augment.py       # 오프라인 augmentation
│   ├── train.py         # 메인 학습 (2단계)
│   └── evaluate.py      # F1, Confusion Matrix
├── data/
│   ├── splits/          # 기존 CSVs
│   └── noise/           # 배경소음 WAV (신규 확보 필요)
├── checkpoints/         # .gitignore
├── results/             # .gitignore
├── tfhub_cache/         # .gitignore
└── docs/
    ├── dataset_plan.md
    └── finetune_plan.md
```

### 모델 저장 형식

```python
# 1. 학습 중 체크포인트
ModelCheckpoint('checkpoints/best_model.h5')

# 2. 최종 저장 (SavedModel)
classifier.save('saved_model/')

# 3. TFLite 변환 (INT8 양자화 - Android/iOS 배포용)
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open('model_quantized.tflite', 'wb').write(tflite_model)
```

> **ESP32 직접 배포 불가:** YAMNet 3.7MB > ESP32 520KB SRAM.  
> ESP32는 추론 결과 수신 전용. 실제 추론은 Android/iOS TFLite 앱 또는 Raspberry Pi 담당.

---

## 전체 실행 순서 (1차 Fine-tuning 로드맵)

| Phase | 내용 | 예상 소요 |
|-------|------|----------|
| Phase 0 | 배경소음 WAV 2종 확보, TF-metal 설치 확인 | 1일 |
| Phase 1 | augment.py 구현 + 오프라인 augmentation 실행 | 2~3시간 |
| Phase 2 | train/*.py 구현 (dataset, model, train, evaluate) | 2~3시간 |
| Phase 3 | 1단계 학습 (YAMNet frozen, lr=1e-3) | 30~60분 |
| Phase 4 | 2단계 학습 (부분 언프리징, lr=1e-4) | 20~30분 |
| Phase 5 | 평가 + 직접 녹음 우선순위 결정 | 30분 |
| Phase 6 | SavedModel → TFLite INT8 양자화 변환 | 30분 |
| **합계** | | **1.5~2일** |

---

## 현재 Blocking 항목

- [ ] `data/noise/corridor_noise.wav`, `data/noise/tv_noise.wav` 미확보 → noise-mix augmentation 불가
- [ ] 직접 녹음 미완료 → 2차 Fine-tuning 대기
- [ ] `step3_yamnet_evaluate.py` 기존 코드와 충돌 가능성 → 사전 검토 필요
- [ ] val/test set에 직접 녹음 데이터 없음 → 1차 평가는 서양+AI Hub 도메인 내 성능만 반영
