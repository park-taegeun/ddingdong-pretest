# ML 학습 데이터 수급 계획 (2026-04-27)

> 띵동 (청각장애인용 현관 부착형 소리 분류 알림 시스템)  
> 분류 대상: 초인종 / 노크 / 화재경보  
> ML 아키텍처: YAMNet Fine-tuning (TensorFlow + Keras)  
> 목표 정확도: 90% 이상

---

## STEP 1. 데이터셋 실태 조사 (firecrawl-mcp 실측 크롤링)

> 추정치는 (추정) 표기 / 확인 불가 항목은 사유 명시 / 크롤링 완료일: 2026-04-27

### 1-A. FSD50K

| 항목 | 내용 |
|------|------|
| 출처 | Zenodo (zenodo.org/records/4060432) |
| 총 클립 수 | 51,197 클립 / 108.3시간 |
| 분할 | dev(40,966 / 80.4h) + eval(10,231 / 27.9h) |
| 초인종 클립 수 | **실측**: dev 107 + eval 37 = **144개** |
| 노크 클립 수 | **실측**: dev 270 + eval 103 = **373개** |
| 화재경보 클립 수 | Alarm 435개 자동 필터 → **제외 확정** (차량 사이렌 위주) |
| 클립 평균 길이 | dev 7.1s / eval 9.8s (0.3~30s 가변) |
| 오디오 포맷 | WAV, PCM 16bit, 44.1kHz, mono |
| 라이선스 | CC-BY (데이터셋 전체) |
| 한국 환경음 | **없음** |

---

### 1-B. AudioSet

| 클래스 | 다운로드 성공 | 성공률 | 비고 |
|--------|------------|-------|------|
| Doorbell | **88개** | 73% | mid: /m/03wwcy |
| Knock | **110개** | 92% | mid: /m/0dxrf |
| Fire alarm | **100개** | 83% | mid: /m/07pp_mv |
| **합계** | **298개** | 82.5% | 성공률 82.5% (영상 삭제/비공개 정상) |

> AudioSet fire_alarm 100개: RMS 분석 결과 65% 차량 사이렌 → **제외 확정**

---

### 1-C. AI Hub S_103

| 항목 | 내용 |
|------|------|
| 파일 수 | **171개** MP3 (실측 확정) |
| 평균 길이 | 24.87초 |
| 3초 클립 분할 후 | **~1,341개** |
| 특징 | 한국 KC 인증 단독경보형 감지기 실측 |
| 저장 위치 | 01_extracted/fire_alarm/ |

---

## STEP 2. 확정 데이터 현황 (2026-04-28 기준)

### 01_extracted/ 확정 수량

| 클래스 | 수량 | 구성 |
|--------|------|------|
| doorbell | **195개** | FSD50K 107 + AudioSet 88 |
| knock | **380개** | FSD50K 270 + AudioSet 110 |
| fire_alarm | **171개 확정 + 240개 예정** | AI Hub 171 + 직접 녹음 240 예정 |

### 제외 확정 데이터

| 데이터 | 사유 |
|--------|------|
| FSD50K Alarm 435개 | 표본 청취 결과 차량 사이렌 위주 |
| AudioSet fire_alarm 100개 | RMS 분석 결과 65% 차량 사이렌 |
| YouTube 추출 | 저작권 문제 |

---

## STEP 3. 직접 녹음 계획 (확정)

| 클래스 | 조건 조합 | 횟수 | 예정 클립 | 우선순위 |
|--------|---------|------|---------|--------|
| 초인종 | 인터폰 3종 × 거리 3단계 | 10회 | 90개 | 🔴 필수 |
| 화재경보 | 장소 2 × 거리 4 × 배경소음 3 | 10회 | 240개 | 🔴 필수 |
| 노크 | 문재질 3 × 세기 3 × 거리 2 | 10회 | 180개 | 🟠 필요 |

> 초인종 녹음 시 **"문 닫힌 상태" 시나리오 추가 권장** (실제 배포 환경)

---

## STEP 4. 전처리 기준 (확정)

| 항목 | 기준 |
|------|------|
| 샘플레이트 | 16kHz mono 변환 |
| 클립 분할 | 3초 고정 (stride=3) |
| YAMNet 입력 | **raw waveform 직접 입력** (YAMNet 내부에서 64-bin mel-spec 생성) |
| ~~librosa mel-spec~~ | ~~128 mel bin~~ → **삭제: YAMNet과 호환 불가** |
| train/val 분할 | **파일 단위 분할 필수** (클립 단위 시 data leakage 발생) |

### YAMNet Fine-tuning 구조 (확정)

```python
import tensorflow_hub as hub
import tensorflow as tf

# YAMNet 로드 및 Fine-tuning 구조
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

waveform_input = tf.keras.Input(shape=(None,), dtype=tf.float32)
_, embeddings, _ = yamnet_model(waveform_input)  # embeddings: (batch, time, 1024)
pooled = tf.keras.layers.GlobalAveragePooling1D()(embeddings)
output = tf.keras.layers.Dense(3, activation='softmax')(pooled)
model = tf.keras.Model(inputs=waveform_input, outputs=output)

# EarlyStopping 필수
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)
```

---

## STEP 5. Augmentation 전략 (확정)

| 기법 | 파라미터 | 생성 수 | 적용 대상 |
|------|---------|--------|----------|
| time-stretching | ×0.85, ×1.15 | 2개 | 전체 |
| background noise | SNR 10/20dB | 4개 | 초인종·노크 |
| background noise | SNR 25/35dB | 4개 | 화재경보 |
| volume scaling | -6dB | 1개 | 전체 |
| pitch-shifting | -2/+2 semitone | 2개 | 한국 환경음만 |
| **SpecAugment** | freq_mask=10, time_mask=5 | 권장 추가 | 전체 |

- 서양 클립: 원본 포함 8개 / 한국 클립: 원본 포함 10개

---

## STEP 6. 학습 순서

1. **1차 Fine-tuning**: 공개 데이터 (현재 확보량으로 즉시 시작 가능)
2. **검증**: 클래스별 precision/recall/F1 + Confusion Matrix
3. **2차 Fine-tuning**: 직접 녹음 추가 후 재학습

### 클래스 불균형 처리

- `class_weight` 적용 (sklearn.utils.class_weight.compute_class_weight)
- 한국 환경음 샘플에 `sample_weight` 1.5~2.0배

---

## STEP 7. 최종 점검 결과 (2026-04-28)

> sequential-thinking 7-step 분석 + Firecrawl 문헌 검색 기반

### ❗ Critical 수정 사항

| 항목 | 오류 내용 | 수정 내용 |
|------|---------|--------|
| 전처리 기준 | librosa mel-spec 128 bin 입력 → YAMNet 동작 불가 | raw waveform 입력으로 수정 (YAMNet 내부 처리) |
| train/val 분할 | 분할 방식 미명시 | 파일 단위 분할 필수 명시 |

### ⚠️ 보완 권장

| 항목 | 권장 조치 |
|------|----------|
| Fine-tuning 방식 | 상위 레이어 언프리징 + EarlyStopping(patience=3) |
| 초인종 녹음 | 문 닫힌 상태 시나리오 추가 |
| Augmentation | SpecAugment 추가 |
| 검증 set | 한국 환경음 포함 비율 명시 |
| 평가 지표 | 클래스별 F1 + Confusion Matrix 추가 |

### 정확도 예측 (추정)

| 시나리오 | 예상 정확도 |
|---------|------------|
| 현재 계획 + 직접 녹음 완료 | 87~92% (추정) |
| Critical 수정 + 직접 녹음 완료 | **90~93%** (추정) |
| 직접 녹음 없을 경우 | 75~82% (추정) |

### 문헌 근거 (Firecrawl)

| 논문/출처 | 핵심 수치 | 우리 계획과 비교 |
|---------|---------|----------------|
| arxiv 2504.19030 (2025) | YAMNet embedding → 12클래스 **95.28%** (32,465샘플) | 방향 동일, 규모 유사 |
| Springer 2025 | 제한된 데이터로 **94.21%** (ResNet/VGG/AST 초과) | 소량 + YAMNet = 90%+ 가능 |
| TF 공식 튜토리얼 | raw waveform → embedding(1024) → Dense 구조 | Critical 수정 방향 확인 |
