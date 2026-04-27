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
| 초인종 클립 수 | **확인불가** (FSD50K.ground_truth/dev.csv 직접 파싱 필요) |
| 노크 클립 수 | **확인불가** (동일 사유) |
| 화재경보 클립 수 | **확인불가** (동일 사유) |
| 클립 평균 길이 | dev 7.1s / eval 9.8s (0.3~30s 가변) |
| 오디오 포맷 | WAV, PCM 16bit, 44.1kHz, mono |
| 라이선스 | CC-BY (데이터셋 전체) / 개별 클립 혼합 (CC0/CC-BY/CC-BY-NC/CC Sampling+) |
| 다운로드 방법 | Zenodo 직접 다운로드 (총 24.7GB, 분할 zip) |
| 한국 환경음 | **없음** (Freesound 원본 기반, 서양 환경음 위주) |

**다운로드 명령어:**
```bash
# dev audio 분할 파일 6개 다운로드 후 병합
for i in z01 z02 z03 z04 z05; do
  wget "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.$i?download=1" -O FSD50K.dev_audio.$i
done
wget "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.zip?download=1" -O FSD50K.dev_audio.zip
zip -s 0 FSD50K.dev_audio.zip --out unsplit.zip && unzip unsplit.zip

# ground truth CSV로 클래스별 클립 수 확인
wget "https://zenodo.org/records/4060432/files/FSD50K.ground_truth.zip?download=1"
unzip FSD50K.ground_truth.zip
grep -c "Doorbell" FSD50K.ground_truth/dev.csv
grep -c "Knock" FSD50K.ground_truth/dev.csv
grep -c "Fire_alarm" FSD50K.ground_truth/dev.csv
grep -c "Smoke_detector" FSD50K.ground_truth/dev.csv
```

---

### 1-B. AudioSet

크롤링 URL: research.google.com/audioset/dataset/ 각 클래스 페이지

| 클래스 | Eval | Balanced Train | Unbalanced Train | **전체** | 품질 평가 |
|--------|------|----------------|------------------|----------|-----------|
| Doorbell | 60 | 60 | 211 | **331** | 높음 (78%) |
| Knock | 60 | 60 | 202 | **322** | 중간 (67%) |
| Smoke detector/alarm | 62 | 60 | 426 | **548** | 중간 (44%) |
| Fire alarm | 60 | 60 | 801 | **921** | 높음 (90%) |

| 항목 | 내용 |
|------|------|
| 클립 길이 | 10초 고정 (YouTube 세그먼트) |
| 라이선스 | CC BY 4.0 |
| 직접 다운로드 | **불가** (YouTube ID 기반) |
| 대안 도구 | yt-dlp 파이프라인 필요 |
| 한국 환경음 | **없음** |

**다운로드 파이프라인:**
```bash
# 1. CSV 다운로드
wget http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv
wget http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv
wget http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv
wget http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv

# 2. 클래스 ID 확인
# Doorbell: /m/0160x5, Knock: /m/0dxrf, Fire alarm: /m/0d6p0, Smoke detector: /m/01g50p

# 3. yt-dlp로 오디오 다운로드 (클래스 필터링 후)
pip install yt-dlp
python scripts/audioset_download.py \
  --class_ids /m/0160x5 /m/0dxrf /m/0d6p0 /m/01g50p \
  --output_dir data/raw/ \
  --format wav
```
> **주의**: YouTube 동영상 삭제로 실제 다운로드 성공률 약 60~70% (balanced train/eval 우선)

---

### 1-C. AI Hub — 자연 및 인공적 발생 非언어적 소리 데이터

크롤링 URL: aihub.or.kr/aihubdata/data/view.do?dataSetSn=644

| 항목 | 내용 |
|------|------|
| 총 클립 수 | 44,810 클립 |
| 포맷 | **MP3** (WAV 아님) |
| 초인종 클립 수 | **없음** (125개 소분류 전체 확인 — 해당 항목 없음) |
| 노크 클립 수 | **없음** (해당 항목 없음) |
| 화재경보 클립 수 | **확인불가(알람 소분류별 세부 미공개)** / 알람 대분류 1,259 클립 중 약 **157개 (추정)** |
| 화재경보 유형 | 화재경보 경종 + **독립형 화재경보 소리** 포함 확인 |
| 샘플링 주파수 | 확인불가(직접 로그인 확인 필요) |
| 녹음 환경 | 실내 50% / 실외 50%, 인접(10m) 46% / 근거리(50m) 30% / 원거리(100m) 24% |
| 라이선스 | 확인불가(직접 로그인 이용정책 확인 필요) |
| 접근 방법 | 회원가입 + 다운로드 신청 (내국인만 가능) / 승인 후 AI Hub Shell API |
| 한국 환경음 | **있음** (KC 인증 단독경보형 감지기 포함 확인) |

**신청 절차:**
1. aihub.or.kr 회원가입
2. dataSetSn=644 페이지 → '다운로드' 버튼 클릭 → 신청서 작성
3. 승인 소요: **3~7 영업일** (일반 데이터셋 기준)
4. 승인 후 AI Hub Shell API: `TS_5.알람.zip` (388MB) 선택 다운로드

---

### 1-D. Freesound

크롤링 URL: freesound.org/search/ 검색어별 직접 확인

| 검색어 | 결과 수 (WAV) | 한국 환경음 | 주요 특징 |
|--------|-------------|------------|----------|
| doorbell | **290** | **1건** (sarena6487528, "Korea, Doorbell, Sound" 태그) | 서양 차임벨/기계식 위주 |
| knock door | **779** | 확인불가 | 목재 문 위주, 철재/ABS 일부 |
| fire alarm | **201** | **없음** | 서양 지속 사이렌/경보 위주 |
| smoke detector beep | **9** | **없음** | 극소수 |
| korean doorbell | **0** | — | 검색 결과 없음 |

| 항목 | 내용 |
|------|------|
| 라이선스 | 혼합 (CC0 / CC-BY / CC-BY-NC — 상위 결과 약 60% CC0+CC-BY 추정) |
| WAV 직접 다운로드 | **가능** (로그인 필요) |
| API | Freesound API v2 — 로그인 후 API key 발급 |

---

### STEP 1 종합 요약표

| 데이터셋 | 초인종 클립 수 | 노크 클립 수 | 화재경보 클립 수 | 라이선스 | 접근 방법 | 한국 환경음 포함 |
|---------|------------|-----------|--------------|---------|---------|--------------|
| FSD50K | 확인불가(CSV 파싱) | 확인불가 | 확인불가 | CC-BY (전체) | Zenodo 직접 24.7GB | 없음 |
| AudioSet (balanced) | 60 | 60 | 120 (smoke60+fire60) | CC BY 4.0 | yt-dlp 파이프라인 | 없음 |
| AudioSet (unbalanced) | 211 | 202 | 1,227 (smoke426+fire801) | CC BY 4.0 | yt-dlp 파이프라인 | 없음 |
| AI Hub | **없음** | **없음** | ~157 **(추정)** | 확인불가(로그인) | 회원가입+신청(내국인) | **있음** |
| Freesound | 290 (WAV) | 779 (WAV) | 210 (WAV) | 혼합 | API/수동 | 1건 |

---

## STEP 2. 수급 전략 및 직접 녹음 프로토콜

### 문제 1: Fine-tuning 최소 요건 충족 여부 (클래스당 최소 200개 원본)

AudioSet YouTube 다운로드 성공률 70% 적용:

| 클래스 | AudioSet 실효 | Freesound 가용(CC0+BY) | AI Hub | FSD50K (추정) | **합계** | 200개 충족 |
|--------|-------------|----------------------|--------|--------------|---------|-----------|
| 초인종 | 232 (84+148) | ~70 | 없음 | ~100 | **~402** | ✅ |
| 노크 | 225 (84+141) | ~150 | 없음 | ~150 | **~525** | ✅ |
| 화재경보 | 1,249 | ~126 | ~157 | ~150 | **~1,682** | ✅ |

> **결론**: 수치상 충족 가능. 단, 화재경보·초인종의 도메인 불일치(서양↔한국)로 단순 수량 충족만으로는 90% 정확도 달성 불가 — 직접 녹음 필수.

---

### 문제 2: 화재경보 도메인 불일치 보완

**서양 vs 한국 화재경보 비교:**

| 구분 | 서양 (AudioSet/FSD50K) | 한국 KC 인증 단독경보형 |
|------|----------------------|----------------------|
| 패턴 | 지속음 사이렌 / 저주파 워블 | 간헐 비프 "삐-삐-삐" |
| 주파수 | 가변 (연속 스위핑) | 3,100~3,500Hz |
| 온/오프 | 지속 | 0.5s on / 0.5s off |
| 규격 | NFPA / UK BS | 한국 소방청 KFI |

**한국 환경음 30% 목표 달성 계획:**
- 전체 가용(보수적): ~1,682 클립 → 30% = **504개** 이상 필요
- AI Hub: ~157개 (추정) / 부족분: **347개**
- 직접 녹음 목표: **360클립** (버퍼 포함)

---

### 문제 3: 초인종 도메인 불일치 보완

**서양 vs 한국 초인종 비교:**

| 구분 | 서양 차임벨 | 한국 아파트 인터폰 |
|------|------------|----------------|
| 음형 | 멜로디형 (Ding-Dong 2음, Westminster) | 단음 전자음 "딩동" |
| 지속 시간 | 1~3초 | 0.3~0.8초 (매우 짧음) |
| 주파수 특성 | 복합 배음 구조 | 단순 파형, 고주파 |

**Pitch-shifting으로 서양→한국 변환 가능 여부: ❌ 불가**
- 멜로디 구조 자체가 다름 → 직접 녹음이 유일한 해결책

**한국 환경음 30% 목표:**
- 현재 확보: ~1건 (Freesound Korea 태그)
- 직접 녹음: **135클립** → 비율 = 135/402 = 33.6% → ✅ 충족

---

### 문제 4: 직접 녹음 프로토콜

#### 화재경보 녹음 프로토콜

| 항목 | 내용 |
|------|------|
| 장소 | 실내(현관 앞) / 실외(아파트 복도) |
| 거리 | 0.5m / 1.0m / 1.5m / 2.0m |
| 배경 소음 | 무음 / TV 소리(~50dB) / 복도 잡음 |
| 조건 조합 | 2 × 4 × 3 = **24 조합** |
| 조건당 횟수 | 10회 |
| 예상 원본 클립 | **240개** |
| 클립 길이 | 3~5초 (비프 사이클 최소 1회 완전 포함) |
| 장비 | INMP441 (배포 환경 동일) |
| 필요 장비 | KC 인증 단독경보형 감지기 (TEST 버튼 활용) |

#### 노크 녹음 프로토콜

| 항목 | 내용 |
|------|------|
| 문 재질 | 목재(방문) / 철재(현관문) / ABS 플라스틱(강의실 문) |
| 세기 | 약 / 중 / 강 |
| 거리 | 0.5m / 1.0m |
| 조건 조합 | 3 × 3 × 2 = **18 조합** |
| 조건당 횟수 | 10회 |
| 예상 원본 클립 | **180개** |
| 클립 길이 | 3~5초 (노크 3~5회 포함) |
| 장비 | INMP441 |

#### 초인종 녹음 프로토콜

| 항목 | 내용 |
|------|------|
| 제조사/모델 | 현대HT / 코맥스(또는 LG U+) / 기타 아파트 인터폰 |
| 거리 | 0.5m / 1.0m / 1.5m |
| 조건 조합 | 3 × 3 = **9 조합** |
| 조건당 횟수 | 15회 (한국 환경음 30% 달성 위해 상향) |
| 예상 원본 클립 | **135개** |
| 클립 길이 | 3~5초 (벨 울림 1~2회 포함) |
| 장비 | INMP441 |

#### 직접 녹음 프로토콜 요약표

| 클래스 | 조건 조합 수 | 조건당 횟수 | 예상 원본 클립 수 | 장비 | 소요 시간 |
|--------|-----------|----------|--------------|------|----------|
| 화재경보 | 24 | 10 | **240** | INMP441 | 약 8시간 (2일) |
| 노크 | 18 | 10 | **180** | INMP441 | 약 5시간 (1일) |
| 초인종 | 9 | 15 | **135** | INMP441 | 약 4시간 (1일) |
| **합계** | **51** | — | **555** | INMP441 | **약 17시간 (4일)** |

---

### 문제 5: Data Augmentation 계획

| 기법 | 파라미터 | 생성 버전 수 | 적용 대상 |
|------|---------|-----------|----------|
| time-stretching | ×0.8, ×0.9, ×1.1, ×1.2 | 4 | 전체 클립 |
| pitch-shifting | -2, -1, +1, +2 semitone | 4 | **한국 환경음 클립만** (서양 클립 적용 시 도메인 악화 우려) |
| background noise | 백색잡음/복도소음/TV소음 × SNR 10dB/20dB | 6 | 전체 클립 |
| volume scaling | -6dB / +6dB | 2 | 전체 클립 |

- 한국 환경음 1개 → **17개** (1+4+4+6+2)
- 서양 환경음 1개 → **13개** (1+4+0+6+2)
- 목표 배율(최소 4배): ✅ **달성**

### STEP 2 종합 표

| 클래스 | 공개 데이터 확보량 | 직접 녹음 목표량 | augmentation 후 최종량 (추정) | 200개 기준 충족 |
|--------|----------------|--------------|---------------------------|---------------|
| 초인종 | ~402 (추정 포함) | 135 | **~6,981** | ✅ |
| 노크 | ~525 (추정 포함) | 180 | **~9,165** | ✅ |
| 화재경보 | ~1,682 (추정 포함) | 240 | **~25,038** | ✅ |

> **클래스 불균형 주의**: 화재경보 >> 노크 > 초인종  
> Fine-tuning 시 `class_weight` 또는 oversampling 필수 적용

---

## STEP 3. 수집 실행 계획

### 3-1. 데이터셋 다운로드 우선순위

**Priority 1 — FSD50K (즉시 착수, 승인 불필요)**
- Zenodo 직접 다운로드, 로그인 불필요
- 다운로드 명령어: 위 1-A 항목 참조

**Priority 2 — AudioSet (FSD50K 병행)**
- CSV 다운로드 후 yt-dlp 파이프라인 실행
- balanced train + eval 우선, unbalanced는 이후 보완
- 다운로드 명령어: 위 1-B 항목 참조

**Priority 3 — AI Hub (즉시 신청 권장)**
- 승인 소요 3~7 영업일 → 대기 중 FSD50K/AudioSet 처리
- 신청 절차: 위 1-C 항목 참조
- 타겟 파일: `TS_5.알람.zip` (388MB Training) + `VS_5.알람.zip` (50MB Validation)

**Priority 4 — Freesound (보완용)**
- API key 발급 후 CC0+CC-BY 필터 배치 다운로드
- 한국 초인종 보완: YouTube "아파트 인터폰 소리" yt-dlp 수동 수집 (라이선스 확인 필수)

### 3-2. 직접 녹음 일정

| 날 | 항목 | 장소 | 예상 시간 |
|----|------|------|----------|
| Day 1 | 노크 녹음 (18조합×10회) | 학교 강의실(ABS) + 현관문(철재) + 방문(목재) | 5시간 |
| Day 2 | 초인종 녹음 (9조합×15회) | 아파트 3종 인터폰 확보 후 현장 방문 | 4시간 |
| Day 3 | 화재경보 녹음 — 실내 (12조합×10회) | 자택 현관 앞 | 4시간 |
| Day 4 | 화재경보 녹음 — 실외 (12조합×10회) | 아파트 복도 | 4시간 |

**장비 준비 체크리스트:**
- [ ] INMP441 마이크 모듈 + 테스트 회로 (배포 환경 동일)
- [ ] KC 인증 단독경보형 감지기 (TEST 버튼 작동 확인)
- [ ] 아파트 인터폰 3종 접근 확보 (현대HT / 코맥스 또는 LG U+ / 기타)
- [ ] 녹음 소프트웨어: `python -c "import sounddevice; print(sounddevice.query_devices())"`

### 3-3. 전처리 파이프라인 설계

```
수집 오디오 (WAV/MP3, 다양한 샘플레이트)
    ↓ librosa.load(sr=16000, mono=True)
16kHz mono 변환
    ↓ 묵음 구간 제거 (librosa.effects.trim)
유효 구간 추출
    ↓ 3초 클립 분할 (stride 1.5초, 묵음 클립 제거)
3초 고정 클립 (48,000 샘플)
    ↓ librosa.feature.melspectrogram(n_mels=128, hop_length=512, n_fft=2048)
Mel-spectrogram (128 × 94 프레임)
    ↓ librosa.power_to_db(ref=np.max)
dB 스케일 변환
    ↓ np.save()
.npy 저장
```

**디렉토리 구조:**
```
data/
├── raw/               # 원본 수집 오디오 (변경 금지)
│   ├── doorbell/      # FSD50K + AudioSet + Freesound + 직접녹음
│   │   ├── fsd50k/
│   │   ├── audioset/
│   │   ├── freesound/
│   │   └── recorded/
│   ├── knock/
│   └── fire_alarm/
│       ├── western/   # AudioSet/FSD50K 서양 사이렌
│       └── korean/    # AI Hub + 직접녹음 한국 비프패턴
├── processed/         # 16kHz mono 3초 클립
│   ├── doorbell/
│   ├── knock/
│   └── fire_alarm/
└── features/          # .npy mel-spectrogram (128×94)
    ├── doorbell/
    ├── knock/
    └── fire_alarm/
```

**자동화 스크립트**: `scripts/preprocess.py` 작성 필요 (librosa 배치 처리)

---

## 참고: 주요 URL 및 클래스 ID

| 데이터셋 | 클래스 | ID / URL |
|---------|--------|---------|
| AudioSet | Doorbell | /m/0160x5 |
| AudioSet | Knock | /m/0dxrf |
| AudioSet | Fire alarm | /m/0d6p0 |
| AudioSet | Smoke detector | /m/01g50p |
| FSD50K | 전체 | zenodo.org/records/4060432 |
| AI Hub | 비언어적 소리 | aihub.or.kr/aihubdata/data/view.do?dataSetSn=644 |
| Freesound | doorbell (WAV) | freesound.org/search/?q=doorbell&f=type:wav |
| Freesound | knock door (WAV) | freesound.org/search/?q=knock+door&f=type:wav |
| Freesound | fire alarm (WAV) | freesound.org/search/?q=fire+alarm&f=type:wav |
