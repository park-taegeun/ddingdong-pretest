# 띵동 사전 테스트 (ddingdong-pretest)

청각장애인용 현관 부착형 소리 분류 알림 시스템 **띵동**의 파이프라인 사전 검증 레포지토리입니다.

## 테스트 목적

하드웨어 구매 전, 핵심 파이프라인의 실측 성능 수치를 확보하여 리스크를 정량화합니다.

| 테스트 | 목적 |
|--------|------|
| A — YAMNet pre-trained 성능 | fine-tuning 없이 3종 분류 가능성 파악 |
| B — SP/DTW + cosine 변별력 | 초인종 vs 노크/화재경보 DTW 거리 분리 마진 확인 |
| C — 엔드투엔드 지연시간 | WAV 수신~추론 5초 목표 달성 가능성 |
| D — 신뢰도 임계값 최적화 | 70% 임계값의 오탐/미탐 트레이드오프 |

## 데이터셋

- **출처**: FSD50K (Zenodo) 메타데이터 → Freesound CDN preview
- **클래스**: doorbell / knock / fire_alarm (각 10개, 16kHz mono WAV)

## 환경

- Python 3.9.6
- 상세 버전: `requirements.txt` 참조

## 실행 순서

```bash
pip install -r requirements.txt
python step1_download_samples.py   # 샘플 확보
python step3_yamnet_test.py        # 테스트 A
python step4_dtw_test.py           # 테스트 B
python step5_latency_test.py       # 테스트 C
python step6_threshold_test.py     # 테스트 D
```

## 시스템 구성

- **하드웨어**: ESP32-WROOM-32 + INMP441(I2S) + HC-SR501(PIR) + VL53L0X(ToF)
- **서버**: AWS EC2 t2.micro / Flask + Gunicorn(워커 2) + Nginx + Let's Encrypt
- **모델**: YAMNet fine-tuning + SP/DTW 2차 필터
- **알림**: 카카오톡 푸시
