#!/usr/bin/env bash
# FSD50K dev audio 다운로드 + 압축 해제 + Doorbell/Knock 분류
# Zenodo record 4060432

set -e

BASE=~/Desktop/서경대학교/시험\ 준비/26-1/공학종합설계1/ML\ 학습\ 데이터/ddingdong_dataset
RAW="$BASE/00_source_raw/fsd50k"
EXTRACTED_DB="$BASE/01_extracted/doorbell"
EXTRACTED_KN="$BASE/01_extracted/knock"
PRETEST=/Users/xorms/Desktop/xorms/프로젝트/ddingdong/pretest

ZENODO="https://zenodo.org/record/4060432/files"

echo "[1/4] FSD50K dev audio 다운로드 (24.7GB, 약 1~2시간 소요)"
cd "$RAW"
for f in FSD50K.dev_audio.z01 FSD50K.dev_audio.z02 FSD50K.dev_audio.z03 \
          FSD50K.dev_audio.z04 FSD50K.dev_audio.z05 FSD50K.dev_audio.zip; do
    if [ -f "$f" ]; then
        echo "  [SKIP] $f 이미 존재"
    else
        echo "  다운로드 중: $f"
        curl -L --retry 3 --continue-at - -O "$ZENODO/$f"
    fi
done

echo "[2/4] split zip 병합 후 압축 해제"
mkdir -p "$RAW/dev_audio"
zip -s 0 FSD50K.dev_audio.zip --out FSD50K.dev_audio_merged.zip
unzip -q FSD50K.dev_audio_merged.zip -d "$RAW/dev_audio"
echo "  압축 해제 완료: $(find "$RAW/dev_audio" -name '*.wav' | wc -l | tr -d ' ')개 WAV"

echo "[3/4] FSD50K ground truth 읽어 Doorbell/Knock WAV 분류 복사"
python3 - <<'PYEOF'
import csv, shutil
from pathlib import Path

pretest   = Path('/Users/xorms/Desktop/xorms/프로젝트/ddingdong/pretest')
audio_dir = Path.home() / 'Desktop/서경대학교/시험 준비/26-1/공학종합설계1/ML 학습 데이터/ddingdong_dataset/00_source_raw/fsd50k/dev_audio/FSD50K.dev_audio'
dst = {
    'doorbell':   Path.home() / 'Desktop/서경대학교/시험 준비/26-1/공학종합설계1/ML 학습 데이터/ddingdong_dataset/01_extracted/doorbell',
    'knock':      Path.home() / 'Desktop/서경대학교/시험 준비/26-1/공학종합설계1/ML 학습 데이터/ddingdong_dataset/01_extracted/knock',
}

gt_path = pretest / 'data/FSD50K.ground_truth/dev.csv'
counts = {k: 0 for k in dst}

with open(gt_path) as f:
    for row in csv.DictReader(f):
        labels = row['labels'].split(',')
        fname  = str(row['fname'])
        wav    = audio_dir / f'{fname}.wav'
        if not wav.exists():
            continue
        for cls, dpath in dst.items():
            kw = 'Doorbell' if cls == 'doorbell' else 'Knock'
            if any(kw.lower() in l.lower() for l in labels):
                shutil.copy2(wav, dpath / wav.name)
                counts[cls] += 1
                break

for cls, c in counts.items():
    print(f'  {cls}: {c}개 복사')
PYEOF

echo "[4/4] 완료"
echo "  01_extracted/doorbell: $(ls "$EXTRACTED_DB" | wc -l | tr -d ' ')개"
echo "  01_extracted/knock:    $(ls "$EXTRACTED_KN" | wc -l | tr -d ' ')개"
