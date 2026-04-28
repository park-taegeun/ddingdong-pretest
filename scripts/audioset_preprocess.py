"""
AudioSet 10초 WAV → 3초 WAV 클립 분할 스크립트

입력: 00_source_raw/audioset/{class}/*.wav  (이미 16kHz mono)
출력: 01_clips/{class}/{stem}_{start_ms:07d}.wav

클래스: doorbell, knock, fire_alarm
"""

import os
import subprocess
import sys
from pathlib import Path

SOURCE_BASE = Path.home() / 'Desktop/서경대학교/시험 준비/26-1/공학종합설계1/ML 학습 데이터/ddingdong_dataset/00_source_raw/audioset'
OUTPUT_BASE = Path.home() / 'Desktop/서경대학교/시험 준비/26-1/공학종합설계1/ML 학습 데이터/ddingdong_dataset/01_clips'

CLASSES   = ['doorbell', 'knock', 'fire_alarm']
CLIP_SEC  = 3.0
SAMPLE_RATE = 16000


def get_duration(path: Path) -> float:
    r = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
         '-of', 'csv=p=0', str(path)],
        capture_output=True, text=True
    )
    return float(r.stdout.strip())


def split_clip(src: Path, start: float, out_path: Path) -> bool:
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-ss', f'{start:.3f}',
        '-t',  f'{CLIP_SEC:.3f}',
        '-i',  str(src),
        '-ar', str(SAMPLE_RATE),
        '-ac', '1',
        '-sample_fmt', 's16',
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def main():
    grand_total = grand_created = grand_skipped = grand_failed = 0

    for cls in CLASSES:
        src_dir = SOURCE_BASE / cls
        out_dir = OUTPUT_BASE / cls
        if not src_dir.exists():
            print(f'[SKIP] {src_dir} 없음')
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        wavs = sorted(src_dir.glob('*.wav'))
        total = created = skipped = failed = 0

        for wav in wavs:
            duration = get_duration(wav)
            n_clips = int(duration // CLIP_SEC)
            total += n_clips

            for i in range(n_clips):
                start = i * CLIP_SEC
                start_ms = int(start * 1000)
                out_name = f'{wav.stem}_{start_ms:07d}.wav'
                out_path = out_dir / out_name

                if out_path.exists():
                    skipped += 1
                    continue

                ok = split_clip(wav, start, out_path)
                if ok:
                    created += 1
                else:
                    failed += 1
                    print(f'  [FAIL] {wav.name} @ {start:.1f}s')

        grand_total   += total
        grand_created += created
        grand_skipped += skipped
        grand_failed  += failed
        print(f'{cls}: {len(wavs)}파일 → {total}클립  (신규 {created} / 스킵 {skipped} / 실패 {failed})')

    print(f'\n=== 전체 완료: {grand_total}클립  신규 {grand_created} / 스킵 {grand_skipped} / 실패 {grand_failed} ===')
    print(f'출력 경로: {OUTPUT_BASE}')


if __name__ == '__main__':
    main()
