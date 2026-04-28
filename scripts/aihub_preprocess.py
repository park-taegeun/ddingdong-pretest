"""
AI Hub 화재경보 MP3 → 3초 WAV 클립 분할 스크립트

입력: 00_source_raw/aihub_alarm/*.mp3  (평균 ~25초)
출력: 01_clips/fire_alarm/{stem}_{start_ms:07d}.wav  (3초, 16kHz mono)

사전 준비:
  ffmpeg 설치 (brew install ffmpeg)
"""

import os
import subprocess
import sys
from pathlib import Path

SOURCE_DIR = Path.home() / 'Desktop/서경대학교/시험 준비/26-1/공학종합설계1/ML 학습 데이터/ddingdong_dataset/00_source_raw/aihub_alarm'
OUTPUT_DIR = Path.home() / 'Desktop/서경대학교/시험 준비/26-1/공학종합설계1/ML 학습 데이터/ddingdong_dataset/01_clips/fire_alarm'
CLIP_SEC   = 3.0
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mp3_files = sorted(SOURCE_DIR.glob('*.mp3'))
    if not mp3_files:
        print(f'[ERROR] MP3 파일 없음: {SOURCE_DIR}')
        sys.exit(1)

    total_clips = created = skipped = failed = 0

    for mp3 in mp3_files:
        duration = get_duration(mp3)
        n_clips = int(duration // CLIP_SEC)  # 마지막 불완전 클립 제외
        total_clips += n_clips

        for i in range(n_clips):
            start = i * CLIP_SEC
            start_ms = int(start * 1000)
            out_name = f'{mp3.stem}_{start_ms:07d}.wav'
            out_path = OUTPUT_DIR / out_name

            if out_path.exists():
                skipped += 1
                continue

            ok = split_clip(mp3, start, out_path)
            if ok:
                created += 1
            else:
                failed += 1
                print(f'[FAIL] {mp3.name} @ {start:.1f}s')

        print(f'  {mp3.name}: {n_clips}클립 ({duration:.1f}s)')

    print(f'\n=== 완료 ===')
    print(f'  총 클립: {total_clips}  신규: {created}  스킵: {skipped}  실패: {failed}')
    print(f'  출력 경로: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
