"""
AudioSet 오디오 다운로드 스크립트 (yt-dlp 기반)
필터링된 CSV에서 YouTube ID를 읽어 WAV로 다운로드

사전 준비:
  pip install yt-dlp
  python scripts/audioset_filter.py  # 먼저 필터링 실행
"""

import csv
import os
import subprocess
import sys
import time
from pathlib import Path

FILTERED_DIR = 'data/filtered'
OUTPUT_BASE  = str(Path.home() / 'Desktop/서경대학교/시험 준비/26-1/공학종합설계1/ML 학습 데이터/ddingdong_dataset/00_source_raw/audioset')

CLASS_MAP = {
    'doorbell':   'doorbell',
    'knock':      'knock',
    'fire_alarm': 'fire_alarm',
}

SPLITS = ['balanced_train', 'eval']

YTDLP_OPTS = [
    '--no-playlist',
    '--quiet',
    '--extract-audio',
    '--audio-format', 'wav',
    '--audio-quality', '0',
    '--postprocessor-args', '-ar 16000 -ac 1',  # 16kHz mono
]


def download_clip(ytid: str, start: float, end: float, out_path: str) -> bool:
    """yt-dlp로 단일 클립 다운로드 (start~end 초 구간)"""
    url = f'https://www.youtube.com/watch?v={ytid}'
    duration = end - start

    cmd = [
        'yt-dlp',
        *YTDLP_OPTS,
        '--download-sections', f'*{start:.1f}-{end:.1f}',
        '--force-keyframes-at-cuts',
        '-o', out_path,
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return result.returncode == 0


def main():
    success_total = 0
    fail_total    = 0

    for class_name in CLASS_MAP:
        out_dir = Path(OUTPUT_BASE) / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        clips: list[dict] = []
        for split in SPLITS:
            csv_path = Path(FILTERED_DIR) / f'{split}_{class_name}.csv'
            if not csv_path.exists():
                print(f"[SKIP] {csv_path} not found — run audioset_filter.py first")
                continue
            with open(csv_path) as f:
                clips.extend(list(csv.DictReader(f)))

        if not clips:
            continue

        print(f"\n=== {class_name}: {len(clips)} clips ===")
        success = fail = 0

        for i, row in enumerate(clips, 1):
            ytid  = row['ytid']
            start = float(row['start'])
            end   = float(row['end'])
            fname = f"{ytid}_{start:.1f}_{end:.1f}.wav"
            out_path = str(out_dir / fname)

            if os.path.exists(out_path):
                success += 1
                continue

            ok = download_clip(ytid, start, end, out_path)
            if ok:
                success += 1
                print(f"  [{i:4d}/{len(clips)}] OK  {fname}")
            else:
                fail += 1
                print(f"  [{i:4d}/{len(clips)}] FAIL {ytid}")

            time.sleep(0.5)  # rate limit

        print(f"  완료: {success} 성공 / {fail} 실패")
        success_total += success
        fail_total    += fail

    print(f"\n=== 전체 결과: {success_total} 성공 / {fail_total} 실패 ===")
    if fail_total > 0:
        rate = success_total / (success_total + fail_total) * 100
        print(f"  성공률: {rate:.1f}% (YouTube 영상 삭제/비공개 등으로 실패 발생 정상)")


if __name__ == '__main__':
    main()
