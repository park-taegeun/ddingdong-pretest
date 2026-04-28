"""
FSD50K Doorbell/Knock WAV → 3초 클립 전처리 스크립트

입력: 01_extracted/{doorbell,knock}/*.wav  (44.1kHz mono, 가변 길이)
출력: 01_clips/{doorbell,knock}/{stem}_{start_ms:07d}.wav  (16kHz mono, 3초)

처리 규칙:
  - 파일 길이 ≥ 3s : 3초 비중첩 분할, 마지막 불완전 클립 제외
  - 파일 길이 < 3s : 16kHz 리샘플 후 zero-pad → 1클립 생성
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

EXTRACTED_BASE = Path.home() / 'Desktop/서경대학교/시험 준비/26-1/공학종합설계1/ML 학습 데이터/ddingdong_dataset/01_extracted'
OUTPUT_BASE    = Path.home() / 'Desktop/서경대학교/시험 준비/26-1/공학종합설계1/ML 학습 데이터/ddingdong_dataset/01_clips'

CLASSES     = ['doorbell', 'knock']
CLIP_SEC    = 3.0
SAMPLE_RATE = 16000
CLIP_SAMPLES = int(CLIP_SEC * SAMPLE_RATE)  # 48000


def get_duration(path: Path) -> float:
    r = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
         '-of', 'csv=p=0', str(path)],
        capture_output=True, text=True
    )
    return float(r.stdout.strip())


def split_to_clip(src: Path, start: float, out_path: Path) -> bool:
    """3초 구간 잘라서 16kHz mono WAV로 저장"""
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


def resample_and_pad(src: Path, out_path: Path) -> bool:
    """16kHz mono 변환 후 3초로 zero-pad (짧은 파일 전용)"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # 리샘플
        r1 = subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', str(src),
            '-ar', str(SAMPLE_RATE), '-ac', '1', '-sample_fmt', 's16',
            tmp_path,
        ], capture_output=True, text=True)
        if r1.returncode != 0:
            return False

        # zero-pad to exactly CLIP_SAMPLES
        r2 = subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', tmp_path,
            '-af', f'apad=whole_len={CLIP_SAMPLES}',
            '-t', f'{CLIP_SEC:.3f}',
            '-ar', str(SAMPLE_RATE), '-ac', '1', '-sample_fmt', 's16',
            str(out_path),
        ], capture_output=True, text=True)
        return r2.returncode == 0
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    grand_total = grand_created = grand_skipped = grand_failed = 0
    grand_padded = 0

    for cls in CLASSES:
        src_dir = EXTRACTED_BASE / cls
        out_dir = OUTPUT_BASE / cls
        if not src_dir.exists():
            print(f'[SKIP] {src_dir} 없음')
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        wavs = sorted(src_dir.glob('*.wav'))
        total = created = skipped = failed = padded = 0

        for wav in wavs:
            if wav.stat().st_size == 0:
                print(f'  [SKIP-empty] {wav.name}')
                continue
            duration = get_duration(wav)

            if duration < CLIP_SEC:
                # zero-pad → 1클립
                out_name = f'{wav.stem}_0000000.wav'
                out_path = out_dir / out_name
                total += 1
                if out_path.exists():
                    skipped += 1
                    continue
                ok = resample_and_pad(wav, out_path)
                if ok:
                    created += 1
                    padded += 1
                else:
                    failed += 1
                    print(f'  [FAIL-pad] {wav.name}')
            else:
                # 3초 비중첩 분할
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
                    ok = split_to_clip(wav, start, out_path)
                    if ok:
                        created += 1
                    else:
                        failed += 1
                        print(f'  [FAIL] {wav.name} @ {start:.1f}s')

        grand_total   += total
        grand_created += created
        grand_skipped += skipped
        grand_failed  += failed
        grand_padded  += padded
        print(f'{cls}: {len(wavs)}파일 → {total}클립  (신규 {created} / 스킵 {skipped} / 실패 {failed} | zero-pad {padded})')

    print(f'\n=== 완료: {grand_total}클립  신규 {grand_created} / 스킵 {grand_skipped} / 실패 {grand_failed} | zero-pad {grand_padded} ===')


if __name__ == '__main__':
    main()
