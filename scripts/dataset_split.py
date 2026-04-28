"""
파일 단위 train/val/test 분할 스크립트

입력:
  01_clips/doorbell/*.wav    (FSD50K + AudioSet)
  01_clips/knock/*.wav       (FSD50K + AudioSet)
  01_clips/fire_alarm/*.wav  (AI Hub + AudioSet)

출력:
  data/splits/train.csv
  data/splits/val.csv
  data/splits/test.csv
  각 CSV: path, label, source

분할 비율: 70 / 15 / 15
데이터 누수 방지: 모든 소스의 클립은 원본 파일 단위로 그룹핑 후 분할
  파일명 규칙: {source_stem}_{start_ms:07d}.wav → source_stem이 그룹키
"""

import csv
import os
import random
from collections import defaultdict
from pathlib import Path

CLIPS_BASE = Path.home() / 'Desktop/서경대학교/시험 준비/26-1/공학종합설계1/ML 학습 데이터/ddingdong_dataset/01_clips'
OUTPUT_DIR = Path('/Users/xorms/Desktop/xorms/프로젝트/ddingdong/pretest/data/splits')

CLASSES = ['doorbell', 'knock', 'fire_alarm']
SPLIT_RATIO = (0.70, 0.15, 0.15)  # train, val, test
SEED = 42


def get_source(stem: str) -> str:
    if stem.startswith('S-'):
        return 'aihub'
    # FSD50K IDs are purely numeric (e.g. 100634_0000000)
    first_part = stem.split('_')[0]
    if first_part.lstrip('-').isdigit():
        return 'fsd50k'
    return 'audioset'


def get_source_group(stem: str) -> str:
    """원본 파일 단위 그룹: {source_stem}_{start_ms} → {source_stem} (마지막 _XXXXXXX 제거)"""
    return '_'.join(stem.split('_')[:-1])


def split_list(items: list, ratios: tuple, seed: int) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows, test_rows = [], [], []

    for label in CLASSES:
        clip_dir = CLIPS_BASE / label
        if not clip_dir.exists():
            print(f'[SKIP] {clip_dir} 없음')
            continue

        wavs = sorted(clip_dir.glob('*.wav'))
        if not wavs:
            print(f'[SKIP] {label}: WAV 없음')
            continue

        # 그룹별로 묶기 (data leakage 방지)
        groups: dict[str, list[Path]] = defaultdict(list)
        for w in wavs:
            g = get_source_group(w.stem)
            groups[g].append(w)

        group_keys = sorted(groups.keys())
        tr_keys, va_keys, te_keys = split_list(group_keys, SPLIT_RATIO, SEED)

        def rows_from_keys(keys):
            rows = []
            for k in keys:
                for p in groups[k]:
                    rows.append({'path': str(p), 'label': label, 'source': get_source(p.stem)})
            return rows

        tr = rows_from_keys(tr_keys)
        va = rows_from_keys(va_keys)
        te = rows_from_keys(te_keys)

        train_rows.extend(tr)
        val_rows.extend(va)
        test_rows.extend(te)

        print(f'{label}: {len(wavs)}클립 / {len(group_keys)}그룹 → train {len(tr)} / val {len(va)} / test {len(te)}')

    fieldnames = ['path', 'label', 'source']
    for split_name, rows in [('train', train_rows), ('val', val_rows), ('test', test_rows)]:
        out = OUTPUT_DIR / f'{split_name}.csv'
        with open(out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f'  {split_name}.csv: {len(rows)}행 → {out}')

    total = len(train_rows) + len(val_rows) + len(test_rows)
    print(f'\n총 {total}개 클립 분할 완료')


if __name__ == '__main__':
    main()
