"""
AudioSet CSV 필터링 스크립트
balanced_train_segments.csv / eval_segments.csv 에서 4개 클래스 행만 추출
"""

import csv
import os
import sys

TARGET = {
    '/m/03wwcy': 'doorbell',
    '/m/0dxrf':  'knock',
    '/m/01g50p': 'smoke_detector',
    '/m/07pp_mv': 'fire_alarm',
}

CSV_FILES = {
    'balanced_train': 'data/balanced_train_segments.csv',
    'eval':           'data/eval_segments.csv',
}

OUTPUT_DIR = 'data/filtered'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def filter_csv(split: str, filepath: str) -> dict[str, list]:
    results: dict[str, list] = {name: [] for name in TARGET.values()}

    with open(filepath) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(', ')
            if len(parts) < 4:
                continue
            ytid, start, end, labels = parts[0], parts[1], parts[2], parts[3]
            label_list = [l.strip().strip('"') for l in labels.split(',')]
            for mid, name in TARGET.items():
                if mid in label_list:
                    results[name].append({'ytid': ytid, 'start': start, 'end': end})

    return results


def save_filtered(split: str, results: dict[str, list]) -> None:
    for name, rows in results.items():
        out_path = os.path.join(OUTPUT_DIR, f'{split}_{name}.csv')
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['ytid', 'start', 'end'])
            writer.writeheader()
            writer.writerows(rows)


def main():
    all_counts: dict[str, dict[str, int]] = {}

    for split, filepath in CSV_FILES.items():
        if not os.path.exists(filepath):
            print(f"[ERROR] {filepath} not found. Download first:", file=sys.stderr)
            print(f"  curl -O http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/{os.path.basename(filepath)}", file=sys.stderr)
            continue

        results = filter_csv(split, filepath)
        save_filtered(split, results)
        all_counts[split] = {name: len(rows) for name, rows in results.items()}

    print("\n=== AudioSet 클래스별 클립 수 ===")
    header = f"{'class':20s}  {'balanced_train':>15s}  {'eval':>6s}  {'합계':>6s}"
    print(header)
    print('-' * len(header))
    for name in TARGET.values():
        bt  = all_counts.get('balanced_train', {}).get(name, 0)
        ev  = all_counts.get('eval', {}).get(name, 0)
        tot = bt + ev
        print(f"{name:20s}  {bt:>15d}  {ev:>6d}  {tot:>6d}")

    print(f"\n필터링 결과 저장: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
