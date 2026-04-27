import os
import glob
import numpy as np
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import cosine

def load_mel_spectrogram(filename):
    """
    Load a WAV file and compute its Mel-spectrogram.
    librosa.feature.melspectrogram is used as per the requirement (128 mel bins).
    """
    y, sr = librosa.load(filename, sr=16000, mono=True)
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    # (n_mels, T) -> (T, n_mels)
    return log_mel_spec.T

def calculate_dtw_distance(mel1, mel2):
    """ Calculate DTW distance with cosine distance metric. """
    distance, path = fastdtw(mel1, mel2, dist=cosine)
    return distance

def main():
    base_dir = 'samples'
    
    doorbell_files = glob.glob(os.path.join(base_dir, 'doorbell', '*.wav'))
    knock_files = glob.glob(os.path.join(base_dir, 'knock', '*.wav'))
    fire_alarm_files = glob.glob(os.path.join(base_dir, 'fire_alarm', '*.wav'))
    
    if len(doorbell_files) == 0:
        print("초인종 샘플이 없어 DTW 테스트를 진행할 수 없습니다.")
        return
        
    print("멜스펙트로그램(128 bins) 추출 및 SP/DTW(cosine) 거리를 계산 중입니다...")
    
    # 템플릿: 첫 번째 초인종 샘플
    template_file = doorbell_files[0]
    template_mel = load_mel_spectrogram(template_file)
    
    # 평가 대상 분리 (템플릿 제외한 초인종)
    doorbell_test_files = doorbell_files[1:]
    
    results = {
        'doorbell': [],
        'knock': [],
        'fire_alarm': []
    }
    
    # 1. 템플릿 vs 같은 클래스 초인종 (intra-class)
    for f in doorbell_test_files:
        try:
            mel = load_mel_spectrogram(f)
            dist = calculate_dtw_distance(template_mel, mel)
            results['doorbell'].append(dist)
        except Exception as e:
            print(f"오류 ({f}): {e}")
            
    # 2. 템플릿 vs 노크 (inter-class)
    for f in knock_files:
        try:
            mel = load_mel_spectrogram(f)
            dist = calculate_dtw_distance(template_mel, mel)
            results['knock'].append(dist)
        except Exception as e:
            print(f"오류 ({f}): {e}")
            
    # 3. 템플릿 vs 화재경보 (inter-class)
    for f in fire_alarm_files:
        try:
            mel = load_mel_spectrogram(f)
            dist = calculate_dtw_distance(template_mel, mel)
            results['fire_alarm'].append(dist)
        except Exception as e:
            print(f"오류 ({f}): {e}")
            
    # 결과 요약
    print("\n| 비교 쌍 | 평균 DTW 거리 | 최솟값 | 최댓값 | 표준편차 |")
    print("|--------|-------------|------|------|--------|")
    
    targets = [
        ('초인종 vs 초인종 (같은 집)', 'doorbell'),
        ('초인종 vs 노크', 'knock'),
        ('초인종 vs 화재경보', 'fire_alarm')
    ]
    
    for name, key in targets:
        dists = results[key]
        if len(dists) > 0:
            avg = np.mean(dists)
            min_d = np.min(dists)
            max_d = np.max(dists)
            std = np.std(dists)
            print(f"| {name} | {avg:.3f} | {min_d:.3f} | {max_d:.3f} | {std:.3f} |")
        else:
            print(f"| {name} | 측정 불가 | 측정 불가 | 측정 불가 | 측정 불가 |")

if __name__ == '__main__':
    main()
