import os
import glob
import time
import numpy as np
import librosa
import tensorflow_hub as hub
from fastdtw import fastdtw
from scipy.spatial.distance import cosine

def load_wav(filename):
    wav, sr = librosa.load(filename, sr=16000, mono=True)
    return wav, sr

def to_mel(wav, sr):
    mel_spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec.T

def compute_dtw(mel1, mel2):
    dist, _ = fastdtw(mel1, mel2, dist=cosine)
    return dist

def main():
    print("YAMNet 모델을 로딩 중입니다...")
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, 'samples')
    doorbell_files = glob.glob(os.path.join(base_dir, 'doorbell', '*.wav'))
    if len(doorbell_files) < 2:
        print("초인종 샘플 부족으로 테스트를 진행할 수 없습니다.")
        return
        
    template_wav, template_sr = load_wav(doorbell_files[0])
    template_mel = to_mel(template_wav, template_sr)
    test_files = doorbell_files[1:]
    
    # 100회 반복 측정을 위한 샘플 리스트 생성 (test_files 반복)
    eval_files = (test_files * (100 // len(test_files) + 1))[:100]
    
    times = {
        'load': [],
        'mel': [],
        'yamnet': [],
        'dtw': [],
        'total': []
    }
    
    print("100회 지연시간 측정 중...")
    for f in eval_files:
        try:
            # 1. WAV Loading
            t0 = time.time()
            wav, sr = load_wav(f)
            t1 = time.time()
            
            # 2. Mel processing
            mel = to_mel(wav, sr)
            t2 = time.time()
            
            # 3. YAMNet inference
            scores, embeddings, spectrogram = model(wav)
            t3 = time.time()
            
            # 4. DTW calculation
            dist = compute_dtw(template_mel, mel)
            t4 = time.time()
            
            times['load'].append((t1 - t0) * 1000)
            times['mel'].append((t2 - t1) * 1000)
            times['yamnet'].append((t3 - t2) * 1000)
            times['dtw'].append((t4 - t3) * 1000)
            times['total'].append((t4 - t0) * 1000)
            
        except Exception as e:
            pass

    print("\n| 단계 | 평균(ms) | 최댓값(ms) | 비고 |")
    print("|------|---------|----------|------|")
    
    t_mean = lambda k: np.mean(times[k])
    t_max = lambda k: np.max(times[k])
    
    print(f"| WAV 전처리 | {t_mean('load'):.1f} | {t_max('load'):.1f} | |")
    print(f"| 멜스펙트로그램 변환 | {t_mean('mel'):.1f} | {t_max('mel'):.1f} | |")
    print(f"| YAMNet 추론 | {t_mean('yamnet'):.1f} | {t_max('yamnet'):.1f} | |")
    print(f"| SP/DTW 계산 | {t_mean('dtw'):.1f} | {t_max('dtw'):.1f} | |")
    print(f"| 전체 파이프라인 | {t_mean('total'):.1f} | {t_max('total'):.1f} | |")

if __name__ == '__main__':
    main()
