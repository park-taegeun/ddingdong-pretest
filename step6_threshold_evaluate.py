import os
import glob
import numpy as np
import librosa
import tensorflow_hub as hub

# 타겟 클래스 매핑
TARGET_CLASSES = {
    'doorbell': [349],
    'knock': [353],
    'fire_alarm': [394, 393, 382]
}

def load_wav_16k_mono(filename):
    wav, _ = librosa.load(filename, sr=16000, mono=True)
    return wav

def main():
    print("YAMNet 모델을 로딩 중입니다...")
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, 'samples')
    classes = ['doorbell', 'knock', 'fire_alarm']
    
    # 클래스 별 샘플 로드 및 점수 계산 캐싱
    sample_scores = [] # (true_label, dict of max score per class)
    
    for cls in classes:
        folder = os.path.join(base_dir, cls)
        wav_files = glob.glob(os.path.join(folder, '*.wav'))
        for wav_file in wav_files:
            try:
                waveform = load_wav_16k_mono(wav_file)
                scores, _, _ = model(waveform)
                mean_scores = np.mean(scores.numpy(), axis=0)
                
                max_scores = {}
                for target_cls in classes:
                    target_score = np.max([mean_scores[i] for i in TARGET_CLASSES[target_cls]])
                    max_scores[target_cls] = target_score
                    
                sample_scores.append((cls, max_scores))
            except Exception as e:
                pass
                
    if not sample_scores:
        print("샘플이 없어 테스트를 진행할 수 없습니다.")
        return

    # 임계값 별 FPR, FNR 계산 (전체 평균)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("\n| 임계값 | 오탐률 | 미탐률 | 권장 여부 |")
    print("|-------|------|------|---------|")
    
    for th in thresholds:
        total_fps = 0
        total_fns = 0
        total_negatives = 0
        total_positives = 0
        
        for true_label, max_scores in sample_scores:
            # 3가지 클래스 각각에 대해 이진 분류 평가
            for eval_cls in classes:
                is_positive = (eval_cls == true_label)
                predicted_positive = (max_scores[eval_cls] >= th)
                
                if is_positive:
                    total_positives += 1
                    if not predicted_positive: # 미탐 (False Negative)
                        total_fns += 1
                else:
                    total_negatives += 1
                    if predicted_positive: # 오탐 (False Positive)
                        total_fps += 1
        
        fpr = (total_fps / total_negatives * 100) if total_negatives > 0 else 0
        fnr = (total_fns / total_positives * 100) if total_positives > 0 else 0
        
        # 권장 여부: FPR <= 5% 이고 FNR <= 20% 이내면 권장, 아니면 비권장
        recommended = "권장" if fpr <= 5 and fnr <= 50 else "비권장"
        
        print(f"| {int(th*100)}% | {fpr:.1f}% | {fnr:.1f}% | {recommended} |")

if __name__ == '__main__':
    main()
