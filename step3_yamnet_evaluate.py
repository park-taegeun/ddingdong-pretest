import os
import glob
import numpy as np
import librosa
import tensorflow_hub as hub

# 타겟 클래스 매핑 (YAMNet 클래스 ID 기준)
# 349: Doorbell, 353: Knock, 394: Fire alarm, 393: Smoke detector/alarm, 382: Alarm
TARGET_CLASSES = {
    'doorbell': [349],
    'knock': [353],
    'fire_alarm': [394, 393, 382]
}

def load_wav_16k_mono(filename):
    """ WAV 파일을 읽어서 16kHz, mono 형식으로 변환합니다. """
    wav, _ = librosa.load(filename, sr=16000, mono=True)
    return wav

def main():
    print("YAMNet 모델을 로딩 중입니다...")
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    base_dir = 'samples'
    classes = ['doorbell', 'knock', 'fire_alarm']
    
    results = []
    
    for cls in classes:
        folder = os.path.join(base_dir, cls)
        wav_files = glob.glob(os.path.join(folder, '*.wav'))
        
        sample_count = len(wav_files)
        
        if sample_count == 0:
            results.append({
                'class': cls, 'count': 0, 'top1_acc': 0, 'avg_conf': 0, 'pass_70': 0
            })
            continue

        correct_top1_count = 0
        total_confidence = 0.0
        pass_70_count = 0
        
        actual_sample_count = sample_count
        
        for wav_file in wav_files:
            try:
                waveform = load_wav_16k_mono(wav_file)
                # YAMNet 추론 (scores, embeddings, spectrogram)
                scores, _, _ = model(waveform)
                
                # 전체 클립에 대해 프레임별 점수 평균 계산
                mean_scores = np.mean(scores.numpy(), axis=0)
                
                top1_class_id = np.argmax(mean_scores)
                top1_confidence = mean_scores[top1_class_id]
                
                # 타겟 클래스의 점수 확인
                target_score = np.max([mean_scores[i] for i in TARGET_CLASSES[cls]])
                total_confidence += target_score
                
                is_correct = top1_class_id in TARGET_CLASSES[cls]
                if is_correct:
                    correct_top1_count += 1
                    
                # 70% 임계값 검증: 타겟 클래스의 신뢰도가 70% 이상인지
                if target_score >= 0.70:
                    pass_70_count += 1
                    
            except Exception as e:
                print(f"[{wav_file}] 처리 오류: {e}")
                actual_sample_count -= 1
                
        if actual_sample_count > 0:
            results.append({
                'class': cls,
                'count': actual_sample_count,
                'top1_acc': correct_top1_count / actual_sample_count * 100,
                'avg_conf': total_confidence / actual_sample_count,
                'pass_70': pass_70_count / actual_sample_count * 100
            })
        else:
            results.append({
                'class': cls, 'count': 0, 'top1_acc': 0, 'avg_conf': 0, 'pass_70': 0
            })
            
    # 전체 마크다운 표 출력
    print("\n| 클래스 | 샘플 수 | Top-1 정확도 | 평균 신뢰도 | 70% 임계값 통과율 |")
    print("|--------|--------|------------|-----------|----------------|")
    
    class_name_map = {
        'doorbell': '초인종',
        'knock': '노크',
        'fire_alarm': '화재경보'
    }
    
    for res in results:
        kr_name = class_name_map[res['class']]
        cnt = res['count']
        acc = f"{res['top1_acc']:.1f}%" if cnt > 0 else "측정 불가"
        conf = f"{res['avg_conf']:.3f}" if cnt > 0 else "측정 불가"
        pass70 = f"{res['pass_70']:.1f}%" if cnt > 0 else "측정 불가"
        print(f"| {kr_name} | {cnt} | {acc} | {conf} | {pass70} |")

if __name__ == '__main__':
    main()
