import urllib.request, re, os, subprocess, time

FFMPEG = "/opt/homebrew/bin/ffmpeg"
BASE_DIR = "/Users/xorms/Desktop/xorms/프로젝트/ddingdong/pretest/samples"

# FSD50K에서 추출한 클래스별 ID (상위 10개씩)
SAMPLES = {
    "doorbell": ["69184", "361564", "203941", "203942", "203945", "203947", "275629", "192761", "196378", "196379"],
    "knock":    ["256513", "336390", "336391", "336392", "76813", "233486", "342551", "342552", "85016", "83479"],
    "fire_alarm": ["270643", "393524", "25032", "82797", "128021", "372087", "66973", "316843", "165542", "55031"],
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

def get_preview_url(sound_id):
    embed_url = f"https://freesound.org/embed/sound/iframe/{sound_id}/simple/large/"
    req = urllib.request.Request(embed_url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        urls = re.findall(r'https://cdn\.freesound\.org/previews/[^"<> ]+\.mp3', html)
        if urls:
            # prefer hq
            hq = [u for u in urls if "-hq." in u]
            return hq[0] if hq else urls[0]
    except Exception as e:
        print(f"  [임베드 오류 {sound_id}] {e}")
    return None

results = {}
for cls, ids in SAMPLES.items():
    out_dir = os.path.join(BASE_DIR, cls)
    os.makedirs(out_dir, exist_ok=True)
    results[cls] = []
    print(f"\n=== {cls} ===")
    
    for sid in ids:
        mp3_path = os.path.join(out_dir, f"{sid}.mp3")
        wav_path = os.path.join(out_dir, f"{sid}.wav")
        
        if os.path.exists(wav_path):
            print(f"  {sid}: 이미 존재 ({os.path.getsize(wav_path)//1024}KB)")
            results[cls].append(wav_path)
            continue
        
        # 1. preview URL 가져오기
        url = get_preview_url(sid)
        if not url:
            print(f"  {sid}: preview URL 없음 — 건너뜀")
            continue
        
        # 2. MP3 다운로드
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                mp3_data = resp.read()
            with open(mp3_path, "wb") as f:
                f.write(mp3_data)
            print(f"  {sid}: MP3 {len(mp3_data)//1024}KB 다운로드")
        except Exception as e:
            print(f"  {sid}: MP3 다운로드 실패 {e}")
            continue
        
        # 3. WAV 변환 (16kHz mono 16-bit)
        cmd = [FFMPEG, "-y", "-i", mp3_path,
               "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
               wav_path, "-loglevel", "error"]
        ret = subprocess.run(cmd, capture_output=True)
        if ret.returncode == 0:
            size = os.path.getsize(wav_path)
            print(f"  {sid}: WAV 변환 완료 ({size//1024}KB)")
            os.remove(mp3_path)
            results[cls].append(wav_path)
        else:
            print(f"  {sid}: WAV 변환 실패 {ret.stderr.decode()[:100]}")
        
        time.sleep(0.3)  # 서버 부하 방지

# 결과 요약
print("\n\n=== 다운로드 결과 ===")
for cls, files in results.items():
    print(f"{cls}: {len(files)}개 확보")
