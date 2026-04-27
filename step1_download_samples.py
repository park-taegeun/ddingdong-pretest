"""STEP 1 — FSD50K 기반 Freesound 샘플 다운로드
출처: FSD50K 메타데이터에서 클래스 ID 추출 → Freesound embed iframe preview URL 파싱 → 16kHz WAV 변환
"""
import urllib.request, re, os, subprocess, time

FFMPEG = "/opt/homebrew/bin/ffmpeg"  # Mac: brew install ffmpeg / Linux: apt install ffmpeg
BASE_DIR = "samples"                  # 실행 경로 기준

# FSD50K 메타데이터(collection_dev/eval.csv)에서 추출한 Freesound 클립 ID (클래스당 10개)
SAMPLES = {
    "doorbell":   ["69184",  "361564", "203941", "203942", "203945",
                   "203947", "275629", "192761", "196378", "196379"],
    "knock":      ["256513", "336390", "336391", "336392", "76813",
                   "233486", "342551", "342552", "85016",  "83479"],
    "fire_alarm": ["270643", "393524", "25032",  "82797",  "128021",
                   "372087", "66973",  "316843", "165542", "55031"],
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def get_preview_url(sound_id: str) -> str | None:
    """Freesound embed iframe에서 CDN preview MP3 URL 추출"""
    embed_url = f"https://freesound.org/embed/sound/iframe/{sound_id}/simple/large/"
    req = urllib.request.Request(embed_url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        urls = re.findall(r"https://cdn\.freesound\.org/previews/[^\"<> ]+\.mp3", html)
        if urls:
            hq = [u for u in urls if "-hq." in u]
            return hq[0] if hq else urls[0]
    except Exception as e:
        print(f"  [embed error {sound_id}] {e}")
    return None


def download_wav(sound_id: str, out_dir: str) -> str | None:
    wav_path = os.path.join(out_dir, f"{sound_id}.wav")
    if os.path.exists(wav_path):
        print(f"  {sound_id}: already exists ({os.path.getsize(wav_path)//1024}KB)")
        return wav_path

    url = get_preview_url(sound_id)
    if not url:
        print(f"  {sound_id}: preview URL not found — skip")
        return None

    mp3_path = os.path.join(out_dir, f"{sound_id}.mp3")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        with open(mp3_path, "wb") as f:
            f.write(data)
        print(f"  {sound_id}: MP3 {len(data)//1024}KB downloaded")
    except Exception as e:
        print(f"  {sound_id}: download failed — {e}")
        return None

    # 16kHz mono 16-bit WAV 변환 (YAMNet 입력 형식)
    cmd = [FFMPEG, "-y", "-i", mp3_path,
           "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
           wav_path, "-loglevel", "error"]
    ret = subprocess.run(cmd, capture_output=True)
    os.remove(mp3_path)
    if ret.returncode == 0:
        print(f"  {sound_id}: WAV OK ({os.path.getsize(wav_path)//1024}KB)")
        return wav_path
    print(f"  {sound_id}: ffmpeg conversion failed")
    return None


if __name__ == "__main__":
    for cls, ids in SAMPLES.items():
        out_dir = os.path.join(BASE_DIR, cls)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== {cls} ===")
        for sid in ids:
            download_wav(sid, out_dir)
            time.sleep(0.3)  # 서버 부하 방지
