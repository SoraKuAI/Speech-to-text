import re
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm

# ===== 参数 =====
WAV_DIR = Path("wavs")
TARGET_SR = 24000
TARGET_CH = 1  # mono
PREFIX = "northsky"

# ===== 扫描文件并分组 =====
groups = {}
all_original_files = set()

for wav in WAV_DIR.glob("*.wav"):
    all_original_files.add(wav.resolve())
    name = wav.stem

    # northsky_59_0 这种
    m = re.match(rf"({PREFIX}_\d+)_([0-9]+)$", name)
    if m:
        key = m.group(1)
        idx = int(m.group(2))
        groups.setdefault(key, []).append((idx, wav))
    else:
        # 普通 northsky_10.wav
        groups.setdefault(name, []).append((0, wav))

# ===== 合并 + 重采样 + 单声道 =====
processed = []

for key, items in tqdm(groups.items(), desc="Processing audio"):
    items = sorted(items, key=lambda x: x[0])

    audio = AudioSegment.empty()
    for _, wav_path in items:
        seg = AudioSegment.from_wav(wav_path)
        seg = seg.set_frame_rate(TARGET_SR).set_channels(TARGET_CH)
        audio += seg

    processed.append((key, audio))

# ===== 按数字顺序重新编号 =====
processed.sort(key=lambda x: int(re.search(r'\d+', x[0]).group()))

new_files = set()

for i, (_, audio) in enumerate(processed):
    out_name = f"{PREFIX}_{i:04d}.wav"
    out_path = WAV_DIR / out_name
    audio.export(out_path, format="wav")
    new_files.add(out_path.resolve())

# ===== 删除原始音频（不删新文件）=====
for wav in all_original_files:
    if wav not in new_files and wav.exists():
        wav.unlink()

print("✅ 处理完成：已统一格式并清理原始音频")
