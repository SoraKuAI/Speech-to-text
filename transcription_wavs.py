import asyncio
import re
from pathlib import Path
import httpx
from tqdm import tqdm

# 参数设置
BASE_URL = "http://10.189.3.18:8000/v1/audio/transcriptions"
MODEL_NAME = "Qwen3-ASR"

WAV_DIR = Path("northsky_datasets")
OUT_FILE = Path("northsky.list")

CONCURRENCY = 8
TIMEOUT = 300

AUDIO_PREFIX = "./northsky_datasets"
SPEAKER = "northsky"
LANG = "ZH"


def extract_index(line: str) -> int:
    m = re.search(r'northsky_(\d+)\.wav', line)
    return int(m.group(1)) if m else -1


async def transcribe(client: httpx.AsyncClient, wav: Path, sem: asyncio.Semaphore):
    async with sem:
        try:
            with open(wav, "rb") as f:
                files = {
                    "file": (wav.name, f, "audio/wav")
                }
                data = {
                    "model": MODEL_NAME
                }

                resp = await client.post(
                    BASE_URL,
                    data=data,
                    files=files,
                )
                resp.raise_for_status()
                text = resp.json().get("text", "").strip()

        except Exception as e:
            print(f"[ERROR] {wav.name}: {e}")
            return None

        # 标准输出格式
        rel_path = f"{AUDIO_PREFIX}/{wav.name}"
        return f"{rel_path}|{SPEAKER}|{LANG}|{text}"


async def main():
    wavs = sorted(WAV_DIR.glob("*.wav"))
    sem = asyncio.Semaphore(CONCURRENCY)

    results = []

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        tasks = [
            transcribe(client, wav, sem)
            for wav in wavs
        ]

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            res = await coro
            if res:
                results.append(res)

    # 排序
    results.sort(key=extract_index)

    OUT_FILE.write_text("\n".join(results), encoding="utf-8")
    print(f"✅ Done {len(results)} 条")


if __name__ == "__main__":
    asyncio.run(main())
