from openai import OpenAI

client = OpenAI(
    base_url="http://10.189.3.18:8000/v1",
    api_key="EMPTY"
)

wav_path = "wavs/northsky_0000.wav"

with open(wav_path, "rb") as f:
    transcription = client.audio.transcriptions.create(
        model="Qwen3-ASR",
        file=f,
    )

print(transcription.text)
