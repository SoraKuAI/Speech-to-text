import os
import re
import time
import argparse
import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def remove_emoji(text):
    emoji_pattern = re.compile(
        "["  # 常见 emoji unicode 范围
        "\U0001f600-\U0001f64f"  # 表情符号
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def parse_args():
    parser = argparse.ArgumentParser(description="Auto Data Labeling with FunASR")
    parser.add_argument(
        "-i", "--input_dir", required=True, help="Input directory of audio files"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="speaker.list",
        help="Output filename (default: speaker.list)",
    )
    parser.add_argument(
        "--is-split",
        action="store_true",
        help="Split output by language (e.g., speaker_zh.list, speaker_jp.list)",
    )
    return parser.parse_args()


def detect_lang_code(raw_text):
    match = re.match(r"<\|([a-z]{2})\|>", raw_text)
    lang_code = match.group(1).upper() if match else "UNK"
    lang_map = {"ZH": "ZH", "JA": "JP", "EN": "EN", "KO": "KO", "YUE": "YUE"}
    return lang_map.get(lang_code, lang_code)


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_path = args.output
    is_split = args.is_split

    model_dir = "./models/SenseVoiceSmall"
    model = AutoModel(
        model=model_dir,
        trust_remote_code=False,
        disable_update=True,
        vad_model="./models/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
    )

    start_time = time.time()
    all_lines = []
    split_outputs = {}  # {"ZH": [...], "JP": [...]}

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".wav"):
            full_path = os.path.join(input_dir, filename)
            relative_path = f"./raw_audio/{filename}"
            speaker = filename.split("_")[0]

            try:
                res = model.generate(
                    input=full_path,
                    cache={},
                    language="auto",
                    use_itn=True,
                    batch_size_s=64,
                    merge_vad=True,
                    merge_length_s=15,
                )

                raw_text = res[0]["text"]
                text = rich_transcription_postprocess(raw_text).strip()
                text = remove_emoji(text)
                lang = detect_lang_code(raw_text)
                line = f"{relative_path}|{speaker}|{lang}|{text}"
                all_lines.append(line)

                if is_split:
                    if lang not in split_outputs:
                        split_outputs[lang] = []
                    split_outputs[lang].append(line)

            except Exception as e:
                all_lines.append(f"{relative_path}|{speaker}|ERROR|{str(e)}")

    total_time = time.time() - start_time

    # 写入主输出文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))

    print(f"\n✅ 输出完成: {output_path}（共 {len(all_lines)-1} 条）")
    print(f"🕒 总耗时: {total_time:.2f} 秒")

    # 可选语言拆分输出
    if is_split:
        base_name, _ = os.path.splitext(output_path)
        for lang, lines in split_outputs.items():
            split_file = f"{base_name.lower()}_{lang.lower()}.list"
            with open(split_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"📁 拆分输出: {split_file}（{len(lines)} 条）")


if __name__ == "__main__":
    main()
