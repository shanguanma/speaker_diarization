# copy and modified from https://github.com/vasistalodagala/whisper-finetune/blob/master/custom_data/data_prep.py

import argparse
import json
import os
from datasets import Dataset, Audio, Value




def get_parser():
    parser = argparse.ArgumentParser(description='Preliminary data preparation script before Whisper Fine-tuning.')
    parser.add_argument('--source_data', type=str, required=True, default=False, help='Path to dataset jsonl format')
    parser.add_argument('--output_data_dir', type=str, required=False, default='/maduo/datasets/mlc-slm/train/train_asr', help='Output data directory path.')

    args = parser.parse_args()
    return args
def process_jsonl(args):
    """
    {"key": "0295_001_1297_95_1300_86_O1", "path": "/F00120240032/mlc-slm_task2/data/English/American English/0295/asr/wav/0295_001_1297_95_1300_86_O1.wav", "target": "it's like why why would I wish for them to do better you know"}
    -> {"key": "0295_001_1297_95_1300_86_O1", "path": "/F00120240032/mlc-slm_task2/data/English/American_English/0295/asr/wav/0295_001_1297_95_1300_86_O1.wav", "target": "it's like why why would I wish for them to do better you know"}
    """
    os.makedirs(args.output_data_dir, exist_ok=True)
    out = args.output_data_dir + "/raw.jsonl"
    with open(args.source_data) as f, open(out, 'w') as fw:  # 修改为 .jsonl 扩展名以保持一致性
        for line in f:
            # 解析 JSON 行
            d = json.loads(line.strip())
            path = d['path']
            print(f"原始路径: {path}")

            # 处理路径（移除空格并用下划线连接）
            new_path = "_".join(path.split())
            d["path"] = new_path
            print(f'新路径: {d["path"]}')

            # 将字典转换回 JSON 字符串并写入文件
            result = json.dumps(d, ensure_ascii=False)
            fw.write(f"{result}\n")
    return out
def prepared_hf_dataset_format(out: str, output_data_dir: str):
    entries = open(out,'r').read().splitlines()
    audio_dataset = Dataset.from_dict({"audio": [json.loads(line)["path"] for line in entries],
                    "sentence": [ json.loads(line)["target"] for line in entries]})

    audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16_000))
    audio_dataset = audio_dataset.cast_column("sentence", Value("string"))
    audio_dataset.save_to_disk(args.output_data_dir)
    print('Data preparation done')

if __name__ == "__main__":
    args = get_parser()
    out = process_jsonl(args)
    prepared_hf_dataset_format(out, args.output_data_dir)



