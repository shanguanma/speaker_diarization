#!/usr/bin/env python3

import glob
import os
import argparse

import json
import pandas as pd
import tqdm

def merge_audio_process(args):
    # i.e. /maduo/datasets/mlc-slm/train/English/American_English/target_audio/American_English_0265_002/all.wav
    # i.e. /maduo/datasets/mlc-slm/dev/English/American/target_audio/American_0517_007/all.wav
    audio_file_lists=glob.glob(f'{args.dataset_path}/*/*/*/*.wav', recursive=False)
    #os.symlink(audio_file, audio_output_path)
    os.makedirs(args.output_dir,exist_ok=True)
    for audio_file in audio_file_lists:
        print(f"audio_file: {audio_file}")
        target_wav_name=os.path.basename(audio_file)
        target_wav_folder = audio_file.split("/")[-2]
        audio_output_dir = os.path.join(args.output_dir,target_wav_folder)
        print(f"audio_output_dir: {audio_output_dir}")
        os.makedirs(audio_output_dir,exist_ok=True)
        audio_output_path = os.path.join(audio_output_dir,target_wav_name)
        os.symlink(audio_file, audio_output_path)
        print(f"audio_output_path: {audio_output_path}")

def merge_label_json(args):
    # i.e. /maduo/datasets/mlc-slm/train/English/American_English/train.json
    label_json_file_lists=glob.glob(f'{args.dataset_path}/*/*.json', recursive=False)
    output_path=f"{args.output_json_dir}/{args.part}_english.json"
    
    outs = open(output_path, 'w')
    for label_json_file in label_json_file_lists:
        lines = open(label_json_file).read().splitlines()
        for line in tqdm.tqdm(lines):
            res = json.loads(line)
            json.dump(res, outs)
            outs.write("\n")
    outs.close()




def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='/maduo/datasets/mlc-slm/train/English/')
    parser.add_argument('--output-dir', type=str, default='/maduo/datasets/mlc-slm/train/train_english/target_audio')
    parser.add_argument('--output-json-dir', type=str, default='/maduo/datasets/mlc-slm/train/train_english/')
    parser.add_argument("--part",type=str, default="train")
    args=parser.parse_args()
    return args
if __name__ == "__main__":
    args = get_args()
    merge_audio_process(args)
    merge_label_json(args)

