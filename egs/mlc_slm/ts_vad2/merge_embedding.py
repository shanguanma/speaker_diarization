#!/usr/bin/env python3

import glob
import os
import argparse

import json
import pandas as pd
import tqdm


def merge_embed(args):
    # i.e. /maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/dev/English/American/cam++_en_zh_advanced_feature_dir/American_0517_007/1.pt
    embedding_file_lists=glob.glob(f'{args.in_embed_dir}/*/*/*/*.pt', recursive=False)
    for embedding_file in embedding_file_lists:
        
        target_embedding_name=os.path.basename(embedding_file)
        
        target_embedding_folder ="/".join(embedding_file.split("/")[-3:-1]) # cam++_en_zh_advanced_feature_dir/American_0517_007
        
        embedding_output_dir = os.path.join(args.out_embed_dir,target_embedding_folder)
        print(f"embedding_output_dir: {embedding_output_dir}")
        
        os.makedirs(embedding_output_dir,exist_ok=True)
        
        embedding_output_path = os.path.join(embedding_output_dir,target_embedding_name)
        
        os.symlink(embedding_file, embedding_output_path)
        print(f"embedding_output_path: {embedding_output_path}")


def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--in-embed-dir', type=str, default='/maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/dev/English')
    parser.add_argument('--out-embed-dir', type=str, default='/maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/dev/dev_english')
    args=parser.parse_args()
    return args
if __name__ == "__main__":
    args = get_args()
    merge_embed(args)  
