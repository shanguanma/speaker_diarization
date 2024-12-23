#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn
import glob
import sys
import os
import pathlib
import re
import logging
def embed_scp(inp:str):
    embeds_list =glob.glob(f"{inp}/*/*.pt")
    embeds_dict = {}
    for embed in embeds_list:
        #print(f"embed: {embed} !!!")
        line = embed.split("/")
        emb = line[-1].split(".")[0]
        assert "_" in emb, f"emb: {emb}!!!"
        #print(f"emb: {emb}")
        embsplit =  emb.split("_")
        uttid=f"{embsplit[0]}-{line[-2]}_{embsplit[-1]}"
        embeds_dict[uttid] = embed # embeds_dict["SPK8049-R8008_M8013_MS807_137"]= /mntcephfs/lab_data/maduo/datasets/alimeeting/Train_Ali_far/non_overlap_segment_6s_spk_emb/alimeeting/SpeakerEmbedding/Train/cam++_zh-cn_200k_feature_dir/R8008_M8013_MS807/SPK8049_137.pt
        #f.write(f"{uttid} {embed}\n")
    return embeds_dict
#SPK8049-R8008_M8013_MS807_137_SPK8068-R8007_M8011_MS806_62_SPK8068-R8007_M8011_MS806_83_SPK8001-R8003_M8001_MS801_78/SPK8049.wav
#SPK8049-R8008_M8013_MS807_137_SPK8068-R8007_M8011_MS806_62_SPK8068-R8007_M8011_MS806_83_SPK8001-R8003_M8001_MS801_78/SPK8049_137.wav
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    non_overlaps_target_embeds_dir = sys.argv[1]
    target_audios_file = sys.argv[2]
    save_dir=sys.argv[3]
    embeds_dict = embed_scp(non_overlaps_target_embeds_dir)
    with open(target_audios_file, 'r') as f:
        for line in f:
            #print(f"line: {line}!!!")
            wav_path = line.strip()
            folder_name = os.path.dirname(wav_path).split("/")[-1] # SPK8049-R8008_M8013_MS807_137_SPK8068-R8007_M8011_MS806_62_SPK8068-R8007_M8011_MS806_83_SPK8001-R8003_M8001_MS801_78
            embedding_name=os.path.basename(wav_path).rsplit('.',1)[0] # SPK8049_137
            #spk_name = embedding_name.split("_")[0]
            #seg_id = embedding_name.split("_")[-1]
            for key in embeds_dict.keys():
                #if re.match(key,folder_name) and re.match(embedding_name, key):
                if key in folder_name and embedding_name.split("_")[0] in key and embedding_name.split("_")[-1] in key:
                    dest_dir = os.path.join(save_dir,folder_name)
                    dest_dir = pathlib.Path(dest_dir)
                    dest_dir.mkdir(exist_ok=True, parents=True)
                    logging.info(f"target_embed: {embeds_dict[key]},target_wav: {wav_path}!!!")
                    save_path=dest_dir / f'{embedding_name}.pt'
                    cmd = f"ln -svf {embeds_dict[key]} {save_path}"
                    os.system(cmd)

