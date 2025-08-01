from funasr import AutoModel # pip install funasr # only for simu data
import numpy as np
import librosa
import soundfile as sf
import json
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

def vad_func(wav,sr):
    fsmn_vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", disable_update=True)
    if wav.dtype != np.int16:
        wav = (wav * 32767).astype(np.int16)
        result = fsmn_vad_model.generate(wav, fs=sr)
        time_stamp = result[0]['value']
        return time_stamp # in ms

def write_vad_list():

def spktochunks(args):
    voxceleb2_dataset_dir=args.voxceleb2_dataset_dir
    wavscp =  f"{args.voxceleb2_dataset_dir}/wav.scp"
    spk2utt = f"{args.voxceleb2_dataset_dir}/spk2utt"
    spk2wav = defaultdict(list)
    wav2scp = {}
    with open(wavscp,'r')as fscp:
        for line in fscp:
            line = line.strip().split()
            key = line[0]
            wav2scp[key] = line[1]

    with open(spk2utt, 'r')as fspk:
        for line in fspk:
            line = line.strip().split()
            key = line[0]
            paths = [wav2scp[i] for i in line[1:]]
            if key in spk2wav:
                spk2wav[key].append(paths)
            else:
                spk2wav[key] = paths

    #spk2chunks=defaultdict(list)
    outs = open(args.out_text, "w")
    for spk_id in spk2wav.keys():
        spk2chunks=defaultdict(list)
        for wav_path in spk2wav[spk_id]:
            wav, sr = sf.read(wav_path)
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            time_stamp_list = vad_func(wav,sr=16000)
            # in ms ->(/1000) in second ->(*16000) in sample points
            #speech_chunks = [wav[int(s*16):int(e*16)] for s, e in time_stamp_list]

            if spk_id in spk2chunks:
                spk2chunks[spk_id].append(time_stamp_list)
            else:
                spk2chunks[spk_id] = time_stamp_list
        res = {
                'spk_id' : spk_id,
                'results': spk2chunks,
             }
        json.dump(res, outs)
        outs.write("\n")
    outs.close()

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--voxceleb2-dataset-dir",type=str, default="/maduo/datasets/voxceleb2/vox2_dev/", help="the path for voxceleb2 kaldi format")
    parser.add_argument("--out-text", type=str, default="/maduo/datasets/voxceleb2/vox2_dev/train.json", help="the path for spk2chunks json files, it is remove silent clips")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    spktochunks(args)
    

