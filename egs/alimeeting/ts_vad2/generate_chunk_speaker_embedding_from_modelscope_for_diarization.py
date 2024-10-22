# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script will download pretrained models from modelscope (https://www.modelscope.cn/models)
based on the given model id, and extract embeddings from input audio.
Please pre-install "modelscope" follow the command:
    pip install modelscope
Usage:

    """

import os
import sys
import re
import pathlib
import numpy as np
import soundfile
import wave
import argparse
import torch
import logging
import torchaudio
import torchaudio.compliance.kaldi as Kaldi


from modelscope.hub.snapshot_download import snapshot_download
#from modelscope.pipelines.util import is_official_hub_path
def get_args():
    parser = argparse.ArgumentParser(description='Extract speaker embeddings.')
    parser.add_argument('--model_id', default='', type=str, help='Model id in modelscope')
    parser.add_argument('--wavs', nargs='+', type=str, help='Wavs')
    #parser.add_argument('--local_model_dir', default='pretrained', type=str, help='Local model dir')
    parser.add_argument('--save_dir',type=str, default="model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Train/cam++_en_zh_feature_dir", help='speaker embedding dir')
    parser.add_argument(
            "--length_embedding",
            type=float,
            default=6,
            help="length of embeddings, seconds",
    )
    parser.add_argument(
        "--step_embedding", type=float, default=1, help="step of embeddings, seconds"
    )
    parser.add_argument(
        "--batch_size", type=int, default=96, help="step of embeddings, seconds"
    )
    args = parser.parse_args()
    return args



CAMPPLUS_VOX = {
    'obj': 'ts_vad2.cam_pplus_wespeaker.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
    },
}

CAMPPLUS_COMMON = {
    'obj': 'ts_vad2.cam_pplus_wespeaker.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_VOX = {
    'obj': 'ts_vad2.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_COMMON = {
    'obj': 'ts_vad2.ERes2Net_huge.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_base_COMMON = {
    'obj': 'ts_vad2.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}

ERes2Net_Base_3D_Speaker = {
    'obj': 'ts_vad2.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}

ERes2Net_Large_3D_Speaker = {
    'obj': 'ts_vad2.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 64,
    },
}
ERes2Netv2w24s4ep4_Large_3D_Speaker={
    'obj': "ts_vad2.ERes2NetV2.ERes2NetV2",
    'args':{
    # it is modified from https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common/file/view/master?fileName=configuration.json&status=1
        #"sample_rate": 16000,
        "embedding_size": 192,
        "baseWidth": 24,
        "scale":4,
        "expansion": 4
    },
}
EPACA_CNCeleb = {
    'obj': 'ts_vad2.ECAPA_TDNN.ECAPA_TDNN',
    'args': {
        'input_size': 80,
        'lin_neurons': 192,
        'channels': [1024, 1024, 1024, 1024, 3072],
    },
}

supports = {
    # eres2netv2w24s4ep4 trained on 200k labeled speakers
    'iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common':{
       'revision': 'v1.0.1',
       'model': ERes2Netv2w24s4ep4_Large_3D_Speaker,
       'model_pt': "pretrained_eres2netv2w24s4ep4.ckpt",
    },

    # CAM++ trained on 200k labeled speakers
    'iic/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0',
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
    },
    # ERes2Net trained on 200k labeled speakers
    'iic/speech_eres2net_sv_zh-cn_16k-common': {
        'revision': 'v1.0.5',
        'model': ERes2Net_COMMON,
        'model_pt': 'pretrained_eres2net_aug.ckpt',
    },
    # ERes2Net_Base trained on 200k labeled speakers
    'iic/speech_eres2net_base_200k_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0',
        'model': ERes2Net_base_COMMON,
        'model_pt': 'pretrained_eres2net.pt',
    },
    # CAM++ trained on a large-scale Chinese-English corpus
    'iic/speech_campplus_sv_zh_en_16k-common_advanced': {
        'revision': 'v1.0.0',
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_en_common.pt',
    },
    # CAM++ trained on VoxCeleb
    'iic/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2',
        'model': CAMPPLUS_VOX,
        'model_pt': 'campplus_voxceleb.bin',
    },
    # ERes2Net trained on VoxCeleb
    'iic/speech_eres2net_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2',
        'model': ERes2Net_VOX,
        'model_pt': 'pretrained_eres2net.ckpt',
    },
    # ERes2Net_Base trained on 3dspeaker
    'iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.1',
        'model': ERes2Net_Base_3D_Speaker,
        'model_pt': 'eres2net_base_model.ckpt',
    },
    # ERes2Net_large trained on 3dspeaker
    'iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.0',
        'model': ERes2Net_Large_3D_Speaker,
        'model_pt': 'eres2net_large_model.ckpt',
    },
    # ECAPA-TDNN trained on CNCeleb
    'iic/speech_ecapa-tdnn_sv_zh-cn_cnceleb_16k': {
        'revision': 'v1.0.0',
        'model': EPACA_CNCeleb,
        'model_pt': 'ecapa-tdnn.ckpt',
    },
    # ECAPA-TDNN trained on 3dspeaker
    'iic/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.0',
        'model': EPACA_CNCeleb,
        'model_pt': 'ecapa-tdnn.ckpt',
    },
    # ECAPA-TDNN trained on VoxCeleb
    'iic/speech_ecapa-tdnn_sv_en_voxceleb_16k': {
        'revision': 'v1.0.1',
        'model': EPACA_CNCeleb,
        'model_pt': 'ecapa_tdnn.bin',
    },
}

def dynamic_import(import_path):
    import importlib
    module_name, obj_name = import_path.rsplit('.', 1)
    m = importlib.import_module(module_name)
    return getattr(m, obj_name)




def extract_embeddings(args,batch):
    if args.model_id.startswith('damo/'):
        args.model_id = args.model_id.replace('damo/','iic/', 1)
    assert args.model_id in supports, "Model id not currently supported."

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    conf = supports[args.model_id]
    # download models from modelscope according to model_id
    cache_dir = snapshot_download(
                args.model_id,
                revision=conf['revision'],
                )
    cache_dir = pathlib.Path(cache_dir)


    # link
    download_files = ['examples', conf['model_pt']]
    for src in cache_dir.glob('*'):
        if re.search('|'.join(download_files), src.name):
            dst = save_dir / src.name
            try:
                dst.unlink()
            except FileNotFoundError:
                pass
            dst.symlink_to(src)

    pretrained_model = save_dir / conf['model_pt']
    pretrained_state = torch.load(pretrained_model, map_location='cpu')

    if torch.cuda.is_available():
        msg = 'Using gpu for inference.'
        logging.info(f'{msg}')
        device = torch.device('cuda')
    else:
        msg = 'No cuda device is detected. Using cpu.'
        logging.info(f'{msg}')
        device = torch.device('cpu')

    # load model
    model = conf['model']
    model = dynamic_import(model['obj'])(**model['args'])
    model.load_state_dict(pretrained_state)
    model.to(device)
    model.eval()

    batch = torch.stack(batch) # expect B,T,F
    # compute embedding
    embeddings = model.forward(batch.to(device))#(B,D)
    embeddings = embeddings.detach() ## it will remove requires_grad=True of output of model
    assert embeddings.requires_grad==False
    logging.info(f"embeddings shape: {embeddings.shape} !")
    return embeddings


def extract_embed(args, file, feature_extractor):
    batch = []
    embeddings = []
    wav_length = wave.open(file, "rb").getnframes()  # entire length for target speech
    if wav_length > int(args.length_embedding * 16000):
        for start in range(
            0,
            wav_length - int(args.length_embedding * 16000),
            int(args.step_embedding * 16000),
        ):
            stop = start + int(args.length_embedding * 16000)
            target_speech, _ = soundfile.read(file, start=start, stop=stop)
            target_speech = torch.FloatTensor(np.array(target_speech))
            # because 3d-speaker and wespeaker are offer speaker models which are not include Fbank module,
            # We should perform fbank feature extraction before sending it to the network

            # compute feat
            feat = feature_extractor(target_speech) # (T,F)
            # compute embedding
            batch.append(feat)#[(T,F),(T,F),...]
            if len(batch) == args.batch_size:
                embeddings.extend(extract_embeddings(args, batch))
                batch = []
    else:
        embeddings.extend(
            extract_embeddings(
               args, [torch.FloatTensor(np.array(soundfile.read(file)[0]))]
            )
        )
    if len(batch) != 0:
        embeddings.extend(extract_embeddings(args,batch))

    embeddings = torch.stack(embeddings)
    return embeddings


class FBank(object):
    def __init__(self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):
        sr = 16000
        assert sr==self.sample_rate
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        # select single channel
        if wav.shape[0] > 1:
            wav = wav[0, :]
        assert len(wav.shape) == 2 and wav.shape[0]==1
        feat = Kaldi.fbank(wav, num_mel_bins=self.n_mels,
            sample_frequency=sr, dither=dither)
        # feat: [T, N]
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat


def main():
    args = get_args()
    feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
    logging.info(f'Extracting embeddings...')
    # input is wav list
    wav_list_file = args.wavs[0]
    with open(wav_list_file,'r') as f:
        for line in f:
            wav_path = line.strip()

            embedding = extract_embed(args,wav_path,feature_extractor)

            dest_dir=os.path.join(args.save_dir,os.path.dirname(wav_path).split("/")[-1])
            dest_dir = pathlib.Path(dest_dir)
            dest_dir.mkdir(exist_ok=True, parents=True)
            embedding_name=os.path.basename(wav_path).rsplit('.',1)[0]
            save_path=dest_dir / f'{embedding_name}.pt'
            torch.save(embedding, save_path)
            logging.info(f'The extracted embedding from {wav_path} is saved to {save_path}.')


def setup_logging(verbose=1):
    """Make logging setup with a given log level."""
    if verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

if __name__ == '__main__':
    setup_logging(verbose=1)
    main()
