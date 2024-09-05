#!/usr/bin/env python3

# Copyright (c) 2022 Xu Xiang
#               2022 Zhengyang Chen (chenzhengyang117@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import kaldiio
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

import onnxruntime as ort
from utils import validate_path

import logging
import torch
import re
import pathlib
from modelscope.hub.snapshot_download import snapshot_download

## >>> import modelscope
#>>> modelscope.__version__
#'1.17.1'


logging.basicConfig(level=logging.INFO,format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

CAMPPLUS_VOX = {
    'obj': 'umap_cluster.cam_pplus_wespeaker.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
    },
}

CAMPPLUS_COMMON = {
    'obj': 'umap_cluster.cam_pplus_wespeaker.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_VOX = {
    'obj': 'umap_cluster.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}
ERes2Net_COMMON = {
    'obj': 'umap_cluster.ERes2Net_huge.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_base_COMMON = {
    'obj': 'umap_cluster.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}

ERes2Net_Base_3D_Speaker = {
    'obj': 'umap_cluster.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}
ERes2Net_Large_3D_Speaker = {
    'obj': 'umap_cluster.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 64,
        },
}

EPACA_CNCeleb = {
    'obj': 'umap_cluster.ECAPA_TDNN.ECAPA_TDNN',
    'args': {
        'input_size': 80,
        'lin_neurons': 192,
        'channels': [1024, 1024, 1024, 1024, 3072],
    },
}

supports = {
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




def load_model(args):
    if args.model_id.startswith('damo/'):
        args.model_id = args.model_id.replace('damo/','iic/', 1)
    assert args.model_id in supports, "Model id not currently supported."

    #save_dir = pathlib.Path(args.save_dir)
    #save_dir.mkdir(exist_ok=True, parents=True)
    save_dir=os.path.dirname(args.ark_path) # str
    save_dir = pathlib.Path(save_dir)

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
    return model, device


def read_fbank(scp_file):
    fbank_dict = OrderedDict()

    for utt, fbank in kaldiio.load_scp_sequential(scp_file):
        fbank_dict[utt] = fbank
    return fbank_dict


def subsegment(fbank, seg_id, window_fs, period_fs, frame_shift):
    subsegs = []
    subseg_fbanks = []

    seg_begin, seg_end = seg_id.split('-')[-2:]
    seg_length = (int(seg_end) - int(seg_begin)) // frame_shift

    # We found that the num_frames + 2 equals to seg_length, which is caused
    # by the implementation of torchaudio.compliance.kaldi.fbank.
    # Thus, here seg_length is used to get the subsegs.
    num_frames, feat_dim = fbank.shape
    if seg_length <= window_fs:
        subseg = seg_id + "-{:08d}-{:08d}".format(0, seg_length)
        subseg_fbank = np.resize(fbank, (window_fs, feat_dim))

        subsegs.append(subseg)
        subseg_fbanks.append(subseg_fbank)
    else:
        max_subseg_begin = seg_length - window_fs + period_fs
        for subseg_begin in range(0, max_subseg_begin, period_fs):
            subseg_end = min(subseg_begin + window_fs, seg_length)
            subseg = seg_id + "-{:08d}-{:08d}".format(subseg_begin, subseg_end)
            subseg_fbank = np.resize(fbank[subseg_begin:subseg_end],
                                     (window_fs, feat_dim))

            subsegs.append(subseg)
            subseg_fbanks.append(subseg_fbank)

    return subsegs, subseg_fbanks


def extract_embeddings(fbanks, batch_size, model, subseg_cmn,device):
    fbanks_array = np.stack(fbanks)
    if subseg_cmn:
        fbanks_array = fbanks_array - np.mean(
            fbanks_array, axis=1, keepdims=True)

    embeddings = []
    for i in tqdm(range(0, fbanks_array.shape[0], batch_size)):
        batch_feats = fbanks_array[i:i + batch_size] #(batch_size, window_fs, fbank_dim) i.e.(96,150,80)
        logging.info(f"model input shape: {batch_feats.shape}")
        batch_feats = torch.from_numpy(batch_feats)
        batch_embs = model.forward(batch_feats.to(device)).squeeze()#  (batch_size, model_dim)

        batch_embs = batch_embs.detach().cpu().numpy()
        logging.info(f"model output shape: {batch_embs.shape}")
        embeddings.append(batch_embs)
    embeddings = np.vstack(embeddings) # (subsegment_nums, model_dim)
    logging.info(f"np.vstack(embeddings) shape: {embeddings.shape}")
    return embeddings


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--scp', required=True, help='wav scp')
    parser.add_argument('--ark-path',
                        required=True,
                        help='path to store embedding ark')
    parser.add_argument('--model-id', required=True, help='modelscope speaker model id ,i.e. iic/speech_campplus_sv_zh_en_16k-common_advanced')
    #parser.add_argument('--device',
    #                    default='cuda',
    #                    help='inference device type: cpu or cuda')
    parser.add_argument('--batch-size',
                        type=int,
                        default=96,
                        help='batch size for embedding extraction')
    parser.add_argument('--frame-shift',
                        type=int,
                        default=10,
                        help='frame shift in fbank extraction (ms)')
    parser.add_argument('--window-secs',
                        type=float,
                        default=1.50,
                        help='the window seconds in embedding extraction')
    parser.add_argument('--period-secs',
                        type=float,
                        default=0.75,
                        help='the shift seconds in embedding extraction')
    parser.add_argument('--subseg-cmn',
                        default=True,
                        type=lambda x: x.lower() == 'true',
                        help='do cmn after or before fbank sub-segmentation')
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    # transform duration to frame number
    window_fs = int(args.window_secs * 1000) // args.frame_shift
    period_fs = int(args.period_secs * 1000) // args.frame_shift

    #session = init_session(args.source, args.device)
    model,device = load_model(args)
    fbank_dict = read_fbank(args.scp)

    subsegs, subseg_fbanks = [], []
    for seg_id, fbank in fbank_dict.items():
        tmp_subsegs, tmp_subseg_fbanks = subsegment(fbank, seg_id, window_fs,
                                                    period_fs,
                                                    args.frame_shift)
        subsegs.extend(tmp_subsegs)
        subseg_fbanks.extend(tmp_subseg_fbanks)
    embeddings = extract_embeddings(subseg_fbanks, args.batch_size, model,
                                    args.subseg_cmn, device)

    validate_path(args.ark_path)
    emb_ark = os.path.abspath(args.ark_path)
    emb_scp = emb_ark[:-3] + "scp"

    with kaldiio.WriteHelper('ark,scp:' + emb_ark + "," + emb_scp) as writer:
        for i, subseg_id in enumerate(subsegs):
            writer(subseg_id, embeddings[i])


if __name__ == '__main__':
    main()
