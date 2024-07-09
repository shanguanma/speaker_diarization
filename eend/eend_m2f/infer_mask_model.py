#!/usr/bin/env python3
#
# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Duo Ma
# Licensed under the MIT license.
#
import os
import h5py
import logging
import numpy as np
from scipy.ndimage import shift

import torch
import torch.nn as nn

#from eend.eend.pytorch_backend.models import TransformerEdaModel, EendEdaModel
from eend import feature
from eend import kaldi_data
from eend.eend_m2f.train_mask_model import get_backbone_model
from eend.eend_m2f.train_mask_model import get_pixel_decoder
from eend.eend_m2f.train_mask_model import get_transformer_decoder
from eend.eend_m2f.model import EendM2F

def _gen_chunk_indices(data_len, chunk_size):
    step = chunk_size
    start = 0
    while start < data_len:
        end = min(data_len, start + chunk_size)
        yield start, end
        start += step


def infer(args):
    logging.basicConfig(level=logging.INFO,format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    logging.info(f"args: {str(args)}")

    os.environ['PYTORCH_SEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    torch.manual_seed(args.seed)
    # Prepare model

    input_dim = feature.get_input_dim(
            args.frame_size,
            args.context_size,
            args.input_transform)

    backbone = get_backbone_model(args,input_dim)
    pixel_decoder = get_pixel_decoder(args)
    transformer_decoder = get_transformer_decoder(args)

    if args.model_type == 'eend_m2f':
        model = EendM2F(
            backbone=backbone,
            pixel_decoder=pixel_decoder,
            transformer_decoder=transformer_decoder,
            num_queries=args.transformer_decoder_num_queries,
            deep_supervision=True,
            no_object_weight=0.1,
            class_weight=2.0,
            mask_weight=5.0,
            dice_weight=5.0,
            location_weight=1000.0,
            proposal_weight=20.0,
            train_num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
        )
    elif args.model_type == 'eend_fastinst':
        pass
    else:
        raise ValueError('Possible model_type is "eend_m2f"')


    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"Device: {device}")


    model = model.to(device)
    states = torch.load(args.model_file,map_location=device)
    model.load_state_dict(states)
    model.eval()

    kaldi_obj = kaldi_data.KaldiData(args.data_dir)
    for recid in kaldi_obj.wavs:
        data, rate = kaldi_obj.load_wav(recid)
        Y = feature.stft(data, args.frame_size, args.frame_shift)
        logging.info(f"args.sampling_rate: {args.sampling_rate}")
        Y = feature.transform(Y, transform_type=args.input_transform,sample_rate=args.sampling_rate)
        Y = feature.splice(Y, context_size=args.context_size)
        Y = Y[::args.subsampling]
        out_chunks = []
        with torch.no_grad():
            hs = None
            for start, end in _gen_chunk_indices(len(Y), args.chunk_size):
                Y_chunked = Y[start:end]
                ## why add .copy(), because it will solve the below Warning:
                # UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors.
                # This means writing to this tensor will result in undefined behavior.
                # You may want to copy the array to protect its data or make it writable before converting it to a tensor.
                # This type of warning will be suppressed for the rest of this program.
                # (Triggered internally at ../torch/csrc/utils/tensor_numpy.cpp:206.)
                Y_chunked = torch.from_numpy(Y_chunked.copy())
                #Y_chunked.to(device)
                #ys = model.infer([Y_chunked.to(device)],threshold_discard=args.threshold_discard)
                ys = model.infer2([Y_chunked.to(device)],threshold_discard=args.threshold_discard)
                out_chunks.append(ys[0])
                #if args.save_attention_weight == 1:
                #    raise NotImplementedError()
        outfname = recid + '.h5'
        outpath = os.path.join(args.out_dir, outfname)
        if args.label_delay != 0:
            outdata = shift(np.vstack(out_chunks), (-args.label_delay, 0))
        else:
            outdata = np.vstack(out_chunks) # Splicing time clips on the time axis, outdata shape: (T,args.num_speakers)
        print(f"start write predict of network into h5 file")
        with h5py.File(outpath, 'w') as wf:
            wf.create_dataset('T_hat', data=outdata)
    print(f"infer finish!!!")
