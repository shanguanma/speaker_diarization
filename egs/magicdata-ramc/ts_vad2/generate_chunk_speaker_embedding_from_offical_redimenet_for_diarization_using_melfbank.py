#!/usr/bin/env bash

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
import torch.nn as nn
from typing import Optional

from redimnet_wespeaker import ReDimNetB0,ReDimNetB1,ReDimNetB2,ReDimNetB3,ReDimNetB4,ReDimNetB5,ReDimNetB6
#from features import MelBanks
def get_args():
    parser = argparse.ArgumentParser(description="Extract speaker embeddings.")
    parser.add_argument(
        "--redimnet_hubconfig_file_dir", default="ts_vad2/redimnet/", type=str, help="hubconfig.py path  in offical redimnet repo"
    )
    parser.add_argument(
        "--model_name", default="ReDimNetB2", type=str, help="Model name  in offical redimnet repo"
    )
    parser.add_argument("--wavs", nargs="+", type=str, help="Wavs")
    # parser.add_argument('--local_model_dir', default='pretrained', type=str, help='Local model dir')
    parser.add_argument(
        "--save_dir",
        type=str,
        default="model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Train/cam++_en_zh_feature_dir",
        help="speaker embedding dir",
    )
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

def extract_embeddings(args, batch):

    if torch.cuda.is_available():
        msg = "Using gpu for inference."
        logging.info(f"{msg}")
        device = torch.device("cuda")
    else:
        msg = "No cuda device is detected. Using cpu."
        logging.info(f"{msg}")
        device = torch.device("cpu")

    #pretrained_state = torch.load(args.pretrained_model, map_location=device, weights_only=False)
    #pretrained_state = torch.load(args.pretrained_model, map_location=torch.device("cpu"), weights_only=False)
    # Instantiate model(TODO) maduo add model choice
    #model=ECAPA_TDNN_GLOB_c1024(feat_dim=80,embed_dim=192,pooling_func="ASTP")
    model: Optional[nn.Module] = None

    #(NOTE maduo) ReDimNetB4,ReDimNetB5,ReDimNetB6 are oom in tsvad in SRIBD, ReDimNetB0,ReDimNetB1, ReDimNetB are not better than ReDimNetB2.
    if args.model_name=="ReDimNetB2":
        model=torch.hub.load(args.redimnet_hubconfig_file_dir, 'ReDimNet',model_name="b2",train_type="ft_lm",dataset='vox2',source="local")
    model.to(device)
    model.eval()

    batch = torch.stack(batch)  # expect B,T
    #logging.info(f"batch shape: {batch.shape}")
    # compute embedding
    embeddings = model.forward(batch.to(device))  # (B,D)
    if isinstance(embeddings, tuple): #  redimnet model
        embeddings = embeddings[-1]

    embeddings = (
        embeddings.detach()
        )  ## it will remove requires_grad=True of output of model
    assert embeddings.requires_grad == False
    logging.info(f"embeddings shape: {embeddings.shape} !")
    return embeddings


def extract_embed(args, file):
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
            #logging.info(f"target_speech shape: {target_speech.shape}")
            # compute feat
            #target_speech = target_speech.unsqueeze(0)
            #logging.info(f"after add batch dim, target_speech shape: {target_speech.shape}")
            #feat = feature_extractor(target_speech)  #(1,T) ->(1,F,T')
            #logging.info(f"after feature_extactor, feat shape: {feat.shape}")
            #feat = feat.permute(0,2,1).squeeze(0) # (1,F,T') -> (T',F)
            # compute embedding
            #feat = feature_extractor(target_speech) # (T,F)
            batch.append(target_speech)  # [(T'),(T'),...]
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
        embeddings.extend(extract_embeddings(args, batch))

    embeddings = torch.stack(embeddings)
    return embeddings


def main():
    args = get_args()
    #feature_extractor = MelBanks(n_mels=72) # its output shape: (1,F,T')
    #feature_extractor = FBank(72, sample_rate=16000, mean_nor=True)
    logging.info(f'Extracting embeddings...')
    # input is wav list
    wav_list_file = args.wavs[0]
    with open(wav_list_file, "r") as f:
        for line in f:
            wav_path = line.strip()

            embedding = extract_embed(args, wav_path)

            dest_dir = os.path.join(
                args.save_dir, os.path.dirname(wav_path).split("/")[-1]
            )
            dest_dir = pathlib.Path(dest_dir)
            dest_dir.mkdir(exist_ok=True, parents=True)
            embedding_name = os.path.basename(wav_path).rsplit(".", 1)[0]
            save_path = dest_dir / f"{embedding_name}.pt"
            torch.save(embedding, save_path)
            logging.info(
                f"The extracted embedding from {wav_path} is saved to {save_path}."
            )


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


if __name__ == "__main__":
    setup_logging(verbose=1)
    main()
