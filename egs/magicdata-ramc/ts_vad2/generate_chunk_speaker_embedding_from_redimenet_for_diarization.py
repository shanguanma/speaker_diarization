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

from ecapa_tdnn_wespeaker import (
    ECAPA_TDNN_c1024,
    ECAPA_TDNN_GLOB_c1024,
    ECAPA_TDNN_c512,
    ECAPA_TDNN_GLOB_c512,
)
from resnet_wespeaker import  (
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNet221,
    ResNet293,
)
from samresnet_wespeaker import SimAM_ResNet34_ASP, SimAM_ResNet100_ASP

from redimnet_wespeaker import ReDimNetB0,ReDimNetB1,ReDimNetB2,ReDimNetB3,ReDimNetB4,ReDimNetB5,ReDimNetB6
from features import MelBanks
def get_args():
    parser = argparse.ArgumentParser(description="Extract speaker embeddings.")
    parser.add_argument(
        "--pretrained_model", default="", type=str, help="Model  in wespeaker"
    )
    parser.add_argument(
        "--model_name", default="ReDimNetB3", type=str, help="Model name  in wespeaker"
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

    pretrained_state = torch.load(args.pretrained_model, map_location=device, weights_only=False)
    #pretrained_state = torch.load(args.pretrained_model, map_location=torch.device("cpu"), weights_only=False)
    # Instantiate model(TODO) maduo add model choice
    #model=ECAPA_TDNN_GLOB_c1024(feat_dim=80,embed_dim=192,pooling_func="ASTP")
    model: Optional[nn.Module] = None

    if args.model_name=="ReDimNetB3":
        model = ReDimNetB3(feat_dim=72,embed_dim=192,pooling_func="ASTP")
    elif args.model_name=="ReDimNetB4":
        model = ReDimNetB4(feat_dim=72,embed_dim=192,pooling_func="ASTP")
    elif args.model_name=="ReDimNetB2":
        model = ReDimNetB2(feat_dim=72,embed_dim=192,pooling_func="ASTP")
    # load weight of model
    model.load_state_dict(pretrained_state,strict=False)
    model.to(device)
    model.eval()

    batch = torch.stack(batch)  # expect B,T,F
    #logging.info(f"batch shape: {batch.shape}")
    # compute embedding
    embeddings = model.forward(batch.to(device))  # (B,D)
    if isinstance(embeddings, tuple): # for Resnet* and redimnet model
        embeddings = embeddings[-1]

    embeddings = (
        embeddings.detach()
        )  ## it will remove requires_grad=True of output of model
    assert embeddings.requires_grad == False
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
            #logging.info(f"target_speech shape: {target_speech.shape}")
            # compute feat
            target_speech = target_speech.unsqueeze(0)
            #logging.info(f"after add batch dim, target_speech shape: {target_speech.shape}")
            feat = feature_extractor(target_speech)  #(1,T) ->(1,F,T')
            #logging.info(f"after feature_extactor, feat shape: {feat.shape}")
            feat = feat.permute(0,2,1).squeeze(0) # (1,F,T') -> (T',F)
            # compute embedding
            batch.append(feat)  # [(T',F),(T',F),...]
            if len(batch) == args.batch_size:
                embeddings.extend(extract_embeddings(args, batch))
                batch = []
    else:
        embeddings.extend(
            extract_embeddings(
                args, [feature_extractor(torch.FloatTensor(np.array(soundfile.read(file)[0])))]
            )
        )
    if len(batch) != 0:
        embeddings.extend(extract_embeddings(args, batch))

    embeddings = torch.stack(embeddings)
    return embeddings


def main():
    args = get_args()
    feature_extractor = MelBanks(n_mels=72) # its output shape: (1,F,T')
    logging.info(f"Extracting embeddings...")
    # input is wav list
    wav_list_file = args.wavs[0]
    with open(wav_list_file, "r") as f:
        for line in f:
            wav_path = line.strip()

            embedding = extract_embed(args, wav_path, feature_extractor)

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
