#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

from torch.utils.data import DataLoader
from ts_vad_dataset2 import TSVADDataset
from dataclasses import dataclass


@dataclass
class TSVADDataConfig:
    data_dir: str = "/mntcephfs/lab_data/maduo/datasets/alimeeting"
    """path to target audio and mixture labels root directory."""

    ts_len: int = 6
    """The number of seconds to capture the target speaker's speech"""

    rs_len: int = 4
    """The number of seconds to capture the reference(mixer) speech"""

    rs_segment_shift: int = 2
    """The number of seconds for sliding to capture the reference(mix) speech"""

    spk_path: str = (
        "/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding"
    )
    """path to target speaker embedding directory, if it is not None, otherwise we will use target speaker wavfrom via `data_dir` """

    speech_encoder_type: str = "cam++"
    """path to pretrained speaker encoder path."""

    speaker_embedding_name_dir: str = "cam++_en_zh_advanced_feature_dir"
    """specify speaker embedding directory name"""

    speaker_embed_dim: int = 192
    """speaker embedding dimension."""

    noise_ratio: float = 0.8
    """noise ratio when adding noise"""

    zero_ratio: float = 0.3
    """the ratio to pad zero vector when shuffle level is 0"""

    sample_rate: int = 16000
    """sample rate for input audio of SE task"""

    max_num_speaker: int = 4
    """max number of speakers"""

    dataset_name: str = "alimeeting"
    """dataset name"""

    embed_input: bool = False
    """embedding input"""

    embed_len: float = 1
    """embedding length for diarization"""

    embed_shift: float = 0.4
    """embedding shift for diarization"""

    label_rate: int = 25
    """diarization label rate"""

    random_channel: bool = False
    """for multi-channel, use a randomly one"""

    random_mask_speaker_prob: float = 0.0
    """whether random mask speaker from input"""

    random_mask_speaker_step: int = 0
    """whether random mask speaker from input"""

    musan_path: str = "/mntcephfs/lee_dataset/asr/musan"
    """musan noise directory."""

    rir_path: str = "/mntcephfs/lee_dataset/asr/RIRS_NOISES"
    """rir path."""
    #target_speaker_input_type: str = 'spk_embed'
    #"""set target speaker input type, it can be choices from `spk_embed`, `wavform``"""

cfg = TSVADDataConfig()
from model import TSVADConfig

model_cfg = TSVADConfig()
cfg.speech_encoder_type = model_cfg.speech_encoder_type


def load_dataset(
    cfg,
    split: str,
):
    spk_path = None
    if cfg.dataset_name == "alimeeting":
        if split == "Test" or split == "Eval":
            #if cfg.target_speaker_input_type == "spk_embed":
            if cfg.spk_path is not None:
                spk_path = f"{cfg.spk_path}/{split}/{cfg.speaker_embedding_name_dir}"  ## speaker embedding directory
            else:
                spk_path = None # we will use target speaker wavform via `audio_path`
            json_path = f"{cfg.data_dir}/{split}_Ali/{split}_Ali_far/{split}.json"  ## offer mixer wavform name,
            audio_path = f"{cfg.data_dir}/{split}_Ali/{split}_Ali_far/target_audio"  ## offer number of speaker, offer mixer wavform name, offer target speaker wav,
        elif split == "Train":
            if cfg.spk_path is not None:
                spk_path = f"{cfg.spk_path}/{split}/{cfg.speaker_embedding_name_dir}"  ## speaker embedding directory
            else:
                spk_path = None # we will use target speaker wavform via `audio_path`
            json_path = f"{cfg.data_dir}/{split}_Ali_far/{split}.json"  ## offer mixer wavform name,
            audio_path = f"{cfg.data_dir}/{split}_Ali_far/target_audio"  ## offer number of speaker, offer mixer wavform name, offer target speaker wav,
    else:
        raise Exception(f"The given dataset {cfg.dataset_name} is not supported.")
    if (
        cfg.speech_encoder_type == "WavLM"
        or cfg.speech_encoder_type == "WavLM_weight_sum"
        or cfg.speech_encoder_type == "whisper"
        or cfg.speech_encoder_type == "hubert"
    ):
        fbank_input = False
    else:
        fbank_input = True
    datasets = TSVADDataset(
        json_path=json_path,
        audio_path=audio_path,
        ts_len=cfg.ts_len,
        rs_len=cfg.rs_len,
        spk_path=spk_path,
        is_train="train" in split.lower(),
        rs_segment_shift=cfg.rs_segment_shift, # cut mixer wavform
        zero_ratio=cfg.zero_ratio,
        max_num_speaker=cfg.max_num_speaker,
        dataset_name=cfg.dataset_name,
        sample_rate=cfg.sample_rate,
        embed_len=cfg.embed_len,     # cut mixer emebdding
        embed_shift=cfg.embed_shift, # cut mixer embedding
        embed_input=cfg.embed_input, # embed of mixer (reference) speech, default is False
        fbank_input=fbank_input, # for type of mixer speech input
        label_rate=cfg.label_rate,
        random_channel=cfg.random_channel,
        random_mask_speaker_prob=cfg.random_mask_speaker_prob,
        random_mask_speaker_step=cfg.random_mask_speaker_step,
        speaker_embed_dim=cfg.speaker_embed_dim,
        musan_path=cfg.musan_path if "train" in split.lower() else None,
        rir_path=cfg.rir_path if "train" in split.lower() else None,
        noise_ratio=cfg.noise_ratio,
    )
    return datasets


if __name__ == "__main__":
    import logging
    logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    cfg.spk_path = None
    eval_dataset = load_dataset(cfg, "Train")

    print(eval_dataset)
    print(
        len(eval_dataset)
    )  # __len__ is the dunder method so this should return the length of the dataset
    print(
        eval_dataset[3]
    )  # accessing data at particular indexes by indexing the dataset instance

    # instantiating a dataloader object
    eval_dataloader = DataLoader(
        dataset=eval_dataset,  # the dataset instance
        batch_size=1,  # automatic batching
        drop_last=False,  # drops the last incomplete batch in case the dataset size is not divisible by 64
        shuffle=True,  # shuffles the dataset before every epoch
        collate_fn=eval_dataset.collater,
    )
    import math

    total_num_itrs = int(math.ceil(len(eval_dataloader) / float(3)))
    print(f"len(eval_dataloader): {len(eval_dataloader)}")
    print(f"grouped total_num_itrs = {total_num_itrs}")
    # iterating over the dataloader instance
    for batch_num, data_dict in enumerate(eval_dataloader):
        if batch_num < 2:
            print(f"data_dict['net_input']: {data_dict['net_input']},data_dict['id']: {data_dict['id']}")
            print(f"target_speech shape: {data_dict['net_input']['target_speech'].shape}") # when cfg.spk_path is not None, (B,4,speaker_embeding_dim)
                                                                 # when cfg.spk_path is None, (B,4,96000) # 96000 = 16000 * 6, 6 means that ts_len=6seconds
            print(f"labels shape: {data_dict['net_input']['labels'].shape}")
        else:
            break
    # input, label = data_dict['input'], data_dict['label']
    # print(f'Batch number {batch_num} has {len(input)} data points and correspondingly {len(label)} labels.')

    #    else:
    #        break
