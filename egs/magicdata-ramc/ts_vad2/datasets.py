#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

from torch.utils.data import DataLoader
from ts_vad_dataset import TSVADDataset
from dataclasses import dataclass


@dataclass
class TSVADDataConfig:
    data_dir: str = "/mntcephfs/lab_data/maduo/datasets/alimeeting"
    """path to target audio and mixture labels root directory."""

    ts_len: int = 6000
    """Input ms of target speaker utterance"""

    rs_len: int = 4
    """Input ms of reference speech"""

    segment_shift: int = 2
    """Speech shift during segmenting"""

    spk_path: str = (
        "/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding"
    )
    """path to target speaker embedding directory"""

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


cfg = TSVADDataConfig()
from model import TSVADConfig

model_cfg = TSVADConfig()
cfg.speech_encoder_type = model_cfg.speech_encoder_type


def load_dataset(
    cfg,
    split: str,
):
    #spk_path:str=None
    #json_path:str=None
    #audio_path:str=None
    if cfg.dataset_name == "alimeeting":
        if split == "Test" or split == "Eval":
            spk_path = f"{cfg.spk_path}/{split}/{cfg.speaker_embedding_name_dir}"  ## speaker embedding directory
            json_path = f"{cfg.data_dir}/{split}_Ali/{split}_Ali_far/{split}.json"  ## offer mixer wavform name,
            audio_path = f"{cfg.data_dir}/{split}_Ali/{split}_Ali_far/target_audio"  ## offer number of speaker, offer mixer wavform name, offer target speaker wav,
        elif split == "Train":
            spk_path = f"{cfg.spk_path}/{split}/{cfg.speaker_embedding_name_dir}"  ## speaker embedding directory
            json_path = f"{cfg.data_dir}/{split}_Ali_far/{split}.json"  ## offer mixer wavform name,
            audio_path = f"{cfg.data_dir}/{split}_Ali_far/target_audio"  ## offer number of speaker, offer mixer wavform name, offer target speaker wav,

    elif cfg.dataset_name == "magicdata-ramc":
        spk_path=f"{cfg.spk_path}/{split}/{cfg.speaker_embedding_name_dir}"  ## speaker embedding directory
        json_path=f"{cfg.data_dir}/{split}/{split}.json"  ## offer mixer wavform name,
        audio_path = f"{cfg.data_dir}/{split}/target_audio" ## offer number of speaker, offer mixer wavform name, offer target speaker wav
    else:
        raise Exception(f"The given dataset {cfg.dataset_name} is not supported.")

    if (
        cfg.speech_encoder_type == "WavLM"
        or cfg.speech_encoder_type == "WavLM_weight_sum"
        or cfg.speech_encoder_type == "whisper"
        or cfg.speech_encoder_type == "hubert"
    ):
        fbank_input = False
    elif cfg.speech_encoder_type=="ReDimNetB3" or cfg.speech_encoder_type=="ReDimNetB2" or cfg.speech_encoder_type=="ReDimNetB4":
        redimnet_input=True
        fbank_input = True
    else:
        fbank_input = True
    datasets = TSVADDataset(
        json_path=json_path,
        audio_path=audio_path,
        ts_len=cfg.ts_len,
        rs_len=cfg.rs_len,
        spk_path=spk_path,
        is_train="train" in split.lower(),
        segment_shift=cfg.segment_shift,
        zero_ratio=cfg.zero_ratio,
        max_num_speaker=cfg.max_num_speaker,
        dataset_name=cfg.dataset_name,
        sample_rate=cfg.sample_rate,
        embed_len=cfg.embed_len,
        embed_shift=cfg.embed_shift,
        embed_input=cfg.embed_input,
        fbank_input=fbank_input,
        redimnet_input=redimnet_input,
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
        batch_size=64,  # automatic batching
        drop_last=False,  # drops the last incomplete batch in case the dataset size is not divisible by 64
        shuffle=True,  # shuffles the dataset before every epoch
        collate_fn=eval_dataset.collater,
    )
    import math

    total_num_itrs = int(math.ceil(len(eval_dataloader) / float(64)))
    print(f"len(eval_dataloader): {len(eval_dataloader)}")
    print(f"grouped total_num_itrs = {total_num_itrs}")
    # iterating over the dataloader instance
    # for batch_num, data_dict in enumerate(eval_dataloader):
    #    if batch_num < 2:
    #        print(data_dict['net_input'],data_dict['id'])
    # input, label = data_dict['input'], data_dict['label']
    # print(f'Batch number {batch_num} has {len(input)} data points and correspondingly {len(label)} labels.')

    #    else:
    #        break
