import logging
import os
from dataclasses import dataclass, field
from typing import Optional
import re

from omegaconf import MISSING

from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import register_task, FairseqTask
import torch

from ts_vad.data.ts_vad_dataset import TSVADDataset


logger = logging.getLogger(__name__)

SPEECH_ENCODER_TYPE = ChoiceEnum(
    ["wavlm", "wavlm_large_ft","ecapa", "fbank", "cam++", "ecapa_wespeaker", "resnet34_wespeaker"]
)
SPEAKER_ENCODER_TYPE = ChoiceEnum(["own", "ecapa", "resnet"])


@dataclass
class TSVADTaskConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory."})
    ts_len: int = field(
        default=6000, metadata={"help": "Input ms of target speaker utterance"}
    )
    rs_len: int = field(default=6000, metadata={"help": "Input ms of reference speech"})
    segment_shift: int = field(
        default=6, metadata={"help": "Speech shift during segmenting"}
    )
    spk_path: Optional[str] = field(
        default=None, metadata={"help": "path to audio directory."}
    )
    aux_path: Optional[str] = field(
        default=None, metadata={"help": "path to aux audio directory."}
    )
    musan_path: Optional[str] = field(default=None, metadata={"help": "musan path."})
    rir_path: Optional[str] = field(default=None, metadata={"help": "rir path."})
    speech_encoder_type: SPEECH_ENCODER_TYPE = field(
        default="ecapa", metadata={"help": "path to pretrained speaker encoder path."}
    )
    speaker_encoder_type: SPEAKER_ENCODER_TYPE = field(
        default="own", metadata={"help": "path to pretrained speaker encoder path."}
    )
    noise_ratio: float = field(
        default=0.5, metadata={"help": "noise ratio when adding noise"}
    )
    zero_ratio: float = field(
        default=0.5,
        metadata={"help": "the ratio to pad zero vector when shuffle level is 0"},
    )
    sample_rate: int = field(
        default=8000, metadata={"help": "sample rate for input audio of SE task"}
    )
    max_num_speaker: int = field(default=4, metadata={"help": "max number of speakers"})
    dataset_name: str = field(default="alimeeting", metadata={"help": "dataset name"})
    collar: float = field(default=0.25, metadata={"help": "collar value"})
    med_filter: int = field(default=21, metadata={"help": "med filter"})
    embed_input: bool = field(default=False, metadata={"help": "embedding input"})
    embed_len: float = field(
        default=1, metadata={"help": "embedding length for diarization"}
    )
    embed_shift: float = field(
        default=0.4, metadata={"help": "embedding shift for diarization"}
    )
    librimix_mode: str = field(default="min", metadata={"help": "librimix mode"})
    librimix_type: str = field(default="both", metadata={"help": "both or clean"})
    mix_noise: bool = field(default=False, metadata={"help": "add noise to mix"})
    aux_noise: bool = field(default=False, metadata={"help": "add noise to aux"})
    ovr_ratio: Optional[float] = field(
        default=None, metadata={"help": "overlap ratio for the SparseLibriMix dataset"}
    )
    label_rate: int = field(default=25, metadata={"help": "diarization label rate"})
    residual_predict: bool = field(
        default=False, metadata={"help": "whether to predict all left audio"}
    )
    random_channel: bool = field(
        default=False, metadata={"help": "for multi-channel, use a randomly one"}
    )
    support_mc: bool = field(
        default=False, metadata={"help": "whether use multi-channel data for icmc"}
    )
    random_mask_speaker_prob: float = field(
        default=0.0, metadata={"help": "whether random mask speaker from input"}
    )
    random_mask_speaker_step: int = field(
        default=0, metadata={"help": "whether random mask speaker from input"}
    )
    spk_dict_postfix: Optional[str] = field(
        default=None, metadata={"help": "dict post fix"}
    )

    # Inf
    min_silence: float = field(default=0.32, metadata={"help": "min silence"})
    min_speech: float = field(default=0.0, metadata={"help": "min speech"})
    inference: bool = field(default=False, metadata={"help": "inference or not"})
    rttm_name: Optional[str] = field(default=None, metadata={"help": "rttm name"})
    fusion_inf: bool = field(
        default=False, metadata={"help": "fusion diar for extraction"}
    )
    inf_use_res_embed: bool = field(
        default=False, metadata={"help": "whether use res embed during inference"}
    )
    sctk_tool_path: str = field(
        default="SCTK-2.4.12", metadata={"help": "specify sctk tool path"}
    )
    rttm_dir: str = field(
        default="SCTK-2.4.12", metadata={"help": "specify reference rttm folder"}
    )
    speaker_embedding_name_dir: str = field(
        default="ecapa_feature_dir",
        metadata={"help": "specify speaker embedding directory name"},
    )
    speaker_embed_dim: int = field(
        default=192, metadata={"help": "speaker embedding dimension."}
    )
    regions: str = field(
        default="all",
        metadata={
            "help": """it is from spyder package help --regions [all|single|overlap|nonoverlap]
                                  Only evaluate on the selected region type.
                                  Default is all.  - all: all regions.  -
                                  single: only single-speaker regions (ignore
                                  silence and multiple speaker).  - overlap:
                                  only regions with multiple speakers in the
                                  reference.  - nonoverlap: only regions
                                  without multiple speakers in the reference.
                                  [default: all]"""
        },
    )


@register_task("ts_vad_task", dataclass=TSVADTaskConfig)
class TSVADTask(FairseqTask):
    """
    This task is responsible for code input tasks.
    If pre-training, then code is the input. No explicit output is provided.
    If fine-tuning, then code is the input, and ltr is the output.
    """

    def __init__(self, cfg: TSVADTaskConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def load_dataset(
        self,
        split: str,
        **kwargs,
    ):
        # For librimix name
        if self.cfg.dataset_name.startswith("libri") and self.cfg.dataset_name.endswith(
            "mix"
        ):
            self.cfg.dataset_name = self.cfg.dataset_name.capitalize()
            self.cfg.dataset_name = (
                self.cfg.dataset_name[:-3] + self.cfg.dataset_name[-3:].capitalize()
            )

        if self.cfg.dataset_name == "alimeeting":
            if split == "Test" or split == "Eval":
                spk_path = f"{self.cfg.spk_path}/{split}/{self.cfg.speaker_embedding_name_dir}"  ## speaker embedding directory
                json_path = f"{self.cfg.data}/{split}_Ali/{split}_Ali_far/{split}.json"  ## offer mixer wavform name,
                audio_path = f"{self.cfg.data}/{split}_Ali/{split}_Ali_far/target_audio"  ## offer number of speaker, offer mixer wavform name, offer target speaker wav,
            elif split == "Train":
                spk_path = f"{self.cfg.spk_path}/{split}/{self.cfg.speaker_embedding_name_dir}"  ## speaker embedding directory
                json_path = f"{self.cfg.data}/{split}_Ali_far/{split}.json"  ## offer mixer wavform name,
                audio_path = f"{self.cfg.data}/{split}_Ali_far/target_audio"  ## offer number of speaker, offer mixer wavform name, offer target speaker wav,

        elif self.cfg.dataset_name == "ami":
            spk_path = f"{self.cfg.spk_path}/{split}"
            json_path = f"{self.cfg.data}/{split}/{split}.json"
            audio_path = f"{self.cfg.data}/{split}"
        elif self.cfg.dataset_name.startswith(
            "Libri"
        ) and self.cfg.dataset_name.endswith("Mix"):
            spk_path = self.cfg.spk_path
            if self.cfg.label_rate != 25:
                json_path = f"{self.cfg.data}/{self.cfg.dataset_name}/{self.cfg.librimix_mode}_json_{self.cfg.label_rate}/{split}.json"
            else:
                json_path = f"{self.cfg.data}/{self.cfg.dataset_name}/{self.cfg.librimix_mode}_json/{split}.json"
            if split == "dev":
                audio_name = "train-100"
            else:
                audio_name = split
            audio_path = f"{self.cfg.data}/{self.cfg.dataset_name}/wav{str(self.cfg.sample_rate)[:-3]}k/{self.cfg.librimix_mode}/{audio_name}/mix_{self.cfg.librimix_type}"
        elif re.match("^SparseLibri(2|3|23)Mix$", self.cfg.dataset_name):
            spk_path = self.cfg.spk_path

            assert (
                self.cfg.ovr_ratio is not None
            ), f"For SparseLibri(2|3|23)Mix dataset, ovr_ratio must be provided."
            if (
                self.cfg.ovr_ratio.is_integer()
            ):  # For some reason ovr_ratio is 1.0 but it should be 1
                ovr_ratio = int(self.cfg.ovr_ratio)
            else:
                ovr_ratio = self.cfg.ovr_ratio

            if self.cfg.label_rate != 25:
                json_path = f"{self.cfg.data}/sparse_23_{ovr_ratio}/json_{self.cfg.label_rate}/{split}.json"
            else:
                json_path = f"{self.cfg.data}/sparse_23_{ovr_ratio}/json/{split}.json"

            audio_path = f"{self.cfg.data}/sparse_23_{ovr_ratio}/{split}/mix_noisy"
        elif self.cfg.dataset_name == "callhome_sim":
            spk_path = f"{self.cfg.spk_path}/{split}/embed"
            json_path = f"{self.cfg.data}/{split}/{split}_{self.cfg.label_rate}.json"
            audio_path = f"{self.cfg.data}/{split}/wavs_16k"
        elif self.cfg.dataset_name == "libri_css_sim":
            spk_path = f"{self.cfg.spk_path}/{split}/embed"
            json_path = f"{self.cfg.data}/SimLibriCSS-{split}/{split}_{self.cfg.label_rate}.json"  # mixer audio coresponding to label
            audio_path = f"{self.cfg.data}/SimLibriCSS-{split}/wav"  ## mixer audio
        elif self.cfg.dataset_name == "icmc":
            spk_path = f"{self.cfg.spk_path}/{split}_embed"
            json_path = f"{self.cfg.data}/{split}/{split}_{self.cfg.label_rate}.json"
            audio_path = f"{self.cfg.data}/{split}"
        else:
            raise Exception(
                f"The given dataset {self.cfg.dataset_name} is not supported."
            )
        if (
            self.cfg.speech_encoder_type == "cam++"
            or self.cfg.speech_encoder_type == "ecapa_wespeaker"
            or self.cfg.speech_encoder_type == "resnet34_wespeaker"
        ):
            fbank_input = True
        else:
            fbank_input = False
        self.datasets[split] = TSVADDataset(
            json_path=json_path,
            audio_path=audio_path,
            ts_len=self.cfg.ts_len,
            rs_len=self.cfg.rs_len,
            spk_path=spk_path,
            aux_path=self.cfg.aux_path,
            is_train="train" in split.lower(),
            segment_shift=self.cfg.segment_shift,
            musan_path=self.cfg.musan_path if "train" in split.lower() else None,
            rir_path=self.cfg.rir_path if "train" in split.lower() else None,
            noise_ratio=self.cfg.noise_ratio,
            zero_ratio=self.cfg.zero_ratio,
            max_num_speaker=self.cfg.max_num_speaker,
            dataset_name=self.cfg.dataset_name,
            sample_rate=self.cfg.sample_rate,
            embed_len=self.cfg.embed_len,
            embed_shift=self.cfg.embed_shift,
            embed_input=self.cfg.embed_input,
            fbank_input=fbank_input,
            label_rate=self.cfg.label_rate,
            random_channel=self.cfg.random_channel,
            support_mc=self.cfg.support_mc,
            random_mask_speaker_prob=self.cfg.random_mask_speaker_prob,
            random_mask_speaker_step=self.cfg.random_mask_speaker_step,
            speaker_embed_dim=self.cfg.speaker_embed_dim,
        )

    def inference_step(self, models, sample, prefix_tokens=None, constraints=None):
        with torch.no_grad():
            return models[0](**sample["net_input"])

    @classmethod
    def setup_task(cls, cfg: TSVADTaskConfig, **kwargs) -> "TSVADTaskConfig":
        return cls(cfg)

    def train_step(
        self, sample, model, criterion, optimizer, update_num, *args, **kwargs
    ):
        loss, sample_size, logging_output = super().train_step(
            sample, model, criterion, optimizer, update_num, *args, **kwargs
        )
        for split in self.datasets:
            if hasattr(self.datasets[split], "update_num"):
                self.datasets[split].update_num = update_num
        return loss, sample_size, logging_output
