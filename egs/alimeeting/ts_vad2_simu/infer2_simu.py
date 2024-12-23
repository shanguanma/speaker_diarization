#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn
from dataclasses import dataclass
import logging
import subprocess
import os
import sys

# from argparse import Namespace
from collections import defaultdict
from scipy import signal
from tqdm import tqdm

import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from checkpoint import (
    average_checkpoints,
    find_checkpoints,
    load_checkpoint,
)
from utils import str2bool

## load dataset
from datasets_simu import TSVADDataConfig
from datasets_simu import load_dataset
from model import TSVADModel
from model import TSVADConfig

# remove the short silence
def change_zeros_to_ones(inputs, min_silence, threshold, frame_len):
    res = []
    num_0 = 0
    thr = int(min_silence // frame_len)
    for i in inputs:
        if i >= threshold:
            if num_0 != 0:
                if num_0 > thr:
                    res.extend([0] * num_0)
                else:
                    res.extend([1] * num_0)
                num_0 = 0
            res.extend([1])
        else:
            num_0 += 1
    if num_0 > thr:
        res.extend([0] * num_0)
    else:
        res.extend([1] * num_0)
    return res


# Combine the short speech segments
def change_ones_to_zeros(inputs, min_speech, threshold, frame_len):
    res = []
    num_1 = 0
    thr = int(min_speech // frame_len)
    for i in inputs:
        if i < threshold:
            if num_1 != 0:
                if num_1 > thr:
                    res.extend([1] * num_1)
                else:
                    res.extend([0] * num_1)
                num_1 = 0
            res.extend([0])
        else:
            num_1 += 1
    if num_1 > thr:
        res.extend([1] * num_1)
    else:
        res.extend([0] * num_1)
    return res


def postprocess(res_dict_all, args):

    eval_dir = f"{args.results_path}/{args.split}"
    os.makedirs(eval_dir, exist_ok=True)
    der_write = open(f"{eval_dir}/der_result", "a")
    rttm_path = eval_dir + "/res_rttm"
    rttms = {}
    for threshold in [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]:
        rttms[threshold] = open(f"{rttm_path}_{threshold}", "w")

    frame_len = 1 / args.label_rate
    logging.info(f"frame_len: {frame_len}!!")
    rttm_name = args.rttm_name  # test dataset groud truth rttm name

    for filename in tqdm(res_dict_all):
        speaker_id = filename.split("-")[-1]
        name = filename[: -len(speaker_id) - 1]
        labels = res_dict_all[filename]
        labels = dict(sorted(labels.items()))
        ave_labels = []
        for key in labels:
            ave_labels.append(np.mean(labels[key]))
        labels = signal.medfilt(ave_labels, args.med_filter)

        for threshold in rttms:
            labels_threshold = change_zeros_to_ones(
                labels, args.min_silence, threshold, frame_len
            )
            labels_threshold = change_ones_to_zeros(
                labels_threshold, args.min_speech, threshold, frame_len
            )
            start, duration = 0, 0
            for i, label in enumerate(labels_threshold):
                if label == 1:
                    duration += frame_len
                else:
                    if duration != 0:
                        line = (
                            "SPEAKER "
                            + str(name)
                            + " 1 %.3f" % (start)
                            + " %.3f " % (duration)
                            + "<NA> <NA> "
                            + str(speaker_id)
                            + " <NA> <NA>\n"
                        )
                        rttms[threshold].write(line)
                        duration = 0
                    start = i * frame_len
            if duration != 0:
                line = (
                    "SPEAKER "
                    + str(name)
                    + " 1 %.3f" % (start)
                    + " %.3f " % (duration)
                    + "<NA> <NA> "
                    + str(speaker_id)
                    + " <NA> <NA>\n"
                )
                rttms[threshold].write(line)
    for threshold in rttms:
        rttms[threshold].close()

    for threshold in rttms:
        ## because the below
        out = subprocess.check_output(
            [
                "perl",
                f"{args.sctk_tool_path}/src/md-eval/md-eval.pl",
                f"-c {args.collar}",
                "-s %s" % (f"{rttm_path}_{threshold}"),
                f"-r {args.rttm_dir}/{rttm_name.lower()}",
            ]
        )
        out = out.decode("utf-8")
        DER, MS, FA, SC = (
            float(out.split("/")[0]),
            float(out.split("/")[1]),
            float(out.split("/")[2]),
            float(out.split("/")[3]),
        )

        print(
            "Eval for threshold %2.2f: DER %2.2f%%, MS %2.2f%%, FA %2.2f%%, SC %2.2f%%\n"
            % (threshold, DER, MS, FA, SC)
        )
        print(
            "Eval for threshold %2.2f: DER %2.2f%%, MS %2.2f%%, FA %2.2f%%, SC %2.2f%%\n"
            % (threshold, DER, MS, FA, SC),
            file=der_write,
        )
    der_write.close()


def setup_logging(verbose=2):
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


def load_model(
    device,
    params,
    model: torch.nn.Module,
    model_file: str = None,
    use_averaged_model: bool = False,
):
    #
    if not use_averaged_model:
        if model_file is not None:
            # case1 load one best checkpoint model
            model = model.to(device)
            model.load_state_dict(
                torch.load(params.model_file, map_location=device)["model"]
            )
            model.eval()
    elif use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
            model.eval()
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
            model.eval()


def main(args):
    setup_logging(verbose=2)

    model_cfg = TSVADConfig()
    data_cfg = TSVADDataConfig()
    data_cfg.speech_encoder_type = (
        args.speech_encoder_type
    )  # for fbank_input parameter determination in dataset preparing stage.
    ## overload config
    data_cfg.rs_len = args.rs_len
    data_cfg.segment_shift = args.segment_shift
    data_cfg.spk_path = args.spk_path
    data_cfg.eval_test_spk_path = args.eval_test_spk_path
    data_cfg.speaker_embedding_name_dir = args.speaker_embedding_name_dir
    data_cfg.data_dir = args.data_dir
    data_cfg.eval_test_data_dir = args.eval_test_data_dir
    data_cfg.speaker_embed_dim = args.speaker_embed_dim
    logging.info(f"infer data_cfg: {data_cfg}")
    logging.info(f"currently, it will infer {args.split} set.")
    # split=args.split # i.e.: Eval , Test
    infer_dataset = load_dataset(data_cfg, args.split)

    infer_dl = DataLoader(
        dataset=infer_dataset,  # the dataset instance
        batch_size=args.batch_size,  # automatic batching, i.e.: batch_size=64
        drop_last=False,  # drops the last incomplete batch in case the dataset size is not divisible by 64
        shuffle=False,  # shuffles the dataset before every epoch
        collate_fn=infer_dataset.collater,
    )

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"Device: {device}")

    # load model
    model_cfg.speech_encoder_type = args.speech_encoder_type  #
    model_cfg.speech_encoder_path = args.speech_encoder_path
    model_cfg.speaker_embed_dim = args.speaker_embed_dim
    model_cfg.select_encoder_layer_nums = (
        args.select_encoder_layer_nums
    )  # only for speech_encoder_type=="WavLm"
    model_cfg.wavlm_fuse_feat_post_norm = args.wavlm_fuse_feat_post_norm # only for self.speech_encoder_type == "WavLM_weight_sum"
    model_cfg.speech_encoder_config = args.speech_encoder_config # only for w2v-bert2 ssl model
    logging.info(f"infer model_cfg: {model_cfg}")
    model = TSVADModel(cfg=model_cfg, task_cfg=data_cfg, device=device)

    # logging.info(f"model: {model}")
    load_model(
        device,
        args,
        model,
        model_file=args.model_file,
        use_averaged_model=args.use_averaged_model,
    )

    ## compute model predict DER
    DER = []
    ACC = []
    res_dict_all = defaultdict(lambda: defaultdict(list))
    # for sample in progress:
    for batch_idx, batch in enumerate(infer_dl):
        ref_speech = batch["net_input"]["ref_speech"].to(device)
        target_speech = batch["net_input"]["target_speech"].to(device)
        labels = batch["net_input"]["labels"].to(device)
        labels_len = batch["net_input"]["labels_len"].to(device)
        with torch.no_grad():
            # print(f"ref_speech: {ref_speech}")
            # print(f"target_speech shape: {target_speech.shape}")
            # print(f"labels shape: {labels.shape}")
            # print(f"labels_len shape: {labels_len.shape}")
            result, res_dict = model.infer(
                ref_speech=ref_speech,
                target_speech=target_speech,
                labels=labels,
                labels_len=labels_len,
                # inference=True,
                file_path=batch["net_input"]["file_path"],
                speaker_ids=batch["net_input"]["speaker_ids"],
                start=batch["net_input"]["start"],
            )
        for filename in res_dict:
            for time_step in res_dict[filename]:
                res_dict_all[filename][time_step].extend(res_dict[filename][time_step])

        DER.append(result["DER"])
        ACC.append(result["ACC"])
    print("Model DER: ", sum(DER) / len(DER))
    print("Model ACC: ", sum(ACC) / len(ACC))

    ## final post process diarization result:
    postprocess(res_dict_all, args)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--split",
        type=str,
        default="Eval",
        help="infer dataset name",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="ts_vad/exp5",
        help="infer directory",
    )
    parser.add_argument(
        "--sctk-tool-path",
        type=str,
        default="./SCTK-2.4.12",
        help="tools(i.e. md_eval) directory of eval diarization performance",
    )

    parser.add_argument(
        "--collar",
        type=float,
        default=0.25,
        help="set 0.25s for md_eval",
    )

    parser.add_argument(
        "--rttm-dir",
        type=str,
        default="/mntcephfs/lab_data/maduo/model_hub/ts_vad",
        help="infer dataset ground truth rttm file directory",
    )

    parser.add_argument(
        "--rttm-name",
        type=str,
        default="alimeeting_eval.rttm",
        help="infer dataset ground truth rttm file ",
    )
    parser.add_argument(
        "--min-silence",
        type=float,
        default="0.32",
        help="min silence",
    )
    parser.add_argument(
        "--min-speech",
        type=float,
        default="0.0",
        help="min speech",
    )
    parser.add_argument(
        "--label-rate",
        type=int,
        default=25,
        help="diarization label rate",
    )
    parser.add_argument(
        "--rs-len",
        type=int,
        default=4,
        help="for infer",
    )
    parser.add_argument(
        "--segment-shift",
        type=int,
        default=1,
        help="for infer",
    )

    parser.add_argument(
        "--model-file",
        type=str,
        default=None,
        help="pretrain ts_vad model path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="batch size for infer",
    )
    parser.add_argument(
        "--med-filter",
        type=int,
        default=21,
        help="med filter",
    )
    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=False,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=9,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="ts_vad/exp",
        help="The experiment dir",
    )
    from train_accelerate_ddp2_debug2 import add_model_arguments, add_data_model_common_arguments, add_data_arguments
    add_data_arguments(parser)
    add_model_arguments(parser)
    add_data_model_common_arguments(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
