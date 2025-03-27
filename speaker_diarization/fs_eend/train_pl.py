import os
import argparse
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--OMP_NUM_THREADS', type=int, default=1)
temp_args, _ = parser.parse_known_args()
os.environ["OMP_NUM_THREADS"] = str(temp_args.OMP_NUM_THREADS)

import random
import torch
import yaml
import hyperpyyaml

import numpy as np
import pytorch_lightning as pl

from pytorch_lightning.profilers import AdvancedProfiler

from functools import partial
from collections import defaultdict
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


from fs_eend import OnlineTransformerDADiarization
from scheduler import NoamScheduler
from dataset import KaldiDiarizationDataset, my_collate
from model_pl import SpeakerDiarization

import warnings
warnings.filterwarnings("ignore")


def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def train(configs, args):

    train_set = KaldiDiarizationDataset(
            data_dir=configs["data"]["train_data_dir"],
            chunk_size=configs["data"]["chunk_size"],
            context_size=configs["data"]["context_recp"],
            input_transform=configs["data"]["feat_type"],
            frame_size=configs["data"]["feat"]["win_length"],
            frame_shift=configs["data"]["feat"]["hop_length"],
            subsampling=configs["data"]["subsampling"],
            rate=configs["data"]["feat"]["sample_rate"],
            label_delay=configs["data"]["label_delay"],
            n_speakers=configs["data"]["num_speakers"],
            use_last_samples=configs["data"]["use_last_samples"],
            shuffle=configs["data"]["shuffle"])

    val_set = KaldiDiarizationDataset(
            data_dir=configs["data"]["val_data_dir"],
            chunk_size=configs["data"]["chunk_size"],
            context_size=configs["data"]["context_recp"],
            input_transform=configs["data"]["feat_type"],
            frame_size=configs["data"]["feat"]["win_length"],
            frame_shift=configs["data"]["feat"]["hop_length"],
            subsampling=configs["data"]["subsampling"],
            rate=configs["data"]["feat"]["sample_rate"],
            label_delay=configs["data"]["label_delay"],
            n_speakers=configs["data"]["num_speakers"],
            use_last_samples=configs["data"]["use_last_samples"],
            shuffle=configs["data"]["shuffle"])


    test_set = KaldiDiarizationDataset(
            data_dir=configs["data"]["test_data_dir"],
            chunk_size=args.test_chunk_size,
            context_size=configs["data"]["context_recp"],
            input_transform=configs["data"]["feat_type"],
            frame_size=configs["data"]["feat"]["win_length"],
            frame_shift=configs["data"]["feat"]["hop_length"],
            subsampling=configs["data"]["subsampling"],
            rate=configs["data"]["feat"]["sample_rate"],
            label_delay=configs["data"]["label_delay"],
            n_speakers=args.test_n_speakers,
            use_last_samples=configs["data"]["use_last_samples"],
            shuffle=configs["data"]["shuffle"])

    datasets = {
        "train": train_set,
        "val": val_set,
        "test": test_set
    }

    collate_func = my_collate

    # Define model
    model = OnlineTransformerDADiarization(
        n_speakers=configs["data"]["num_speakers"],
        in_size=(2 * configs["data"]["context_recp"] + 1) * configs["data"]["feat"]["n_mels"],          # Transformer need to know maximum data length
        **configs["model"]["params"],
    )


    # Define optimizer
    opt_config = {
        "params": model.parameters(),
        "lr": configs["training"]["lr"]
    }
    opt_name = configs["training"]["opt"].lower()
    if opt_name == "adam":
        opt = torch.optim.Adam
    elif opt_name == "sgd":
        opt = torch.optim.SGD
    elif opt_name == "noam":
        opt = partial(
            torch.optim.Adam,
            betas=(0.9, 0.98),
            eps=1e-9
        )
    else:
        NotImplementedError
    opt = opt(**opt_config)

    if configs["training"]["scheduler"]:
        print("Using noam scheduler")
        scheduler = NoamScheduler(opt, configs["model"]["params"]["n_units"], configs["training"]["warm_steps"], scale=configs["training"]["schedule_scale"]) if configs["training"]["scheduler"].lower() == "noam" else NotImplementedError
    else:
        scheduler = None

    # Define the logger
    #logger = TensorBoardLogger(os.path.dirname(configs["log"]["log_dir"]), configs["log"]["model_name"])
    logger = TensorBoardLogger(configs["log"]["log_dir"], name=configs["log"]["model_name"])
    #configs["log"]["log_dir"] = logger.log_dir  # updarte log_dir to: ./logs/model_xx/version_xx
    print("Experiment dir:", configs["log"]["log_dir"])
    os.makedirs(configs["log"]["log_dir"], exist_ok=True)
    with open(configs["log"]["log_dir"] + "/config.yaml", "w") as f:
        docs = yaml.dump(configs, f)
        f.close()
    callbacks = [
        EarlyStopping(monitor="val/obj_metric", patience=configs["training"]["early_stop_epoch"], verbose=True, mode="min"),
        ModelCheckpoint(logger.log_dir, monitor="val/obj_metric", save_top_k=configs["log"]["save_top_k"], mode="min", save_last=True)
    ]


    # Define the training setup
    spk_dia_main = SpeakerDiarization(
        hparams=configs,
        model=model,
        datasets=datasets,
        opt=opt,
        scheduler=scheduler,
        collate_func=collate_func
    )

    # Initialization
    if configs["training"]["init_ckpt"]:
        print("Load from checkpoint {} ... ".format(configs["training"]["init_ckpt"]))
        ckpt_package = torch.load(configs["training"]["init_ckpt"], map_location="cpu")
        # state_dict = ckpt_package["state_dict"]
        spk_dia_main.load_state_dict(ckpt_package)


    # Define the trainer
    from lightning.pytorch.strategies import DDPStrategy
    profiler = AdvancedProfiler(filename="perf_logs")
    trainer = pl.Trainer(
        max_epochs=configs["training"]["max_epochs"],
        callbacks=callbacks,
        #gpus=gpus,
        devices=args.gpus, ##The devices to use. Can be set to a positive number (int or str), a sequence of device indices
                     ## (list or str), the value ``-1`` to indicate all available devices should be used,
        accumulate_grad_batches=configs["training"]["grad_accm"],
        logger=logger,
        gradient_clip_val=configs["training"]["grad_clip"],
        check_val_every_n_epoch=configs["training"]["val_interval"],
        strategy='ddp_find_unused_parameters_true',
        #ckpt_path=args.resume_ckpt,
        **configs["debug"]
    )

    if args.inference:
        eval_model_streaming(trainer, spk_dia_main, args.save_avg_path)
    else:
        # Start training
        trainer.fit(spk_dia_main,ckpt_path=args.resume_ckpt)
        #best_path = trainer.checkpoint_callback.best_model_path
        #print("Best model path:", best_path)
        #test_folder = os.path.dirname(best_path)
        eval_model(configs,trainer,args, spk_dia_main, logger)
        #eval_model_streaming(trainer, spk_dia_main, args.save_avg_path)


def eval_model(configs,trainer,args, spk_dia_main, logger):
    best_path = trainer.checkpoint_callback.best_model_path
    #best_path="/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/fs_eend_base/epoch=83-step=7224.ckpt"
    print("Best model path:", best_path)
    test_folder = os.path.dirname(best_path)
    print(f"dirctory of all ckpt : {test_folder}")
    ckpts=[]
    #for x in all_files:
    for x in os.listdir(test_folder):
        #print(x)
        if "epoch" in x:
            #print(f"x_1: {x}")
            if int(x.split("=")[1].split("-")[0])>=int(configs["log"]["start_epoch"]) and int(x.split("=")[1].split("-")[0])<=int(configs["log"]["end_epoch"]):
                #print(f"x: {x}")
                ckpts.append(x)
    print(f"ckpts: {ckpts}")
    print("Test using ckpts:")
    [print(test_folder + "/" + x) for x in ckpts]
    test_state = defaultdict(float)
    for c in ckpts:
        # state_dict = torch.load(test_folder + "/" + c, map_location=torch.device("cuda:{}".format(gpus[0])))["state_dict"]
        state_dict = torch.load(test_folder + "/" + c, map_location="cpu")["state_dict"]
        for name, param in state_dict.items():
            test_state[name] += param / len(ckpts)

    save_model_path= os.path.join(args.exp_dir, "ave_model.pt")
    torch.save(test_state, save_model_path)
    print(f"test_state: {test_state.keys()}")
    spk_dia_main.load_state_dict(test_state)
    trainer.test(spk_dia_main)
    #return save_model_path

def eval_model_streaming(trainer, spk_dia_main, checkpoint):
    #best_path = trainer.checkpoint_callback.best_model_path
    #print("Best model path:", best_path)
    #test_folder = os.path.dirname(best_path)
    test_state = torch.load(checkpoint, map_location="cpu")
    spk_dia_main.load_state_dict(test_state)
    trainer.test(spk_dia_main)

if __name__ == "__main__":

    torch.set_float32_matmul_precision("high")

    parser = ArgumentParser()

    parser.add_argument('--configs', type=str, help='Configuration file path', required=True)
    parser.add_argument('--gpus', type=int, default=None, help='Device used for training')
    parser.add_argument("--save_avg_path",type=str, default=None,help="average checkpoint is stored at experiment checkpoint directory,it is used to inference")
    parser.add_argument("--test_chunk_size", type=int, default=500, help="you can set it same as train stage and or not.")
    parser.add_argument("--inference", type=str2bool,default=False, help="if it is true, it will run inference stage")
    parser.add_argument("--test_n_speakers", type=int, default=2, help="you should set it less than or equal to train stage number of speaker.")
    #parser.add_argument("--checkpoint_resume", default=None, help="Checkpoint path to resume training")
    parser.add_argument("--exp_dir", type=str, default=None, help="Checkpoint path to test training")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="resume the checkpoint and continue to train model.")
    args = parser.parse_args()

    #args = parser.parse_args()
    with open(args.configs, "r") as f:
        configs = hyperpyyaml.load_hyperpyyaml(f)
        print(f"configs: {configs}")
        f.close()

    # Freeze seed
    seed = configs["training"]["seed"]
    if seed:
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pl.seed_everything(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    ## generate experiments directory and rewrite configs["log"]["log_dir"]
    configs["log"]["log_dir"]=args.exp_dir
    train(configs, args)



