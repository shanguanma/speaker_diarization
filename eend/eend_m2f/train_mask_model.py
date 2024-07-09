# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Duo Ma
# Licensed under the MIT license.
#
import os
import numpy as np
#from tqdm import tqdm
import logging

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

from eend.eend_m2f.lr_scheduler import  NoamScheduler
from eend.eend_m2f.model import EendM2F
from eend.eend_m2f.backbone2 import Backbone
from eend.eend_m2f.pixel_decoder import OneDimTransposedConvolutionUpsampleLayer
from eend.eend_m2f.mask2former_transformer_decoder import OneScaleMaskedTransformerDecoder
from eend.eend_m2f.diarization_dataset_mask2former import KaldiDiarizationDatasetMask2former, my_collate
from eend.eend_m2f.checkpoints import save_state_dict_and_infos
from eend.eend_m2f.checkpoints import keep_best_models


def get_backbone_model(args,input_dim) -> nn.Module:
    backbone = Backbone(
        encoder_type=args.backbone_encoder_type,
        encoder_n_layers=args.backbone_encoder_layers,
        ffn_dim=args.backbone_ffn_dim,
        conformer_depthwise_conv_kernel_size=args.backbone_conformer_depthwise_conv_kernel_size,
        n_heads=args.backbone_num_heads,
        downsample_type=args.backbone_downsample_type,
        input_feat_dim=input_dim,
        output_feat_dim=args.backbone_output_feat_dim,
    )
    return backbone
def get_pixel_decoder(args)->nn.Module:
    pixel_decoder = OneDimTransposedConvolutionUpsampleLayer(
        feat_dim=args.backbone_output_feat_dim,
    )
    return pixel_decoder
def get_transformer_decoder(args)->nn.Module:
    if args.transformer_decoder_name=="OneScaleMaskedTransformerDecoder":
        transformer_decoder=OneScaleMaskedTransformerDecoder(
            in_channels=args.transformer_decoder_input_feat_dim,
            mask_classification=args.transformer_decoder_mask_classification,
            num_classes=args.transformer_decoder_num_classes,
            hidden_dim=args.transformer_decoder_hidden_dim,
            num_queries=args.transformer_decoder_num_queries,
            nheads=args.transformer_decoder_num_heads,
            dim_feedforward=args.transformer_decoder_ffn_dim,
            dec_layers=args.transformer_decoder_num_layers,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False,
            num_feature_levels=1,
        )
    elif args.transformer_decoder_name=="fastinst":
        pass
    return transformer_decoder


def setup_logging(verbose):
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


def train(rank, world_size,args):
    """ Training model with pytorch backend.
    This function is called from eend/bin/train.py with
    parsed command-line arguments.
    """
    setup_logging(args.verbose)
    #logging.basicConfig(level=logging.INFO,format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

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

    train_set = KaldiDiarizationDatasetMask2former(
        data_dir=args.train_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        input_transform=args.input_transform,
        frame_size=args.frame_size,
        frame_shift=args.frame_shift,
        subsampling=args.subsampling,
        rate=args.sampling_rate,
        use_last_samples=True,
        label_delay=args.label_delay,
        n_speakers=args.num_speakers,
        )
    dev_set = KaldiDiarizationDatasetMask2former(
        data_dir=args.valid_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        input_transform=args.input_transform,
        frame_size=args.frame_size,
        frame_shift=args.frame_shift,
        subsampling=args.subsampling,
        rate=args.sampling_rate,
        use_last_samples=True,
        label_delay=args.label_delay,
        n_speakers=args.num_speakers,
        )

    # Prepare model
    Y, _, _ = next(iter(train_set))
    input_dim = Y.shape[1]
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


    if world_size > 1:
        from eend.eend.pytorch_backend.dist import setup_dist
        setup_dist(rank, world_size, agrs.master_port)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    model = model.to(device)
    logging.info('Prepared model')
    logging.info(model)
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param/1000/1000} MB")

    #model = model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)


    # Setup optimizer(TODO)Duo Ma, will add adamw to aligment to mask2former origin recipe.
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'noam':
        # for noam, lr refers to base_lr (i.e. scale), suggest lr=1.0
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    else:
        raise ValueError(args.optimizer)

    # For noam, we use noam scheduler
    if args.optimizer == 'noam':
        scheduler = NoamScheduler(optimizer,
                                  args.transformer_decoder_hidden_dim,
                                  warmup_steps=args.noam_warmup_steps)

    # Init/Resume
    start_epoch=0
    if args.initmodel:
        logging.info(f"Load model from {args.initmodel}")
        start_epoch=int(args.initmodel.split("/")[-1].split(".")[0].split("_")[-1]) # i.e.: /path/to/model_10.pt
        logging.info(f"model start train from {start_epoch} epoch")
        model.load_state_dict(torch.load(args.initmodel))

    train_iter = DataLoader(
            train_set,
            batch_size=args.batchsize,
            shuffle=True,
            #num_workers=16,
            num_workers=8,
            collate_fn=my_collate,
            )

    dev_iter = DataLoader(
            dev_set,
            batch_size=args.batchsize,
            shuffle=False,
            #num_workers=16,
            num_workers=8,
            collate_fn=my_collate,
            )

    # Training
    # y: feats, t: label
    # grad accumulation is according to: https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20
    for epoch in range(start_epoch + 1, args.max_epochs + 1):
        model.train()
        # zero grad here to accumualte gradient
        optimizer.zero_grad()
        loss_epoch = 0
        num_total = 0
        #for step, (y, t) in tqdm(enumerate(train_iter), ncols=100, total=len(train_iter)):
        for step, (y,t,t_label) in enumerate(train_iter):
            #logging.info(f"device: {device}")
            y = [yi.to(device) for yi in y]
            t = [ti.to(device) for ti in t]
            t_label = [ti.to(device) for ti in t_label]

            loss, stats = model(y,t,t_label)
            #loss, label = batch_pit_loss(output, t)
            # clear graph here
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                # noam should be updated on step-level
                if args.optimizer == 'noam':
                    scheduler.step()

                if args.gradclip > 0:
                    ## the problem: RuntimeError: Expected nested_tensorlist[0].size() > 0 to be true, but got false.
                    ## solution is from https://blog.csdn.net/AmbitiousTyj/article/details/136589229
                    for param in model.parameters():
                        if param.grad is not None and param.grad.nelement() > 0:
                            nn.utils.clip_grad_value_(model.parameters(), args.gradclip)
            loss_epoch += loss.item()
            num_total += 1
        loss_epoch /= num_total

        model.eval()
        with torch.no_grad():
            stats_avg = {}
            cnt = 0
            # issue: raise RuntimeError('received %d items of ancdata' % RuntimeError: received 0 items of ancdata
            # reason: pytorch多线程共享tensor是通过打开文件的方式实现的，而打开文件的数量是有限制的。在使用torch.multiprocess时，
            #         由于子进程中进行了文件读写操作，因此出现了RuntimeError: received 0 items of ancdata的错误
            # solution: torch.multiprocessing.set_sharing_strategy('file_system')
            for y, t,t_label in dev_iter:
                y = [yi.to(device) for yi in y]
                t = [ti.to(device) for ti in t]
                t_label = [ti.to(device) for ti in t_label]


                _,stats = model(y,t,t_label)
                for k, v in stats.items():
                    if type(v).__name__=='dict':
                        for k_s, v_s in v.items():
                            stats_avg[k_s] = stats_avg.get(k_s,0) + v_s
                    elif type(v).__name__=='float':
                        stats_avg[k] = stats_avg.get(k, 0) + v

                cnt += 1
            stats_avg = {k:f"{round(v/cnt,5)}" for k,v in stats_avg.items()}
            #logging.info(f"stats_avg: {stats_avg}")
        model_filename = os.path.join(args.model_save_dir, f"model_{epoch}.pt")
        info ={}
        info.update({"epoch":f"{epoch}","tag": "CV"})
        for k, v in stats_avg.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.item()
            else:
                info[k] = v
        save_state_dict_and_infos(model, model_filename, info)
        logging.info(f"Epoch: {epoch:3d}, LR: {optimizer.param_groups[0]['lr']:.7f},\
            Training Loss: {loss_epoch:.5f}, Dev Stats: {stats_avg}")

    logging.info('Finished!')
