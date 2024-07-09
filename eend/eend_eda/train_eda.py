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

from eend.eend_eda.models import TransformerEdaModel,EendEdaModel
from eend.eend_eda.lr_scheduler import NoamScheduler
from eend.eend_eda.diarization_dataset import KaldiDiarizationDataset, my_collate
from eend.eend_eda.checkpoints import save_state_dict_and_infos
from eend.eend_eda.checkpoints import keep_best_models
#from eend.eend.pytorch_backend.loss import batch_pit_loss, report_diarization_error


def train(rank, world_size,args):
    """ Training model with pytorch backend.
    This function is called from eend/bin/train.py with
    parsed command-line arguments.
    """
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

    train_set = KaldiDiarizationDataset(
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
    dev_set = KaldiDiarizationDataset(
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
    Y, T = next(iter(train_set))

    if args.model_type == 'TransformerEda':
        model = TransformerEdaModel(
                n_speakers=args.num_speakers,
                in_size=Y.shape[1],
                n_units=args.hidden_size,
                n_heads=args.transformer_encoder_n_heads,
                n_layers=args.transformer_encoder_n_layers,
                dropout=args.transformer_encoder_dropout,
                has_pos=False,
                diar_weight=args.diar_weight,
                attractor_weight=args.attractor_weight,
                )
    elif args.model_type == 'ConformerEda':
        model = EendEdaModel(
                n_speakers=args.num_speakers,
                in_size=Y.shape[1],
                n_units=args.hidden_size,
                n_heads=args.transformer_encoder_n_heads,
                n_layers=args.transformer_encoder_n_layers,
                dropout=args.transformer_encoder_dropout,
                diar_weight=args.diar_weight,
                attractor_weight=args.attractor_weight,
                encoder_type="conformer",
                eda_type="lstm"
                )
    else:
        raise ValueError('Possible model_type is "TransformerEda"')


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


    # Setup optimizer
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
                                  args.hidden_size,
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
            collate_fn=my_collate
            )

    dev_iter = DataLoader(
            dev_set,
            batch_size=args.batchsize,
            shuffle=False,
            #num_workers=16,
            num_workers=8,
            collate_fn=my_collate
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
        for step, (y,t) in enumerate(train_iter):
            #logging.info(f"device: {device}")
            y = [yi.to(device) for yi in y]
            t = [ti.to(device) for ti in t]

            loss, stats = model(y,t)
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
            for y, t in dev_iter:
                y = [yi.to(device) for yi in y]
                t = [ti.to(device) for ti in t]
                #output = model(y)
                #_, label = batch_pit_loss(output, t)
                #stats = report_diarization_error(output, label)
                _,stats = model(y,t)
                for k, v in stats.items():
                    stats_avg[k] = stats_avg.get(k, 0) + v
                cnt += 1
            stats_avg = {k:v/cnt for k,v in stats_avg.items()}
            #stats_avg['der'] = stats_avg['der']  * 100
            #for k in stats_avg.keys():
            #    stats_avg[k] = round(stats_avg[k], 2)

        model_filename = os.path.join(args.model_save_dir, f"model_{epoch}.pt")
        #torch.save(model.state_dict(), model_filename)
        #stats = os.path.join(args.model_save_dir, f"model_{epoch}.yaml")
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
