import re
import os
import torch
import datetime
import yaml
import sys
import logging
import glob

def checkpoint_paths(path: str, pattern=r"model_(\d+)\.pt", keep_match=False):
    """Retrieves all checkpoints found in `path` directory.

    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)
    #files = PathManager.ls(path)
    files = [x for x in Path(path).iterdir() if x.is_file()]
    entries = []
    for i, f in enumerate(files):
        m = pt_regexp.fullmatch(f)
        if m is not None:
            idx = float(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    if keep_match:
        return [(os.path.join(path, x[1]), x[0]) for x in sorted(entries, reverse=True)]
    else:
        return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]


def keep_last_epochs(path: str, keep_last_epochs: int):
    if keep_last_epochs>0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(
            path, pattern=r"model_(\d+)\.pt",
        )
        for old_chk in checkpoints[keep_last_epochs :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)


def keep_best_models(path: str, keep_best_models: int, min_epoch: int=0, max_epoch=sys.maxsize):
    checkpoints = checkpoint_paths(
        path, pattern=r"model_(\d+)\.pt",
    )
    models = []
    val_scores = []
    yamls = glob.glob('{}/model_*.yaml'.format(path))
    for y in yamls:
        with open(y, 'r') as f:
            dic_yaml = yaml.load(f, Loader=yaml.FullLoader)
            loss = dic_yaml['loss'].item()
            epoch = dic_yaml['epoch']
            tag = dic_yaml['tag']
            if epoch >= min_epoch and epoch <= max_epoch:
                val_scores += [[epoch, loss, tag]]
    sorted_val_scores = sorted(val_scores,key=lambda x: x[1],reverse=False) # increase, The value is getting bigger and bigger.
    print("best val (epoch, loss, tag) = " + str(sorted_val_scores[:keep_best_models]))

    keep_path_list = [path + '/model_{}.pt'.format(score[0])for score in sorted_val_scores[:keep_best_models]]
    remove_models = [path + '/model_{}.pt'.format(score[0])for score in sorted_val_scores[keep_best_models:]]
    print(f"keep model list: {keep_path_list}")

    for old_chk in remove_models:
        if os.path.lexists(old_chk):
            os.remove(old_chk)




def save_state_dict_and_infos(model: torch.nn.Module, path: str, infos=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    rank = int(os.environ.get('RANK', 0))
    logging.info(f'[Rank {rank}: Checkpoint: save to checkpoint {path}')
    torch.save(state_dict, path)
    info_path = re.sub('.pt$', '.yaml', path)
    #logging.info(f"infos: {infos}")
    if infos is None:
        infos = {}
    infos['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    with open(info_path, 'w') as fout:
        data = yaml.dump(infos)
        fout.write(data)


