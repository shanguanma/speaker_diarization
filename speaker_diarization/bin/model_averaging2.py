import re
import yaml
import glob
import os
import sys
import argparse
from pathlib import Path
import torch

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

def avg_model(src_path: str, nums: int,dst_model: str, min_epoch: int=0, max_epoch=sys.maxsize, remove_models: bool=True):
    #checkpoints = checkpoint_paths(
    #    path, pattern=r"model_(\d+)\.pt",
    #)

    models = []
    val_scores = []
    yamls = glob.glob('{}/model_*.yaml'.format(src_path))
    for y in yamls:
        with open(y, 'r') as f:
            dic_yaml = yaml.load(f, Loader=yaml.FullLoader)
            loss = dic_yaml['loss']
            epoch = dic_yaml['epoch']
            tag = dic_yaml['tag']
            if int(epoch) >= min_epoch and int(epoch) <= max_epoch:
                val_scores += [[epoch, loss, tag]]
    sorted_val_scores = sorted(val_scores,key=lambda x: x[1],reverse=False) # increase, The value is getting bigger and bigger.
    print("best val (epoch, loss, tag) = " + str(sorted_val_scores[:nums]))

    path_list = [src_path + '/model_{}.pt'.format(score[0])for score in sorted_val_scores[:nums]]
    
    #remove_models = [path + '/model_{}.pt'.format(score[0])for score in sorted_val_scores[keep_best_models:]]
    print(f"best model list: {path_list}")

    #for old_chk in remove_models:
    #    if os.path.lexists(old_chk):
    #        os.remove(old_chk)
    avg = {}
    assert nums == len(path_list)
    for path in path_list:
        print('Processing {}'.format(path))
        states = torch.load(path, map_location=torch.device('cpu'))
        for k in states.keys():
            if k not in avg.keys():
                avg[k] = states[k].clone()
            else:
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], nums)
    print('Saving to {}'.format(dst_model))
    torch.save(avg, dst_model)

    ### remove not best models
    if remove_models:
        remove_models_list = [src_path + '/model_{}.pt'.format(score[0])for score in sorted_val_scores[nums:]]
        remove_yamls_list = [src_path + '/model_{}.yaml'.format(score[0])for score in sorted_val_scores[nums:]]
        print(f" Will remove models list: {remove_models_list}")
        for old_chk in remove_models_list:
            if os.path.lexists(old_chk):
                os.remove(old_chk)
       
        print(f"Will remove models yaml list: {remove_yamls_list}")
        for old_chk in remove_yamls_list:
            if os.path.lexists(old_chk):
                os.remove(old_chk)
    print(f"Finish!!!")


def get_args():
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--dst_model', required=True, help='averaged model')
    parser.add_argument('--src_path',
                        required=True,
                        help='src model path for average')
    parser.add_argument('--nums',
                        default=5,
                        type=int,
                        help='nums for averaged model')
    parser.add_argument('--remove_models',
                        default=True,
                        type=str2bool,
                        help='remove not best models and to save store space.')
    args = parser.parse_args()
    print(args)
    return args

if __name__ == '__main__':
    args = get_args()
    avg_model(src_path=args.src_path, nums=args.nums,dst_model=args.dst_model,remove_models=args.remove_models)
