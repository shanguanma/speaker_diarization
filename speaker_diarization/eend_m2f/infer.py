#!/usr/bin/env python3
#
# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#
import os
import yamlargparse
from eend.eend.utils import str2bool
def get_parser():
    parser = yamlargparse.ArgumentParser(description='decoding')
    parser.add_argument('-c', '--config', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('data_dir',
                        help='kaldi-style data dir')
    parser.add_argument('model_file',
                        help='best.nnet')
    parser.add_argument('out_dir',
                        help='output directory.')
    parser.add_argument('--model_type', default='TransformerEda', type=str)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--num-speakers', type=int, default=2)
    parser.add_argument('--input-transform', default='',
                        choices=['', 'log', 'logmel',
                                 'logmel23', 'logmel23_swn', 'logmel23_mn','logmel23_espnet'],
                        help='input transform')
    parser.add_argument('--label-delay', default=0, type=int,
                        help='number of frames delayed from original labels'
                             ' for uni-directional rnn to see in the future')
    parser.add_argument('--hidden-size', default=256, type=int)
    parser.add_argument('--chunk-size', default=2000, type=int,
                        help='input is chunked with this size')
    parser.add_argument('--context-size', default=0, type=int,
                        help='frame splicing')
    parser.add_argument('--subsampling', default=1, type=int)
    parser.add_argument('--sampling-rate', default=16000, type=int,
                        help='sampling rate')
    parser.add_argument('--frame-size', default=1024, type=int,
                        help='frame size')
    parser.add_argument('--frame-shift', default=256, type=int,
                        help='frame shift')

    ## network setting
    parser.add_argument('--model-type', default='eend_m2f',
            help='Type of mask style model(i.e. eend_m2f, eend_fastinst )')
    parser.add_argument('--backbone-encoder-type', default='conformer',type=str,help="")
    parser.add_argument('--backbone-encoder-layers', default=6,type=int,help='')
    parser.add_argument('--backbone-ffn-dim', default=1024,type=int,help='')
    parser.add_argument('--backbone-conformer-depthwise-conv-kernel-size',default=49, type=int, help='')
    parser.add_argument('--backbone-num-heads',default=4,type=int,help='')
    parser.add_argument('--backbone-downsample-type',default='depthwise_pointwise_conv_downsample10',type=str, help='')
    parser.add_argument('--backbone-output-feat-dim',default=256,type=int,help='')
    parser.add_argument('--transformer-decoder-name',default='mask2former',type=str, help='')
    parser.add_argument('--transformer-decoder-input-feat-dim',default=256,type=int, help='')
    parser.add_argument('--transformer-decoder-mask-classification', default=True, type=str2bool,help='')
    parser.add_argument('--transformer-decoder-hidden-dim',default=256,type=int, help='')
    parser.add_argument('--transformer-decoder-num-queries',default=50,type=int, help='')
    parser.add_argument('--transformer-decoder-num-heads',default=4,type=int,help='')
    parser.add_argument('--transformer-decoder-ffn-dim',default=1024,type=int,help='')
    parser.add_argument('--transformer-decoder-num-layers',default=6,type=int,help='')
    parser.add_argument('--transformer-decoder-num-classes', default=20, type=int)
    parser.add_argument('--threshold-discard', default=0.8, type=float, help="more than it, will set to True")
    parser.add_argument('--seed', default=777, type=int)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    from eend.eend_m2f.infer_mask_model import infer
    infer(args)

