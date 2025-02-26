#!/usr/bin/env bash

stage=0
stop_stage=1000
nj=32
. utils/parse_options.sh
#. path_for_fairseq_speechtext.sh
#. path_for_fsq_speechtext.sh
. path_for_fsq_sptt.sh
#. path.sh # with kaldi env and fsq_sptt
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

if [ ${stage} -le 1 ]&&[ ${stop_stage} -ge 1 ];then

    echo "Start  conformer eend_eda training"
    exp_name=eend_eda_base_100epoch_conformer_magicdata-RAMC
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    train_conf=$root_dir/eend/conf/base/mini_librispeech_train.yaml
    train_dir=$root_dir/data/magicdata-RAMC/train
    dev_dir=$root_dir/data/magicdata-RAMC/dev
    model_dir=$root_dir/exp/$exp_name
    max_epochs=100
    model_type="ConformerEda"
    sampling_rate=16000
    num_speakers=3
    python eend/eend/bin/train_eda.py -c $train_conf \
            $train_dir \
            $dev_dir \
            $model_dir \
            --max-epochs $max_epochs\
            --model-type $model_type\
	    --sampling-rate $sampling_rate\
	    --num-speakers $num_speakers

fi


if [ ${stage} -le 2 ]&&[ ${stop_stage} -ge 2 ];then
    echo "remove all G00000000 "
    echo "Start  conformer eend_eda training"
    exp_name=eend_eda_base_100epoch_conformer_magicdata-RAMC_nog0
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    train_conf=$root_dir/eend/conf/base/mini_librispeech_train.yaml
    train_dir=$root_dir/data/magicdata-RAMC_nog0/train
    dev_dir=$root_dir/data/magicdata-RAMC_nog0/dev
    model_dir=$root_dir/exp/$exp_name
    max_epochs=100
    model_type="ConformerEda"
    sampling_rate=16000
    num_speakers=2
    python eend/eend/bin/train_eda.py -c $train_conf \
            $train_dir \
            $dev_dir \
            $model_dir \
            --max-epochs $max_epochs\
            --model-type $model_type\
            --sampling-rate $sampling_rate\
            --num-speakers $num_speakers

fi
