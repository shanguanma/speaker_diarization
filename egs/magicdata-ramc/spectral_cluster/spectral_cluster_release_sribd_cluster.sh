#!/usr/bin/env bash

stage=0
stop_stage=1000

. utils/parse_options.sh
 if [ ${stage} -le 100 ] && [ ${stop_stage} -ge 100 ];then
     . path_for_dia_pt2.4.sh
    echo "clustering ......."
    #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
    #vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
    vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=3
    step_size=3
    skip_chunk_size=0.93
    pretrain_speaker_model_ckpt=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
    #for sub in dev test;do
    #testset="dev test"
    testset="cssd_testset"
    for sub in $testset;do
     test_set_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/"
     for name in $vad_threshold;do
      python spectral_cluster/020_spectral_cluster_common.py\
         --pretrain_speaker_model_ckpt $pretrain_speaker_model_ckpt \
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_$name \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type
    done
   done
fi
if [ ${stage} -le 101 ] && [ ${stop_stage} -ge 101 ];then
    . path_for_dia_pt2.4.sh
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
   #vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91"
   vad_type="transformer_vad"
   chunk_size=3
   step_size=3
   skip_chunk_size=0.93
   testset="cssd_testset"
   #. path_for_nn_vad.sh
   for sub in $testset;do
    test_set_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/"
    test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/cssd_testset/rttm_debug_nog0
     for name in $vad_threshold;do
       echo "vad_threshold: $name"
       python cder/score.py  -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
     done
    done
fi
