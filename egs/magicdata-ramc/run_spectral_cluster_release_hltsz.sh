#!/usr/bin/env bash

stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_speaker_diarization_hltsz.sh
if [ ${stage} -le 100 ] && [ ${stop_stage} -ge 100 ];then
    
    echo "clustering ......."
    #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
    #vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
    vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=2
    step_size=2
    skip_chunk_size=0.93
    pretrain_speaker_model_ckpt=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
    #for sub in dev test;do
    #testset="dev test"
    testset="cssd_testset"
    model_name="cam++"
    for sub in $testset;do
     test_set_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/"
     dest_dir=/data/maduo/exp/speaker_diarization/spectral_cluster_magicdata-ramc/$sub
     mkdir -p $dest_dir
     for name in $vad_threshold;do
      python spectral_cluster/020_spectral_cluster_common.py\
         --pretrain_speaker_model_ckpt $pretrain_speaker_model_ckpt \
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_$name \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $dest_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_cn_en_cam++\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type\
         --model_name $model_name
    done
   done
fi
if [ ${stage} -le 101 ] && [ ${stop_stage} -ge 101 ];then
  
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
   #vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91"
   vad_type="transformer_vad"
   chunk_size=2
   step_size=2
   skip_chunk_size=0.93
   testset="cssd_testset"
   #. path_for_nn_vad.sh
   for sub in $testset;do
    test_set_dir="/data/maduo/exp/speaker_diarization/spectral_cluster_magicdata-ramc/$sub"
    test_set_dir_groundtruth=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/rttm_debug_nog0
     for name in $vad_threshold;do
       echo "vad_threshold: $name"
       python cder/score.py  -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_cn_en_cam++
     done
    done
fi
# grep -r Avg logs/run_spectral_cluster_release_hltsz_stage100-101.log
#Avg CDER : 0.227
#Avg CDER : 0.146
#Avg CDER : 0.133
#Avg CDER : 0.133
#Avg CDER : 0.134
#Avg CDER : 0.134
#Avg CDER : 0.133
#Avg CDER : 0.132
#Avg CDER : 0.130
#Avg CDER : 0.130
#Avg CDER : 0.130
#Avg CDER : 0.129
#Avg CDER : 0.129
#Avg CDER : 0.127
#Avg CDER : 0.125
#Avg CDER : 0.124
#Avg CDER : 0.124
#Avg CDER : 0.122
#Avg CDER : 0.120
#Avg CDER : 0.119
#Avg CDER : 0.119
#Avg CDER : 0.118
#Avg CDER : 0.115
#Avg CDER : 0.111
#Avg CDER : 0.100
#Avg CDER : 0.108
#Avg CDER : 0.106

if [ ${stage} -le 102 ] && [ ${stop_stage} -ge 102 ];then

    echo "clustering ......."
    #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
    #vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
    vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=2
    step_size=2
    skip_chunk_size=0.93
    pretrain_speaker_model_ckpt=/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin
    #for sub in dev test;do
    #testset="dev test"
    testset="cssd_testset"
    model_name="cam++"
    for sub in $testset;do
     test_set_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/"
     dest_dir=/data/maduo/exp/speaker_diarization/spectral_cluster_magicdata-ramc/$sub
     mkdir -p $dest_dir
     for name in $vad_threshold;do
      python spectral_cluster/020_spectral_cluster_common.py\
         --pretrain_speaker_model_ckpt $pretrain_speaker_model_ckpt \
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_$name \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $dest_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_cam++_zh_200k\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type\
         --model_name $model_name
    done
   done
fi
if [ ${stage} -le 103 ] && [ ${stop_stage} -ge 103 ];then

   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
   #vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91"
   vad_type="transformer_vad"
   chunk_size=2
   step_size=2
   skip_chunk_size=0.93
   testset="cssd_testset"
   #. path_for_nn_vad.sh
   for sub in $testset;do
    dest_dir="/data/maduo/exp/speaker_diarization/spectral_cluster_magicdata-ramc/$sub"
    test_set_dir_groundtruth=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/rttm_debug_nog0
     for name in $vad_threshold;do
       echo "vad_threshold: $name"
       python cder/score.py  -r $test_set_dir_groundtruth -s $dest_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_cam++_zh_200k
     done
    done
fi
#grep -r Avg logs/run_spectral_cluster_release_hltsz_stage102-103.log
#Avg CDER : 0.242
#Avg CDER : 0.153
#Avg CDER : 0.152
#Avg CDER : 0.149
#Avg CDER : 0.148
#Avg CDER : 0.135
#Avg CDER : 0.135
#Avg CDER : 0.132
#Avg CDER : 0.131
#Avg CDER : 0.129
#Avg CDER : 0.129
#Avg CDER : 0.129
#Avg CDER : 0.129
#Avg CDER : 0.128
#Avg CDER : 0.126
#Avg CDER : 0.123
#Avg CDER : 0.124
#Avg CDER : 0.123
#Avg CDER : 0.121
#Avg CDER : 0.121
#Avg CDER : 0.120
#Avg CDER : 0.120
#Avg CDER : 0.117
#Avg CDER : 0.109
#Avg CDER : 0.098
#Avg CDER : 0.105
#Avg CDER : 0.103


if [ ${stage} -le 104 ] && [ ${stop_stage} -ge 104 ];then

    echo "clustering ......."
    #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
    #vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
    vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=2
    step_size=2
    skip_chunk_size=0.93
    pretrain_speaker_model_ckpt=/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_eres2netv2_sv_zh-cn_16k-common/pretrained_eres2netv2.ckpt
    #for sub in dev test;do
    #testset="dev test"
    testset="cssd_testset"
    model_name="ERes2NetV2_COMMON"
    for sub in $testset;do
     test_set_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/"
     dest_dir=/data/maduo/exp/speaker_diarization/spectral_cluster_magicdata-ramc/$sub
     mkdir -p $dest_dir
     for name in $vad_threshold;do
      python spectral_cluster/020_spectral_cluster_common.py\
         --pretrain_speaker_model_ckpt $pretrain_speaker_model_ckpt \
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_$name \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $dest_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_ERes2NetV2_COMMON_zh_200k\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type\
         --model_name $model_name
    done
   done
fi
if [ ${stage} -le 105 ] && [ ${stop_stage} -ge 105 ];then

   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
   #vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91"
   vad_type="transformer_vad"
   chunk_size=2
   step_size=2
   skip_chunk_size=0.93
   testset="cssd_testset"
   #. path_for_nn_vad.sh
   for sub in $testset;do
    dest_dir="/data/maduo/exp/speaker_diarization/spectral_cluster_magicdata-ramc/$sub"
    test_set_dir_groundtruth=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/rttm_debug_nog0
     for name in $vad_threshold;do
       echo "vad_threshold: $name"
       python cder/score.py  -r $test_set_dir_groundtruth -s $dest_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_ERes2NetV2_COMMON_zh_200k
     done
    done
fi
# grep -r Avg logs/run_spectral_cluster_release_hltsz_stage104-105_again.log
#Avg CDER : 0.225
#Avg CDER : 0.145
#Avg CDER : 0.141
#Avg CDER : 0.142
#Avg CDER : 0.139
#Avg CDER : 0.125
#Avg CDER : 0.125
#Avg CDER : 0.127
#Avg CDER : 0.127
#Avg CDER : 0.125
#Avg CDER : 0.121
#Avg CDER : 0.123
#Avg CDER : 0.123
#Avg CDER : 0.122
#Avg CDER : 0.120
#Avg CDER : 0.119
#Avg CDER : 0.118
#Avg CDER : 0.118
#Avg CDER : 0.113
#Avg CDER : 0.115
#Avg CDER : 0.115
#Avg CDER : 0.115
#Avg CDER : 0.111
#Avg CDER : 0.106
#Avg CDER : 0.096
#Avg CDER : 0.105
#Avg CDER : 0.103



if [ ${stage} -le 106 ] && [ ${stop_stage} -ge 106 ];then

    echo "clustering ......."
    #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
    #vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
    vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=2
    step_size=2
    skip_chunk_size=0.93
    pretrain_speaker_model_ckpt=/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common/pretrained_eres2netv2w24s4ep4.ckpt
    #for sub in dev test;do
    #testset="dev test"
    testset="cssd_testset"
    model_name="ERes2NetV2_w24s4ep4_COMMON"
    for sub in $testset;do
     test_set_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/"
     dest_dir=/data/maduo/exp/speaker_diarization/spectral_cluster_magicdata-ramc/$sub
     mkdir -p $dest_dir
     for name in $vad_threshold;do
      python spectral_cluster/020_spectral_cluster_common.py\
         --pretrain_speaker_model_ckpt $pretrain_speaker_model_ckpt \
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_$name \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $dest_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_ERes2NetV2_w24s4ep4_COMMON_zh_200k\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type\
         --model_name $model_name
    done
   done
fi
if [ ${stage} -le 107 ] && [ ${stop_stage} -ge 107 ];then

   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
   #vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91"
   vad_type="transformer_vad"
   chunk_size=2
   step_size=2
   skip_chunk_size=0.93
   testset="cssd_testset"
   #. path_for_nn_vad.sh
   for sub in $testset;do
    dest_dir="/data/maduo/exp/speaker_diarization/spectral_cluster_magicdata-ramc/$sub"
    test_set_dir_groundtruth=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/rttm_debug_nog0
     for name in $vad_threshold;do
       echo "vad_threshold: $name"
       python cder/score.py  -r $test_set_dir_groundtruth -s $dest_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_ERes2NetV2_w24s4ep4_COMMON_zh_200k
     done
    done
fi

#grep -r Avg logs/run_spectral_cluster_release_hltsz_stage106-107_again.log
#Avg CDER : 0.254
#Avg CDER : 0.149
#Avg CDER : 0.145
#Avg CDER : 0.143
#Avg CDER : 0.142
#Avg CDER : 0.132
#Avg CDER : 0.128
#Avg CDER : 0.127
#Avg CDER : 0.126
#Avg CDER : 0.125
#Avg CDER : 0.125
#Avg CDER : 0.124
#Avg CDER : 0.124
#Avg CDER : 0.123
#Avg CDER : 0.121
#Avg CDER : 0.120
#Avg CDER : 0.120
#Avg CDER : 0.120
#Avg CDER : 0.117
#Avg CDER : 0.116
#Avg CDER : 0.116
#Avg CDER : 0.116
#Avg CDER : 0.111
#Avg CDER : 0.105
#Avg CDER : 0.096
#Avg CDER : 0.103
#Avg CDER : 0.101


if [ ${stage} -le 108 ] && [ ${stop_stage} -ge 108 ];then

    echo "clustering ......."
    #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
    #vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
    vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=3
    step_size=1.5
    skip_chunk_size=0.93
    pretrain_speaker_model_ckpt=/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common/pretrained_eres2netv2w24s4ep4.ckpt
    #for sub in dev test;do
    #testset="dev test"
    testset="cssd_testset"
    model_name="ERes2NetV2_w24s4ep4_COMMON"
    for sub in $testset;do
     test_set_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/"
     dest_dir=/data/maduo/exp/speaker_diarization/spectral_cluster_magicdata-ramc/$sub
     mkdir -p $dest_dir
     for name in $vad_threshold;do
      python spectral_cluster/020_spectral_cluster_common.py\
         --pretrain_speaker_model_ckpt $pretrain_speaker_model_ckpt \
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_$name \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $dest_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_ERes2NetV2_w24s4ep4_COMMON_zh_200k\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type\
         --model_name $model_name
    done
   done
fi
if [ ${stage} -le 109 ] && [ ${stop_stage} -ge 109 ];then

   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
   #vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91"
   vad_type="transformer_vad"
   chunk_size=3
   step_size=1.5
   skip_chunk_size=0.93
   testset="cssd_testset"
   #. path_for_nn_vad.sh
   for sub in $testset;do
    dest_dir="/data/maduo/exp/speaker_diarization/spectral_cluster_magicdata-ramc/$sub"
    test_set_dir_groundtruth=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/rttm_debug_nog0
     for name in $vad_threshold;do
       echo "vad_threshold: $name"
       python cder/score.py  -r $test_set_dir_groundtruth -s $dest_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_ERes2NetV2_w24s4ep4_COMMON_zh_200k
     done
    done
fi

#grep -r Avg logs/run_spectral_cluster_release_hltsz_stage108-109.log
#Avg CDER : 0.189
#Avg CDER : 0.151
#Avg CDER : 0.131
#Avg CDER : 0.130
#Avg CDER : 0.132
#Avg CDER : 0.119
#Avg CDER : 0.117
#Avg CDER : 0.115
#Avg CDER : 0.114
#Avg CDER : 0.115
#Avg CDER : 0.118
#Avg CDER : 0.117
#Avg CDER : 0.116
#Avg CDER : 0.115
#Avg CDER : 0.113
#Avg CDER : 0.111
#Avg CDER : 0.110
#Avg CDER : 0.109
#Avg CDER : 0.109
#Avg CDER : 0.109
#Avg CDER : 0.109
#Avg CDER : 0.109
#Avg CDER : 0.107
#Avg CDER : 0.099
#Avg CDER : 0.095
#Avg CDER : 0.103
#Avg CDER : 0.091

if [ ${stage} -le 110 ] && [ ${stop_stage} -ge 110 ];then

    echo "clustering ......."
    #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
    #vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
    vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=3
    step_size=0.75
    skip_chunk_size=0.93
    pretrain_speaker_model_ckpt=/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common/pretrained_eres2netv2w24s4ep4.ckpt
    #for sub in dev test;do
    #testset="dev test"
    testset="cssd_testset"
    model_name="ERes2NetV2_w24s4ep4_COMMON"
    for sub in $testset;do
     test_set_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/"
     dest_dir=/data/maduo/exp/speaker_diarization/spectral_cluster_magicdata-ramc/$sub
     mkdir -p $dest_dir
     for name in $vad_threshold;do
      python spectral_cluster/020_spectral_cluster_common.py\
         --pretrain_speaker_model_ckpt $pretrain_speaker_model_ckpt \
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_$name \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $dest_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_ERes2NetV2_w24s4ep4_COMMON_zh_200k\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type\
         --model_name $model_name
    done
   done
fi
if [ ${stage} -le 111 ] && [ ${stop_stage} -ge 111 ];then

   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
   #vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91"
   vad_type="transformer_vad"
   chunk_size=3
   step_size=0.75
   skip_chunk_size=0.93
   testset="cssd_testset"
   #. path_for_nn_vad.sh
   for sub in $testset;do
    dest_dir="/data/maduo/exp/speaker_diarization/spectral_cluster_magicdata-ramc/$sub"
    test_set_dir_groundtruth=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/rttm_debug_nog0
     for name in $vad_threshold;do
       echo "vad_threshold: $name"
       python cder/score.py  -r $test_set_dir_groundtruth -s $dest_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_ERes2NetV2_w24s4ep4_COMMON_zh_200k
     done
    done
fi
# grep -r Avg logs/run_spectral_cluster_release_hltsz_stage110-111.log
#Avg CDER : 0.228
#Avg CDER : 0.154
#Avg CDER : 0.153
#Avg CDER : 0.130
#Avg CDER : 0.128
#Avg CDER : 0.128
#Avg CDER : 0.128
#Avg CDER : 0.116
#Avg CDER : 0.118
#Avg CDER : 0.116
#Avg CDER : 0.116
#Avg CDER : 0.118
#Avg CDER : 0.117
#Avg CDER : 0.117
#Avg CDER : 0.116
#Avg CDER : 0.117
#Avg CDER : 0.117
#Avg CDER : 0.116
#Avg CDER : 0.116
#Avg CDER : 0.116
#Avg CDER : 0.117
#Avg CDER : 0.113
#Avg CDER : 0.110
#Avg CDER : 0.108
#Avg CDER : 0.105
#Avg CDER : 0.112
#Avg CDER : 0.109




# sota setting(best CDER=8.9) for cssd_test util 2025-6-6
if [ ${stage} -le 113 ] && [ ${stop_stage} -ge 113 ];then

    echo "clustering ......."
    #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
    #vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
    vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=3
    step_size=3
    skip_chunk_size=0.93
    pretrain_speaker_model_ckpt=/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common/pretrained_eres2netv2w24s4ep4.ckpt
    #for sub in dev test;do
    #testset="dev test"
    testset="cssd_testset"
    model_name="ERes2NetV2_w24s4ep4_COMMON"
    for sub in $testset;do
     test_set_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/"
     dest_dir=/data/maduo/exp/speaker_diarization/spectral_cluster_magicdata-ramc/$sub
     mkdir -p $dest_dir
     for name in $vad_threshold;do
      python spectral_cluster/020_spectral_cluster_common.py\
         --pretrain_speaker_model_ckpt $pretrain_speaker_model_ckpt \
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_$name \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $dest_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_ERes2NetV2_w24s4ep4_COMMON_zh_200k\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type\
         --model_name $model_name
    done
   done
fi
if [ ${stage} -le 114 ] && [ ${stop_stage} -ge 114 ];then

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
    dest_dir="/data/maduo/exp/speaker_diarization/spectral_cluster_magicdata-ramc/$sub"
    test_set_dir_groundtruth=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/rttm_debug_nog0
     for name in $vad_threshold;do
       echo "vad_threshold: $name"
       python cder/score.py  -r $test_set_dir_groundtruth -s $dest_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_ERes2NetV2_w24s4ep4_COMMON_zh_200k
     done
    done
fi
#grep -r Avg logs/run_spectral_cluster_release_hltsz_stage113-115.log
#Avg CDER : 0.196
#Avg CDER : 0.142
#Avg CDER : 0.126
#Avg CDER : 0.122
#Avg CDER : 0.115
#Avg CDER : 0.111
#Avg CDER : 0.111
#Avg CDER : 0.109
#Avg CDER : 0.109
#Avg CDER : 0.108
#Avg CDER : 0.108
#Avg CDER : 0.107
#Avg CDER : 0.106
#Avg CDER : 0.106
#Avg CDER : 0.106
#Avg CDER : 0.104
#Avg CDER : 0.105
#Avg CDER : 0.104
#Avg CDER : 0.103
#Avg CDER : 0.103
#Avg CDER : 0.104
#Avg CDER : 0.103
#Avg CDER : 0.100
#Avg CDER : 0.095
#Avg CDER : 0.089
#Avg CDER : 0.101
#Avg CDER : 0.100


if [ ${stage} -le 115 ] && [ ${stop_stage} -ge 115 ];then

   echo "DER score"
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91" # total 27
   vad_type="transformer_vad"
   chunk_size=3
   step_size=3
   skip_chunk_size=0.93
   testset="cssd_testset"
   collar="0.0 0.25"
   #. path_for_nn_vad.sh
   for c in $collar;do
    for sub in $testset;do
     dest_dir="/data/maduo/exp/speaker_diarization/spectral_cluster_magicdata-ramc/$sub"
     test_set_dir_groundtruth=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/rttm_debug_nog0
      for name in $vad_threshold;do
        echo "vad_threshold: $name"
        perl SCTK-2.4.12/src/md-eval/md-eval.pl -c $c -r $test_set_dir_groundtruth -s $dest_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_ERes2NetV2_w24s4ep4_COMMON_zh_200k
     done
    done
   done
fi

#grep -r DER logs/run_spectral_cluster_release_hltsz_stage113-115.log
#DER score
# collar=0.0
#DER=45.57, miss=3.77, falarm=28.11, confusion=13.69
#DER=38.02, miss=4.32, falarm=21.87, confusion=11.82
#DER=37.04, miss=4.42, falarm=21.15, confusion=11.47
#DER=36.68, miss=4.54, falarm=20.46, confusion=11.67
#DER=35.80, miss=4.61, falarm=20.05, confusion=11.14
#DER=34.75, miss=4.87, falarm=18.81, confusion=11.08
#DER=34.23, miss=4.99, falarm=18.27, confusion=10.98
#DER=33.77, miss=5.08, falarm=17.82, confusion=10.87
#DER=33.45, miss=5.17, falarm=17.44, confusion=10.84
#DER=33.16, miss=5.28, falarm=17.08, confusion=10.80
#DER=32.81, miss=5.35, falarm=16.72, confusion=10.73
#DER=32.27, miss=5.54, falarm=16.06, confusion=10.67
#DER=31.66, miss=5.78, falarm=15.38, confusion=10.49
#DER=31.14, miss=5.98, falarm=14.77, confusion=10.39
#DER=30.71, miss=6.19, falarm=14.25, confusion=10.28
#DER=29.78, miss=6.79, falarm=13.09, confusion=9.90
#DER=29.61, miss=6.98, falarm=12.83, confusion=9.79
#DER=29.50, miss=7.12, falarm=12.62, confusion=9.75
#DER=29.08, miss=7.58, falarm=11.97, confusion=9.53
#DER=29.00, miss=7.76, falarm=11.76, confusion=9.47
#DER=28.94, miss=7.94, falarm=11.56, confusion=9.44
#DER=28.80, miss=8.11, falarm=11.35, confusion=9.34
#DER=27.71, miss=10.01, falarm=8.77, confusion=8.93
#DER=27.93, miss=13.86, falarm=5.80, confusion=8.26
#DER=31.92, miss=23.16, falarm=2.84, confusion=5.93
#DER=50.88, miss=48.12, falarm=0.74, confusion=2.01
#DER=53.81, miss=51.41, falarm=0.62, confusion=1.77

# collar=0.25
#DER=24.88, miss=1.20, falarm=15.25, confusion=8.43
#DER=15.52, miss=1.42, falarm=7.55, confusion=6.54
#DER=14.50, miss=1.46, falarm=6.83, confusion=6.21
#DER=14.16, miss=1.51, falarm=6.20, confusion=6.45
#DER=13.30, miss=1.54, falarm=5.88, confusion=5.89
#DER=12.81, miss=1.65, falarm=5.26, confusion=5.89
#DER=12.48, miss=1.71, falarm=4.97, confusion=5.80
#DER=12.20, miss=1.74, falarm=4.74, confusion=5.72
#DER=12.02, miss=1.78, falarm=4.54, confusion=5.70
#DER=11.87, miss=1.83, falarm=4.35, confusion=5.69
#DER=11.62, miss=1.85, falarm=4.15, confusion=5.62
#DER=11.38, miss=1.95, falarm=3.82, confusion=5.61
#DER=11.08, miss=2.07, falarm=3.54, confusion=5.47
#DER=10.88, miss=2.18, falarm=3.28, confusion=5.41
#DER=10.71, miss=2.30, falarm=3.08, confusion=5.33
#DER=10.37, miss=2.69, falarm=2.64, confusion=5.04
#DER=10.34, miss=2.82, falarm=2.56, confusion=4.96
#DER=10.38, miss=2.92, falarm=2.50, confusion=4.96
#DER=10.33, miss=3.21, falarm=2.30, confusion=4.82  # vad=0.5 # best DER
#DER=10.37, miss=3.34, falarm=2.25, confusion=4.78  # vad=0.51
#DER=10.44, miss=3.46, falarm=2.20, confusion=4.78  # vad=0.52
#DER=10.41, miss=3.59, falarm=2.14, confusion=4.69  # vad=0.53
#DER=11.06, miss=4.86, falarm=1.71, confusion=4.49  # vad=0.6
#DER=12.92, miss=7.59, falarm=1.19, confusion=4.14  # vad=0.7
#DER=18.21, miss=14.69, falarm=0.66, confusion=2.86 #vad=0.8 # Best CDER
#DER=40.45, miss=39.41, falarm=0.16, confusion=0.88 # vad=0.9
#DER=43.93, miss=43.04, falarm=0.13, confusion=0.76 # vad=0.91

