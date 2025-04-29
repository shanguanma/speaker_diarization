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
    chunk_size=3
    step_size=3
    skip_chunk_size=0.93
    pretrain_speaker_model_ckpt=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
    #for sub in dev test;do
    #testset="dev test"
    testset="cssd_testset"
    model_name="cam++"
    for sub in $testset;do
     test_set_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/"
     dest_dir=/data/maduo/exp/speaker_diarization/spectral_cluster_magicdata-ramc/$sub
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
   chunk_size=3
   step_size=3
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


if [ ${stage} -le 102 ] && [ ${stop_stage} -ge 102 ];then

    echo "clustering ......."
    #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
    #vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
    vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=3
    step_size=3
    skip_chunk_size=0.93
    pretrain_speaker_model_ckpt=/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin
    #for sub in dev test;do
    #testset="dev test"
    testset="cssd_testset"
    model_name="cam++"
    for sub in $testset;do
     test_set_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/"
     dest_dir=/data/maduo/exp/speaker_diarization/spectral_cluster_magicdata-ramc/$sub
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
       python cder/score.py  -r $test_set_dir_groundtruth -s $dest_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_cam++_zh_200k
     done
    done
fi

if [ ${stage} -le 104 ] && [ ${stop_stage} -ge 104 ];then

    echo "clustering ......."
    #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
    #vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
    vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=3
    step_size=3
    skip_chunk_size=0.93
    pretrain_speaker_model_ckpt=/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_eres2netv2_sv_zh-cn_16k-common/pretrained_eres2netv2.ckpt
    #for sub in dev test;do
    #testset="dev test"
    testset="cssd_testset"
    model_name="ERes2NetV2_COMMON"
    for sub in $testset;do
     test_set_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$sub/"
     dest_dir=/data/maduo/exp/speaker_diarization/spectral_cluster_magicdata-ramc/$sub
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
       python cder/score.py  -r $test_set_dir_groundtruth -s $dest_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_ERes2NetV2_COMMON_zh_200k
     done
    done
fi


if [ ${stage} -le 106 ] && [ ${stop_stage} -ge 106 ];then

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
if [ ${stage} -le 107 ] && [ ${stop_stage} -ge 107 ];then

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
       python cder/score.py  -r $test_set_dir_groundtruth -s $dest_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_ERes2NetV2_COMMON_zh_200k
     done
    done
fi
