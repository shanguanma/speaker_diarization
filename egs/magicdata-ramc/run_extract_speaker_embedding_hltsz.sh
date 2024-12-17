#!/usr/bin/env bash

stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_speaker_diarization_hltsz.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "generate speaker embedding"
   dest_dir=/data/maduo/model_hub
   feature_name=cam++_zh-cn_200k_feature_dir
   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
   model_id=iic/speech_campplus_sv_zh-cn_16k-common
   subsets="dev test train"
   for name in $subsets;do
    #if [ $name = "Train" ];then
     #echo "extract Train settrain target speaker embedding"
     # 提取embedding
     #input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/${name}_Ali_far/target_audio/
     #wav_path=$input_dir/wavs.txt
     #else
     echo "extract $name target speaker embedding"
     # 提取embedding
     input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/${name}_Ali/${name}_Ali_far/target_audio/
     wav_path=$input_dir/wavs.txt
     find $input_dir -name "*.wav" | grep -v "all.wav" >$file
     head $file
     save_dir=$dest_dir/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding/$name/$feature_name
     python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size 32
   done
fi
