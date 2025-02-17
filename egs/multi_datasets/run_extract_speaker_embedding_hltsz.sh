#!/usr/bin/env bash

stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_speaker_diarization_hltsz.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "generate speaker embedding for aishell-4"
   dest_dir=/data/maduo/model_hub
   feature_name=cam++_en_zh_advanced_feature_dir
   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
   #model_id=iic/speech_campplus_sv_zh-cn_16k-common
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
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
     input_dir=/data/maduo/datasets/aishell-4/data_processed/$name/target_audio/
     wav_path=$input_dir/wavs.txt
     find $input_dir -name "*.wav" | grep -v "all.wav" >$wav_path
     head $wav_path
     save_dir=$dest_dir/ts_vad/spk_embed/aishell_4/SpeakerEmbedding/$name/$feature_name
     python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size 96
   done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "generate speaker embedding for ami"
   dest_dir=/data/maduo/model_hub
   feature_name=cam++_en_zh_advanced_feature_dir
   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
   #model_id=iic/speech_campplus_sv_zh-cn_16k-common
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
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
     input_dir=/data/maduo/datasets/ami/ami_version1.6.2/data_processed/data/ami/$name/target_audio/
     wav_path=$input_dir/wavs.txt
     find $input_dir -name "*.wav" | grep -v "all.wav" >$wav_path
     head $wav_path
     save_dir=$dest_dir/ts_vad/spk_embed/ami/SpeakerEmbedding/$name/$feature_name
     python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size 96
   done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "generate speaker embedding for aishell-4"
   dest_dir=/data/maduo/model_hub
   feature_name=cam++_zh-cn_200k_feature_dir
   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
   model_id=iic/speech_campplus_sv_zh-cn_16k-common
   #model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
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
     input_dir=/data/maduo/datasets/aishell-4/data_processed/$name/target_audio/
     wav_path=$input_dir/wavs.txt
     find $input_dir -name "*.wav" | grep -v "all.wav" >$wav_path
     head $wav_path
     save_dir=$dest_dir/ts_vad/spk_embed/aishell_4/SpeakerEmbedding/$name/$feature_name
     python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size 96
   done
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "generate speaker embedding for ami"
   dest_dir=/data/maduo/model_hub
   #feature_name=cam++_en_zh_advanced_feature_dir
   feature_name=cam++_zh-cn_200k_feature_dir
   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
   model_id=iic/speech_campplus_sv_zh-cn_16k-common
   #model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
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
     input_dir=/data/maduo/datasets/ami/ami_version1.6.2/data_processed/data/ami/$name/target_audio/
     wav_path=$input_dir/wavs.txt
     find $input_dir -name "*.wav" | grep -v "all.wav" >$wav_path
     head $wav_path
     save_dir=$dest_dir/ts_vad/spk_embed/ami/SpeakerEmbedding/$name/$feature_name
     python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size 96
   done
fi

## magicdata-ramc speaker embedding, you can see  ../magicdata-ramc/run_extract_speaker_embedding_hltsz.sh
## alimeeting speaker embedding

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "generate speaker embedding for msdwild "
   dest_dir=/data/maduo/model_hub
   #feature_name=cam++_en_zh_advanced_feature_dir
   feature_name=cam++_zh-cn_200k_feature_dir
   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
   model_id=iic/speech_campplus_sv_zh-cn_16k-common
   #model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   subsets="dev few_val  many_val train"
   for name in $subsets;do
    #if [ $name = "Train" ];then
     #echo "extract Train settrain target speaker embedding"
     # 提取embedding
     #input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/${name}_Ali_far/target_audio/
     #wav_path=$input_dir/wavs.txt
     #else
     echo "extract $name target speaker embedding"
     # 提取embedding
     input_dir=/data/maduo/datasets/MSDWild/data_processed/$name/target_audio/
     wav_path=$input_dir/wavs.txt
     find $input_dir -name "*.wav" | grep -v "all.wav" >$wav_path
     head $wav_path
     save_dir=$dest_dir/ts_vad/spk_embed/MSDWild/SpeakerEmbedding/$name/$feature_name
     python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size 96
   done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "generate speaker embedding for VoxConverse  "
   dest_dir=/data/maduo/model_hub
   #feature_name=cam++_en_zh_advanced_feature_dir
   feature_name=cam++_zh-cn_200k_feature_dir
   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
   model_id=iic/speech_campplus_sv_zh-cn_16k-common
   #model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
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
     input_dir=/data/maduo/datasets/VoxConverse/data_processed/$name/target_audio/
     wav_path=$input_dir/wavs.txt
     find $input_dir -name "*.wav" | grep -v "all.wav" >$wav_path
     head $wav_path
     save_dir=$dest_dir/ts_vad/spk_embed/VoxConverse/SpeakerEmbedding/$name/$feature_name
     python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size 96
   done
fi
