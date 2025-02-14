#!/usr/bin/env bash

stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_dia_pt2.4.sh


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "prepared magicdata-ramc kaldi format data"
   ## it has been removed G00000000 utt in rttm file
   # based on the paper "The X-Lance Speaker Diarization System for the Conversational Short-phrase Speaker Diarization Challenge 2022"
   source_data_dir=/mntcephfs/lee_dataset/asr/MagicData-RAMC/
   output_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/
   python3 magicdata_ramc_prepared_180h_with_g0.py $source_data_dir $output_dir

   data_dir=$output_dir
   ## remove  G00000000
   for name in dev test train;do
       grep -v "G00000000" $data_dir/$name/rttm_debug > $data_dir/$name/rttm_debug_nog0
   done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
  echo "get target audio and json label file from rttm file"
  datasets="dev test train"
  #datasets="dev"
  source_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
  for name in $datasets;do
    oracle_rttm=$source_dir/$name/rttm_debug_nog0
    wavscp=$source_dir/$name/wav.scp
    dest_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
    python3 ts_vad2/oracle_rttm_to_generate_target_speaker_wav_and_label_for_ts_vad.py\
         --oracle_rttm $oracle_rttm\
         --wavscp $wavscp\
         --dest_dir $dest_dir\
         --type $name
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "generate speaker embedding"
   dest_dir=/mntcephfs/lab_data/maduo/model_hub
   feature_name=redimnet_b4-vox2-ft_lm_feature_dir
   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
   model_path=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b4-vox2-ft_lm.pt
   model_name="ReDimNetB3"
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
     input_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/${name}/target_audio
     wav_path=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/${name}/wavs.txt
     find $input_dir -name "*.wav" | grep -v "all.wav" >$file
     head $file
     save_dir=$dest_dir/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding/$name/$feature_name
     python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --pretrained_model $model_path\
           --model_name $model_name\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size 32
   done
fi
