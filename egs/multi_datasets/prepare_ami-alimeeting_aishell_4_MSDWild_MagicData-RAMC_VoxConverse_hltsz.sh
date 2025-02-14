#!/usr/bin/env bash

stage=0
stop_stage=1000
. path_for_speaker_diarization_hltsz.sh
. utils/parse_options.sh

# how to prepared wav.scp and rttm and you can see /data/maduo/datasets/aishell-4/data_processed
# prepared aishell-4
# num_speaker:3-7 in one audio
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
  datasets="dev test train"
  #datasets="dev"
  source_dir=/data/maduo/datasets/aishell-4/data_processed
  for name in $datasets;do
    oracle_rttm=$source_dir/$name/${name}.rttm
    wavscp=$source_dir/$name/wav.scp
    dest_dir=$source_dir
    python3 ts_vad2/oracle_rttm_to_generate_target_speaker_wav_and_label_for_ts_vad.py\
             --oracle_rttm $oracle_rttm\
             --wavscp $wavscp\
             --dest_dir $dest_dir\
             --type $name
  done
fi

# prepared ami tsvad data format: label data: $name.json, target_audio, how to prepared wav.scp and rttm, you can see
# /data/maduo/datasets/ami/ami_version1.6.2/data_processed/prepare_ami.sh
# num_speakers: 3-5 in one audio ,4 spks are most
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
  datasets="dev test train"
  #datasets="dev"
  source_dir=/data/maduo/datasets/ami/ami_version1.6.2/data_processed/data/ami
  for name in $datasets;do
    oracle_rttm=$source_dir/$name/$name.rttm
    wavscp=$source_dir/$name/wav.scp
    dest_dir=$source_dir
    python3 ts_vad2/oracle_rttm_to_generate_target_speaker_wav_and_label_for_ts_vad.py\
             --oracle_rttm $oracle_rttm\
             --wavscp $wavscp\
             --dest_dir $dest_dir\
             --type $name
  done
fi

# prepared alimeeting, you  see ../alimeeting/prepared_alimeeting.sh
# num_speakers : 2-4 in one audio

# prepared magicdata-ramc, you can see ../magicdata-ramc/run_ts_vad2_hltsz.sh
# num_speakers : 2 in one audio

#  how to prepared wav.scp and rttm of MSDWild, you can see /data/maduo/datasets/MSDWild/data_processed/run.sh
# num_speakers : 2-10 in one audio, 2 and 3 speakers are most.
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
  datasets="dev few_val  many_val train"
  #datasets="dev"
  source_dir=/data/maduo/datasets/MSDWild/data_processed/
  for name in $datasets;do
    oracle_rttm=$source_dir/$name/$name.rttm
    wavscp=$source_dir/$name/wav.scp
    dest_dir=$source_dir
    rm -r $dest_dir/${name}/$name.json
    rm -r $dest_dir/${name}/target_audio
    python3 ts_vad2/oracle_rttm_to_generate_target_speaker_wav_and_label_for_ts_vad.py\
             --oracle_rttm $oracle_rttm\
             --wavscp $wavscp\
             --dest_dir $dest_dir\
             --type $name
  done
fi

#  how to prepared wav.scp and rttm of VoxConverse, you can see /data/maduo/datasets/VoxConverse/data_processed/run.sh
# num_speakers: 1-21 in one audio
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
  datasets="dev test train"
  #datasets="dev"
  source_dir=/data/maduo/datasets/VoxConverse/data_processed
  for name in $datasets;do
    oracle_rttm=$source_dir/$name/$name.rttm
    wavscp=$source_dir/$name/wav.scp
    dest_dir=$source_dir
    rm -r $dest_dir/${name}/$name.json
    rm -r $dest_dir/${name}/target_audio
    python3 ts_vad2/oracle_rttm_to_generate_target_speaker_wav_and_label_for_ts_vad.py\
             --oracle_rttm $oracle_rttm\
             --wavscp $wavscp\
             --dest_dir $dest_dir\
             --type $name
  done
fi
