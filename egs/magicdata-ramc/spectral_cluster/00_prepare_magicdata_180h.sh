#!/usr/bin/env bash

stage=0
stop_stage=1000

. utils/parse_options.sh
#. path_for_nn_vad.sh
. path_for_fsq_sptt.sh # hltsz
#. path_for_fsq_speechtext.sh # sribd

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "prepared kaldi data"
   echo "note: here segments file format is as follows: uttid wavid start_time end_time"
   txt_dir="/data/MagicData-RAMC/MDT2021S003/TXT"
   wav_dir="/data/MagicData-RAMC/MDT2021S003/WAV"
   output_dir="/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC"
   python scripts/magicdata-RAMC/prepare_magicdata_180h.py \
	   $txt_dir\
	   $wav_dir\
	   $output_dir
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "generate spk2utt"
   output_dir="/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC"
   dataset="train dev test"
   for name in  $dataset;do
       source_md/utt2spk_to_spk2utt.pl $output_dir/$name/utt2spk > $output_dir/$name/spk2utt
   done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
  echo "generate reco2dur"
  output_dir="/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC"
  dataset="train dev test"
  ## need kaldi environment
  . path.sh
  for name in $dataset;do
    utils/data/get_reco2dur.sh $output_dir/$name
  done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
  echo "generate segments"
  output_dir="/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC"
  dataset="train dev test"
  for name in $dataset;do
   cat $output_dir/$name/segment>$output_dir/$name/segments 
  done

fi

## we should remove G00000000 releated segment
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "remove G00000000"
   echo "prepared kaldi data"
   echo "note: here segments file format is as follows: uttid wavid start_time end_time"
   txt_dir="/data/MagicData-RAMC/MDT2021S003/TXT"
   wav_dir="/data/MagicData-RAMC/MDT2021S003/WAV"
   dest_dir="/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC_nog0"
   #dataset="train dev test"
   mkdir -p $dest_dir
   python scripts/magicdata-RAMC/prepare_magicdata_180h_nog0.py\
	   $txt_dir\
	   $wav_dir\
	   $dest_dir
    
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "generate spk2utt"
   output_dir="/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC_nog0"
   dataset="train dev test"
   for name in  $dataset;do
       source_md/utt2spk_to_spk2utt.pl $output_dir/$name/utt2spk > $output_dir/$name/spk2utt
   done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
  echo "generate reco2dur"
  output_dir="/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC_nog0"
  dataset="train dev test"
  ## need kaldi environment
  . path.sh
  for name in $dataset;do
    utils/data/get_reco2dur.sh $output_dir/$name
  done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
   echo "remove G00000000 for test dev "
   data_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC
   for name in test dev;do
       grep -v G00000000 $data_dir/$name/rttm > $data_dir/$name/rttm_openslr_gt_${name}_nog0
   done

fi
