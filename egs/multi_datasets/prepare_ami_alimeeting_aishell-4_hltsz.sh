#!/usr/bin/env bash
stage=0
stop_stage=1000
. path_for_speaker_diarization_hltsz.sh
. utils/parse_options.sh

# prepared aishell-4
if [ ${stage} -le -5 ] && [ ${stop_stage} -ge -5 ];then
   dest_dir=/data/maduo/datasets/aishell-4/data_processed
   data_dir=/data/maduo/datasets/aishell-4
   mkdir -p $dest_dir
   for name in train_L train_M train_S;do
     find  $data_dir/$name/TextGrid -name "*.rttm" > $dest_dir/${name}_rttmlist.txt
   done
   for name in train_L train_M train_S;do
     find  $data_dir/$name/wav -name "*.flac" > $dest_dir/${name}_wavlist.txt
   done

fi
if [ ${stage} -le -4 ] && [ ${stop_stage} -ge -4 ];then
   dest_dir=/data/maduo/datasets/aishell-4/data_processed
   for name in train_L train_M train_S;do
    cat $dest_dir/${name}_rttmlist.txt >> $dest_dir/train_rttmlist.txt
    cat $dest_dir/${name}_wavlist.txt >>  $dest_dir/train_wavlist.txt
   done
fi
if [ ${stage} -le -3 ] && [ ${stop_stage} -ge -3 ];then
  dest_dir=/data/maduo/datasets/aishell-4/data_processed
  mkdir -p $dest_dir/train
  mkdir -p $dest_dir/dev

  inp_rttms=$dest_dir/train_rttmlist.txt
  inp_wavs=$dest_dir/train_wavlist.txt
  train_rttms=$dest_dir/train/train.rttm
  train_wavs=$dest_dir/train/wav.scp
  dev_rttms=$dest_dir/dev/dev.rttm
  dev_wavs=$dest_dir/dev/wav.scp
  python3 prepared_train_dev_aishell-4.py \
	  $inp_rttms $inp_wavs\
	  $train_rttms $train_wavs\
	  $dev_rttms $dev_wavs

fi
if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ];then
  inp_dir=/data/maduo/datasets/aishell-4/test
  out_dir=/data/maduo/datasets/aishell-4/data_processed/test
  mkdir -p $out_dir
  python3 prepared_test_aishell-4.py\
	  $inp_dir $out_dir

fi


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
# prepared ami and aishell-4 speaker embedding , you can see run_extract_speaker_embedding_hltsz.sh
# 1. combine label json file
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo ""
   for name in dev train;do
     #if $name="dev";then
     if [ $name = "dev" ];then
      alimeeting_json=/data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/Eval.json
     else
      alimeeting_json=/data/maduo/datasets/alimeeting/Train_Ali_far/Train.json
     fi
     ami_json=/data/maduo/datasets/ami/ami_version1.6.2/data_processed/data/ami/$name/$name.json
     aishell_4_json=/data/maduo/datasets/aishell-4/data_processed/$name/$name.json
     out_json=/data/maduo/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4
     python3 merge_shuff_json.py\
	    $out_json\
	    $name\
	    $alimeeting_json\
	    $ami_json\
	    $aishell_4_json
  done
fi


# 2. combine target_audio file
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
 echo ""
 for name in dev train;do
     #if $name="dev";then
     if [ $name = "dev" ];then
      alimeeting=/data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/target_audio
     else
      alimeeting=/data/maduo/datasets/alimeeting/Train_Ali_far/target_audio
     fi
     ami=/data/maduo/datasets/ami/ami_version1.6.2/data_processed/data/ami/$name/target_audio
     aishell_4=/data/maduo/datasets/aishell-4/data_processed/$name/target_audio
     out=/data/maduo/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4/$name/target_audio
     mkdir -p $out
     python3 combine_files_using_softlink.py\
	     $out\
	     $alimeeting\
	     $ami\
	     $aishell_4

 done
fi

# combine cam++_en_zh_advanced_feature_dir file
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
 echo ""
 for name in dev train;do
     #if $name="dev";then
     if [ $name = "dev" ];then
      alimeeting=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Eval/cam++_en_zh_advanced_feature_dir
     else
      alimeeting=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Train/cam++_en_zh_advanced_feature_dir
     fi
     ami=/data/maduo/model_hub/ts_vad/spk_embed/ami/SpeakerEmbedding/$name/cam++_en_zh_advanced_feature_dir
     aishell_4=/data/maduo/model_hub/ts_vad/spk_embed/aishell_4/SpeakerEmbedding/$name/cam++_en_zh_advanced_feature_dir

     out=/data/maduo/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4/$name/cam++_en_zh_advanced_feature_dir
     mkdir -p $out
     python3 combine_files_using_softlink.py\
             $out\
             $alimeeting\
             $ami\
             $aishell_4

 done
fi


# combine cam++_zh-cn_200k_feature_dir file
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
 echo ""
 for name in dev train;do
     #if $name="dev";then
     if [ $name = "dev" ];then
      alimeeting=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Eval/cam++_zh-cn_200k_feature_dir
     else
      alimeeting=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Train/cam++_zh-cn_200k_feature_dir
     fi
     ami=/data/maduo/model_hub/ts_vad/spk_embed/ami/SpeakerEmbedding/$name/cam++_zh-cn_200k_feature_dir
     aishell_4=/data/maduo/model_hub/ts_vad/spk_embed/aishell_4/SpeakerEmbedding/$name/cam++_zh-cn_200k_feature_dir

     out=/data/maduo/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4/$name/cam++_zh-cn_200k_feature_dir
     mkdir -p $out
     python3 combine_files_using_softlink.py\
             $out\
             $alimeeting\
             $ami\
             $aishell_4

 done
fi
# combine train set of aimeeting, ami and aishell as final train set
# combine dev set of aimeeting, ami and aishell as final dev set


