#!/usr/bin/env bash
stage=0
stop_stage=1000
. path_for_dia_pt2.4.sh
. utils/parse_options.sh

# prepared alimeeting, you  see ../alimeeting/prepared_alimeeting.sh
# prepared ami and aishell-4 speaker embedding , you can see run_extract_speaker_embedding_hltsz.sh
# 1. combine label json file
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo ""
   for name in dev train;do
     #if $name="dev";then
     if [ $name = "dev" ];then
      alimeeting_json=/mntcephfs/lab_data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/Eval.json
     else
      alimeeting_json=/mntcephfs/lab_data/maduo/datasets/alimeeting/Train_Ali_far/Train.json
     fi
     ami_json=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/ami/ami_version1.6.2/data_processed/data/ami/$name/$name.json
     aishell_4_json=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/aishell-4/data_processed/$name/$name.json
     out_json=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4
     mkdir -p $out_json
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
      alimeeting=/mntcephfs/lab_data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/target_audio
     else
      alimeeting=/mntcephfs/lab_data/maduo/datasets/alimeeting/Train_Ali_far/target_audio
     fi
     ami=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/ami/ami_version1.6.2/data_processed/data/ami/$name/target_audio
     aishell_4=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/aishell-4/data_processed/$name/target_audio
     out=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4/$name/target_audio
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
      alimeeting=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Eval/cam++_en_zh_advanced_feature_dir
     else
      alimeeting=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Train/cam++_en_zh_advanced_feature_dir
     fi
     ami=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/ami/SpeakerEmbedding/$name/cam++_en_zh_advanced_feature_dir
     aishell_4=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/aishell_4/SpeakerEmbedding/$name/cam++_en_zh_advanced_feature_dir

     out=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4/$name/cam++_en_zh_advanced_feature_dir
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
      alimeeting=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Eval/cam++_zh-cn_200k_feature_dir
     else
      alimeeting=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Train/cam++_zh-cn_200k_feature_dir
     fi
     ami=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/ami/SpeakerEmbedding/$name/cam++_zh-cn_200k_feature_dir
     aishell_4=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/aishell_4/SpeakerEmbedding/$name/cam++_zh-cn_200k_feature_dir

     out=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4/$name/cam++_zh-cn_200k_feature_dir
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


