#!/usr/bin/env bash
stage=0
stop_stage=1000
split="Eval" # Eval , Train
. utils/parse_options.sh
. path_for_speaker_diarization.sh

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
   # Here, I assume have system rttm from cluster method.
   # now, I will get target speaker for tsvad train based on system rttm.

   system_rttm="/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/exp/spectral_cluster/alimeeting_eval_system_sad_rttm_cam++_advanced"
   audio_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/audio_dir/" # offical audio directory
   dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1_again
   mkdir -p $dest_dir
   split="Eval"
   [ -d "$dest_dir/Eval_Ali/Eval_Ali_far/target_audio" ] && rm -r $dest_dir/Eval_Ali/Eval_Ali_far/target_audio
   [ -f "$dest_dir/Eval_Ali/Eval_Ali_far/Eval.json" ]  && rm -r $dest_dir/Eval_Ali/Eval_Ali_far/Eval.json
   echo "Make system target speaker and its label and store it under $dest_dir/Eval_Ali/Eval_Ali_far/"
   echo "..."
   ## this script output $dest_dir/Eval_Ali/Eval_Ali_far/target_audio and $target_dir/Eval_Ali/Eval_Ali_far/$split.json
   python3 ts_vad2/system_rttm_to_generate_target_speaker_wav_and_label_for_ts_vad.py\
       --system_rttm $system_rttm\
       --audio_dir $audio_dir\
       --dest_dir $dest_dir\
       --type $split
   echo "Finish , store it under $dest_dir/Eval_Ali/Eval_Ali_far/"
fi
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   # Here, I assume have system rttm from cluster method.
   # i.e. run_spectral_cluster.sh --stage 13 --stop-stage 13
   # now, I will get target speaker for tsvad train based on system rttm.
   system_rttm="/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/exp/spectral_cluster/alimeeting_test_system_sad_rttm_cam++_advanced"
   audio_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting/Test_Ali/Test_Ali_far/audio_dir/" # offical audio directory
   dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1_again
   mkdir -p $dest_dir
   split="Test"
   [ -d "$dest_dir/Test_Ali/Test_Ali_far/target_audio" ] && rm -r $dest_dir/Test_Ali/Test_Ali_far/target_audio
   [ -f "$dest_dir/Test_Ali/Test_Ali_far/Test.json" ]  && rm -r $dest_dir/Test_Ali/Test_Ali_far/Test.json
   echo "Make system target speaker and its label and store it under $dest_dir/Test_Ali/Test_Ali_far/"
   echo "..."
   ## this script output $dest_dir/Test_Ali/Test_Ali_far/target_audio and $target_dir/Test_Ali/Test_Ali_far/$split.json
   python3 ts_vad2/system_rttm_to_generate_target_speaker_wav_and_label_for_ts_vad.py\
       --system_rttm $system_rttm\
       --audio_dir $audio_dir\
       --dest_dir $dest_dir\
       --type $split
   echo "Finish , store it under $dest_dir/Test_Ali/Test_Ali_far/"
fi

## prepared target speaker embedding
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "prepare eval set target audio list"
   for name in Eval Test;do
   input_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1_again/${name}_Ali/${name}_Ali_far/target_audio
   file=$input_dir/wavs.txt
    python3 ts_vad2/prepare_alimeeting_target_audio_list.py \
        $input_dir $file
   done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "generate eval set speaker embedding"
   for name in Eval Test;do
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1_again
    feature_name=cam++_en_zh_advanced_feature_dir
    model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
    # 提取embedding
    input_dir=$exp_dir/${name}_Ali/${name}_Ali_far/target_audio
    file=$input_dir/wavs.txt
    wav_path=$file
    save_dir=$exp_dir/SpeakerEmbedding/${name}/$feature_name
    python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
        --model_id $model_id --wavs $wav_path\
        --save_dir $save_dir
   echo "Finish extract target speaker embedding!!!!"
   done
fi

# 2024-10-14 , I want to improve diarization on system vad.
# 1. I want to infer system vad data via pretrain my tsvad model on oracle data.
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_epoch40_front_fix_seed
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval"
 #infer_sets="Train"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1_again/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

 # # spectral cluster target audio , mix audio and labels path
 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1_again"
 for name in $infer_sets;do
    results_path=$exp_dir/sys_cam++_${name}_again
  python3 ts_vad2/infer2.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --select-encoder-layer-nums 6\
    --data-dir $data_dir
done
# Eval set
#Model DER:  0.3179269075704802
#Model ACC:  0.9187840445304846
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 8.51%, MS 0.98%, FA 7.12%, SC 0.42%
#
#Eval for threshold 0.30: DER 6.10%, MS 1.47%, FA 4.09%, SC 0.54%
#
#Eval for threshold 0.35: DER 5.44%, MS 1.73%, FA 3.09%, SC 0.61%
#
#Eval for threshold 0.40: DER 5.05%, MS 2.00%, FA 2.40%, SC 0.65%
#
#Eval for threshold 0.45: DER 4.93%, MS 2.34%, FA 1.95%, SC 0.65%
#
#Eval for threshold 0.50: DER 4.92%, MS 2.76%, FA 1.52%, SC 0.64%
#
#Eval for threshold 0.55: DER 5.13%, MS 3.32%, FA 1.22%, SC 0.59%
#
#Eval for threshold 0.60: DER 5.48%, MS 3.96%, FA 0.99%, SC 0.53%
#
#Eval for threshold 0.70: DER 6.62%, MS 5.63%, FA 0.63%, SC 0.35%
#
#Eval for threshold 0.80: DER 8.81%, MS 8.16%, FA 0.44%, SC 0.21%

fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_epoch40_front_fix_seed
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Test"
 #infer_sets="Train"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1_again/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

 # # spectral cluster target audio , mix audio and labels path
 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1_again"
 for name in $infer_sets;do
    results_path=$exp_dir/sys_cam++_${name}_again
  python3 ts_vad2/infer2.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --select-encoder-layer-nums 6\
    --data-dir $data_dir
done
# Test set
#Model DER:  0.39102962966035737
#Model ACC:  0.9115267263778823
#100%|██████████| 60/60 [00:57<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 8.02%, MS 1.12%, FA 6.44%, SC 0.46%
#
#Eval for threshold 0.30: DER 5.93%, MS 1.72%, FA 3.65%, SC 0.56%
#
#Eval for threshold 0.35: DER 5.45%, MS 2.03%, FA 2.82%, SC 0.60%
#
#Eval for threshold 0.40: DER 5.14%, MS 2.40%, FA 2.10%, SC 0.64%
#
#Eval for threshold 0.45: DER 5.07%, MS 2.82%, FA 1.59%, SC 0.66%
#
#Eval for threshold 0.50: DER 5.17%, MS 3.32%, FA 1.20%, SC 0.65%
#
#Eval for threshold 0.55: DER 5.45%, MS 3.92%, FA 0.91%, SC 0.61%
#
#Eval for threshold 0.60: DER 5.86%, MS 4.60%, FA 0.72%, SC 0.55%
#
#Eval for threshold 0.70: DER 7.01%, MS 6.25%, FA 0.36%, SC 0.40%
#
#Eval for threshold 0.80: DER 9.20%, MS 8.75%, FA 0.18%, SC 0.26%
fi

## use not better speaker model(i.e. resnet34-LM) to get spectral cluster result
# i.e. run_spectral_cluster.sh --stage 8
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
   # Here, I assume have system rttm from cluster method.
   # now, I will get target speaker for tsvad train based on system rttm.

   system_rttm="/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/exp/spectral_cluster/alimeeting_eval_system_sad_rttm"
   audio_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/audio_dir/" # offical audio directory
   dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1_resnet34-lm_spectral_cluster
   mkdir -p $dest_dir
   split="Eval"
   [ -d "$dest_dir/Eval_Ali/Eval_Ali_far/target_audio" ] && rm -r $dest_dir/Eval_Ali/Eval_Ali_far/target_audio
   [ -f "$dest_dir/Eval_Ali/Eval_Ali_far/Eval.json" ]  && rm -r $dest_dir/Eval_Ali/Eval_Ali_far/Eval.json
   echo "Make system target speaker and its label and store it under $dest_dir/Eval_Ali/Eval_Ali_far/"
   echo "..."
   ## this script output $dest_dir/Eval_Ali/Eval_Ali_far/target_audio and $target_dir/Eval_Ali/Eval_Ali_far/$split.json
   python3 ts_vad2/system_rttm_to_generate_target_speaker_wav_and_label_for_ts_vad.py\
       --system_rttm $system_rttm\
       --audio_dir $audio_dir\
       --dest_dir $dest_dir\
       --type $split
   echo "Finish , store it under $dest_dir/Eval_Ali/Eval_Ali_far/"
fi
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
   # Here, I assume have system rttm from cluster method.
   # i.e. run_spectral_cluster.sh --stage 13 --stop-stage 13
   # now, I will get target speaker for tsvad train based on system rttm.
   system_rttm="/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/exp/spectral_cluster/alimeeting_test_system_sad_rttm"
   audio_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting/Test_Ali/Test_Ali_far/audio_dir/" # offical audio directory
   dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1_resnet34-lm_spectral_cluster
   mkdir -p $dest_dir
   split="Test"
   [ -d "$dest_dir/Test_Ali/Test_Ali_far/target_audio" ] && rm -r $dest_dir/Test_Ali/Test_Ali_far/target_audio
   [ -f "$dest_dir/Test_Ali/Test_Ali_far/Test.json" ]  && rm -r $dest_dir/Test_Ali/Test_Ali_far/Test.json
   echo "Make system target speaker and its label and store it under $dest_dir/Test_Ali/Test_Ali_far/"
   echo "..."
   ## this script output $dest_dir/Test_Ali/Test_Ali_far/target_audio and $target_dir/Test_Ali/Test_Ali_far/$split.json
   python3 ts_vad2/system_rttm_to_generate_target_speaker_wav_and_label_for_ts_vad.py\
       --system_rttm $system_rttm\
       --audio_dir $audio_dir\
       --dest_dir $dest_dir\
       --type $split
   echo "Finish , store it under $dest_dir/Test_Ali/Test_Ali_far/"
 fi

 ## prepared target speaker embedding
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
   echo "prepare eval set target audio list"
   for name in Eval Test;do
   input_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1_resnet34-lm_spectral_cluster/${name}_Ali/${name}_Ali_far/target_audio
   file=$input_dir/wavs.txt
    python3 ts_vad2/prepare_alimeeting_target_audio_list.py \
        $input_dir $file
   done
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   echo "generate eval set speaker embedding"
   for name in Eval Test;do
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1_resnet34-lm_spectral_cluster
    feature_name=cam++_en_zh_advanced_feature_dir
    model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
    # 提取embedding
    input_dir=$exp_dir/${name}_Ali/${name}_Ali_far/target_audio
    file=$input_dir/wavs.txt
    wav_path=$file
    save_dir=$exp_dir/SpeakerEmbedding/${name}/$feature_name
    python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
        --model_id $model_id --wavs $wav_path\
        --save_dir $save_dir
   echo "Finish extract target speaker embedding!!!!"
   done
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_epoch40_front_fix_seed
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Train"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1_resnet34-lm_spectral_cluster/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

 # # spectral cluster target audio , mix audio and labels path
 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1_resnet34-lm_spectral_cluster"
 for name in $infer_sets;do
    results_path=$exp_dir/sys_resnet34-lm_${name}
  python3 ts_vad2/infer2.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --select-encoder-layer-nums 6\
    --data-dir $data_dir
done
# Eval set
# Model DER:  0.324623862690458
#Model ACC:  0.9154933495316746
#100%|██████████| 24/24 [00:21<00:00,  1.13it/s]
#Eval for threshold 0.20: DER 10.41%, MS 1.07%, FA 7.33%, SC 2.01%
#
#Eval for threshold 0.30: DER 7.86%, MS 1.56%, FA 4.22%, SC 2.08%
#
#Eval for threshold 0.35: DER 7.15%, MS 1.83%, FA 3.21%, SC 2.11%
#
#Eval for threshold 0.40: DER 6.74%, MS 2.14%, FA 2.51%, SC 2.09%
#
#Eval for threshold 0.45: DER 6.54%, MS 2.49%, FA 1.94%, SC 2.10%
#
#Eval for threshold 0.50: DER 6.56%, MS 3.01%, FA 1.51%, SC 2.04%
#
#Eval for threshold 0.55: DER 6.81%, MS 3.68%, FA 1.21%, SC 1.91%
#
#Eval for threshold 0.60: DER 7.10%, MS 4.41%, FA 0.98%, SC 1.71%
#
#Eval for threshold 0.70: DER 8.22%, MS 6.22%, FA 0.63%, SC 1.36%
#
#Eval for threshold 0.80: DER 10.33%, MS 8.89%, FA 0.43%, SC 1.00%

# Test set
#Model DER:  0.39062356288665495
#Model ACC:  0.9107878471065629
#100%|██████████| 59/59 [00:53<00:00,  1.11it/s]
#Eval for threshold 0.20: DER 8.27%, MS 1.30%, FA 6.27%, SC 0.70%
#
#Eval for threshold 0.30: DER 6.20%, MS 1.89%, FA 3.53%, SC 0.78%
#
#Eval for threshold 0.35: DER 5.75%, MS 2.21%, FA 2.73%, SC 0.82%
#
#Eval for threshold 0.40: DER 5.47%, MS 2.56%, FA 2.06%, SC 0.85%
#
#Eval for threshold 0.45: DER 5.39%, MS 2.98%, FA 1.56%, SC 0.85%
#
#Eval for threshold 0.50: DER 5.45%, MS 3.44%, FA 1.19%, SC 0.82%
#
#Eval for threshold 0.55: DER 5.70%, MS 4.03%, FA 0.90%, SC 0.78%
#
#Eval for threshold 0.60: DER 6.11%, MS 4.72%, FA 0.69%, SC 0.70%
#
#Eval for threshold 0.70: DER 7.23%, MS 6.33%, FA 0.36%, SC 0.53%
#
#Eval for threshold 0.80: DER 9.30%, MS 8.74%, FA 0.19%, SC 0.37%
fi
