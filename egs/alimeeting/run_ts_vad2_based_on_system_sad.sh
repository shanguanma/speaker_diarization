#!/usr/bin/env bash

stage=0
stop_stage=1000
split="Eval" # Eval , Train
. utils/parse_options.sh
. path_for_speaker_diarization.sh

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   # Here, I assume have system rttm from cluster method.
   # now, I will get target speaker for tsvad train based on system rttm.

   system_rttm="/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/exp/spectral_cluster/alimeeting_eval_system_sad_rttm_cam++_advanced"
   audio_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/audio_dir/" # offical audio directory
   dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1/
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

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   # Here, I assume have system rttm from cluster method.
   # now, I will get target speaker for tsvad train based on system rttm.

   system_rttm="/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/exp/spectral_cluster/alimeeting_train_system_sad_rttm_cam++_advanced"
   audio_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting/Train_Ali_far/audio_dir/" # offical audio directory
   dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1/
   mkdir -p $dest_dir
   split="Train"
   [ -d "$dest_dir/Train_Ali_far/target_audio" ] && rm -r $dest_dir/Train_Ali_far/target_audio
   [ -f "$dest_dir/Train_Ali_far/Eval.json" ]  && rm -r $dest_dir/Train_Ali_far/Eval.json
   echo "Make system target speaker and its label and store it under $dest_dir/Train_Ali_far/"
   echo "..."

   ## this script output ${dest_dir}/Train_Ali_far/target_dir and ${dest_dir}/Train_Ali_far/$split.json
   python3 ts_vad2/system_rttm_to_generate_target_speaker_wav_and_label_for_ts_vad.py\
       --system_rttm $system_rttm\
       --audio_dir $audio_dir\
       --dest_dir $dest_dir\
       --type $split
   echo "Finish , store it under $dest_dir/Train_Ali_far/"
fi

## prepared target speaker embedding
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "prepare eval set target audio list"
   #input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Train_Ali_far/target_audio/
   input_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1/Eval_Ali/Eval_Ali_far/target_audio
   file=$input_dir/wavs.txt
    python3 ts_vad2/prepare_alimeeting_target_audio_list.py \
        $input_dir $file

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "generate eval set speaker embedding"
   exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1/
   feature_name=cam++_en_zh_advanced_feature_dir
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   # 提取embedding
   input_dir=$exp_dir/Eval_Ali/Eval_Ali_far/target_audio
   file=$input_dir/wavs.txt
   wav_path=$file
   save_dir=$exp_dir/SpeakerEmbedding/Eval/$feature_name
   python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
       --model_id $model_id --wavs $wav_path\
       --save_dir $save_dir
   echo "Finish extract target speaker embedding!!!!"
fi



## prepared target speaker embedding
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "prepare train set target audio list"
   #input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Train_Ali_far/target_audio/
   input_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1/Train_Ali_far/target_audio
   file=$input_dir/wavs.txt
    python3 ts_vad2/prepare_alimeeting_target_audio_list.py \
        $input_dir $file

fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "generate train set speaker embedding"
   exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1/
   feature_name=cam++_en_zh_advanced_feature_dir
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   # 提取embedding
   input_dir=$exp_dir/Train_Ali_far/target_audio
   file=$input_dir/wavs.txt
   wav_path=$file
   save_dir=$exp_dir/SpeakerEmbedding/Train/$feature_name
   python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
       --model_id $model_id --wavs $wav_path\
       --save_dir $save_dir
   echo "Finish extract target speaker embedding!!!!"
fi




if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
    # it adds noise and rirs to train tsvad model , no grad-clip and no freeze update.
    # # speech encoder is cam++ , oracle target speaker embedding is from cam++ pretrain model
# this cam++ pretrain model is trained on cn-cnceleb and voxceleb dataset, checkpoint is from https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/files
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1

    ## for data augmentation of mixture audio
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES

    ## this exp directory
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/cam++_advanced_two_gpus_unfreeze_with_musan_rirs_from_spectral_cluster_iter1

    input_data_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1
    ## for tsvad model arch
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    ## for target speaker embedding data
    spk_path=$input_data_dir/SpeakerEmbedding
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
    ## for mixture audio and its labels
    data_dir=$input_data_dir

    # To save storage space
    keep_last_k=2
    keep_last_epoch=2
   CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12683 ts_vad2/train_accelerate_ddp2_with_logger.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --freeze-updates 0\
    --grad-clip false\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --data-dir $data_dir\
    --keep-last-k $keep_last_k\
    --keep-last-epoch $keep_last_epoch \
    --exp-dir $exp_dir
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/cam++_advanced_two_gpus_unfreeze_with_musan_rirs_from_spectral_cluster_iter1
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Train Eval Test "
 #infer_sets="Train"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 #results_path=$exp_dir/
  # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
 for name in $infer_sets;do
    #results_path=$exp_dir/$name
    results_path=$exp_dir
    #mkdir -p $results_path

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
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --split $name
done
# cat logs/run_ts_vad2_based_on_system_sad_stage7.log
# Model DER:  0.35665826529608513
#Model ACC:  0.8651375842501714
#frame_len: 0.04!!
#100%|██████████| 767/767 [12:24<00:00,  1.03it/s]
#Eval for threshold 0.20: DER 41.96%, MS 19.60%, FA 13.76%, SC 8.60%
#
#Eval for threshold 0.30: DER 41.36%, MS 23.30%, FA 9.71%, SC 8.35%
#
#Eval for threshold 0.35: DER 41.88%, MS 24.85%, FA 8.73%, SC 8.30%
#
#Eval for threshold 0.40: DER 42.58%, MS 26.31%, FA 8.03%, SC 8.24%
#
#Eval for threshold 0.45: DER 43.41%, MS 27.77%, FA 7.47%, SC 8.17%
#
#Eval for threshold 0.50: DER 44.39%, MS 29.31%, FA 7.01%, SC 8.06%
#
#Eval for threshold 0.55: DER 45.60%, MS 31.03%, FA 6.65%, SC 7.92%
#
#Eval for threshold 0.60: DER 47.02%, MS 32.92%, FA 6.34%, SC 7.76%
#
#Eval for threshold 0.70: DER 50.70%, MS 37.59%, FA 5.66%, SC 7.45%
#
#Eval for threshold 0.80: DER 57.15%, MS 45.25%, FA 4.86%, SC 7.04%

# Eval set
# Model DER:  0.26572581209673773
#Model ACC:  0.9111179424987488
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 12.64%, MS 9.50%, FA 2.73%, SC 0.41%
#
#Eval for threshold 0.30: DER 13.19%, MS 11.21%, FA 1.64%, SC 0.34%
#
#Eval for threshold 0.35: DER 13.54%, MS 11.93%, FA 1.29%, SC 0.31%
#
#Eval for threshold 0.40: DER 14.01%, MS 12.64%, FA 1.06%, SC 0.31%
#
#Eval for threshold 0.45: DER 14.49%, MS 13.34%, FA 0.86%, SC 0.29%
#
#Eval for threshold 0.50: DER 15.10%, MS 14.12%, FA 0.70%, SC 0.28%
#
#Eval for threshold 0.55: DER 15.78%, MS 14.93%, FA 0.64%, SC 0.22%
#
#Eval for threshold 0.60: DER 16.59%, MS 15.83%, FA 0.60%, SC 0.16%
#
#Eval for threshold 0.70: DER 18.35%, MS 17.76%, FA 0.51%, SC 0.08%
#
#Eval for threshold 0.80: DER 20.82%, MS 20.35%, FA 0.44%, SC 0.04%

# Test set
#Model DER:  0.2522657098243328
#Model ACC:  0.9096486364994082
#frame_len: 0.04!!
#100%|██████████| 60/60 [00:58<00:00,  1.02it/s]
#Eval for threshold 0.20: DER 13.73%, MS 9.84%, FA 3.20%, SC 0.69%
#
#Eval for threshold 0.30: DER 13.80%, MS 11.36%, FA 1.79%, SC 0.65%
#
#Eval for threshold 0.35: DER 14.00%, MS 12.07%, FA 1.27%, SC 0.66%
#
#Eval for threshold 0.40: DER 14.29%, MS 12.72%, FA 0.91%, SC 0.67%
#
#Eval for threshold 0.45: DER 14.60%, MS 13.33%, FA 0.57%, SC 0.69%
#
#Eval for threshold 0.50: DER 15.00%, MS 13.99%, FA 0.31%, SC 0.70%
#
#Eval for threshold 0.55: DER 15.61%, MS 14.80%, FA 0.23%, SC 0.58%
#
#Eval for threshold 0.60: DER 16.32%, MS 15.68%, FA 0.19%, SC 0.45%
#
#Eval for threshold 0.70: DER 18.07%, MS 17.69%, FA 0.14%, SC 0.24%
#
#Eval for threshold 0.80: DER 20.60%, MS 20.41%, FA 0.09%, SC 0.10%

fi

## Now I will use oracle Eval set and system Train set to train tsvad model
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
    # it adds noise and rirs to train tsvad model , no grad-clip and no freeze update.
    # # speech encoder is cam++ , oracle target speaker embedding is from cam++ pretrain model
# this cam++ pretrain model is trained on cn-cnceleb and voxceleb dataset, checkpoint is from https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/files
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1

    ## for data augmentation of mixture audio
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES

    ## this exp directory
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/cam++_advanced_two_gpus_unfreeze_with_musan_rirs_from_spectral_cluster_iter1_1

    input_data_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter1_1
    ## for tsvad model arch
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    ## for target speaker embedding data
    spk_path=$input_data_dir/SpeakerEmbedding
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
    ## for mixture audio and its labels
    data_dir=$input_data_dir

    # To save storage space
    keep_last_k=2
    keep_last_epoch=2
   CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12983 ts_vad2/train_accelerate_ddp2_with_logger.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --freeze-updates 0\
    --grad-clip false\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --data-dir $data_dir\
    --keep-last-k $keep_last_k\
    --keep-last-epoch $keep_last_epoch \
    --exp-dir $exp_dir
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/cam++_advanced_two_gpus_unfreeze_with_musan_rirs_from_spectral_cluster_iter1_1
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Train Eval Test "
 #infer_sets="Train"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
  # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
 for name in $infer_sets;do
    results_path=$exp_dir


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
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --split $name
done
#  cat logs/run_ts_vad2_based_on_system_sad_stage9.log
# Train set
# Model DER:  0.35945024292637867
#Model ACC:  0.8640538965994949
#100%|██████████| 767/767 [12:09<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 41.69%, MS 19.32%, FA 14.12%, SC 8.25%
#
#Eval for threshold 0.30: DER 41.02%, MS 23.10%, FA 9.90%, SC 8.02%
#
#Eval for threshold 0.35: DER 41.53%, MS 24.70%, FA 8.88%, SC 7.95%
#
#Eval for threshold 0.40: DER 42.24%, MS 26.19%, FA 8.15%, SC 7.90%
#
#Eval for threshold 0.45: DER 43.10%, MS 27.69%, FA 7.58%, SC 7.83%
#
#Eval for threshold 0.50: DER 44.13%, MS 29.29%, FA 7.11%, SC 7.73%
#
#Eval for threshold 0.55: DER 45.39%, MS 31.06%, FA 6.75%, SC 7.58%
#
#Eval for threshold 0.60: DER 46.87%, MS 33.01%, FA 6.42%, SC 7.44%
#
#Eval for threshold 0.70: DER 50.79%, MS 37.89%, FA 5.74%, SC 7.16%
#
#Eval for threshold 0.80: DER 57.45%, MS 45.72%, FA 4.92%, SC 6.81%

# Eval set
# Model DER:  0.26362074816169057
#Model ACC:  0.9115590414559556
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 12.45%, MS 9.18%, FA 2.82%, SC 0.45%
#
#Eval for threshold 0.30: DER 12.97%, MS 10.84%, FA 1.73%, SC 0.39%
#
#Eval for threshold 0.35: DER 13.49%, MS 11.69%, FA 1.42%, SC 0.38%
#
#Eval for threshold 0.40: DER 13.97%, MS 12.43%, FA 1.17%, SC 0.37%
#
#Eval for threshold 0.45: DER 14.51%, MS 13.19%, FA 0.97%, SC 0.35%
#
#Eval for threshold 0.50: DER 15.08%, MS 13.93%, FA 0.79%, SC 0.35%
#
#Eval for threshold 0.55: DER 15.76%, MS 14.76%, FA 0.70%, SC 0.30%
#
#Eval for threshold 0.60: DER 16.48%, MS 15.63%, FA 0.64%, SC 0.20%
#
#Eval for threshold 0.70: DER 18.31%, MS 17.68%, FA 0.56%, SC 0.08%
#
#Eval for threshold 0.80: DER 20.73%, MS 20.24%, FA 0.45%, SC 0.03%

# Test set
#Model DER:  0.2524398668681938
#Model ACC:  0.9097430670554212
#100%|██████████| 60/60 [00:58<00:00,  1.02it/s]
#Eval for threshold 0.20: DER 14.21%, MS 9.51%, FA 3.96%, SC 0.74%
#
#Eval for threshold 0.30: DER 13.82%, MS 11.11%, FA 2.06%, SC 0.65%
#
#Eval for threshold 0.35: DER 13.94%, MS 11.83%, FA 1.47%, SC 0.64%
#
#Eval for threshold 0.40: DER 14.20%, MS 12.54%, FA 1.02%, SC 0.65%
#
#Eval for threshold 0.45: DER 14.55%, MS 13.23%, FA 0.65%, SC 0.67%
#
#Eval for threshold 0.50: DER 14.99%, MS 13.95%, FA 0.37%, SC 0.67%
#
#Eval for threshold 0.55: DER 15.63%, MS 14.81%, FA 0.26%, SC 0.56%
#
#Eval for threshold 0.60: DER 16.43%, MS 15.79%, FA 0.23%, SC 0.42%
#
#Eval for threshold 0.70: DER 18.38%, MS 18.01%, FA 0.16%, SC 0.21%
#
#Eval for threshold 0.80: DER 21.20%, MS 21.01%, FA 0.11%, SC 0.08%
fi


# I will use tsvad iter1 output system rttm to do iter2 of tsvad.
if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ];then
   # Here, I assume have system rttm from tsvad iter1
   # now, I will get target speaker for tsvad train based on system rttm.

   system_rttm="/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/cam++_advanced_two_gpus_unfreeze_with_musan_rirs_from_spectral_cluster_iter1/Eval/Eval/res_rttm_0.2"
   audio_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/audio_dir/" # offical audio directory
   dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter2/
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

if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ];then
   # Here, I assume have system rttm from tsvad iter1
   # now, I will get target speaker for tsvad train based on system rttm.

   system_rttm="/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/cam++_advanced_two_gpus_unfreeze_with_musan_rirs_from_spectral_cluster_iter1/Train/Train/res_rttm_0.2"
   audio_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/audio_dir/" # offical audio directory
   dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter2/
   mkdir -p $dest_dir
   split="Train"
   [ -d "$dest_dir/Train_Ali_far/target_audio" ] && rm -r $dest_dir/Train_Ali_far/target_audio
   [ -f "$dest_dir/Train_Ali_far/Eval.json" ]  && rm -r $dest_dir/Train_Ali_far/Train.json
   echo "Make system target speaker and its label and store it under $dest_dir/Train_Ali_far/"
   echo "..."
   ## this script output $dest_dir/Eval_Ali/Eval_Ali_far/target_audio and $target_dir/Eval_Ali/Eval_Ali_far/$split.json
   python3 ts_vad2/system_rttm_to_generate_target_speaker_wav_and_label_for_ts_vad.py\
       --system_rttm $system_rttm\
       --audio_dir $audio_dir\
       --dest_dir $dest_dir\
       --type $split
   echo "Finish , store it under $dest_dir/Train_Ali_far/"
fi

## prepared target speaker embedding
if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
   dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter2
   if [ $split == "Eval" ];then
       echo "prepared $split set target audio list"
       input_dir=$dest_dir/${split}_Ali/${split}_Ali_far/target_audio
       file=$input_dir/wavs.txt
   elif [ $split == "Train" ];then
       echo "prepared $split set target audio list"
       input_dir=$dest_dir/${split}_Ali_far/target_audio
       file=$input_dir/wavs.txt
   fi
    python3 ts_vad2/prepare_alimeeting_target_audio_list.py \
        $input_dir $file

fi

if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
   exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter2
   feature_name=cam++_en_zh_advanced_feature_dir
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   # 提取embedding
   if [ $split == "Eval" ];then
      echo "extract ${split} set target speaker embedding "
      input_dir=$exp_dir/${split}_Ali/${split}_Ali_far/target_audio
      wav_path=$input_dir/wavs.txt
      save_dir=$exp_dir/SpeakerEmbedding/${split}/$feature_name
   elif [ $split == "Train" ];then
      echo "extract ${split} set target speaker embedding "
      input_dir=$exp_dir/${split}_Ali_far/target_audio
      wav_path=$input_dir/wavs.txt
      save_dir=$exp_dir/SpeakerEmbedding/${split}/$feature_name
   fi
   python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
       --model_id $model_id --wavs $wav_path\
       --save_dir $save_dir
   echo "Finish extract target speaker embedding!!!!"
fi

if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ];then
    # it adds noise and rirs to train tsvad model , no grad-clip and no freeze update.
    # # speech encoder is cam++ , oracle target speaker embedding is from cam++ pretrain model
    # this cam++ pretrain model is trained on cn-cnceleb and voxceleb dataset,
    # checkpoint is from https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/files
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    ## for data augmentation of mixture audio
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    ## this exp directory
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/cam++_advanced_two_gpus_unfreeze_with_musan_rirs_from_spectral_cluster_iter2
    input_data_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_based_on_system_sad/exp/iter2
    ## for tsvad model arch
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    ## for target speaker embedding data
    spk_path=$input_data_dir/SpeakerEmbedding
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
    ## for mixture audio and its labels
    data_dir=$input_data_dir
   CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12683 ts_vad2/train_accelerate_ddp2_with_logger.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --freeze-updates 0\
    --grad-clip false\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --data-dir $data_dir\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/cam++_advanced_two_gpus_unfreeze_with_musan_rirs_from_spectral_cluster_iter2
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Train Eval Test "
 #infer_sets="Train"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 results_path=$exp_dir/
  # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
 for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path

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
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --split $name
done
fi
