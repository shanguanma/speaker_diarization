#!/usr/bin/env bash

stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_speaker_diarization_hltsz.sh

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "prepared magicdata-ramc kaldi format data"
   ## it has been removed G00000000 utt in rttm file 
   # based on the paper "The X-Lance Speaker Diarization System for the Conversational Short-phrase Speaker Diarization Challenge 2022"
   python3  /data/maduo/datasets/MagicData-RAMC/maduo_processed/prepare_magicdata_180h.py 
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
  echo "get target audio and json label file from rttm file"
  datasets="test train"
  #datasets="dev"
  source_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
  for name in $datasets;do
    oracle_rttm=$source_dir/$name/rttm_debug
    wavscp=$source_dir/$name/wav.scp
    dest_dir=/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc
    python3 ts_vad2/oracle_rttm_to_generate_target_speaker_wav_and_label_for_ts_vad.py\
	     --oracle_rttm $oracle_rttm\
	     --wavscp $wavscp\
	     --dest_dir $dest_dir\
	     --type $name
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "generate oracle vad speaker embedding"
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
     input_dir=/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc/${name}/target_audio/
     wav_path=/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc/${name}/wavs.txt
     find $input_dir -name "*.wav" | grep -v "all.wav" >$wav_path
     head $wav_path
     save_dir=$dest_dir/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding/$name/$feature_name
     python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size 96
   done
fi

#
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    dataset_name="magicdata-ramc" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12815 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar=0.0
 
 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 
 data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${collar}
  python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/rttm_debug\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name
done

# grep -r 'Eval' logs/run_ts_vad2_hltsz_stage4_collar0.25.log
# dev set
#Eval for threshold 0.20: DER 10.35%, MS 0.05%, FA 8.09%, SC 2.21%
#Eval for threshold 0.30: DER 8.23%, MS 0.09%, FA 5.34%, SC 2.80%
#Eval for threshold 0.35: DER 7.48%, MS 0.15%, FA 4.27%, SC 3.07%
#Eval for threshold 0.40: DER 6.83%, MS 0.22%, FA 3.33%, SC 3.29%
#Eval for threshold 0.45: DER 6.31%, MS 0.30%, FA 2.47%, SC 3.54%
#Eval for threshold 0.50: DER 5.87%, MS 0.42%, FA 1.63%, SC 3.82%
#Eval for threshold 0.55: DER 5.96%, MS 0.98%, FA 1.27%, SC 3.71%
#Eval for threshold 0.60: DER 6.45%, MS 1.89%, FA 1.19%, SC 3.37%
#Eval for threshold 0.70: DER 7.88%, MS 4.11%, FA 1.05%, SC 2.72%
#Eval for threshold 0.80: DER 10.06%, MS 7.05%, FA 0.91%, SC 2.10%
# test set
#Eval for threshold 0.20: DER 10.22%, MS 0.03%, FA 8.92%, SC 1.26%
#Eval for threshold 0.30: DER 8.48%, MS 0.12%, FA 6.82%, SC 1.54%
#Eval for threshold 0.35: DER 7.90%, MS 0.20%, FA 6.01%, SC 1.69%
#Eval for threshold 0.40: DER 7.29%, MS 0.35%, FA 4.91%, SC 2.04%
#Eval for threshold 0.45: DER 6.52%, MS 0.52%, FA 3.36%, SC 2.64%
#Eval for threshold 0.50: DER 6.04%, MS 0.90%, FA 2.33%, SC 2.82%
#Eval for threshold 0.55: DER 6.73%, MS 2.34%, FA 2.13%, SC 2.25%
#Eval for threshold 0.60: DER 7.35%, MS 3.55%, FA 2.01%, SC 1.79%
#Eval for threshold 0.70: DER 8.35%, MS 5.28%, FA 1.76%, SC 1.32%
#Eval for threshold 0.80: DER 10.13%, MS 7.56%, FA 1.51%, SC 1.05%

# grep -r 'Eval' logs/run_ts_vad2_hltsz_stage5_collar0.0_again.log
# Eval set
#Eval for threshold 0.20: DER 18.70%, MS 0.21%, FA 15.95%, SC 2.54%
#Eval for threshold 0.30: DER 15.71%, MS 0.39%, FA 12.09%, SC 3.23%
#Eval for threshold 0.35: DER 14.54%, MS 0.54%, FA 10.48%, SC 3.51%
#Eval for threshold 0.40: DER 13.62%, MS 0.75%, FA 9.11%, SC 3.75%
#Eval for threshold 0.45: DER 12.88%, MS 1.01%, FA 7.87%, SC 4.00%
#Eval for threshold 0.50: DER 12.26%, MS 1.36%, FA 6.69%, SC 4.22%
#Eval for threshold 0.55: DER 12.23%, MS 2.18%, FA 5.99%, SC 4.06%
#Eval for threshold 0.60: DER 12.57%, MS 3.35%, FA 5.56%, SC 3.66%
#Eval for threshold 0.70: DER 13.85%, MS 6.27%, FA 4.67%, SC 2.91%
#Eval for threshold 0.80: DER 16.07%, MS 10.13%, FA 3.73%, SC 2.21%
# Test set
#Eval for threshold 0.20: DER 18.82%, MS 0.22%, FA 16.92%, SC 1.69%
#Eval for threshold 0.30: DER 15.98%, MS 0.49%, FA 13.38%, SC 2.11%
#Eval for threshold 0.35: DER 15.02%, MS 0.71%, FA 12.00%, SC 2.31%
#Eval for threshold 0.40: DER 14.07%, MS 1.03%, FA 10.36%, SC 2.68%
#Eval for threshold 0.45: DER 12.95%, MS 1.44%, FA 8.23%, SC 3.28%
#Eval for threshold 0.50: DER 12.22%, MS 2.14%, FA 6.65%, SC 3.42%
#Eval for threshold 0.55: DER 12.76%, MS 3.95%, FA 6.02%, SC 2.79%
#Eval for threshold 0.60: DER 13.29%, MS 5.51%, FA 5.53%, SC 2.25%
#Eval for threshold 0.70: DER 14.34%, MS 8.08%, FA 4.62%, SC 1.64%
#Eval for threshold 0.80: DER 16.42%, MS 11.48%, FA 3.70%, SC 1.24%
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8"
   predict_rttm_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4/magicdata-ramc_collar0.0
   oracle_rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug
    done
   done
# logs/run_ts_vad2_hltsz_stage5_cder_again.log
# dev set
# Avg CDER : 0.422 on threshold=0.5
# Avg CDER : 0.237 on  threshold=0.6
# Avg CDER : 0.156 on threshold=0.7
# Avg CDER : 0.130 on threshold=0.8 
# test set
# Avg CDER : 0.125 on threshold=0.6 
# Avg CDER : 0.157 on threshold=0.5

fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25

 dataset_name="alimeeting" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir/$dataset_name/
  python3 ts_vad2/infer.py \
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
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name
done
#cat logs/run_ts_vad2_hltsz_stage6.log
#Eval set
#Model DER:  0.30812769135726464
#Model ACC:  0.89278416906717
#100%|██████████| 25/25 [00:26<00:00,  1.04s/it]
#Eval for threshold 0.20: DER 18.46%, MS 13.51%, FA 3.72%, SC 1.23%
#
#Eval for threshold 0.30: DER 18.67%, MS 15.76%, FA 1.77%, SC 1.13%
#
#Eval for threshold 0.35: DER 19.15%, MS 16.84%, FA 1.23%, SC 1.07%
#
#Eval for threshold 0.40: DER 19.68%, MS 17.76%, FA 0.90%, SC 1.03%
#
#Eval for threshold 0.45: DER 20.29%, MS 18.70%, FA 0.59%, SC 1.00%
#
#Eval for threshold 0.50: DER 21.00%, MS 19.70%, FA 0.38%, SC 0.92%
#
#Eval for threshold 0.55: DER 22.00%, MS 20.93%, FA 0.36%, SC 0.71%
#
#Eval for threshold 0.60: DER 22.99%, MS 22.11%, FA 0.35%, SC 0.54%
#
#Eval for threshold 0.70: DER 25.39%, MS 24.76%, FA 0.32%, SC 0.31%
#
#Eval for threshold 0.80: DER 29.07%, MS 28.63%, FA 0.30%, SC 0.13%
#Test set
#Model DER:  0.3230883997371865
#Model ACC:  0.8758940689100059
#100%|██████████| 60/60 [01:07<00:00,  1.12s/it]
#Eval for threshold 0.20: DER 20.75%, MS 12.44%, FA 4.36%, SC 3.95%
#
#Eval for threshold 0.30: DER 21.00%, MS 14.76%, FA 2.26%, SC 3.98%
#
#Eval for threshold 0.35: DER 21.34%, MS 15.76%, FA 1.55%, SC 4.04%
#
#Eval for threshold 0.40: DER 21.76%, MS 16.74%, FA 0.92%, SC 4.11%
#
#Eval for threshold 0.45: DER 22.29%, MS 17.70%, FA 0.51%, SC 4.09%
#
#Eval for threshold 0.50: DER 22.95%, MS 18.77%, FA 0.16%, SC 4.02%
#
#Eval for threshold 0.55: DER 23.87%, MS 20.03%, FA 0.06%, SC 3.77%
#
#Eval for threshold 0.60: DER 24.97%, MS 21.42%, FA 0.06%, SC 3.50%
#
#Eval for threshold 0.70: DER 27.50%, MS 24.52%, FA 0.05%, SC 2.94%
#
#Eval for threshold 0.80: DER 30.97%, MS 28.45%, FA 0.03%, SC 2.48%
fi


# compared with stage3-4, stage10-11 add alimeeting data into train and dev set.
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    dataset_name="magicdata-ramc" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/data/maduo/exp/speaker_diarization/ts_vad2/data/alimeeting_and_magicdata-ramc/ts_vad/spk_embed/alimeeting_and_magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/alimeeting_and_magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/alimeeting_and_magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12815 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/alimeeting_and_magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 #collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${collar}
  python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/rttm_debug\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name
done

# grep -r 'Eval' logs/run_ts_vad2_hltsz_stage10-11.log
# collar=0.0
# dev set of magicdata-ramc
# Eval for threshold 0.20: DER 17.74%, MS 0.26%, FA 15.07%, SC 2.42%
#Eval for threshold 0.30: DER 14.15%, MS 0.45%, FA 10.44%, SC 3.26%
#Eval for threshold 0.35: DER 13.10%, MS 0.63%, FA 8.92%, SC 3.55%
#Eval for threshold 0.40: DER 12.40%, MS 0.88%, FA 7.75%, SC 3.76%
#Eval for threshold 0.45: DER 11.93%, MS 1.23%, FA 6.84%, SC 3.86%
#Eval for threshold 0.50: DER 11.65%, MS 1.64%, FA 6.16%, SC 3.85%
#Eval for threshold 0.55: DER 11.59%, MS 2.23%, FA 5.64%, SC 3.71%
#Eval for threshold 0.60: DER 11.73%, MS 3.02%, FA 5.19%, SC 3.53%
#Eval for threshold 0.70: DER 12.59%, MS 5.25%, FA 4.29%, SC 3.06%
#Eval for threshold 0.80: DER 15.12%, MS 9.50%, FA 3.40%, SC 2.21%
## test set of magicdata-ramc
#Eval for threshold 0.20: DER 18.15%, MS 0.24%, FA 16.19%, SC 1.72%
#Eval for threshold 0.30: DER 15.56%, MS 0.53%, FA 12.89%, SC 2.14%
#Eval for threshold 0.35: DER 14.76%, MS 0.80%, FA 11.71%, SC 2.24%
#Eval for threshold 0.40: DER 14.19%, MS 1.17%, FA 10.69%, SC 2.33%
#Eval for threshold 0.45: DER 13.61%, MS 1.66%, FA 9.50%, SC 2.45%
#Eval for threshold 0.50: DER 12.38%, MS 2.68%, FA 6.78%, SC 2.92%
#Eval for threshold 0.55: DER 13.15%, MS 5.06%, FA 5.86%, SC 2.22%
#Eval for threshold 0.60: DER 13.42%, MS 6.06%, FA 5.39%, SC 1.97%
#Eval for threshold 0.70: DER 14.24%, MS 8.06%, FA 4.51%, SC 1.66%
#Eval for threshold 0.80: DER 16.04%, MS 11.10%, FA 3.62%, SC 1.31%


#grep -r 'Eval' logs/run_ts_vad2_hltsz_stage11_collar0.25.log
# Eval set of magicdata-ramc
#Eval for threshold 0.20: DER 9.48%, MS 0.04%, FA 7.35%, SC 2.09%
#Eval for threshold 0.30: DER 6.84%, MS 0.07%, FA 3.92%, SC 2.85%
#Eval for threshold 0.35: DER 6.12%, MS 0.13%, FA 2.88%, SC 3.11%
#Eval for threshold 0.40: DER 5.69%, MS 0.23%, FA 2.16%, SC 3.31%
#Eval for threshold 0.45: DER 5.46%, MS 0.39%, FA 1.64%, SC 3.43%
#Eval for threshold 0.50: DER 5.38%, MS 0.56%, FA 1.36%, SC 3.46%
#Eval for threshold 0.55: DER 5.46%, MS 0.84%, FA 1.23%, SC 3.39%
#Eval for threshold 0.60: DER 5.68%, MS 1.27%, FA 1.15%, SC 3.27%
#Eval for threshold 0.70: DER 6.58%, MS 2.65%, FA 1.01%, SC 2.92%
#Eval for threshold 0.80: DER 9.08%, MS 6.03%, FA 0.90%, SC 2.15%
#Eval for threshold 0.20: DER 9.68%, MS 0.03%, FA 8.36%, SC 1.29%

# test set of magicdata-ramc
#Eval for threshold 0.30: DER 8.15%, MS 0.12%, FA 6.48%, SC 1.56%
#Eval for threshold 0.35: DER 7.75%, MS 0.25%, FA 5.89%, SC 1.61%
#Eval for threshold 0.40: DER 7.49%, MS 0.42%, FA 5.40%, SC 1.67%
#Eval for threshold 0.45: DER 7.23%, MS 0.65%, FA 4.78%, SC 1.79%
#Eval for threshold 0.50: DER 6.26%, MS 1.30%, FA 2.64%, SC 2.32%
#Eval for threshold 0.55: DER 7.23%, MS 3.43%, FA 2.13%, SC 1.68%
#Eval for threshold 0.60: DER 7.55%, MS 4.06%, FA 2.00%, SC 1.49%
#Eval for threshold 0.70: DER 8.25%, MS 5.15%, FA 1.77%, SC 1.32%
#Eval for threshold 0.80: DER 9.70%, MS 7.06%, FA 1.53%, SC 1.12%
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8"
   predict_rttm_dir=/data/maduo/exp/speaker_diarization/ts_vad2/alimeeting_and_magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4/magicdata-ramc_collar0.0
   oracle_rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug
    done
   done
   
#  cat logs/run_ts_vad2_hltsz_stage12_cder.log
# dev set
# Avg CDER : 0.186 on threshold=0.6    
# Avg CDER : 0.143 on threshold=0.7   
# Avg CDER : 0.115 on threshold=0.8    

# test set   
# Avg CDER : 0.133  on threshold=0.6
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25

 dataset_name="alimeeting" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${collar}
  python3 ts_vad2/infer.py \
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
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name
done
#  cat logs/run_ts_vad2_hltsz_stage13.log
# Eval set
## Model DER:  0.30794058000259933
#Model ACC:  0.8928656015263309
#100%|██████████| 25/25 [00:26<00:00,  1.05s/it]
#Eval for threshold 0.20: DER 18.45%, MS 13.53%, FA 3.71%, SC 1.22%
#
#Eval for threshold 0.30: DER 18.68%, MS 15.76%, FA 1.79%, SC 1.14%
#
#Eval for threshold 0.35: DER 19.12%, MS 16.80%, FA 1.25%, SC 1.07%
#
#Eval for threshold 0.40: DER 19.68%, MS 17.73%, FA 0.92%, SC 1.03%
#
#Eval for threshold 0.45: DER 20.28%, MS 18.71%, FA 0.58%, SC 0.99%
#
#Eval for threshold 0.50: DER 20.99%, MS 19.71%, FA 0.38%, SC 0.90%
#
#Eval for threshold 0.55: DER 22.02%, MS 20.96%, FA 0.36%, SC 0.71%
#
#Eval for threshold 0.60: DER 23.04%, MS 22.17%, FA 0.35%, SC 0.53%
#
#Eval for threshold 0.70: DER 25.37%, MS 24.75%, FA 0.32%, SC 0.30%
#
#Eval for threshold 0.80: DER 29.05%, MS 28.62%, FA 0.30%, SC 0.13%
#
# Test set
#Model DER:  0.32313878303933546
#Model ACC:  0.8758894112054267
#100%|██████████| 60/60 [01:03<00:00,  1.07s/it]
#Eval for threshold 0.20: DER 20.76%, MS 12.44%, FA 4.39%, SC 3.94%
#
#Eval for threshold 0.30: DER 20.97%, MS 14.76%, FA 2.25%, SC 3.97%
#
#Eval for threshold 0.35: DER 21.31%, MS 15.76%, FA 1.53%, SC 4.02%
#
#Eval for threshold 0.40: DER 21.73%, MS 16.71%, FA 0.93%, SC 4.09%
#
#Eval for threshold 0.45: DER 22.28%, MS 17.69%, FA 0.51%, SC 4.08%
#
#Eval for threshold 0.50: DER 22.97%, MS 18.79%, FA 0.17%, SC 4.01%
#
#Eval for threshold 0.55: DER 23.85%, MS 20.03%, FA 0.07%, SC 3.76%
#
#Eval for threshold 0.60: DER 24.98%, MS 21.46%, FA 0.06%, SC 3.46%
#
#Eval for threshold 0.70: DER 27.49%, MS 24.50%, FA 0.05%, SC 2.94%
#
#Eval for threshold 0.80: DER 30.95%, MS 28.44%, FA 0.04%, SC 2.47%
fi

if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    dataset_name="magicdata-ramc" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    source_dir=/data/maduo/exp/speaker_diarization/ts_vad2/alimeeting_and_magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/alimeeting_and_magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_continue_ft_on_magicdata-ramc
    mkdir -p $exp_dir
    cp -r $source_dir/best-valid-der.pt $exp_dir/
    mv $exp_dir/best-valid-der.pt $exp_dir/epoch-0.pt
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12815 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 1e-5\
    --do-finetune true\
    --finetune-ckpt $exp_dir/epoch-0.pt\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name
fi


if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/alimeeting_and_magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_continue_ft_on_magicdata-ramc
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 #collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${collar}
  python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/rttm_debug\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name
done
fi
