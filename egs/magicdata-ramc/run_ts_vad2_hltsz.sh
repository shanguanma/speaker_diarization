#!/usr/bin/env bash

stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_speaker_diarization_hltsz.sh

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "prepared magicdata-ramc kaldi format data"
   ## it has been removed G00000000 utt in rttm file 
   # based on the paper "The X-Lance Speaker Diarization System for the Conversational Short-phrase Speaker Diarization Challenge 2022"
   source_data_dir=/data/maduo/datasets/MagicData-RAMC/
   output_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format 
   python3 magicdata_ramc_prepared_180h_with_g0.py $source_data_dir $output_dir
   
   data_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   ## remove  G00000000
   for name in dev test train;do
	   grep -v "G00000000" $data_dir/$name/rttm_debug > $data_dir/$name/rttm_debug_nog0  
   done 
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
  echo "get target audio and json label file from rttm file"
  #datasets="test train"
  datasets="dev"
  source_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
  for name in $datasets;do
    oracle_rttm=$source_dir/$name/rttm_debug_nog0
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
   #subsets="dev test train"
   subsets="dev"
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
    --rttm-name ${name}/rttm_debug_nog0\
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

# bash run_ts_vad2_hltsz.sh --stage 4  --stop-stage 4 
# collar=0.25
# Eval set
# Eval for threshold 0.5 DER=5.34, miss=0.56, falarm=1.30, confusion=3.49 as report , it is best
# Test set
# Eval for threshold 0.5 DER=6.18, miss=1.34, falarm=2.26, confusion=2.58 as report , it is best 

# collar=0.0
# Eval set
# Eval for threshold 0.5 DER=11.49, miss=1.65, falarm=5.99, confusion=3.84 as report , it is best
# Test set
# Eval for threshold 0.5 DER=12.19, miss=2.76, falarm=6.26, confusion=3.17 as report , it is best
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
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
# bash run_ts_vad2_hltsz.sh --stage 5  --stop-stage 5
# dev set
# Avg CDER : 0.105 on threshold=0.8 
# test set
# Avg CDER : 0.138 on threshold=0.55

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
# bash run_ts_vad2_hltsz.sh --stage 6 --stop-stage 6
# Eval set
#Model DER:  0.338978389301755
#Model ACC:  0.8823294161146735
#2024-12-20 14:13:37,962 (infer:89) INFO: frame_len: 0.04!!
#100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25/25 [00:28<00:00,  1.15s/it]
#Eval for threshold 0.2 DER=20.92, miss=14.15, falarm=4.29, confusion=2.47
#
#
#Eval for threshold 0.3 DER=21.43, miss=17.11, falarm=2.09, confusion=2.23
#
#
#Eval for threshold 0.35 DER=22.06, miss=18.50, falarm=1.43, confusion=2.13
#
#
#Eval for threshold 0.4 DER=22.68, miss=19.78, falarm=0.91, confusion=1.99
#
#
#Eval for threshold 0.45 DER=23.54, miss=21.18, falarm=0.56, confusion=1.80
#
#
#Eval for threshold 0.5 DER=24.47, miss=22.54, falarm=0.35, confusion=1.58
#
#
#Eval for threshold 0.55 DER=25.81, miss=24.20, falarm=0.33, confusion=1.28
#
#
#Eval for threshold 0.6 DER=27.13, miss=25.76, falarm=0.32, confusion=1.05
#
#
#Eval for threshold 0.7 DER=30.24, miss=29.25, falarm=0.31, confusion=0.68
#
#
#Eval for threshold 0.8 DER=34.24, miss=33.59, falarm=0.29, confusion=0.36
#
## Test set
#Model DER:  0.3083555472119542
#Model ACC:  0.8867874452284269
#2024-12-20 14:40:27,874 (infer:89) INFO: frame_len: 0.04!!
#100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [01:12<00:00,  1.22s/it]
#Eval for threshold 0.2 DER=20.29, miss=12.49, falarm=5.46, confusion=2.33
#
#
#Eval for threshold 0.3 DER=20.50, miss=15.06, falarm=2.98, confusion=2.46
#
#
#Eval for threshold 0.35 DER=20.86, miss=16.25, falarm=2.07, confusion=2.55
#
#
#Eval for threshold 0.4 DER=21.38, miss=17.43, falarm=1.33, confusion=2.62
#
#
#Eval for threshold 0.45 DER=21.99, miss=18.63, falarm=0.73, confusion=2.63
#
#
#Eval for threshold 0.5 DER=22.68, miss=19.90, falarm=0.22, confusion=2.57
#
#
#Eval for threshold 0.55 DER=23.85, miss=21.50, falarm=0.06, confusion=2.29
#
#
#Eval for threshold 0.6 DER=25.23, miss=23.20, falarm=0.05, confusion=1.98
#
#
#Eval for threshold 0.7 DER=28.38, miss=26.96, falarm=0.04, confusion=1.38
#
#
#Eval for threshold 0.8 DER=32.44, miss=31.58, falarm=0.04, confusion=0.83
#

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
    dataset_name="magicdata-ramc" # dataset name # data structure is same as magicdata-ramc, not real magicadata-ramc
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
    --rttm-name ${name}/rttm_debug_nog0\
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
# grep -r 'Eval' logs/run_ts_vad2_hltsz_stage15-16.log
# Eval set
#Eval for threshold 0.20: DER 10.02%, MS 0.05%, FA 7.88%, SC 2.09%
#Eval for threshold 0.30: DER 7.50%, MS 0.09%, FA 4.71%, SC 2.70%
#Eval for threshold 0.35: DER 6.67%, MS 0.15%, FA 3.54%, SC 2.98%
#Eval for threshold 0.40: DER 6.13%, MS 0.27%, FA 2.63%, SC 3.23%
#Eval for threshold 0.45: DER 5.77%, MS 0.39%, FA 1.93%, SC 3.45%
#Eval for threshold 0.50: DER 5.63%, MS 0.57%, FA 1.50%, SC 3.56%
#Eval for threshold 0.55: DER 5.66%, MS 0.90%, FA 1.24%, SC 3.52%
#Eval for threshold 0.60: DER 5.88%, MS 1.36%, FA 1.15%, SC 3.38%
#Eval for threshold 0.70: DER 6.83%, MS 2.92%, FA 1.02%, SC 2.89%
#Eval for threshold 0.80: DER 9.30%, MS 6.23%, FA 0.90%, SC 2.16%
## Test set
#Eval for threshold 0.20: DER 9.68%, MS 0.03%, FA 8.37%, SC 1.28%
#Eval for threshold 0.30: DER 8.17%, MS 0.13%, FA 6.50%, SC 1.54%
#Eval for threshold 0.35: DER 7.77%, MS 0.26%, FA 5.90%, SC 1.62%
#Eval for threshold 0.40: DER 7.51%, MS 0.43%, FA 5.39%, SC 1.69%
#Eval for threshold 0.45: DER 7.23%, MS 0.65%, FA 4.73%, SC 1.85%
#Eval for threshold 0.50: DER 6.36%, MS 1.00%, FA 2.80%, SC 2.56%
#Eval for threshold 0.55: DER 6.93%, MS 2.80%, FA 2.10%, SC 2.04%
#Eval for threshold 0.60: DER 7.52%, MS 4.00%, FA 1.97%, SC 1.55%
#Eval for threshold 0.70: DER 8.27%, MS 5.22%, FA 1.74%, SC 1.31%
#Eval for threshold 0.80: DER 9.70%, MS 7.08%, FA 1.51%, SC 1.11%

#grep -r 'Eval' logs/run_ts_vad2_hltsz_stage16_collar0.log
# Eval set
#Eval for threshold 0.20: DER 18.17%, MS 0.26%, FA 15.48%, SC 2.43%
#Eval for threshold 0.30: DER 14.74%, MS 0.46%, FA 11.14%, SC 3.14%
#Eval for threshold 0.35: DER 13.61%, MS 0.65%, FA 9.54%, SC 3.41%
#Eval for threshold 0.40: DER 12.76%, MS 0.92%, FA 8.16%, SC 3.68%
#Eval for threshold 0.45: DER 12.18%, MS 1.23%, FA 7.08%, SC 3.87%
#Eval for threshold 0.50: DER 11.88%, MS 1.67%, FA 6.28%, SC 3.93%
#Eval for threshold 0.55: DER 11.78%, MS 2.28%, FA 5.67%, SC 3.83%
#Eval for threshold 0.60: DER 11.95%, MS 3.08%, FA 5.24%, SC 3.63%
#Eval for threshold 0.70: DER 12.80%, MS 5.34%, FA 4.41%, SC 3.05%
#Eval for threshold 0.80: DER 15.34%, MS 9.51%, FA 3.58%, SC 2.25%
# Test set
#Eval for threshold 0.20: DER 18.14%, MS 0.25%, FA 16.18%, SC 1.71%
#Eval for threshold 0.30: DER 15.55%, MS 0.56%, FA 12.88%, SC 2.12%
#Eval for threshold 0.35: DER 14.77%, MS 0.84%, FA 11.68%, SC 2.25%
#Eval for threshold 0.40: DER 14.20%, MS 1.19%, FA 10.67%, SC 2.34%
#Eval for threshold 0.45: DER 13.60%, MS 1.65%, FA 9.44%, SC 2.52%
#Eval for threshold 0.50: DER 12.46%, MS 2.32%, FA 6.96%, SC 3.18%
#Eval for threshold 0.55: DER 12.89%, MS 4.49%, FA 5.86%, SC 2.55%
#Eval for threshold 0.60: DER 13.42%, MS 6.02%, FA 5.39%, SC 2.02%
#Eval for threshold 0.70: DER 14.26%, MS 8.10%, FA 4.52%, SC 1.64%
#Eval for threshold 0.80: DER 16.04%, MS 11.09%, FA 3.66%, SC 1.29%
fi

if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/alimeeting_and_magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_continue_ft_on_magicdata-ramc
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 #collar=0.0

 dataset_name="alimeeting" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
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


if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
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
    dataset_name="alimeeting+magicdata-ramc" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
    data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
    
    alimeeting_spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    alimeeting_data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path 

    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc_with_prob0.5_alimeeting-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
    mkdir -p $exp_dir
   rs_len=4
   segment_shift=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12815 \
   ts_vad2/train_accelerate_ddp_multi.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-4\
    --alimeeting-prob 0.5\
    --alimeeting-spk-path $alimeeting_spk_path\
    --alimeeting-data-dir $alimeeting_data_dir\
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

if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc_with_prob0.5_alimeeting-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
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
fi



if [ ${stage} -le 30 ] && [ ${stop_stage} -ge 30 ];then
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
    dataset_name="alimeeting+magicdata-ramc" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
    data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

    alimeeting_spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    alimeeting_data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc_with_prob0.8_alimeeting-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
    mkdir -p $exp_dir
   rs_len=4
   segment_shift=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 13815 \
   ts_vad2/train_accelerate_ddp_multi.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-4\
    --alimeeting-prob 0.8\
    --alimeeting-spk-path $alimeeting_spk_path\
    --alimeeting-data-dir $alimeeting_data_dir\
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

if [ ${stage} -le 31 ] && [ ${stop_stage} -ge 31 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc_with_prob0.8_alimeeting-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
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
    --rttm-name ${name}/rttm_debug_nog0\
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

# compared with stage3-4, stage40-41 will use mamba network to replace transformer
if [ ${stage} -le 40 ] && [ ${stop_stage} -ge 40 ];then
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
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_both_mamba
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   single_backend_type="mamba"
   multi_backend_type="mamba"
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 13815 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 1e-5\
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type
fi

if [ ${stage} -le 41 ] && [ ${stop_stage} -ge 41 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_both_mamba
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="mamba"
 multi_backend_type="mamba"
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
    --rttm-name ${name}/rttm_debug_nog0\
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type
done
# grep -r 'Eval' logs/run_ts_vad2_hltsz_stage40-41_mamba_1e-5.log
# Eval set
#Eval for threshold 0.2 DER=31.31, miss=0.50, falarm=27.90, confusion=2.91
#Eval for threshold 0.3 DER=24.93, miss=0.93, falarm=19.37, confusion=4.63
#Eval for threshold 0.35 DER=21.68, miss=1.21, falarm=14.86, confusion=5.61
#Eval for threshold 0.4 DER=18.89, miss=1.53, falarm=10.83, confusion=6.52
#Eval for threshold 0.45 DER=17.14, miss=2.04, falarm=8.03, confusion=7.07
#Eval for threshold 0.5 DER=16.37, miss=3.01, falarm=6.10, confusion=7.26
#Eval for threshold 0.55 DER=16.98, miss=5.03, falarm=5.22, confusion=6.73
#Eval for threshold 0.6 DER=18.54, miss=7.96, falarm=4.77, confusion=5.81
#Eval for threshold 0.7 DER=23.08, miss=15.49, falarm=3.96, confusion=3.64
#Eval for threshold 0.8 DER=27.48, miss=22.77, falarm=3.10, confusion=1.61
# Test set
#Eval for threshold 0.2 DER=27.74, miss=0.64, falarm=24.59, confusion=2.52
#Eval for threshold 0.3 DER=22.42, miss=1.16, falarm=17.76, confusion=3.50
#Eval for threshold 0.35 DER=20.23, miss=1.54, falarm=14.57, confusion=4.11
#Eval for threshold 0.4 DER=18.22, miss=2.10, falarm=11.21, confusion=4.91
#Eval for threshold 0.45 DER=16.61, miss=3.09, falarm=7.83, confusion=5.69
#Eval for threshold 0.5 DER=16.59, miss=5.25, falarm=5.95, confusion=5.39
#Eval for threshold 0.55 DER=17.80, miss=8.22, falarm=5.19, confusion=4.39
#Eval for threshold 0.6 DER=19.27, miss=10.98, falarm=4.72, confusion=3.57
#Eval for threshold 0.7 DER=22.25, miss=15.80, falarm=3.97, confusion=2.48
#Eval for threshold 0.8 DER=25.93, miss=20.88, falarm=3.26, confusion=1.79

fi
# compared with stage40-41, stage42-43 will use more length window(i.e.rs_len=10s)
if [ ${stage} -le 42 ] && [ ${stop_stage} -ge 42 ];then
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
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e5_both_mamba_rs_len10
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=10
   segment_shift=2
   single_backend_type="mamba"
   multi_backend_type="mamba"
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 13815 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 1e-5\
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type
fi

if [ ${stage} -le 43 ] && [ ${stop_stage} -ge 43 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e5_both_mamba_rs_len10
 model_file=$exp_dir/best-valid-der.pt
 rs_len=10
 segment_shift=1
 single_backend_type="mamba"
 multi_backend_type="mamba"
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
    --rttm-name ${name}/rttm_debug_nog0\
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type
done
fi

# compared with stage40-41, stage44-45 will use more layer mamba network
if [ ${stage} -le 44 ] && [ ${stop_stage} -ge 44 ];then
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
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e5_both_mamba_4layers_mamba
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   single_backend_type="mamba"
   multi_backend_type="mamba"
   num_transformer_layer=4
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 14815 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 1e-5\
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 45 ] && [ ${stop_stage} -ge 45 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e5_both_mamba_4layers_mamba
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="mamba"
 multi_backend_type="mamba"
 num_transformer_layer=4
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
    --rttm-name ${name}/rttm_debug_nog0\
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer
done
# grep -r 'Eval' logs/run_ts_vad2_hltsz_stage44-45_mamba_1e-5.log
# dev set
# Eval for threshold 0.2 DER=28.52, miss=0.40, falarm=26.64, confusion=1.48
#Eval for threshold 0.3 DER=23.60, miss=0.74, falarm=20.66, confusion=2.20
#Eval for threshold 0.35 DER=21.53, miss=0.98, falarm=17.62, confusion=2.93
#Eval for threshold 0.4 DER=19.39, miss=1.34, falarm=13.88, confusion=4.17
#Eval for threshold 0.45 DER=17.34, miss=1.90, falarm=10.07, confusion=5.37
#Eval for threshold 0.5 DER=16.06, miss=2.91, falarm=7.17, confusion=5.98
#Eval for threshold 0.55 DER=16.13, miss=4.78, falarm=5.62, confusion=5.73
#Eval for threshold 0.6 DER=17.04, miss=7.30, falarm=4.94, confusion=4.80
#Eval for threshold 0.7 DER=20.09, miss=13.25, falarm=4.08, confusion=2.75
#Eval for threshold 0.8 DER=23.93, miss=19.52, falarm=3.18, confusion=1.24
# test set
#Eval for threshold 0.2 DER=26.22, miss=0.52, falarm=23.17, confusion=2.53
#Eval for threshold 0.3 DER=21.62, miss=0.98, falarm=17.18, confusion=3.46
#Eval for threshold 0.35 DER=19.84, miss=1.30, falarm=14.58, confusion=3.96
#Eval for threshold 0.4 DER=18.29, miss=1.74, falarm=11.99, confusion=4.56
#Eval for threshold 0.45 DER=16.98, miss=2.39, falarm=9.49, confusion=5.10
#Eval for threshold 0.5 DER=16.09, miss=3.52, falarm=7.18, confusion=5.40
#Eval for threshold 0.55 DER=16.21, miss=5.41, falarm=5.65, confusion=5.15
#Eval for threshold 0.6 DER=17.10, miss=7.86, falarm=4.96, confusion=4.29
#Eval for threshold 0.7 DER=19.52, miss=12.48, falarm=4.11, confusion=2.93
#Eval for threshold 0.8 DER=22.95, miss=17.76, falarm=3.26, confusion=1.92
fi



if [ ${stage} -le 46 ] && [ ${stop_stage} -ge 46 ];then
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
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_both_mamba_2layers_mamba
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   single_backend_type="mamba"
   multi_backend_type="mamba"
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 14815 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 1e-4\
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 47 ] && [ ${stop_stage} -ge 47 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_both_mamba_2layers_mamba
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="mamba"
 multi_backend_type="mamba"
 num_transformer_layer=2
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
    --rttm-name ${name}/rttm_debug_nog0\
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer
done
fi
# compared with stage3-4, stage40-41 will use mamba v2 network to replace transformer
if [ ${stage} -le 50 ] && [ ${stop_stage} -ge 50 ];then
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
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e5_both_mamba_v2
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   single_backend_type="mamba_v2"
   multi_backend_type="mamba_v2"
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 14815 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 1e-5\
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type
fi

if [ ${stage} -le 51 ] && [ ${stop_stage} -ge 51 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e5_both_mamba_v2
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="mamba_v2"
 multi_backend_type="mamba_v2"
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
    --rttm-name ${name}/rttm_debug_nog0\
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type
done
#grep -r 'Eval' logs/run_ts_vad2_hltsz_stage50-51_mamba_v2_1e-5.log
# Eval set
#Eval for threshold 0.2 DER=31.52, miss=0.53, falarm=27.98, confusion=3.01
#Eval for threshold 0.3 DER=24.98, miss=1.01, falarm=19.33, confusion=4.64
#Eval for threshold 0.35 DER=22.08, miss=1.33, falarm=15.47, confusion=5.27
#Eval for threshold 0.4 DER=19.42, miss=1.82, falarm=11.51, confusion=6.09
#Eval for threshold 0.45 DER=17.71, miss=2.58, falarm=8.32, confusion=6.80
#Eval for threshold 0.5 DER=17.26, miss=3.99, falarm=6.25, confusion=7.02 as report
#Eval for threshold 0.55 DER=18.12, miss=6.49, falarm=5.20, confusion=6.43
#Eval for threshold 0.6 DER=19.99, miss=10.01, falarm=4.64, confusion=5.34
#Eval for threshold 0.7 DER=24.19, miss=16.85, falarm=3.73, confusion=3.61
#Eval for threshold 0.8 DER=29.40, miss=24.68, falarm=2.79, confusion=1.94
# Test set
#Eval for threshold 0.2 DER=30.24, miss=0.63, falarm=27.22, confusion=2.39
#Eval for threshold 0.3 DER=23.15, miss=1.23, falarm=18.66, confusion=3.27
#Eval for threshold 0.35 DER=20.47, miss=1.69, falarm=14.98, confusion=3.80
#Eval for threshold 0.4 DER=18.43, miss=2.42, falarm=11.59, confusion=4.41
#Eval for threshold 0.45 DER=17.16, miss=3.70, falarm=8.62, confusion=4.85
#Eval for threshold 0.5 DER=17.01, miss=5.70, falarm=6.52, confusion=4.80 as report
#Eval for threshold 0.55 DER=17.96, miss=8.47, falarm=5.26, confusion=4.22
#Eval for threshold 0.6 DER=19.91, miss=11.97, falarm=4.67, confusion=3.27
#Eval for threshold 0.7 DER=24.32, miss=18.38, falarm=3.87, confusion=2.07
#Eval for threshold 0.8 DER=29.83, miss=25.13, falarm=3.11, confusion=1.59
fi

