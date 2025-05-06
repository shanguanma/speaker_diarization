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
  datasets="dev test train"
  #datasets="dev"
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
   subsets="dev test train"
   #subsets="dev"
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
 #infer_sets="cssd_testset"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path # for dev test
 #data_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path # for cssd_testset
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
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
    --collar $c\
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
 done
fi

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

#grep -r Eval logs/run_ts_vad2_hltsz_stage4_again_3.log
#collar=0.0
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=16.26, miss=0.29, falarm=13.04, confusion=2.93
#Eval for threshold 0.3 DER=13.38, miss=0.52, falarm=9.23, confusion=3.62
#Eval for threshold 0.35 DER=12.60, miss=0.70, falarm=8.14, confusion=3.76
#Eval for threshold 0.4 DER=12.10, miss=0.95, falarm=7.32, confusion=3.82
#Eval for threshold 0.45 DER=11.70, miss=1.26, falarm=6.59, confusion=3.85
#Eval for threshold 0.5 DER=11.50, miss=1.67, falarm=5.99, confusion=3.84
#Eval for threshold 0.55 DER=11.53, miss=2.23, falarm=5.58, confusion=3.72
#Eval for threshold 0.6 DER=11.64, miss=2.88, falarm=5.19, confusion=3.57
#Eval for threshold 0.7 DER=12.32, miss=4.74, falarm=4.39, confusion=3.18
#Eval for threshold 0.8 DER=14.58, miss=8.67, falarm=3.48, confusion=2.44
# test of magicdata-ramc
#Eval for threshold 0.2 DER=17.95, miss=0.33, falarm=15.93, confusion=1.69
#Eval for threshold 0.3 DER=15.52, miss=0.68, falarm=12.77, confusion=2.07
#Eval for threshold 0.35 DER=14.72, miss=0.95, falarm=11.55, confusion=2.22
#Eval for threshold 0.4 DER=14.16, miss=1.28, falarm=10.54, confusion=2.33
#Eval for threshold 0.45 DER=13.58, miss=1.71, falarm=9.42, confusion=2.45
#Eval for threshold 0.5 DER=12.18, miss=2.76, falarm=6.26, confusion=3.16
#Eval for threshold 0.55 DER=13.25, miss=5.44, falarm=5.65, confusion=2.16
#Eval for threshold 0.6 DER=13.49, miss=6.32, falarm=5.23, confusion=1.95
#Eval for threshold 0.7 DER=14.46, miss=8.51, falarm=4.38, confusion=1.56
#Eval for threshold 0.8 DER=16.66, miss=12.03, falarm=3.46, confusion=1.17
# collar=0.25
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=8.22, miss=0.05, falarm=5.51, confusion=2.66
#Eval for threshold 0.3 DER=6.25, miss=0.11, falarm=2.87, confusion=3.27
#Eval for threshold 0.35 DER=5.83, miss=0.16, falarm=2.27, confusion=3.39
#Eval for threshold 0.4 DER=5.57, miss=0.25, falarm=1.87, confusion=3.44
#Eval for threshold 0.45 DER=5.40, miss=0.37, falarm=1.56, confusion=3.47
#Eval for threshold 0.5 DER=5.34, miss=0.55, falarm=1.30, confusion=3.49
#Eval for threshold 0.55 DER=5.46, miss=0.85, falarm=1.21, confusion=3.40
#Eval for threshold 0.6 DER=5.61, miss=1.16, falarm=1.13, confusion=3.32
#Eval for threshold 0.7 DER=6.32, miss=2.25, falarm=1.00, confusion=3.07
#Eval for threshold 0.8 DER=8.43, miss=5.17, falarm=0.86, confusion=2.41
# test of magicdata-ramc
#Eval for threshold 0.2 DER=9.71, miss=0.05, falarm=8.40, confusion=1.27
#Eval for threshold 0.3 DER=8.27, miss=0.17, falarm=6.57, confusion=1.52
#Eval for threshold 0.35 DER=7.84, miss=0.28, falarm=5.94, confusion=1.62
#Eval for threshold 0.4 DER=7.57, miss=0.41, falarm=5.45, confusion=1.70
#Eval for threshold 0.45 DER=7.29, miss=0.62, falarm=4.88, confusion=1.79
#Eval for threshold 0.5 DER=6.16, miss=1.34, falarm=2.25, confusion=2.57
#Eval for threshold 0.55 DER=7.39, miss=3.71, falarm=2.06, confusion=1.62
#Eval for threshold 0.6 DER=7.62, miss=4.20, falarm=1.94, confusion=1.49
#Eval for threshold 0.7 DER=8.44, miss=5.49, falarm=1.69, confusion=1.26
#Eval for threshold 0.8 DER=10.32, miss=7.89, falarm=1.43, confusion=1.00


# 2025-2-27
#  python3 cder/score.py -s /data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4/magicdata-ramc_collar0.0/dev/res_rttm_0.5 -r  /data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/dev/rttm_debug_nog0
# Avg CDER : 0.196
#  python3 cder/score.py -s /data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4/magicdata-ramc_collar0.0/test/res_rttm_0.5 -r  /data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/test/rttm_debug_nog0
# Avg CDER : 0.146

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4/magicdata-ramc_collar0.0
   oracle_rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi
# bash run_ts_vad2_hltsz.sh --stage 5  --stop-stage 5
# dev set
# Avg CDER : 0.105 on threshold=0.8
# test set
# Avg CDER : 0.138 on threshold=0.55


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
 collar="0.0 0.25"

 dataset_name="alimeeting" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_$c/
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
    --collar $c\
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
done
fi
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

# 2025-3-20
#grep -r Eval logs/run_ts_vad2_hltsz_20250320_alimeeting.log
# Eval of alimeeting, collar=0.0
#Eval for threshold 0.2 DER=31.23, miss=21.77, falarm=5.83, confusion=3.63
#Eval for threshold 0.3 DER=32.27, miss=25.91, falarm=3.37, confusion=2.98
#Eval for threshold 0.35 DER=33.23, miss=27.86, falarm=2.63, confusion=2.73
#Eval for threshold 0.4 DER=34.10, miss=29.57, falarm=2.06, confusion=2.46
#Eval for threshold 0.45 DER=35.15, miss=31.30, falarm=1.67, confusion=2.18
#Eval for threshold 0.5 DER=36.27, miss=33.02, falarm=1.39, confusion=1.86
#Eval for threshold 0.55 DER=37.78, miss=35.00, falarm=1.27, confusion=1.51
#Eval for threshold 0.6 DER=39.23, miss=36.83, falarm=1.16, confusion=1.24
#Eval for threshold 0.7 DER=42.69, miss=41.00, falarm=0.92, confusion=0.76
#Eval for threshold 0.8 DER=46.98, miss=45.85, falarm=0.70, confusion=0.43
#Eval for threshold 0.9 DER=52.75, miss=52.12, falarm=0.48, confusion=0.15

# Test of alimeeting, collar=0.0
#Eval for threshold 0.2 DER=30.10, miss=19.62, falarm=7.01, confusion=3.47
#Eval for threshold 0.3 DER=31.04, miss=23.74, falarm=4.18, confusion=3.13
#Eval for threshold 0.35 DER=31.79, miss=25.62, falarm=3.17, confusion=2.99
#Eval for threshold 0.4 DER=32.64, miss=27.38, falarm=2.39, confusion=2.88
#Eval for threshold 0.45 DER=33.64, miss=29.16, falarm=1.75, confusion=2.73
#Eval for threshold 0.5 DER=34.75, miss=30.98, falarm=1.22, confusion=2.56
#Eval for threshold 0.55 DER=36.19, miss=33.01, falarm=0.98, confusion=2.20
#Eval for threshold 0.6 DER=37.79, miss=35.08, falarm=0.86, confusion=1.85
#Eval for threshold 0.7 DER=41.33, miss=39.48, falarm=0.63, confusion=1.22
#Eval for threshold 0.8 DER=45.73, miss=44.61, falarm=0.41, confusion=0.71
#Eval for threshold 0.9 DER=52.12, miss=51.63, falarm=0.21, confusion=0.28

# Eval of alimeeting, collar=0.25
#Eval for threshold 0.2 DER=20.82, miss=14.09, falarm=4.26, confusion=2.47 as report
#Eval for threshold 0.3 DER=21.44, miss=17.13, falarm=2.08, confusion=2.23
#Eval for threshold 0.35 DER=22.04, miss=18.49, falarm=1.41, confusion=2.13
#Eval for threshold 0.4 DER=22.66, miss=19.77, falarm=0.90, confusion=1.99
#Eval for threshold 0.45 DER=23.56, miss=21.18, falarm=0.58, confusion=1.81
#Eval for threshold 0.5 DER=24.47, miss=22.54, falarm=0.35, confusion=1.58
#Eval for threshold 0.55 DER=25.79, miss=24.18, falarm=0.33, confusion=1.27
#Eval for threshold 0.6 DER=27.13, miss=25.78, falarm=0.32, confusion=1.04
#Eval for threshold 0.7 DER=30.25, miss=29.29, falarm=0.31, confusion=0.65
#Eval for threshold 0.8 DER=34.26, miss=33.61, falarm=0.29, confusion=0.36
#Eval for threshold 0.9 DER=39.93, miss=39.54, falarm=0.27, confusion=0.12

# Test of alimeeting, collar=0.25
#Eval for threshold 0.2 DER=20.30, miss=12.49, falarm=5.47, confusion=2.34 as report
#Eval for threshold 0.3 DER=20.55, miss=15.08, falarm=2.99, confusion=2.47
#Eval for threshold 0.35 DER=20.86, miss=16.26, falarm=2.06, confusion=2.53
#Eval for threshold 0.4 DER=21.38, miss=17.42, falarm=1.34, confusion=2.61
#Eval for threshold 0.45 DER=21.95, miss=18.63, falarm=0.70, confusion=2.62
#Eval for threshold 0.5 DER=22.67, miss=19.88, falarm=0.22, confusion=2.57
#Eval for threshold 0.55 DER=23.85, miss=21.49, falarm=0.06, confusion=2.29
#Eval for threshold 0.6 DER=25.20, miss=23.20, falarm=0.05, confusion=1.96
#Eval for threshold 0.7 DER=28.39, miss=26.99, falarm=0.04, confusion=1.36
#Eval for threshold 0.8 DER=32.48, miss=31.61, falarm=0.04, confusion=0.84
#Eval for threshold 0.9 DER=38.71, miss=38.34, falarm=0.03, confusion=0.35

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 #infer_sets="dev test"
 infer_sets="cssd_testset"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path # for dev test
 data_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path # for cssd_testset
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
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
    --collar $c\
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
 done
fi
# logs/run_ts_vad2_hltsz_20250320_cssd_testset.log
# grep -r Eval logs/run_ts_vad2_hltsz_20250320_cssd_testset.log
# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=26.62, miss=3.01, falarm=22.28, confusion=1.33
#Eval for threshold 0.3 DER=22.31, miss=4.34, falarm=15.79, confusion=2.18
#Eval for threshold 0.35 DER=20.99, miss=5.17, falarm=13.28, confusion=2.55
#Eval for threshold 0.4 DER=20.11, miss=6.17, falarm=11.10, confusion=2.84
#Eval for threshold 0.45 DER=19.70, miss=7.43, falarm=9.31, confusion=2.96 as report
#Eval for threshold 0.5 DER=19.83, miss=9.09, falarm=8.00, confusion=2.74
#Eval for threshold 0.55 DER=20.53, miss=11.12, falarm=7.15, confusion=2.25
#Eval for threshold 0.6 DER=21.39, miss=13.19, falarm=6.38, confusion=1.82
#Eval for threshold 0.7 DER=23.83, miss=17.97, falarm=4.80, confusion=1.06
#Eval for threshold 0.8 DER=27.79, miss=24.00, falarm=3.24, confusion=0.55
#Eval for threshold 0.9 DER=35.33, miss=33.29, falarm=1.79, confusion=0.24

# cssd_testset of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=8.39, miss=0.87, falarm=7.07, confusion=0.46
#Eval for threshold 0.3 DER=6.14, miss=1.36, falarm=3.86, confusion=0.92
#Eval for threshold 0.35 DER=5.51, miss=1.68, falarm=2.67, confusion=1.16
#Eval for threshold 0.4 DER=5.13, miss=2.05, falarm=1.68, confusion=1.40
#Eval for threshold 0.45 DER=5.06, miss=2.63, falarm=0.88, confusion=1.54 as report
#Eval for threshold 0.5 DER=5.40, miss=3.49, falarm=0.42, confusion=1.50
#Eval for threshold 0.55 DER=6.21, miss=4.68, falarm=0.33, confusion=1.20
#Eval for threshold 0.6 DER=7.12, miss=5.91, falarm=0.27, confusion=0.93
#Eval for threshold 0.7 DER=9.58, miss=8.90, falarm=0.15, confusion=0.53
#Eval for threshold 0.8 DER=13.37, miss=13.03, falarm=0.07, confusion=0.27
#Eval for threshold 0.9 DER=20.80, miss=20.62, falarm=0.03, confusion=0.14



if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4/alimeeting_collar0.0
   oracle_rttm_dir=/data/maduo/model_hub/ts_vad
   infer_sets="Eval Test"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/alimeeting_${name,,}.rttm
    done
   done
fi
# 2025-3-20
#grep -r Avg logs/run_ts_vad2_hltsz_20250320_stage8.log
# Eval of alimeeting,
#Avg CDER : 0.244
#Avg CDER : 0.201
#Avg CDER : 0.180
#Avg CDER : 0.158
#Avg CDER : 0.175
#Avg CDER : 0.160
#Avg CDER : 0.153
#Avg CDER : 0.147
#Avg CDER : 0.138
#Avg CDER : 0.125
#Avg CDER : 0.194

# Test of alimeeting
#Avg CDER : 0.247
#Avg CDER : 0.214
#Avg CDER : 0.195
#Avg CDER : 0.186
#Avg CDER : 0.172
#Avg CDER : 0.155
#Avg CDER : 0.141
#Avg CDER : 0.134
#Avg CDER : 0.116
#Avg CDER : 0.125
#Avg CDER : 0.145

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
 #infer_sets="dev test"
 infer_sets="dev test cssd_testset"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 data_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels  of 'dev test cssd_testset'
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
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
    --collar $c\
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
done
fi
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


# without nog0(2024-12-26 result)
#  grep -r 'Eval'  logs/run_ts_vad2_hltsz_stage10-11_wo_nog0.log
# collar=0.25
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=9.02, miss=0.06, falarm=6.52, confusion=2.44
#Eval for threshold 0.3 DER=6.61, miss=0.09, falarm=3.42, confusion=3.10
#Eval for threshold 0.35 DER=6.03, miss=0.15, falarm=2.62, confusion=3.27
#Eval for threshold 0.4 DER=5.70, miss=0.27, falarm=2.05, confusion=3.37
#Eval for threshold 0.45 DER=5.49, miss=0.40, falarm=1.64, confusion=3.45
#Eval for threshold 0.5 DER=5.40, miss=0.60, falarm=1.34, confusion=3.46
#Eval for threshold 0.55 DER=5.53, miss=0.97, falarm=1.20, confusion=3.36
#Eval for threshold 0.6 DER=5.79, miss=1.43, falarm=1.12, confusion=3.23
#Eval for threshold 0.7 DER=6.76, miss=2.83, falarm=1.00, confusion=2.93
#Eval for threshold 0.8 DER=9.00, miss=5.77, falarm=0.89, confusion=2.35

# test of magicdata-ramc
#Eval for threshold 0.2 DER=9.41, miss=0.05, falarm=8.06, confusion=1.30
#Eval for threshold 0.3 DER=8.03, miss=0.15, falarm=6.32, confusion=1.56
#Eval for threshold 0.35 DER=7.65, miss=0.27, falarm=5.74, confusion=1.64
#Eval for threshold 0.4 DER=7.41, miss=0.46, falarm=5.27, confusion=1.68
#Eval for threshold 0.45 DER=6.93, miss=0.68, falarm=4.35, confusion=1.90
#Eval for threshold 0.5 DER=6.18, miss=1.33, falarm=2.45, confusion=2.41
#Eval for threshold 0.55 DER=7.11, miss=3.36, falarm=2.07, confusion=1.68
#Eval for threshold 0.6 DER=7.57, miss=4.11, falarm=1.95, confusion=1.51
#Eval for threshold 0.7 DER=8.32, miss=5.24, falarm=1.75, confusion=1.33
#Eval for threshold 0.8 DER=9.84, miss=7.21, falarm=1.50, confusion=1.13

# 2025-3-20
#grep -r Eval logs/run_ts_vad2_hltsz_20250320_stage11.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=17.23, miss=0.27, falarm=14.24, confusion=2.72
#Eval for threshold 0.3 DER=13.87, miss=0.50, falarm=9.92, confusion=3.45
#Eval for threshold 0.35 DER=12.93, miss=0.69, falarm=8.57, confusion=3.67
#Eval for threshold 0.4 DER=12.30, miss=0.98, falarm=7.55, confusion=3.77
#Eval for threshold 0.45 DER=11.89, miss=1.33, falarm=6.73, confusion=3.83
#Eval for threshold 0.5 DER=11.62, miss=1.78, falarm=6.02, confusion=3.81
#Eval for threshold 0.55 DER=11.63, miss=2.46, falarm=5.52, confusion=3.66
#Eval for threshold 0.6 DER=11.79, miss=3.29, falarm=5.04, confusion=3.47
#Eval for threshold 0.7 DER=12.69, miss=5.49, falarm=4.16, confusion=3.04
#Eval for threshold 0.8 DER=15.06, miss=9.40, falarm=3.28, confusion=2.37
#Eval for threshold 0.9 DER=20.72, miss=17.19, falarm=2.30, confusion=1.23

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=17.64, miss=0.29, falarm=15.60, confusion=1.75
#Eval for threshold 0.3 DER=15.26, miss=0.66, falarm=12.46, confusion=2.14
#Eval for threshold 0.35 DER=14.50, miss=0.95, falarm=11.29, confusion=2.26
#Eval for threshold 0.4 DER=13.94, miss=1.35, falarm=10.27, confusion=2.32
#Eval for threshold 0.45 DER=13.16, miss=1.85, falarm=8.76, confusion=2.55
#Eval for threshold 0.5 DER=12.18, miss=2.87, falarm=6.33, confusion=2.98 as report
#Eval for threshold 0.55 DER=12.99, miss=5.21, falarm=5.56, confusion=2.22
#Eval for threshold 0.6 DER=13.39, miss=6.31, falarm=5.11, confusion=1.97
#Eval for threshold 0.7 DER=14.27, miss=8.35, falarm=4.28, confusion=1.64
#Eval for threshold 0.8 DER=16.16, miss=11.44, falarm=3.41, confusion=1.31
#Eval for threshold 0.9 DER=20.41, miss=17.07, falarm=2.46, confusion=0.88

# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=28.09, miss=3.07, falarm=23.81, confusion=1.21
#Eval for threshold 0.3 DER=22.65, miss=4.32, falarm=16.54, confusion=1.78
#Eval for threshold 0.35 DER=20.72, miss=5.09, falarm=13.51, confusion=2.12
#Eval for threshold 0.4 DER=19.39, miss=5.99, falarm=10.93, confusion=2.47
#Eval for threshold 0.45 DER=18.67, miss=7.26, falarm=8.78, confusion=2.63 as report
#Eval for threshold 0.5 DER=18.89, miss=9.17, falarm=7.42, confusion=2.29
#Eval for threshold 0.55 DER=19.69, miss=11.43, falarm=6.49, confusion=1.77
#Eval for threshold 0.6 DER=20.75, miss=13.76, falarm=5.64, confusion=1.35
#Eval for threshold 0.7 DER=23.72, miss=18.81, falarm=4.11, confusion=0.80
#Eval for threshold 0.8 DER=28.02, miss=24.87, falarm=2.70, confusion=0.45
#Eval for threshold 0.9 DER=35.75, miss=34.08, falarm=1.47, confusion=0.21

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=9.03, miss=0.06, falarm=6.53, confusion=2.44
#Eval for threshold 0.3 DER=6.62, miss=0.09, falarm=3.43, confusion=3.10
#Eval for threshold 0.35 DER=6.04, miss=0.15, falarm=2.63, confusion=3.26
#Eval for threshold 0.4 DER=5.70, miss=0.27, falarm=2.06, confusion=3.38
#Eval for threshold 0.45 DER=5.50, miss=0.40, falarm=1.65, confusion=3.45
#Eval for threshold 0.5 DER=5.39, miss=0.59, falarm=1.34, confusion=3.46 as report
#Eval for threshold 0.55 DER=5.54, miss=0.99, falarm=1.20, confusion=3.35
#Eval for threshold 0.6 DER=5.80, miss=1.45, falarm=1.12, confusion=3.23
#Eval for threshold 0.7 DER=6.76, miss=2.83, falarm=1.00, confusion=2.93
#Eval for threshold 0.8 DER=8.99, miss=5.76, falarm=0.89, confusion=2.34
#Eval for threshold 0.9 DER=14.56, miss=12.58, falarm=0.74, confusion=1.24

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=9.42, miss=0.05, falarm=8.08, confusion=1.30
#Eval for threshold 0.3 DER=8.02, miss=0.15, falarm=6.31, confusion=1.56
#Eval for threshold 0.35 DER=7.65, miss=0.27, falarm=5.74, confusion=1.65
#Eval for threshold 0.4 DER=7.42, miss=0.46, falarm=5.27, confusion=1.68
#Eval for threshold 0.45 DER=6.93, miss=0.68, falarm=4.35, confusion=1.90
#Eval for threshold 0.5 DER=6.18, miss=1.32, falarm=2.45, confusion=2.41 as report
#Eval for threshold 0.55 DER=7.11, miss=3.36, falarm=2.07, confusion=1.68
#Eval for threshold 0.6 DER=7.57, miss=4.12, falarm=1.95, confusion=1.51
#Eval for threshold 0.7 DER=8.32, miss=5.25, falarm=1.75, confusion=1.32
#Eval for threshold 0.8 DER=9.85, miss=7.23, falarm=1.50, confusion=1.12
#Eval for threshold 0.9 DER=13.45, miss=11.45, falarm=1.19, confusion=0.82

# cssd_testset of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=9.90, miss=0.89, falarm=8.73, confusion=0.29
#Eval for threshold 0.3 DER=6.57, miss=1.31, falarm=4.72, confusion=0.54
#Eval for threshold 0.35 DER=5.49, miss=1.58, falarm=3.15, confusion=0.76
#Eval for threshold 0.4 DER=4.77, miss=1.90, falarm=1.82, confusion=1.05
#Eval for threshold 0.45 DER=4.52, miss=2.45, falarm=0.85, confusion=1.22 as report
#Eval for threshold 0.5 DER=5.01, miss=3.53, falarm=0.43, confusion=1.05
#Eval for threshold 0.55 DER=5.95, miss=4.91, falarm=0.31, confusion=0.72
#Eval for threshold 0.6 DER=7.12, miss=6.39, falarm=0.23, confusion=0.50
#Eval for threshold 0.7 DER=10.11, miss=9.70, falarm=0.13, confusion=0.28
#Eval for threshold 0.8 DER=14.19, miss=13.96, falarm=0.07, confusion=0.17
#Eval for threshold 0.9 DER=21.56, miss=21.42, falarm=0.03, confusion=0.11

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8"
   predict_rttm_dir=/data/maduo/exp/speaker_diarization/ts_vad2/alimeeting_and_magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4/magicdata-ramc_collar0.0
   oracle_rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi
#  cat logs/run_ts_vad2_hltsz_stage12_cder.log
# dev set
# Avg CDER : 0.186 on threshold=0.6
# Avg CDER : 0.143 on threshold=0.7
# Avg CDER : 0.115 on threshold=0.8

# test set
# Avg CDER : 0.133  on threshold=0.6

# 2025-3-20
#grep -r Avg logs/run_ts_vad2_hltsz_20250320_stage12.log
# dev of magicdata-ram
#Avg CDER : 0.967
#Avg CDER : 0.617
#Avg CDER : 0.522
#Avg CDER : 0.424
#Avg CDER : 0.354
#Avg CDER : 0.270 # threshold=0.5
#Avg CDER : 0.199
#Avg CDER : 0.166
#Avg CDER : 0.128
#Avg CDER : 0.110

# test of magicdata-ranc
#Avg CDER : 0.459
#Avg CDER : 0.331
#Avg CDER : 0.282
#Avg CDER : 0.256
#Avg CDER : 0.243
#Avg CDER : 0.185 # threshold=0.5
#Avg CDER : 0.129
#Avg CDER : 0.137
#Avg CDER : Error!
#Avg CDER : Error!

# cssd_testset of magicdata-ramc
#Avg CDER : 0.289
#Avg CDER : 0.218
#Avg CDER : 0.183
#Avg CDER : 0.154
#Avg CDER : 0.126 # threshold=0.45
#Avg CDER : 0.107
#Avg CDER : 0.099
#Avg CDER : 0.096
#Avg CDER : 0.087 
#Avg CDER : 0.096

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
 #exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/alimeeting_and_magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
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
 collar="0.0 0.25"

 dataset_name="alimeeting" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
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
    --collar $c\
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
done
fi
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
# 2025-3-20
#grep -r Eval logs/run_ts_vad2_hltsz_20250320_stage13.log
# Eval of alimeeting, collar=0.0
#Eval for threshold 0.2 DER=24.05, miss=1.97, falarm=21.06, confusion=1.01
#Eval for threshold 0.3 DER=17.87, miss=3.31, falarm=13.29, confusion=1.27
#Eval for threshold 0.35 DER=16.10, miss=4.08, falarm=10.71, confusion=1.32
#Eval for threshold 0.4 DER=14.92, miss=4.86, falarm=8.74, confusion=1.31
#Eval for threshold 0.45 DER=14.19, miss=5.78, falarm=7.14, confusion=1.26
#Eval for threshold 0.5 DER=13.81, miss=6.81, falarm=5.77, confusion=1.23
#Eval for threshold 0.55 DER=13.69, miss=7.89, falarm=4.66, confusion=1.15
#Eval for threshold 0.6 DER=13.84, miss=9.09, falarm=3.72, confusion=1.03
#Eval for threshold 0.7 DER=15.05, miss=11.90, falarm=2.37, confusion=0.79
#Eval for threshold 0.8 DER=17.74, miss=15.84, falarm=1.39, confusion=0.50
#Eval for threshold 0.9 DER=22.96, miss=22.05, falarm=0.69, confusion=0.22

# Test of alimeeting, collar=0.0
#Eval for threshold 0.2 DER=21.43, miss=2.07, falarm=18.24, confusion=1.12
#Eval for threshold 0.3 DER=16.56, miss=3.43, falarm=11.76, confusion=1.37
#Eval for threshold 0.35 DER=15.26, miss=4.23, falarm=9.59, confusion=1.43
#Eval for threshold 0.4 DER=14.39, miss=5.07, falarm=7.84, confusion=1.48
#Eval for threshold 0.45 DER=13.87, miss=6.03, falarm=6.38, confusion=1.47
#Eval for threshold 0.5 DER=13.66, miss=7.08, falarm=5.16, confusion=1.43
#Eval for threshold 0.55 DER=13.74, miss=8.20, falarm=4.17, confusion=1.37
#Eval for threshold 0.6 DER=14.05, miss=9.47, falarm=3.31, confusion=1.26
#Eval for threshold 0.7 DER=15.59, miss=12.59, falarm=2.00, confusion=0.99
#Eval for threshold 0.8 DER=18.65, miss=16.88, falarm=1.09, confusion=0.69
#Eval for threshold 0.9 DER=24.57, miss=23.73, falarm=0.48, confusion=0.37

# Eval of alimeeting, collar=0.25
#Eval for threshold 0.2 DER=12.36, miss=0.69, falarm=11.36, confusion=0.31
#Eval for threshold 0.3 DER=8.16, miss=1.24, falarm=6.48, confusion=0.44
#Eval for threshold 0.35 DER=6.95, miss=1.56, falarm=4.94, confusion=0.45
#Eval for threshold 0.4 DER=6.18, miss=1.90, falarm=3.83, confusion=0.45
#Eval for threshold 0.45 DER=5.75, miss=2.38, falarm=2.96, confusion=0.41
#Eval for threshold 0.5 DER=5.59, miss=2.94, falarm=2.27, confusion=0.39 as report
#Eval for threshold 0.55 DER=5.61, miss=3.50, falarm=1.73, confusion=0.38
#Eval for threshold 0.6 DER=5.72, miss=4.09, falarm=1.29, confusion=0.33
#Eval for threshold 0.7 DER=6.68, miss=5.67, falarm=0.77, confusion=0.25
#Eval for threshold 0.8 DER=8.68, miss=8.06, falarm=0.46, confusion=0.16
#Eval for threshold 0.9 DER=12.65, miss=12.30, falarm=0.30, confusion=0.04

# Test of alimeeting, collar=0.25
#Eval for threshold 0.2 DER=10.62, miss=0.91, falarm=9.28, confusion=0.43
#Eval for threshold 0.3 DER=7.33, miss=1.57, falarm=5.21, confusion=0.55
#Eval for threshold 0.35 DER=6.54, miss=1.95, falarm=4.02, confusion=0.57
#Eval for threshold 0.4 DER=6.03, miss=2.38, falarm=3.06, confusion=0.59
#Eval for threshold 0.45 DER=5.78, miss=2.85, falarm=2.34, confusion=0.60 
#Eval for threshold 0.5 DER=5.72, miss=3.38, falarm=1.74, confusion=0.60 as report
#Eval for threshold 0.55 DER=5.84, miss=3.98, falarm=1.29, confusion=0.58
#Eval for threshold 0.6 DER=6.11, miss=4.66, falarm=0.93, confusion=0.53
#Eval for threshold 0.7 DER=7.25, miss=6.43, falarm=0.44, confusion=0.39
#Eval for threshold 0.8 DER=9.42, miss=8.99, falarm=0.20, confusion=0.24
#Eval for threshold 0.9 DER=13.68, miss=13.50, falarm=0.08, confusion=0.10

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/data/maduo/exp/speaker_diarization/ts_vad2/alimeeting_and_magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4/alimeeting_collar0.0
   oracle_rttm_dir=/data/maduo/model_hub/ts_vad
   infer_sets="Eval Test"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/alimeeting_${name,,}.rttm
    done
   done
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
 #infer_sets="dev test"
 infer_sets="cssd_testset"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 #collar=0.0
 collar="0.0 0.25"
 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 data_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels  of cssd_testset path
 for c in $collar;do
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
done
fi
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

# it comsumes about more than 4 days in training stage.
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


# compared with stage40-41, stage46-47 will increase lr_rate from 1e-5 to 1e-4 (note: lr_rate=2e-4, loss will have nan)
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
#grep -r 'Eval' logs/run_ts_vad2_hltsz_stage46-47_mamba_1e-4.log
# collar=0.0
# dev set
#Eval for threshold 0.2 DER=16.90, miss=1.15, falarm=12.82, confusion=2.93
#Eval for threshold 0.3 DER=14.41, miss=1.86, falarm=8.83, confusion=3.72
#Eval for threshold 0.35 DER=13.71, miss=2.32, falarm=7.44, confusion=3.96
#Eval for threshold 0.4 DER=13.37, miss=2.91, falarm=6.40, confusion=4.06
#Eval for threshold 0.45 DER=13.22, miss=3.62, falarm=5.54, confusion=4.06
#Eval for threshold 0.5 DER=13.42, miss=4.53, falarm=4.95, confusion=3.94
#Eval for threshold 0.55 DER=13.92, miss=5.67, falarm=4.55, confusion=3.70
#Eval for threshold 0.6 DER=14.52, miss=6.89, falarm=4.18, confusion=3.44
#Eval for threshold 0.7 DER=16.32, miss=9.85, falarm=3.53, confusion=2.95
#Eval for threshold 0.8 DER=19.38, miss=14.30, falarm=2.83, confusion=2.25
# test set
#Eval for threshold 0.2 DER=18.88, miss=1.44, falarm=15.38, confusion=2.06
#Eval for threshold 0.3 DER=16.67, miss=2.55, falarm=11.52, confusion=2.60
#Eval for threshold 0.35 DER=16.14, miss=3.29, falarm=10.08, confusion=2.77
#Eval for threshold 0.4 DER=15.81, miss=4.17, falarm=8.79, confusion=2.86
#Eval for threshold 0.45 DER=15.63, miss=5.26, falarm=7.38, confusion=2.98
#Eval for threshold 0.5 DER=15.61, miss=6.74, falarm=5.69, confusion=3.18
#Eval for threshold 0.55 DER=16.42, miss=9.19, falarm=4.46, confusion=2.77
#Eval for threshold 0.6 DER=18.12, miss=12.06, falarm=3.98, confusion=2.08
#Eval for threshold 0.7 DER=21.30, miss=16.53, falarm=3.28, confusion=1.49
#Eval for threshold 0.8 DER=25.98, miss=22.40, falarm=2.54, confusion=1.03

fi


# compared with stage46-47, stage48-49 multi_backend_type will use transformer network.
if [ ${stage} -le 48 ] && [ ${stop_stage} -ge 48 ];then
    . path_for_speaker_diarization_hltsz.sh
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
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17815 \
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

if [ ${stage} -le 49 ] && [ ${stop_stage} -ge 49 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 num_transformer_layer=2
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer
done
#grep -r Eval logs/run_ts_vad2_hltsz_stage48-49_mamba2_1e-4.log
#collar=0.0
#dev of magicdata-ramc
#Eval for threshold 0.2 DER=15.94, miss=0.23, falarm=12.62, confusion=3.09
#Eval for threshold 0.3 DER=13.47, miss=0.44, falarm=9.44, confusion=3.58
#Eval for threshold 0.35 DER=12.76, miss=0.62, falarm=8.44, confusion=3.70
#Eval for threshold 0.4 DER=12.20, miss=0.83, falarm=7.60, confusion=3.77
#Eval for threshold 0.45 DER=11.81, miss=1.14, falarm=6.90, confusion=3.78
#Eval for threshold 0.5 DER=11.59, miss=1.55, falarm=6.29, confusion=3.75
#Eval for threshold 0.55 DER=11.49, miss=2.07, falarm=5.80, confusion=3.63
#Eval for threshold 0.6 DER=11.47, miss=2.69, falarm=5.30, confusion=3.48
#Eval for threshold 0.7 DER=11.98, miss=4.43, falarm=4.38, confusion=3.16
#Eval for threshold 0.8 DER=13.65, miss=7.57, falarm=3.42, confusion=2.66
## test of magicdata-ramc
#Eval for threshold 0.2 DER=18.17, miss=0.30, falarm=16.18, confusion=1.69
#Eval for threshold 0.3 DER=15.73, miss=0.64, falarm=12.97, confusion=2.11
#Eval for threshold 0.35 DER=14.92, miss=0.90, falarm=11.78, confusion=2.24
#Eval for threshold 0.4 DER=14.22, miss=1.25, falarm=10.57, confusion=2.40
#Eval for threshold 0.45 DER=13.37, miss=1.72, falarm=8.93, confusion=2.72
#Eval for threshold 0.5 DER=12.33, miss=2.46, falarm=6.60, confusion=3.27
#Eval for threshold 0.55 DER=12.94, miss=4.65, falarm=5.78, confusion=2.51
#Eval for threshold 0.6 DER=13.46, miss=6.14, falarm=5.27, confusion=2.05
#Eval for threshold 0.7 DER=14.40, miss=8.46, falarm=4.31, confusion=1.63
#Eval for threshold 0.8 DER=16.43, miss=11.82, falarm=3.38, confusion=1.24



#grep -r Eval logs/run_ts_vad2_hltsz_stage49.log
## collar=0.25
## dev of magicdata-ramc
#Eval for threshold 0.2 DER=7.35, miss=0.05, falarm=4.51, confusion=2.79
#Eval for threshold 0.3 DER=5.86, miss=0.12, falarm=2.56, confusion=3.19
#Eval for threshold 0.35 DER=5.52, miss=0.17, falarm=2.07, confusion=3.28
#Eval for threshold 0.4 DER=5.29, miss=0.23, falarm=1.72, confusion=3.33
#Eval for threshold 0.45 DER=5.17, miss=0.34, falarm=1.49, confusion=3.35
#Eval for threshold 0.5 DER=5.14, miss=0.49, falarm=1.30, confusion=3.35
#Eval for threshold 0.55 DER=5.20, miss=0.70, falarm=1.20, confusion=3.30
#Eval for threshold 0.6 DER=5.31, miss=0.98, falarm=1.11, confusion=3.22
#Eval for threshold 0.7 DER=5.90, miss=1.89, falarm=0.98, confusion=3.03
#Eval for threshold 0.8 DER=7.45, miss=3.94, falarm=0.86, confusion=2.65
#
## test of magicdata-ramc
#Eval for threshold 0.2 DER=9.42, miss=0.07, falarm=8.09, confusion=1.27
#Eval for threshold 0.3 DER=8.10, miss=0.19, falarm=6.38, confusion=1.53
#Eval for threshold 0.35 DER=7.72, miss=0.30, falarm=5.82, confusion=1.60
#Eval for threshold 0.4 DER=7.42, miss=0.45, falarm=5.24, confusion=1.73
#Eval for threshold 0.45 DER=6.91, miss=0.66, falarm=4.21, confusion=2.04
#Eval for threshold 0.5 DER=6.08, miss=1.03, falarm=2.39, confusion=2.66
#Eval for threshold 0.55 DER=6.86, miss=2.90, falarm=2.00, confusion=1.97
#Eval for threshold 0.6 DER=7.45, miss=4.01, falarm=1.87, confusion=1.57
#Eval for threshold 0.8 DER=9.96, miss=7.48, falarm=1.39, confusion=1.08
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

# stage63-64 single_backend will use transformer network.
if [ ${stage} -le 63 ] && [ ${stop_stage} -ge 63 ];then
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
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_transformer_multi_backend_mamba2
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   single_backend_type="transformer"
   multi_backend_type="mamba2"
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 18815 \
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

if [ ${stage} -le 64 ] && [ ${stop_stage} -ge 64 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_transformer_multi_backend_mamba2
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="mamba2"
 num_transformer_layer=2
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer
done
#grep -r 'Eval' logs/run_ts_vad2_hltsz_stage63-64_mamba2_1e-4.log
#collar=0.0
#dev of magicdata-ramc
#Eval for threshold 0.2 DER=20.87, miss=0.20, falarm=18.62, confusion=2.06
#Eval for threshold 0.3 DER=17.42, miss=0.35, falarm=14.64, confusion=2.42
#Eval for threshold 0.35 DER=16.12, miss=0.46, falarm=13.08, confusion=2.58
#Eval for threshold 0.4 DER=15.01, miss=0.61, falarm=11.62, confusion=2.77
#Eval for threshold 0.45 DER=14.00, miss=0.82, falarm=10.21, confusion=2.98
#Eval for threshold 0.5 DER=13.10, miss=1.11, falarm=8.77, confusion=3.22
#Eval for threshold 0.55 DER=12.36, miss=1.54, falarm=7.35, confusion=3.47
#Eval for threshold 0.6 DER=11.90, miss=2.24, falarm=6.01, confusion=3.65
#Eval for threshold 0.7 DER=13.02, miss=5.53, falarm=4.82, confusion=2.67
#Eval for threshold 0.8 DER=15.07, miss=9.11, falarm=3.89, confusion=2.06
#
#test of magicdata-ramc
#Eval for threshold 0.2 DER=21.17, miss=0.20, falarm=19.64, confusion=1.33
#Eval for threshold 0.3 DER=17.75, miss=0.41, falarm=15.59, confusion=1.75
#Eval for threshold 0.35 DER=16.54, miss=0.57, falarm=13.99, confusion=1.98
#Eval for threshold 0.4 DER=15.53, miss=0.79, falarm=12.56, confusion=2.18
#Eval for threshold 0.45 DER=14.61, miss=1.07, falarm=11.14, confusion=2.40
#Eval for threshold 0.5 DER=13.69, miss=1.46, falarm=9.38, confusion=2.84
#Eval for threshold 0.55 DER=12.65, miss=2.01, falarm=7.28, confusion=3.36
#Eval for threshold 0.6 DER=12.36, miss=2.84, falarm=6.20, confusion=3.31
#Eval for threshold 0.7 DER=13.73, miss=6.51, falarm=5.08, confusion=2.14
#Eval for threshold 0.8 DER=15.80, miss=10.41, falarm=4.07, confusion=1.33

#grep -r Eval logs/run_ts_vad2_hltsz_stage64.log
## collar=0.25
## dev of magicdata-ramc
#Eval for threshold 0.2 DER=12.05, miss=0.04, falarm=10.30, confusion=1.71
#Eval for threshold 0.3 DER=9.63, miss=0.07, falarm=7.62, confusion=1.95
#Eval for threshold 0.35 DER=8.77, miss=0.09, falarm=6.61, confusion=2.06
#Eval for threshold 0.4 DER=7.99, miss=0.13, falarm=5.66, confusion=2.20
#Eval for threshold 0.45 DER=7.28, miss=0.19, falarm=4.72, confusion=2.37
#Eval for threshold 0.5 DER=6.60, miss=0.30, falarm=3.70, confusion=2.60
#Eval for threshold 0.55 DER=6.00, miss=0.46, falarm=2.65, confusion=2.89
#Eval for threshold 0.6 DER=5.61, miss=0.85, falarm=1.56, confusion=3.19
#Eval for threshold 0.7 DER=6.99, miss=3.58, falarm=1.08, confusion=2.33
#Eval for threshold 0.8 DER=9.08, miss=6.29, falarm=0.94, confusion=1.85
#
## test of magicdata-ramc
#Eval for threshold 0.2 DER=11.99, miss=0.03, falarm=11.05, confusion=0.91
#Eval for threshold 0.3 DER=9.71, miss=0.07, falarm=8.43, confusion=1.21
#Eval for threshold 0.35 DER=8.95, miss=0.13, falarm=7.46, confusion=1.37
#Eval for threshold 0.4 DER=8.33, miss=0.21, falarm=6.60, confusion=1.52
#Eval for threshold 0.45 DER=7.76, miss=0.33, falarm=5.75, confusion=1.69
#Eval for threshold 0.5 DER=7.15, miss=0.51, falarm=4.56, confusion=2.09
#Eval for threshold 0.55 DER=6.33, miss=0.76, falarm=2.90, confusion=2.67
#Eval for threshold 0.6 DER=6.12, miss=1.20, falarm=2.20, confusion=2.72
#Eval for threshold 0.7 DER=7.60, miss=3.99, falarm=1.85, confusion=1.76
#Eval for threshold 0.8 DER=9.49, miss=6.84, falarm=1.56, confusion=1.08
fi

## 2025-2-8 
if [ ${stage} -le 65 ] && [ ${stop_stage} -ge 65 ];then
    . path_for_speaker_diarization_hltsz.sh
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
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_conformer_multi_backend_conformer_rs_len6
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=6
   segment_shift=2
   single_backend_type="conformer"
   multi_backend_type="conformer"
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17815 \
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
  
fi

if [ ${stage} -le 66 ] && [ ${stop_stage} -ge 66 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_conformer_multi_backend_conformer_rs_len6
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="conformer"
 multi_backend_type="conformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
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
 for c in $collar ;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_${name}_collar${c}
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
    --collar $c\
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
done
fi
#grep -r Eval logs/run_ts_vad2_hltsz_stage66_wo_nog0.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=18.79, miss=0.35, falarm=16.29, confusion=2.15
#Eval for threshold 0.3 DER=13.98, miss=0.63, falarm=10.10, confusion=3.25
#Eval for threshold 0.35 DER=12.85, miss=0.83, falarm=8.52, confusion=3.50
#Eval for threshold 0.4 DER=12.13, miss=1.08, falarm=7.39, confusion=3.66
#Eval for threshold 0.45 DER=11.63, miss=1.38, falarm=6.54, confusion=3.71
#Eval for threshold 0.5 DER=11.39, miss=1.83, falarm=5.85, confusion=3.71
#Eval for threshold 0.55 DER=11.52, miss=2.55, falarm=5.40, confusion=3.57
#Eval for threshold 0.6 DER=11.78, miss=3.42, falarm=5.00, confusion=3.37
#Eval for threshold 0.7 DER=13.16, miss=6.26, falarm=4.17, confusion=2.72
#Eval for threshold 0.8 DER=17.22, miss=12.28, falarm=3.27, confusion=1.68

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=19.65, miss=0.52, falarm=17.73, confusion=1.39
#Eval for threshold 0.3 DER=15.98, miss=1.01, falarm=13.08, confusion=1.89
#Eval for threshold 0.35 DER=14.94, miss=1.31, falarm=11.55, confusion=2.08
#Eval for threshold 0.4 DER=14.26, miss=1.71, falarm=10.36, confusion=2.19
#Eval for threshold 0.45 DER=13.78, miss=2.22, falarm=9.29, confusion=2.26
#Eval for threshold 0.5 DER=12.72, miss=3.05, falarm=6.79, confusion=2.88
#Eval for threshold 0.55 DER=13.64, miss=6.24, falarm=5.40, confusion=2.01
#Eval for threshold 0.6 DER=14.10, miss=7.31, falarm=4.94, confusion=1.85
#Eval for threshold 0.7 DER=15.86, miss=10.33, falarm=4.12, confusion=1.42
#Eval for threshold 0.8 DER=19.44, miss=15.30, falarm=3.27, confusion=0.87

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=10.96, miss=0.13, falarm=8.96, confusion=1.87
#Eval for threshold 0.3 DER=7.00, miss=0.23, falarm=3.85, confusion=2.92
#Eval for threshold 0.35 DER=6.19, miss=0.29, falarm=2.73, confusion=3.17
#Eval for threshold 0.4 DER=5.74, miss=0.40, falarm=2.04, confusion=3.31
#Eval for threshold 0.45 DER=5.43, miss=0.51, falarm=1.54, confusion=3.37
#Eval for threshold 0.5 DER=5.37, miss=0.72, falarm=1.25, confusion=3.40
#Eval for threshold 0.55 DER=5.63, miss=1.17, falarm=1.16, confusion=3.30
#Eval for threshold 0.6 DER=6.01, miss=1.78, falarm=1.10, confusion=3.13
#Eval for threshold 0.7 DER=7.45, miss=3.89, falarm=0.97, confusion=2.59
#Eval for threshold 0.8 DER=11.63, miss=9.21, falarm=0.84, confusion=1.58

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=11.46, miss=0.21, falarm=10.24, confusion=1.02
#Eval for threshold 0.3 DER=8.78, miss=0.41, falarm=6.97, confusion=1.40
#Eval for threshold 0.35 DER=8.13, miss=0.54, falarm=6.06, confusion=1.53
#Eval for threshold 0.4 DER=7.78, miss=0.73, falarm=5.44, confusion=1.61
#Eval for threshold 0.45 DER=7.59, miss=0.99, falarm=4.94, confusion=1.66
#Eval for threshold 0.5 DER=6.78, miss=1.45, falarm=3.01, confusion=2.33
#Eval for threshold 0.55 DER=7.82, miss=4.36, falarm=1.95, confusion=1.51
#Eval for threshold 0.6 DER=8.28, miss=5.05, falarm=1.81, confusion=1.41
#Eval for threshold 0.7 DER=9.92, miss=7.24, falarm=1.56, confusion=1.12
#Eval for threshold 0.8 DER=13.27, miss=11.25, falarm=1.30, confusion=0.72


if [ ${stage} -le 67 ] && [ ${stop_stage} -ge 67 ];then
    . path_for_speaker_diarization_hltsz.sh
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
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_conformer_multi_backend_transformer_rs_len6
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=6
   segment_shift=2
   single_backend_type="conformer"
   multi_backend_type="transformer"
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17815 \
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\

fi


if [ ${stage} -le 68 ] && [ ${stop_stage} -ge 68 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_conformer_multi_backend_transformer_rs_len6
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="conformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
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
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_${name}_collar${c}
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
    --collar $c\
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
done
fi
#grep -r Eval logs/run_ts_vad2_hltsz_stage67-68_wo_nog0.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=17.82, miss=0.32, falarm=15.04, confusion=2.46
#Eval for threshold 0.3 DER=13.49, miss=0.60, falarm=9.59, confusion=3.30
#Eval for threshold 0.35 DER=12.57, miss=0.79, falarm=8.25, confusion=3.52
#Eval for threshold 0.4 DER=11.95, miss=1.02, falarm=7.29, confusion=3.64
#Eval for threshold 0.45 DER=11.58, miss=1.31, falarm=6.57, confusion=3.70
#Eval for threshold 0.5 DER=11.41, miss=1.66, falarm=6.03, confusion=3.71
#Eval for threshold 0.55 DER=11.39, miss=2.16, falarm=5.62, confusion=3.61
#Eval for threshold 0.6 DER=11.51, miss=2.80, falarm=5.22, confusion=3.49
#Eval for threshold 0.7 DER=12.40, miss=4.92, falarm=4.42, confusion=3.06
#Eval for threshold 0.8 DER=15.71, miss=10.04, falarm=3.53, confusion=2.13

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=19.52, miss=0.49, falarm=17.63, confusion=1.39
#Eval for threshold 0.3 DER=16.33, miss=0.94, falarm=13.53, confusion=1.86
#Eval for threshold 0.35 DER=15.36, miss=1.23, falarm=12.11, confusion=2.03
#Eval for threshold 0.4 DER=14.62, miss=1.58, falarm=10.84, confusion=2.19
#Eval for threshold 0.45 DER=14.08, miss=2.02, falarm=9.76, confusion=2.30
#Eval for threshold 0.5 DER=12.31, miss=2.68, falarm=6.26, confusion=3.36
#Eval for threshold 0.55 DER=13.67, miss=5.98, falarm=5.55, confusion=2.14
#Eval for threshold 0.6 DER=14.13, miss=7.08, falarm=5.11, confusion=1.94
#Eval for threshold 0.7 DER=15.62, miss=9.94, falarm=4.21, confusion=1.47
#Eval for threshold 0.8 DER=18.64, miss=14.31, falarm=3.35, confusion=0.99

# dev of magicdata-rmac, collar=0.25
#Eval for threshold 0.2 DER=9.81, miss=0.10, falarm=7.50, confusion=2.20
#Eval for threshold 0.3 DER=6.38, miss=0.19, falarm=3.19, confusion=3.01
#Eval for threshold 0.35 DER=5.77, miss=0.25, falarm=2.31, confusion=3.21
#Eval for threshold 0.4 DER=5.41, miss=0.32, falarm=1.78, confusion=3.31
#Eval for threshold 0.45 DER=5.24, miss=0.42, falarm=1.46, confusion=3.35
#Eval for threshold 0.5 DER=5.18, miss=0.55, falarm=1.26, confusion=3.37
#Eval for threshold 0.55 DER=5.28, miss=0.80, falarm=1.17, confusion=3.31
#Eval for threshold 0.6 DER=5.49, miss=1.15, falarm=1.11, confusion=3.24
#Eval for threshold 0.7 DER=6.44, miss=2.55, falarm=0.99, confusion=2.90
#Eval for threshold 0.8 DER=9.76, miss=6.82, falarm=0.86, confusion=2.08

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=11.19, miss=0.18, falarm=9.98, confusion=1.03
#Eval for threshold 0.3 DER=9.01, miss=0.36, falarm=7.27, confusion=1.37
#Eval for threshold 0.35 DER=8.42, miss=0.48, falarm=6.44, confusion=1.50
#Eval for threshold 0.4 DER=7.96, miss=0.63, falarm=5.70, confusion=1.62
#Eval for threshold 0.45 DER=7.65, miss=0.83, falarm=5.11, confusion=1.71
#Eval for threshold 0.5 DER=6.19, miss=1.16, falarm=2.23, confusion=2.81
#Eval for threshold 0.55 DER=7.76, miss=4.15, falarm=1.99, confusion=1.62
#Eval for threshold 0.6 DER=8.21, miss=4.87, falarm=1.87, confusion=1.47
#Eval for threshold 0.7 DER=9.61, miss=6.87, falarm=1.58, confusion=1.15
#Eval for threshold 0.8 DER=12.40, miss=10.29, falarm=1.30, confusion=0.82


if [ ${stage} -le 69 ] && [ ${stop_stage} -ge 69 ];then
    . path_for_speaker_diarization_hltsz.sh
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
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_conformer_multi_backend_transformer_rs_len8
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=8
   segment_shift=2
   single_backend_type="conformer"
   multi_backend_type="transformer"
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17815 \
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\

fi
# note: 2 rtx3090(24GB) oom. so I run it on A100 on sribd

if [ ${stage} -le 70 ] && [ ${stop_stage} -ge 70 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_conformer_multi_backend_transformer_rs_len8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=1
 single_backend_type="conformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
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
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_${name}_collar${c}
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
    --collar $c\
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
done
fi


if [ ${stage} -le 71 ] && [ ${stop_stage} -ge 71 ];then
    . path_for_speaker_diarization_hltsz.sh
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
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=6
   segment_shift=2
   single_backend_type="transformer"
   multi_backend_type="transformer"
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17815 \
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\

fi

if [ ${stage} -le 72 ] && [ ${stop_stage} -ge 72 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
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
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_${name}_collar${c}
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
    --collar $c\
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
done
fi

# grep -r Eval  logs/run_ts_vad2_hltsz_stage71-72_wo_nog0.log
# dev of magicdata-ramc, collar=0.0
# Eval for threshold 0.2 DER=15.87, miss=0.26, falarm=13.07, confusion=2.54
#Eval for threshold 0.3 DER=13.21, miss=0.44, falarm=9.47, confusion=3.30
#Eval for threshold 0.35 DER=12.54, miss=0.57, falarm=8.45, confusion=3.52
#Eval for threshold 0.4 DER=12.05, miss=0.73, falarm=7.68, confusion=3.64
#Eval for threshold 0.45 DER=11.66, miss=0.95, falarm=7.01, confusion=3.70
#Eval for threshold 0.5 DER=11.39, miss=1.19, falarm=6.48, confusion=3.72
#Eval for threshold 0.55 DER=11.29, miss=1.60, falarm=6.06, confusion=3.63
#Eval for threshold 0.6 DER=11.26, miss=2.09, falarm=5.68, confusion=3.50
#Eval for threshold 0.7 DER=11.69, miss=3.63, falarm=4.94, confusion=3.12
#Eval for threshold 0.8 DER=13.35, miss=6.94, falarm=4.08, confusion=2.34

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=18.03, miss=0.35, falarm=16.03, confusion=1.66
#Eval for threshold 0.3 DER=15.67, miss=0.67, falarm=13.03, confusion=1.97
#Eval for threshold 0.35 DER=14.89, miss=0.88, falarm=11.91, confusion=2.11
#Eval for threshold 0.4 DER=14.30, miss=1.13, falarm=10.96, confusion=2.21
#Eval for threshold 0.45 DER=13.82, miss=1.47, falarm=10.05, confusion=2.30
#Eval for threshold 0.5 DER=12.22, miss=1.99, falarm=6.98, confusion=3.25
#Eval for threshold 0.55 DER=13.19, miss=5.00, falarm=6.02, confusion=2.17
#Eval for threshold 0.6 DER=13.40, miss=5.82, falarm=5.58, confusion=1.99
#Eval for threshold 0.7 DER=14.31, miss=7.89, falarm=4.75, confusion=1.67
#Eval for threshold 0.8 DER=16.37, miss=11.26, falarm=3.83, confusion=1.28

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=7.85, miss=0.07, falarm=5.55, confusion=2.23
#Eval for threshold 0.3 DER=5.92, miss=0.12, falarm=2.85, confusion=2.95
#Eval for threshold 0.35 DER=5.54, miss=0.16, falarm=2.21, confusion=3.18
#Eval for threshold 0.4 DER=5.31, miss=0.20, falarm=1.83, confusion=3.27
#Eval for threshold 0.45 DER=5.17, miss=0.28, falarm=1.56, confusion=3.33
#Eval for threshold 0.5 DER=5.10, miss=0.36, falarm=1.37, confusion=3.37
#Eval for threshold 0.55 DER=5.12, miss=0.54, falarm=1.27, confusion=3.31
#Eval for threshold 0.6 DER=5.18, miss=0.77, falarm=1.19, confusion=3.22
#Eval for threshold 0.7 DER=5.69, miss=1.69, falarm=1.07, confusion=2.92
#Eval for threshold 0.8 DER=7.41, miss=4.24, falarm=0.95, confusion=2.21

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=9.74, miss=0.10, falarm=8.40, confusion=1.24
#Eval for threshold 0.3 DER=8.24, miss=0.20, falarm=6.59, confusion=1.45
#Eval for threshold 0.35 DER=7.80, miss=0.28, falarm=5.98, confusion=1.54
#Eval for threshold 0.4 DER=7.51, miss=0.38, falarm=5.53, confusion=1.60
#Eval for threshold 0.45 DER=7.32, miss=0.53, falarm=5.13, confusion=1.66
#Eval for threshold 0.5 DER=6.07, miss=0.79, falarm=2.64, confusion=2.63
#Eval for threshold 0.55 DER=7.27, miss=3.53, falarm=2.15, confusion=1.59
#Eval for threshold 0.6 DER=7.52, miss=4.00, falarm=2.03, confusion=1.49
#Eval for threshold 0.7 DER=8.37, miss=5.27, falarm=1.79, confusion=1.31
#Eval for threshold 0.8 DER=10.24, miss=7.66, falarm=1.50, confusion=1.08

if [ ${stage} -le 73 ] && [ ${stop_stage} -ge 73 ];then
    . path_for_speaker_diarization_hltsz.sh
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
    #speech_encoder_type="CAM++"
    #speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    speech_encoder_type="ReDimNetB2"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_redimnetb2_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_cam++_200k_zh_cn_speaker_embedding_lr3e-4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=6
   segment_shift=2
   single_backend_type="transformer"
   multi_backend_type="transformer"
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17915 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 3e-4\
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
    --num-transformer-layer $num_transformer_layer\

fi
  # two rtx 3090(24GB), oom. run it on sribd
if [ ${stage} -le 74 ] && [ ${stop_stage} -ge 74 ];then
 #exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_redimnetb2_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_cam++_200k_zh_cn_speaker_embedding_lr3e-4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 #speech_encoder_type="CAM++"
 #speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 speech_encoder_type="ReDimNetB2"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_${name}_collar${c}
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
    --collar $c\
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
done
fi


if [ ${stage} -le 75 ] && [ ${stop_stage} -ge 75 ];then
    . path_for_speaker_diarization_hltsz.sh
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
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_max_num_speakers2
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=6
   segment_shift=2
   single_backend_type="transformer"
   multi_backend_type="transformer"
   num_transformer_layer=2
   max_num_speaker=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17815 \
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --max-num-speaker $max_num_speaker

fi

if [ ${stage} -le 76 ] && [ ${stop_stage} -ge 76 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_max_num_speakers2
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 max_num_speaker=2
 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_${name}_collar${c}
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
    --collar $c\
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
    --num-transformer-layer $num_transformer_layer\
    --max-num-speaker $max_num_speaker
 done
done
fi


if [ ${stage} -le 77 ] && [ ${stop_stage} -ge 77 ];then
    . path_for_speaker_diarization_hltsz.sh
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
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_variable_num_speaker
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=6
   segment_shift=2
   single_backend_type="transformer"
   multi_backend_type="transformer"
   num_transformer_layer=2
   #max_num_speaker=2
   support_variable_number_speakers=true
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17815 \
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --support-variable-number-speakers $support_variable_number_speakers

fi

if [ ${stage} -le 78 ] && [ ${stop_stage} -ge 78 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_variable_num_speaker
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 #max_num_speaker=2
 support_variable_number_speakers=true
 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_${name}_collar${c}
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
    --collar $c\
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
    --num-transformer-layer $num_transformer_layer\
    --support-variable-number-speakers $support_variable_number_speakers
 done
done
fi


if [ ${stage} -le 90 ] && [ ${stop_stage} -ge 90 ];then
  echo "get target audio and json label file from rttm file"
  datasets="dev test train"
  #datasets="dev"
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

if [ ${stage} -le 91 ] && [ ${stop_stage} -ge 91 ];then
   echo "generate oracle vad speaker embedding"
   dest_dir=/data/maduo/model_hub
   feature_name=cam++_en_zh_advanced_feature_dir
   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
   #model_id=iic/speech_campplus_sv_zh-cn_16k-common
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   subsets="dev test train"
   #subsets="dev"
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


if [ ${stage} -le 92 ] && [ ${stop_stage} -ge 92 ];then
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
    #speech_encoder_type="CAM++"
    #speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_sv_zh_en_16k-common_advanced_w2v-bert2.0_epoch40_front_fix_seed_lr1e4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 13815 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 40\
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
    --speech-encoder-config $speech_encoder_config\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name
fi

if [ ${stage} -le 93 ] && [ ${stop_stage} -ge 93 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_sv_zh_en_16k-common_advanced_w2v-bert2.0_epoch40_front_fix_seed_lr1e4
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
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 #speech_encoder_type="CAM++"
 #speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

 data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
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
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name
  done
 done
fi
#grep -r 'Eval' logs/run_ts_vad2_hltsz_stage92-93.log
# collar=0.0
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=17.32, miss=0.33, falarm=14.78, confusion=2.22
#Eval for threshold 0.3 DER=14.87, miss=0.64, falarm=11.74, confusion=2.50
#Eval for threshold 0.35 DER=13.53, miss=0.85, falarm=9.86, confusion=2.82
#Eval for threshold 0.4 DER=12.24, miss=1.11, falarm=7.82, confusion=3.30
#Eval for threshold 0.45 DER=11.45, miss=1.44, falarm=6.43, confusion=3.57
#Eval for threshold 0.5 DER=11.18, miss=1.88, falarm=5.69, confusion=3.61
#Eval for threshold 0.55 DER=11.20, miss=2.53, falarm=5.21, confusion=3.46
#Eval for threshold 0.6 DER=11.51, miss=3.58, falarm=4.74, confusion=3.19
#Eval for threshold 0.7 DER=13.54, miss=7.44, falarm=3.89, confusion=2.21
#Eval for threshold 0.8 DER=15.93, miss=11.24, falarm=3.01, confusion=1.68

# test of magicdata-ramc
#Eval for threshold 0.2 DER=17.60, miss=0.35, falarm=15.37, confusion=1.88
#Eval for threshold 0.3 DER=15.34, miss=0.76, falarm=12.29, confusion=2.29
#Eval for threshold 0.35 DER=14.62, miss=1.05, falarm=11.12, confusion=2.46
#Eval for threshold 0.4 DER=13.86, miss=1.45, falarm=9.70, confusion=2.71
#Eval for threshold 0.45 DER=12.97, miss=1.96, falarm=7.90, confusion=3.11
#Eval for threshold 0.5 DER=12.21, miss=2.83, falarm=5.92, confusion=3.46
#Eval for threshold 0.55 DER=12.96, miss=4.96, falarm=5.27, confusion=2.72
#Eval for threshold 0.6 DER=13.59, miss=6.51, falarm=4.79, confusion=2.29
#Eval for threshold 0.7 DER=14.93, miss=9.24, falarm=3.92, confusion=1.77
#Eval for threshold 0.8 DER=17.46, miss=13.05, falarm=3.05, confusion=1.36

# collar=0.25
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=9.55, miss=0.06, falarm=7.63, confusion=1.87
#Eval for threshold 0.3 DER=7.97, miss=0.14, falarm=5.75, confusion=2.08
#Eval for threshold 0.35 DER=6.93, miss=0.20, falarm=4.35, confusion=2.39
#Eval for threshold 0.4 DER=5.87, miss=0.28, falarm=2.67, confusion=2.92
#Eval for threshold 0.45 DER=5.26, miss=0.39, falarm=1.64, confusion=3.24
#Eval for threshold 0.5 DER=5.15, miss=0.53, falarm=1.29, confusion=3.33
#Eval for threshold 0.55 DER=5.27, miss=0.85, falarm=1.18, confusion=3.24
#Eval for threshold 0.6 DER=5.65, miss=1.52, falarm=1.11, confusion=3.01
#Eval for threshold 0.7 DER=7.80, miss=4.76, falarm=0.97, confusion=2.07
#Eval for threshold 0.8 DER=10.08, miss=7.65, falarm=0.83, confusion=1.61

# test of magicdata-ramc
#Eval for threshold 0.2 DER=9.61, miss=0.05, falarm=8.11, confusion=1.45
#Eval for threshold 0.3 DER=8.38, miss=0.20, falarm=6.44, confusion=1.74
#Eval for threshold 0.35 DER=8.02, miss=0.31, falarm=5.84, confusion=1.87
#Eval for threshold 0.4 DER=7.63, miss=0.46, falarm=5.05, confusion=2.11
#Eval for threshold 0.45 DER=6.99, miss=0.67, falarm=3.79, confusion=2.54
#Eval for threshold 0.5 DER=6.34, miss=1.14, falarm=2.22, confusion=2.98
#Eval for threshold 0.55 DER=7.21, miss=2.90, falarm=2.00, confusion=2.31
#Eval for threshold 0.6 DER=7.84, miss=4.04, falarm=1.87, confusion=1.94
#Eval for threshold 0.7 DER=9.00, miss=5.84, falarm=1.63, confusion=1.53
#Eval for threshold 0.8 DER=11.08, miss=8.46, falarm=1.37, confusion=1.25



if [ ${stage} -le 99 ] && [ ${stop_stage} -ge 99 ];then
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
    #speech_encoder_type="CAM++"
    #speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh-cn_200k_w2v-bert2.0_epoch40_front_fix_seed_lr1e4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 14815 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 40\
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
    --speech-encoder-config $speech_encoder_config\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name
fi

if [ ${stage} -le 100 ] && [ ${stop_stage} -ge 100 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh-cn_200k_w2v-bert2.0_epoch40_front_fix_seed_lr1e4
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
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 #speech_encoder_type="CAM++"
 #speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
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
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name
  done
 done
fi
# cat /data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh-cn_200k_w2v-bert2.0_epoch40_front_fix_seed_lr1e4/magicdata-ramc_collar*/*/der_result
#grep -r Eval logs/run_ts_vad2_hltsz_stage100_cam++200k_w2v-bert2.log
# collar=0.0
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=15.78, miss=0.23, falarm=12.86, confusion=2.69
#Eval for threshold 0.3 DER=13.09, miss=0.42, falarm=9.18, confusion=3.49
#Eval for threshold 0.35 DER=12.44, miss=0.58, falarm=8.26, confusion=3.60
#Eval for threshold 0.4 DER=11.96, miss=0.78, falarm=7.53, confusion=3.65
#Eval for threshold 0.45 DER=11.58, miss=1.04, falarm=6.88, confusion=3.66
#Eval for threshold 0.5 DER=11.35, miss=1.38, falarm=6.34, confusion=3.62
#Eval for threshold 0.55 DER=11.22, miss=1.81, falarm=5.88, confusion=3.54
#Eval for threshold 0.6 DER=11.16, miss=2.34, falarm=5.39, confusion=3.43
#Eval for threshold 0.7 DER=11.38, miss=3.70, falarm=4.49, confusion=3.19
#Eval for threshold 0.8 DER=12.96, miss=6.86, falarm=3.59, confusion=2.51

# test of magicdata-ramc
#Eval for threshold 0.2 DER=17.93, miss=0.25, falarm=15.69, confusion=1.99
#Eval for threshold 0.3 DER=15.77, miss=0.54, falarm=12.78, confusion=2.45
#Eval for threshold 0.35 DER=15.02, miss=0.77, falarm=11.66, confusion=2.59
#Eval for threshold 0.4 DER=14.37, miss=1.09, falarm=10.56, confusion=2.72
#Eval for threshold 0.45 DER=13.44, miss=1.51, falarm=8.76, confusion=3.17
#Eval for threshold 0.5 DER=12.34, miss=2.14, falarm=6.45, confusion=3.76
#Eval for threshold 0.55 DER=12.98, miss=4.25, falarm=5.77, confusion=2.96
#Eval for threshold 0.6 DER=13.44, miss=5.69, falarm=5.28, confusion=2.47
#Eval for threshold 0.7 DER=14.27, miss=7.79, falarm=4.41, confusion=2.06
#Eval for threshold 0.8 DER=15.99, miss=10.83, falarm=3.55, confusion=1.61

# collar=0.25
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=7.49, miss=0.05, falarm=4.99, confusion=2.45
#Eval for threshold 0.3 DER=5.69, miss=0.09, falarm=2.40, confusion=3.20
#Eval for threshold 0.35 DER=5.39, miss=0.14, falarm=1.96, confusion=3.28
#Eval for threshold 0.4 DER=5.22, miss=0.20, falarm=1.70, confusion=3.32
#Eval for threshold 0.45 DER=5.12, miss=0.29, falarm=1.51, confusion=3.32
#Eval for threshold 0.5 DER=5.09, miss=0.41, falarm=1.36, confusion=3.32
#Eval for threshold 0.55 DER=5.13, miss=0.58, falarm=1.28, confusion=3.27
#Eval for threshold 0.6 DER=5.22, miss=0.77, falarm=1.21, confusion=3.23
#Eval for threshold 0.7 DER=5.56, miss=1.39, falarm=1.07, confusion=3.10
#Eval for threshold 0.8 DER=7.02, miss=3.59, falarm=0.93, confusion=2.51

# test of magicdata-ramc
#Eval for threshold 0.2 DER=9.56, miss=0.03, falarm=7.91, confusion=1.62
#Eval for threshold 0.3 DER=8.48, miss=0.11, falarm=6.41, confusion=1.95
#Eval for threshold 0.35 DER=8.13, miss=0.20, falarm=5.87, confusion=2.06
#Eval for threshold 0.4 DER=7.83, miss=0.33, falarm=5.33, confusion=2.17
#Eval for threshold 0.45 DER=7.29, miss=0.51, falarm=4.17, confusion=2.61
#Eval for threshold 0.5 DER=6.41, miss=0.80, falarm=2.33, confusion=3.28
#Eval for threshold 0.55 DER=7.22, miss=2.57, falarm=2.10, confusion=2.55
#Eval for threshold 0.6 DER=7.75, miss=3.66, falarm=1.99, confusion=2.10
#Eval for threshold 0.7 DER=8.49, miss=4.91, falarm=1.75, confusion=1.83
#Eval for threshold 0.8 DER=9.80, miss=6.78, falarm=1.51, confusion=1.51






if [ ${stage} -le 120 ] && [ ${stop_stage} -ge 120 ];then
    . path_for_speaker_diarization_hltsz.sh
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
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len8_shift0.8
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=8
   segment_shift=0.8
   single_backend_type="transformer"
   multi_backend_type="transformer"
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17815 \
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\

fi

if [ ${stage} -le 121 ] && [ ${stop_stage} -ge 121 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len8_shift0.8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=0.8
 # rs_len=8
 # segment_shift=0.4
 # rs_len=8
 # segment_shift=1

 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test cssd_testset"
 #infer_sets="Test"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 data_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format"
 for c in $collar;do
  for name in $infer_sets;do
      results_path=$exp_dir/${dataset_name}_${name}_collar${c}_infer_rs_len${rs_len}_shift_${segment_shift}
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
    --collar $c\
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
done
fi
# rs_len=8,segment_shift=0.8
# grep -r Eval logs/run_ts_vad2_hltsz_stage121_rs_len8_shift0.8.log
# # dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=16.27, miss=0.19, falarm=13.65, confusion=2.42
#Eval for threshold 0.3 DER=13.62, miss=0.37, falarm=10.11, confusion=3.14
#Eval for threshold 0.35 DER=12.87, miss=0.50, falarm=8.88, confusion=3.49
#Eval for threshold 0.4 DER=12.32, miss=0.68, falarm=7.97, confusion=3.67
#Eval for threshold 0.45 DER=11.94, miss=0.91, falarm=7.22, confusion=3.81
#Eval for threshold 0.5 DER=11.66, miss=1.21, falarm=6.60, confusion=3.85
#Eval for threshold 0.55 DER=11.54, miss=1.61, falarm=6.16, confusion=3.77
#Eval for threshold 0.6 DER=11.51, miss=2.17, falarm=5.75, confusion=3.58
#Eval for threshold 0.7 DER=11.80, miss=3.79, falarm=4.94, confusion=3.07
#Eval for threshold 0.8 DER=13.19, miss=6.88, falarm=4.06, confusion=2.25
#Eval for threshold 0.9 DER=16.10, miss=11.45, falarm=2.96, confusion=1.69

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=18.23, miss=0.24, falarm=15.97, confusion=2.02
#Eval for threshold 0.3 DER=16.09, miss=0.53, falarm=13.13, confusion=2.43
#Eval for threshold 0.35 DER=15.35, miss=0.76, falarm=12.01, confusion=2.58
#Eval for threshold 0.4 DER=14.73, miss=1.03, falarm=11.01, confusion=2.69
#Eval for threshold 0.45 DER=13.46, miss=1.41, falarm=8.65, confusion=3.40
#Eval for threshold 0.5 DER=12.61, miss=2.22, falarm=6.75, confusion=3.64
#Eval for threshold 0.55 DER=13.24, miss=4.20, falarm=6.08, confusion=2.97
#Eval for threshold 0.6 DER=13.76, miss=5.78, falarm=5.61, confusion=2.37
#Eval for threshold 0.7 DER=14.44, miss=7.80, falarm=4.71, confusion=1.93
#Eval for threshold 0.8 DER=16.07, miss=10.79, falarm=3.80, confusion=1.47
#Eval for threshold 0.9 DER=20.25, miss=16.62, falarm=2.74, confusion=0.90

# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=29.54, miss=2.87, falarm=24.65, confusion=2.03
#Eval for threshold 0.3 DER=25.04, miss=3.88, falarm=18.04, confusion=3.12
#Eval for threshold 0.35 DER=23.40, miss=4.41, falarm=15.37, confusion=3.62
#Eval for threshold 0.4 DER=22.05, miss=5.07, falarm=12.90, confusion=4.08
#Eval for threshold 0.45 DER=21.11, miss=5.88, falarm=10.79, confusion=4.44
#Eval for threshold 0.5 DER=20.77, miss=7.22, falarm=9.16, confusion=4.39
#Eval for threshold 0.55 DER=21.19, miss=9.22, falarm=8.21, confusion=3.76
#Eval for threshold 0.6 DER=21.80, miss=11.37, falarm=7.35, confusion=3.09
#Eval for threshold 0.7 DER=23.60, miss=15.91, falarm=5.72, confusion=1.97
#Eval for threshold 0.8 DER=26.60, miss=21.47, falarm=4.07, confusion=1.05
#Eval for threshold 0.9 DER=32.63, miss=29.93, falarm=2.28, confusion=0.42

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=8.11, miss=0.04, falarm=6.03, confusion=2.03
#Eval for threshold 0.3 DER=6.23, miss=0.09, falarm=3.44, confusion=2.70
#Eval for threshold 0.35 DER=5.78, miss=0.14, falarm=2.60, confusion=3.04
#Eval for threshold 0.4 DER=5.47, miss=0.21, falarm=2.05, confusion=3.21
#Eval for threshold 0.45 DER=5.32, miss=0.29, falarm=1.66, confusion=3.37
#Eval for threshold 0.5 DER=5.24, miss=0.40, falarm=1.40, confusion=3.44
#Eval for threshold 0.55 DER=5.26, miss=0.56, falarm=1.28, confusion=3.41
#Eval for threshold 0.6 DER=5.36, miss=0.86, falarm=1.21, confusion=3.29
#Eval for threshold 0.7 DER=5.80, miss=1.84, falarm=1.08, confusion=2.88
#Eval for threshold 0.8 DER=7.29, miss=4.22, falarm=0.96, confusion=2.11
#Eval for threshold 0.9 DER=9.98, miss=7.54, falarm=0.79, confusion=1.65

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=9.73, miss=0.05, falarm=8.14, confusion=1.54
#Eval for threshold 0.3 DER=8.56, miss=0.16, falarm=6.61, confusion=1.80
#Eval for threshold 0.35 DER=8.18, miss=0.25, falarm=6.02, confusion=1.91
#Eval for threshold 0.4 DER=7.89, miss=0.37, falarm=5.54, confusion=1.98
#Eval for threshold 0.45 DER=6.96, miss=0.53, falarm=3.75, confusion=2.69
#Eval for threshold 0.5 DER=6.35, miss=1.01, falarm=2.34, confusion=2.99
#Eval for threshold 0.55 DER=7.17, miss=2.69, falarm=2.10, confusion=2.38
#Eval for threshold 0.6 DER=7.80, miss=3.97, falarm=1.99, confusion=1.85
#Eval for threshold 0.7 DER=8.46, miss=5.15, falarm=1.74, confusion=1.58
#Eval for threshold 0.8 DER=9.87, miss=7.13, falarm=1.48, confusion=1.26
#Eval for threshold 0.9 DER=13.54, miss=11.56, falarm=1.16, confusion=0.82

# cssd_testset of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=10.74, miss=0.86, falarm=9.01, confusion=0.87
#Eval for threshold 0.3 DER=8.16, miss=1.24, falarm=5.39, confusion=1.53
#Eval for threshold 0.35 DER=7.25, miss=1.42, falarm=3.92, confusion=1.91
#Eval for threshold 0.4 DER=6.50, miss=1.64, falarm=2.57, confusion=2.29
#Eval for threshold 0.45 DER=6.09, miss=1.98, falarm=1.46, confusion=2.65
#Eval for threshold 0.5 DER=6.08, miss=2.67, falarm=0.71, confusion=2.70
#Eval for threshold 0.55 DER=6.68, miss=3.90, falarm=0.52, confusion=2.26
#Eval for threshold 0.6 DER=7.48, miss=5.30, falarm=0.42, confusion=1.76
#Eval for threshold 0.7 DER=9.52, miss=8.18, falarm=0.29, confusion=1.06
#Eval for threshold 0.8 DER=12.57, miss=11.89, falarm=0.17, confusion=0.51
#Eval for threshold 0.9 DER=18.38, miss=18.08, falarm=0.08, confusion=0.22


# rs_len=8, shift=0.4
#grep -r Eval logs/run_ts_vad2_hltsz_stage121_rs_len8_shift0.4.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=16.27, miss=0.19, falarm=13.66, confusion=2.43
#Eval for threshold 0.3 DER=13.62, miss=0.37, falarm=10.12, confusion=3.13
#Eval for threshold 0.35 DER=12.86, miss=0.50, falarm=8.87, confusion=3.48
#Eval for threshold 0.4 DER=12.35, miss=0.68, falarm=8.00, confusion=3.67
#Eval for threshold 0.45 DER=11.94, miss=0.92, falarm=7.21, confusion=3.81
#Eval for threshold 0.5 DER=11.68, miss=1.21, falarm=6.61, confusion=3.85
#Eval for threshold 0.55 DER=11.54, miss=1.62, falarm=6.15, confusion=3.77
#Eval for threshold 0.6 DER=11.52, miss=2.18, falarm=5.75, confusion=3.58
#Eval for threshold 0.7 DER=11.81, miss=3.79, falarm=4.96, confusion=3.07
#Eval for threshold 0.8 DER=13.20, miss=6.87, falarm=4.07, confusion=2.25
#Eval for threshold 0.9 DER=16.08, miss=11.42, falarm=2.98, confusion=1.68

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=18.19, miss=0.24, falarm=15.94, confusion=2.01
#Eval for threshold 0.3 DER=16.07, miss=0.54, falarm=13.10, confusion=2.44
#Eval for threshold 0.35 DER=15.34, miss=0.75, falarm=12.00, confusion=2.59
#Eval for threshold 0.4 DER=14.72, miss=1.03, falarm=11.01, confusion=2.69
#Eval for threshold 0.45 DER=13.47, miss=1.41, falarm=8.64, confusion=3.42
#Eval for threshold 0.5 DER=12.60, miss=2.21, falarm=6.75, confusion=3.64
#Eval for threshold 0.55 DER=13.25, miss=4.22, falarm=6.07, confusion=2.96
#Eval for threshold 0.6 DER=13.76, miss=5.78, falarm=5.61, confusion=2.37
#Eval for threshold 0.7 DER=14.45, miss=7.80, falarm=4.71, confusion=1.94
#Eval for threshold 0.8 DER=16.05, miss=10.77, falarm=3.80, confusion=1.48
#Eval for threshold 0.9 DER=20.21, miss=16.58, falarm=2.74, confusion=0.89


# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=29.55, miss=2.87, falarm=24.65, confusion=2.03
#Eval for threshold 0.3 DER=25.04, miss=3.86, falarm=18.06, confusion=3.12
#Eval for threshold 0.35 DER=23.40, miss=4.43, falarm=15.37, confusion=3.61
#Eval for threshold 0.4 DER=22.04, miss=5.05, falarm=12.93, confusion=4.07
#Eval for threshold 0.45 DER=21.12, miss=5.88, falarm=10.80, confusion=4.43
#Eval for threshold 0.5 DER=20.74, miss=7.18, falarm=9.15, confusion=4.41
#Eval for threshold 0.55 DER=21.17, miss=9.17, falarm=8.22, confusion=3.78
#Eval for threshold 0.6 DER=21.81, miss=11.37, falarm=7.36, confusion=3.07
#Eval for threshold 0.7 DER=23.60, miss=15.91, falarm=5.73, confusion=1.96
#Eval for threshold 0.8 DER=26.59, miss=21.46, falarm=4.07, confusion=1.05
#Eval for threshold 0.9 DER=32.56, miss=29.87, falarm=2.28, confusion=0.41

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=8.12, miss=0.04, falarm=6.04, confusion=2.04
#Eval for threshold 0.3 DER=6.23, miss=0.10, falarm=3.45, confusion=2.68
#Eval for threshold 0.35 DER=5.76, miss=0.15, falarm=2.59, confusion=3.03
#Eval for threshold 0.4 DER=5.48, miss=0.20, falarm=2.07, confusion=3.21
#Eval for threshold 0.45 DER=5.33, miss=0.30, falarm=1.66, confusion=3.37
#Eval for threshold 0.5 DER=5.25, miss=0.40, falarm=1.40, confusion=3.45
#Eval for threshold 0.55 DER=5.26, miss=0.57, falarm=1.28, confusion=3.41
#Eval for threshold 0.6 DER=5.37, miss=0.86, falarm=1.22, confusion=3.30
#Eval for threshold 0.7 DER=5.81, miss=1.86, falarm=1.09, confusion=2.86
#Eval for threshold 0.8 DER=7.31, miss=4.24, falarm=0.96, confusion=2.11
#Eval for threshold 0.9 DER=10.00, miss=7.57, falarm=0.79, confusion=1.64

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=9.70, miss=0.05, falarm=8.12, confusion=1.53
#Eval for threshold 0.3 DER=8.54, miss=0.16, falarm=6.58, confusion=1.81
#Eval for threshold 0.35 DER=8.18, miss=0.25, falarm=6.03, confusion=1.91
#Eval for threshold 0.4 DER=7.88, miss=0.36, falarm=5.54, confusion=1.99
#Eval for threshold 0.45 DER=6.97, miss=0.52, falarm=3.74, confusion=2.71
#Eval for threshold 0.5 DER=6.35, miss=1.01, falarm=2.35, confusion=2.99
#Eval for threshold 0.55 DER=7.19, miss=2.70, falarm=2.11, confusion=2.38
#Eval for threshold 0.6 DER=7.80, miss=3.97, falarm=1.99, confusion=1.85
#Eval for threshold 0.7 DER=8.46, miss=5.16, falarm=1.73, confusion=1.57
#Eval for threshold 0.8 DER=9.87, miss=7.13, falarm=1.48, confusion=1.26
#Eval for threshold 0.9 DER=13.53, miss=11.55, falarm=1.16, confusion=0.81

# cssd_testset of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=10.76, miss=0.86, falarm=9.05, confusion=0.86
#Eval for threshold 0.3 DER=8.16, miss=1.23, falarm=5.39, confusion=1.53
#Eval for threshold 0.35 DER=7.27, miss=1.43, falarm=3.94, confusion=1.90
#Eval for threshold 0.4 DER=6.51, miss=1.64, falarm=2.60, confusion=2.27
#Eval for threshold 0.45 DER=6.10, miss=1.99, falarm=1.46, confusion=2.65
#Eval for threshold 0.5 DER=6.07, miss=2.65, falarm=0.69, confusion=2.72
#Eval for threshold 0.55 DER=6.68, miss=3.89, falarm=0.52, confusion=2.28
#Eval for threshold 0.6 DER=7.53, miss=5.34, falarm=0.42, confusion=1.76
#Eval for threshold 0.7 DER=9.53, miss=8.19, falarm=0.29, confusion=1.05
#Eval for threshold 0.8 DER=12.56, miss=11.88, falarm=0.17, confusion=0.51
#Eval for threshold 0.9 DER=18.37, miss=18.07, falarm=0.08, confusion=0.21


# rs_len=8, shift=1,
#grep -r Eval logs/run_ts_vad2_hltsz_stage121_rs_len8_shift1.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=16.32, miss=0.19, falarm=13.68, confusion=2.45
#Eval for threshold 0.3 DER=13.64, miss=0.37, falarm=10.11, confusion=3.16
#Eval for threshold 0.35 DER=12.88, miss=0.52, falarm=8.89, confusion=3.47
#Eval for threshold 0.4 DER=12.36, miss=0.67, falarm=8.00, confusion=3.68
#Eval for threshold 0.45 DER=11.96, miss=0.92, falarm=7.24, confusion=3.80
#Eval for threshold 0.5 DER=11.68, miss=1.23, falarm=6.60, confusion=3.85
#Eval for threshold 0.55 DER=11.58, miss=1.66, falarm=6.16, confusion=3.75
#Eval for threshold 0.6 DER=11.54, miss=2.24, falarm=5.75, confusion=3.56
#Eval for threshold 0.7 DER=11.81, miss=3.82, falarm=4.94, confusion=3.05
#Eval for threshold 0.8 DER=13.26, miss=6.92, falarm=4.06, confusion=2.28
#Eval for threshold 0.9 DER=16.20, miss=11.57, falarm=2.94, confusion=1.70

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=18.19, miss=0.24, falarm=15.91, confusion=2.04
#Eval for threshold 0.3 DER=16.09, miss=0.54, falarm=13.11, confusion=2.44
#Eval for threshold 0.35 DER=15.35, miss=0.76, falarm=12.00, confusion=2.59
#Eval for threshold 0.4 DER=14.74, miss=1.03, falarm=11.01, confusion=2.70
#Eval for threshold 0.45 DER=13.48, miss=1.42, falarm=8.63, confusion=3.43
#Eval for threshold 0.5 DER=12.59, miss=2.20, falarm=6.74, confusion=3.66
#Eval for threshold 0.55 DER=13.25, miss=4.21, falarm=6.08, confusion=2.97
#Eval for threshold 0.6 DER=13.77, miss=5.78, falarm=5.60, confusion=2.38
#Eval for threshold 0.7 DER=14.46, miss=7.81, falarm=4.71, confusion=1.94
#Eval for threshold 0.8 DER=16.06, miss=10.77, falarm=3.80, confusion=1.48
#Eval for threshold 0.9 DER=20.27, miss=16.63, falarm=2.73, confusion=0.91

# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=29.60, miss=2.88, falarm=24.68, confusion=2.05
#Eval for threshold 0.3 DER=25.04, miss=3.86, falarm=18.07, confusion=3.12
#Eval for threshold 0.35 DER=23.37, miss=4.42, falarm=15.33, confusion=3.61
#Eval for threshold 0.4 DER=22.06, miss=5.06, falarm=12.89, confusion=4.10
#Eval for threshold 0.45 DER=21.10, miss=5.90, falarm=10.75, confusion=4.45
#Eval for threshold 0.5 DER=20.75, miss=7.24, falarm=9.14, confusion=4.38
#Eval for threshold 0.55 DER=21.17, miss=9.18, falarm=8.21, confusion=3.78
#Eval for threshold 0.6 DER=21.83, miss=11.37, falarm=7.37, confusion=3.09
#Eval for threshold 0.7 DER=23.65, miss=15.95, falarm=5.73, confusion=1.97
#Eval for threshold 0.8 DER=26.61, miss=21.48, falarm=4.07, confusion=1.06
#Eval for threshold 0.9 DER=32.65, miss=29.95, falarm=2.28, confusion=0.41

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=8.13, miss=0.04, falarm=6.04, confusion=2.06
#Eval for threshold 0.3 DER=6.25, miss=0.10, falarm=3.45, confusion=2.71
#Eval for threshold 0.35 DER=5.76, miss=0.15, falarm=2.60, confusion=3.01
#Eval for threshold 0.4 DER=5.51, miss=0.21, falarm=2.09, confusion=3.20
#Eval for threshold 0.45 DER=5.34, miss=0.30, falarm=1.70, confusion=3.35
#Eval for threshold 0.5 DER=5.26, miss=0.40, falarm=1.42, confusion=3.44
#Eval for threshold 0.55 DER=5.29, miss=0.60, falarm=1.29, confusion=3.40
#Eval for threshold 0.6 DER=5.37, miss=0.90, falarm=1.21, confusion=3.26
#Eval for threshold 0.7 DER=5.82, miss=1.91, falarm=1.08, confusion=2.84
#Eval for threshold 0.8 DER=7.35, miss=4.25, falarm=0.96, confusion=2.14
#Eval for threshold 0.9 DER=10.09, miss=7.64, falarm=0.79, confusion=1.66

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=9.71, miss=0.05, falarm=8.12, confusion=1.54
#Eval for threshold 0.3 DER=8.55, miss=0.16, falarm=6.59, confusion=1.80
#Eval for threshold 0.35 DER=8.19, miss=0.24, falarm=6.04, confusion=1.91
#Eval for threshold 0.4 DER=7.90, miss=0.36, falarm=5.55, confusion=1.99
#Eval for threshold 0.45 DER=6.98, miss=0.53, falarm=3.73, confusion=2.72
#Eval for threshold 0.5 DER=6.34, miss=0.99, falarm=2.35, confusion=3.00
#Eval for threshold 0.55 DER=7.18, miss=2.68, falarm=2.11, confusion=2.39
#Eval for threshold 0.6 DER=7.79, miss=3.95, falarm=1.98, confusion=1.86
#Eval for threshold 0.7 DER=8.48, miss=5.16, falarm=1.74, confusion=1.58
#Eval for threshold 0.8 DER=9.87, miss=7.12, falarm=1.48, confusion=1.27
#Eval for threshold 0.9 DER=13.57, miss=11.59, falarm=1.16, confusion=0.83

# cssd_testset of magicdat-ramc, collar=0.25
#Eval for threshold 0.2 DER=10.80, miss=0.86, falarm=9.06, confusion=0.89
#Eval for threshold 0.3 DER=8.17, miss=1.23, falarm=5.39, confusion=1.55
#Eval for threshold 0.35 DER=7.24, miss=1.43, falarm=3.91, confusion=1.89
#Eval for threshold 0.4 DER=6.51, miss=1.65, falarm=2.57, confusion=2.29
#Eval for threshold 0.45 DER=6.09, miss=1.98, falarm=1.46, confusion=2.65
#Eval for threshold 0.5 DER=6.08, miss=2.68, falarm=0.69, confusion=2.71
#Eval for threshold 0.55 DER=6.68, miss=3.88, falarm=0.52, confusion=2.28
#Eval for threshold 0.6 DER=7.51, miss=5.31, falarm=0.43, confusion=1.77
#Eval for threshold 0.7 DER=9.57, miss=8.21, falarm=0.30, confusion=1.06
#Eval for threshold 0.8 DER=12.58, miss=11.89, falarm=0.17, confusion=0.52
#Eval for threshold 0.9 DER=18.39, miss=18.09, falarm=0.08, confusion=0.22
i




# lr=2e-4, at Epoch 10, batch_idx_train: 54040, num_updates: 46500, {'loss': nan, 'DER', give up.
if [ ${stage} -le 122 ] && [ ${stop_stage} -ge 122 ];then
    . path_for_speaker_diarization_hltsz.sh
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
    #exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len16_shift0.8
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len16_shift0.8
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=16
   segment_shift=0.8
   single_backend_type="transformer"
   multi_backend_type="transformer"
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17915 \
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
    --num-transformer-layer $num_transformer_layer\

fi


if [ ${stage} -le 123 ] && [ ${stop_stage} -ge 123 ];then
 ##exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len16_shift0.8
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len16_shift0.8

 model_file=$exp_dir/best-valid-der.pt
 rs_len=16
 segment_shift=0.8
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test cssd_testset"
 #infer_sets="Test"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 data_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format"
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_${name}_collar${c}
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
    --collar $c\
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
done
fi

#grep -r Eval logs/run_ts_vad2_hltsz_stage122-123_3.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=14.23, miss=0.29, falarm=10.55, confusion=3.39
#Eval for threshold 0.3 DER=12.76, miss=0.52, falarm=8.61, confusion=3.63
#Eval for threshold 0.35 DER=12.33, miss=0.70, falarm=7.93, confusion=3.69
#Eval for threshold 0.4 DER=11.95, miss=0.91, falarm=7.33, confusion=3.71
#Eval for threshold 0.45 DER=11.70, miss=1.18, falarm=6.80, confusion=3.72
#Eval for threshold 0.5 DER=11.52, miss=1.52, falarm=6.32, confusion=3.69
#Eval for threshold 0.55 DER=11.44, miss=1.91, falarm=5.94, confusion=3.59
#Eval for threshold 0.6 DER=11.41, miss=2.38, falarm=5.53, confusion=3.50
#Eval for threshold 0.7 DER=11.68, miss=3.56, falarm=4.81, confusion=3.31
#Eval for threshold 0.8 DER=12.48, miss=5.47, falarm=3.96, confusion=3.04
#Eval for threshold 0.9 DER=15.22, miss=9.88, falarm=2.93, confusion=2.41

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=17.06, miss=0.41, falarm=14.69, confusion=1.95
#Eval for threshold 0.3 DER=15.26, miss=0.78, falarm=12.20, confusion=2.28
#Eval for threshold 0.35 DER=14.68, miss=1.06, falarm=11.25, confusion=2.37
#Eval for threshold 0.4 DER=14.20, miss=1.38, falarm=10.39, confusion=2.43
#Eval for threshold 0.45 DER=13.80, miss=1.82, falarm=9.49, confusion=2.49
#Eval for threshold 0.5 DER=12.68, miss=2.74, falarm=6.73, confusion=3.21
#Eval for threshold 0.55 DER=13.46, miss=5.40, falarm=5.83, confusion=2.24
#Eval for threshold 0.6 DER=13.68, miss=6.21, falarm=5.41, confusion=2.06
#Eval for threshold 0.7 DER=14.44, miss=8.13, falarm=4.58, confusion=1.73
#Eval for threshold 0.8 DER=15.93, miss=10.86, falarm=3.68, confusion=1.39
#Eval for threshold 0.9 DER=19.85, miss=16.22, falarm=2.67, confusion=0.96

# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=29.46, miss=3.26, falarm=23.92, confusion=2.27
#Eval for threshold 0.3 DER=24.38, miss=4.50, falarm=16.55, confusion=3.33
#Eval for threshold 0.35 DER=22.72, miss=5.27, falarm=13.67, confusion=3.79
#Eval for threshold 0.4 DER=21.60, miss=6.17, falarm=11.33, confusion=4.10
#Eval for threshold 0.45 DER=21.02, miss=7.38, falarm=9.46, confusion=4.18
#Eval for threshold 0.5 DER=21.25, miss=9.17, falarm=8.21, confusion=3.87
#Eval for threshold 0.55 DER=22.07, miss=11.44, falarm=7.37, confusion=3.26
#Eval for threshold 0.6 DER=23.12, miss=13.82, falarm=6.60, confusion=2.71
#Eval for threshold 0.7 DER=25.65, miss=18.89, falarm=5.01, confusion=1.74
#Eval for threshold 0.8 DER=29.98, miss=25.63, falarm=3.40, confusion=0.96
#Eval for threshold 0.9 DER=38.70, miss=36.55, falarm=1.81, confusion=0.35

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=6.17, miss=0.09, falarm=2.93, confusion=3.15
#Eval for threshold 0.3 DER=5.46, miss=0.16, falarm=2.03, confusion=3.27
#Eval for threshold 0.35 DER=5.31, miss=0.21, falarm=1.79, confusion=3.31
#Eval for threshold 0.4 DER=5.19, miss=0.28, falarm=1.60, confusion=3.31
#Eval for threshold 0.45 DER=5.13, miss=0.37, falarm=1.44, confusion=3.32
#Eval for threshold 0.5 DER=5.12, miss=0.49, falarm=1.31, confusion=3.32 as report
#Eval for threshold 0.55 DER=5.15, miss=0.64, falarm=1.24, confusion=3.27
#Eval for threshold 0.6 DER=5.21, miss=0.81, falarm=1.16, confusion=3.24
#Eval for threshold 0.7 DER=5.55, miss=1.34, falarm=1.05, confusion=3.17
#Eval for threshold 0.8 DER=6.30, miss=2.36, falarm=0.92, confusion=3.01
#Eval for threshold 0.9 DER=8.84, miss=5.59, falarm=0.78, confusion=2.47

# test of magicdata-ramc, collar=0.25
##Eval for threshold 0.2 DER=8.85, miss=0.14, falarm=7.26, confusion=1.44
#Eval for threshold 0.3 DER=7.95, miss=0.28, falarm=6.03, confusion=1.64
#Eval for threshold 0.35 DER=7.69, miss=0.39, falarm=5.59, confusion=1.71
#Eval for threshold 0.4 DER=7.50, miss=0.53, falarm=5.23, confusion=1.74
#Eval for threshold 0.45 DER=7.34, miss=0.73, falarm=4.81, confusion=1.80
#Eval for threshold 0.5 DER=6.45, miss=1.30, falarm=2.53, confusion=2.62 as report
#Eval for threshold 0.55 DER=7.42, miss=3.72, falarm=2.03, confusion=1.67
#Eval for threshold 0.6 DER=7.66, miss=4.17, falarm=1.92, confusion=1.57
#Eval for threshold 0.7 DER=8.34, miss=5.24, falarm=1.70, confusion=1.40
#Eval for threshold 0.8 DER=9.61, miss=6.97, falarm=1.44, confusion=1.20
#Eval for threshold 0.9 DER=13.07, miss=11.03, falarm=1.15, confusion=0.89

# cssd_testset of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=11.47, miss=1.01, falarm=9.36, confusion=1.09
#Eval for threshold 0.3 DER=8.10, miss=1.45, falarm=4.88, confusion=1.77
#Eval for threshold 0.35 DER=7.04, miss=1.69, falarm=3.22, confusion=2.13
#Eval for threshold 0.4 DER=6.33, miss=2.00, falarm=1.89, confusion=2.43
#Eval for threshold 0.45 DER=6.02, miss=2.48, falarm=0.93, confusion=2.61 as report
#Eval for threshold 0.5 DER=6.42, miss=3.51, falarm=0.43, confusion=2.47
#Eval for threshold 0.55 DER=7.36, miss=4.93, falarm=0.35, confusion=2.08
#Eval for threshold 0.6 DER=8.51, miss=6.48, falarm=0.28, confusion=1.74
#Eval for threshold 0.7 DER=11.23, miss=9.94, falarm=0.16, confusion=1.13
#Eval for threshold 0.8 DER=15.72, miss=15.03, falarm=0.07, confusion=0.62
#Eval for threshold 0.9 DER=24.73, miss=24.47, falarm=0.03, confusion=0.23

if [ ${stage} -le 124 ] && [ ${stop_stage} -ge 124 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9" # total 11
   predict_rttm_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len16_shift0.8
   oracle_rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   c=0.0
   dataset_name="magicdata-ramc"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/${dataset_name}_${name}_collar${c}/${name}/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi
#grep -r Avg logs/run_ts_vad2_hltsz_stage124.log
# dev of magicdata-ramc
#Avg CDER : 0.373
#Avg CDER : 0.256
#Avg CDER : 0.226
#Avg CDER : 0.192
#Avg CDER : 0.160
#Avg CDER : 0.130
#Avg CDER : 0.113
#Avg CDER : 0.109
#Avg CDER : 0.103
#Avg CDER : 0.096
#Avg CDER : 0.092
# test of magicdata-ramc
#Avg CDER : 0.295
#Avg CDER : 0.233
#Avg CDER : 0.211
#Avg CDER : 0.193
#Avg CDER : 0.175
#Avg CDER : 0.130
#Avg CDER : 0.117
#Avg CDER : Error!
#Avg CDER : Error!
#Avg CDER : Error!
#Avg CDER : Error!

# cssd_testset of magicdata-ramc
#Avg CDER : 0.291
#Avg CDER : 0.217
#Avg CDER : 0.183
#Avg CDER : 0.150
#Avg CDER : 0.123
#Avg CDER : 0.103
#Avg CDER : 0.097
#Avg CDER : 0.091
#Avg CDER : 0.086
#Avg CDER : 0.091
#Avg CDER : 0.105

if [ ${stage} -le 125 ] && [ ${stop_stage} -ge 125 ];then
    . path_for_speaker_diarization_hltsz.sh
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
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len8_shift0.4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=8
   segment_shift=0.4
   single_backend_type="transformer"
   multi_backend_type="transformer"
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 18815 \
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\

fi

if [ ${stage} -le 126 ] && [ ${stop_stage} -ge 126 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len8_shift0.4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=0.4
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test cssd_testset"
 #infer_sets="Test"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 data_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format"
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_${name}_collar${c}
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
    --collar $c\
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
done
fi

if [ ${stage} -le 127 ] && [ ${stop_stage} -ge 127 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9" # total 11
   predict_rttm_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len8_shift0.4
   oracle_rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   c=0.0
   dataset_name="magicdata-ramc"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/${dataset_name}_${name}_collar${c}/${name}/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi

