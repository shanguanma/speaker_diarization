#!/usr/bin/env bash

stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_dia_pt2.4.sh


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    dataset_name="magicdata-ramc" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    speaker_encoder_type="CAM++"
    speaker_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    fusion_type="att_wo_linear"
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_speech_encoder_cam++_speaker_encoder_cam++_fusion_type_att_wo_linear

    data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   batch_size=32
   gradient_accumulation_steps=2
  #CUDA_VISIABLE_DEVICES=0,1 \
  #TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --config_file "ts_vad3/fsdp_config.yaml" --num_processes 2  --main_process_port 12815 \
  CUDA_VISIABLE_DEVICES=0,1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch  --main_process_port 12815 \
  ts_vad3/train_accelerate_ddp.py \
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
    --fusion-type $fusion_type\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --batch-size $batch_size\
    --gradient-accumulation-steps $gradient_accumulation_steps
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_speech_encoder_cam++_speaker_encoder_cam++_fusion_type_att_wo_linear
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 batch_size=32
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test cssd_testset"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 speaker_encoder_type="CAM++"
 speaker_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 fusion_type="att_wo_linear"

 data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path # for dev test cssd_testset

 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad3/infer.py \
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
    --fusion-type $fusion_type\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --batch-size $batch_size
  done
 done
fi

#  grep -r Eval logs/run_ts_vad3_stage1-2_att_wo_linear.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=23.68, miss=0.32, falarm=19.16, confusion=4.21
#Eval for threshold 0.3 DER=18.89, miss=0.60, falarm=13.04, confusion=5.25
#Eval for threshold 0.35 DER=17.22, miss=0.84, falarm=10.68, confusion=5.70
#Eval for threshold 0.4 DER=16.01, miss=1.17, falarm=8.78, confusion=6.06
#Eval for threshold 0.45 DER=15.09, miss=1.66, falarm=7.03, confusion=6.40
#Eval for threshold 0.5 DER=14.70, miss=2.45, falarm=5.74, confusion=6.52
#Eval for threshold 0.55 DER=15.12, miss=3.94, falarm=5.17, confusion=6.01
#Eval for threshold 0.6 DER=15.92, miss=5.73, falarm=4.75, confusion=5.44
#Eval for threshold 0.7 DER=18.34, miss=9.95, falarm=3.90, confusion=4.49
#Eval for threshold 0.8 DER=22.46, miss=15.93, falarm=3.04, confusion=3.49
#Eval for threshold 0.9 DER=28.80, miss=24.23, falarm=2.19, confusion=2.38

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=23.63, miss=0.38, falarm=19.88, confusion=3.36
#Eval for threshold 0.3 DER=19.96, miss=0.81, falarm=14.71, confusion=4.44
#Eval for threshold 0.35 DER=18.56, miss=1.12, falarm=12.45, confusion=5.00
#Eval for threshold 0.4 DER=17.24, miss=1.53, falarm=10.12, confusion=5.60
#Eval for threshold 0.45 DER=16.20, miss=2.10, falarm=8.01, confusion=6.08
#Eval for threshold 0.5 DER=15.52, miss=3.03, falarm=6.16, confusion=6.33
#Eval for threshold 0.55 DER=16.17, miss=5.05, falarm=5.45, confusion=5.67
#Eval for threshold 0.6 DER=17.08, miss=7.16, falarm=4.97, confusion=4.94
#Eval for threshold 0.7 DER=19.17, miss=11.43, falarm=4.08, confusion=3.66
#Eval for threshold 0.8 DER=22.29, miss=16.46, falarm=3.23, confusion=2.60
#Eval for threshold 0.9 DER=27.86, miss=23.84, falarm=2.32, confusion=1.70

# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=36.29, miss=4.51, falarm=25.72, confusion=6.06
#Eval for threshold 0.3 DER=31.16, miss=6.64, falarm=17.24, confusion=7.28
#Eval for threshold 0.35 DER=29.62, miss=7.97, falarm=13.84, confusion=7.80
#Eval for threshold 0.4 DER=28.69, miss=9.56, falarm=10.95, confusion=8.19
#Eval for threshold 0.45 DER=28.44, miss=11.47, falarm=8.54, confusion=8.43
#Eval for threshold 0.5 DER=29.10, miss=14.07, falarm=6.70, confusion=8.32
#Eval for threshold 0.55 DER=30.89, miss=17.59, falarm=5.82, confusion=7.48
#Eval for threshold 0.6 DER=33.13, miss=21.43, falarm=5.05, confusion=6.65
#Eval for threshold 0.7 DER=38.80, miss=30.12, falarm=3.57, confusion=5.10
#Eval for threshold 0.8 DER=46.93, miss=41.05, falarm=2.26, confusion=3.62
#Eval for threshold 0.9 DER=60.41, miss=57.66, falarm=1.01, confusion=1.75

# dev of magicdat-ramc, collar=0.25
#Eval for threshold 0.2 DER=14.86, miss=0.07, falarm=10.99, confusion=3.80
#Eval for threshold 0.3 DER=11.18, miss=0.15, falarm=6.37, confusion=4.66
#Eval for threshold 0.35 DER=10.03, miss=0.23, falarm=4.76, confusion=5.04
#Eval for threshold 0.4 DER=9.17, miss=0.35, falarm=3.43, confusion=5.39
#Eval for threshold 0.45 DER=8.55, miss=0.56, falarm=2.25, confusion=5.74
#Eval for threshold 0.5 DER=8.24, miss=0.95, falarm=1.34, confusion=5.95
#Eval for threshold 0.55 DER=8.76, miss=2.01, falarm=1.15, confusion=5.60
#Eval for threshold 0.6 DER=9.63, miss=3.44, falarm=1.08, confusion=5.11
#Eval for threshold 0.7 DER=12.08, miss=6.75, falarm=0.95, confusion=4.38
#Eval for threshold 0.8 DER=16.27, miss=11.91, falarm=0.80, confusion=3.55
#Eval for threshold 0.9 DER=22.54, miss=19.34, falarm=0.68, confusion=2.52

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=15.09, miss=0.10, falarm=12.18, confusion=2.81
#Eval for threshold 0.3 DER=12.64, miss=0.27, falarm=8.64, confusion=3.73
#Eval for threshold 0.35 DER=11.68, miss=0.41, falarm=7.02, confusion=4.25
#Eval for threshold 0.4 DER=10.78, miss=0.58, falarm=5.38, confusion=4.82
#Eval for threshold 0.45 DER=10.01, miss=0.86, falarm=3.81, confusion=5.34
#Eval for threshold 0.5 DER=9.43, miss=1.33, falarm=2.35, confusion=5.75
#Eval for threshold 0.55 DER=10.13, miss=2.90, falarm=2.01, confusion=5.23
#Eval for threshold 0.6 DER=11.07, miss=4.61, falarm=1.88, confusion=4.58
#Eval for threshold 0.7 DER=13.10, miss=8.02, falarm=1.64, confusion=3.44
#Eval for threshold 0.8 DER=15.91, miss=12.01, falarm=1.41, confusion=2.48
#Eval for threshold 0.9 DER=20.98, miss=18.14, falarm=1.11, confusion=1.72

# cssd_testset of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=19.25, miss=1.48, falarm=13.27, confusion=4.50
#Eval for threshold 0.3 DER=15.32, miss=2.24, falarm=7.35, confusion=5.73
#Eval for threshold 0.35 DER=14.16, miss=2.87, falarm=4.97, confusion=6.32
#Eval for threshold 0.4 DER=13.60, miss=3.75, falarm=3.09, confusion=6.76
#Eval for threshold 0.45 DER=13.55, miss=4.89, falarm=1.47, confusion=7.19
#Eval for threshold 0.5 DER=14.29, miss=6.64, falarm=0.30, confusion=7.35
#Eval for threshold 0.55 DER=16.34, miss=9.52, falarm=0.19, confusion=6.64
#Eval for threshold 0.6 DER=18.82, miss=12.71, falarm=0.14, confusion=5.97
#Eval for threshold 0.7 DER=25.10, miss=20.35, falarm=0.07, confusion=4.68
#Eval for threshold 0.8 DER=34.11, miss=30.64, falarm=0.03, confusion=3.45
#Eval for threshold 0.9 DER=49.67, miss=47.89, falarm=0.01, confusion=1.76


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_speech_encoder_cam++_speaker_encoder_cam++_fusion_type_att_wo_linear/magicdata-ramc_collar0.0
   oracle_rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi
#grep -r Avg logs/run_ts_vad3_stage3_att_wo_linear.log
# dev of magicdata-ramc
#Avg CDER : 1.028
#Avg CDER : 0.765
#Avg CDER : 0.663
#Avg CDER : 0.555
#Avg CDER : 0.460
#Avg CDER : 0.334
#Avg CDER : 0.249
#Avg CDER : 0.206
#Avg CDER : 0.158
#Avg CDER : 0.121
#Avg CDER : 0.103

# test of magicata-ramc
#Avg CDER : 0.903
#Avg CDER : 0.682
#Avg CDER : 0.621
#Avg CDER : 0.553
#Avg CDER : 0.438
#Avg CDER : 0.325
#Avg CDER : 0.249
#Avg CDER : 0.227
#Avg CDER : 0.176
#Avg CDER : 0.176
#Avg CDER : Error!
# cssd_testset of magicata-ramc
#Avg CDER : 0.391
#Avg CDER : 0.307
#Avg CDER : 0.280
#Avg CDER : 0.240
#Avg CDER : 0.218
#Avg CDER : 0.195
#Avg CDER : 0.180
#Avg CDER : 0.172
#Avg CDER : 0.199
#Avg CDER : 0.248
#Avg CDER : 0.286

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    dataset_name="magicdata-ramc" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    speaker_encoder_type="CAM++_per"
    speaker_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    fusion_type="att_wo_linear"
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_speech_encoder_cam++_speaker_encoder_cam++_per_fusion_type_att_wo_linear
    data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   batch_size=32
   gradient_accumulation_steps=2
  #CUDA_VISIABLE_DEVICES=0,1 \
  #TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --config_file "ts_vad3/fsdp_config.yaml" --num_processes 2  --main_process_port 12815 \
  CUDA_VISIABLE_DEVICES=0,1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch  --main_process_port 12615 \
  ts_vad3/train_accelerate_ddp.py \
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
    --fusion-type $fusion_type\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --batch-size $batch_size\
    --gradient-accumulation-steps $gradient_accumulation_steps
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_speech_encoder_cam++_speaker_encoder_cam++_per_fusion_type_att_wo_linear
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 batch_size=32
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test cssd_testset"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 speaker_encoder_type="CAM++_per"
 speaker_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 fusion_type="att_wo_linear"

 data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path # for dev test cssd_testset

 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad3/infer.py \
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
    --fusion-type $fusion_type\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --batch-size $batch_size
  done
 done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_speech_encoder_cam++_speaker_encoder_cam++_per_fusion_type_att_wo_linear/magicdata-ramc_collar0.0
   oracle_rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi



if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    dataset_name="magicdata-ramc" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    speaker_encoder_type="CAM++_per"
    speaker_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    fusion_type="fusion_att_per_speaker_wo_utt_emb"
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_speech_encoder_cam++_speaker_encoder_cam++_per_fusion_type_fusion_att_per_speaker_wo_utt_emb
    data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   batch_size=64
   gradient_accumulation_steps=1
   without_speaker_utt_embed=True
  #CUDA_VISIABLE_DEVICES=0,1 \
  #TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --config_file "ts_vad3/fsdp_config.yaml" --num_processes 2  --main_process_port 12815 \
  CUDA_VISIABLE_DEVICES=0,1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch  --main_process_port 12515 \
  ts_vad3/train_accelerate_ddp.py \
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
    --fusion-type $fusion_type\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --batch-size $batch_size\
    --gradient-accumulation-steps $gradient_accumulation_steps\
    --without-speaker-utt-embed $without_speaker_utt_embed
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_speech_encoder_cam++_speaker_encoder_cam++_per_fusion_type_fusion_att_per_speaker_wo_utt_emb
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 batch_size=64
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test cssd_testset"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 speaker_encoder_type="CAM++_per"
 speaker_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 fusion_type="fusion_att_per_speaker_wo_utt_emb"

 data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path # for dev test cssd_testset
 without_speaker_utt_embed=True

 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad3/infer.py \
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
    --fusion-type $fusion_type\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --batch-size $batch_size\
    --without-speaker-utt-embed $without_speaker_utt_embed
  done
 done
fi
#  grep -r Eval logs/run_ts_vad3_stage9-10_att_wo_linear_cam++_per_wo_utt_emb.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=35.33, miss=0.79, falarm=29.71, confusion=4.83
#Eval for threshold 0.3 DER=28.69, miss=1.52, falarm=20.44, confusion=6.73
#Eval for threshold 0.35 DER=25.69, miss=2.05, falarm=15.81, confusion=7.82
#Eval for threshold 0.4 DER=22.97, miss=2.89, falarm=11.01, confusion=9.07
#Eval for threshold 0.45 DER=21.20, miss=4.23, falarm=6.75, confusion=10.22
#Eval for threshold 0.5 DER=22.20, miss=7.92, falarm=5.01, confusion=9.27
#Eval for threshold 0.55 DER=24.56, miss=12.74, falarm=4.49, confusion=7.33
#Eval for threshold 0.6 DER=27.09, miss=17.32, falarm=3.98, confusion=5.79
#Eval for threshold 0.7 DER=32.41, miss=25.68, falarm=3.06, confusion=3.67
#Eval for threshold 0.8 DER=38.24, miss=34.03, falarm=2.07, confusion=2.13
#Eval for threshold 0.9 DER=47.58, miss=45.46, falarm=1.15, confusion=0.98

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=28.32, miss=1.15, falarm=19.84, confusion=7.33
#Eval for threshold 0.3 DER=23.73, miss=2.07, falarm=12.94, confusion=8.72
#Eval for threshold 0.35 DER=22.37, miss=2.70, falarm=10.46, confusion=9.21
#Eval for threshold 0.4 DER=21.36, miss=3.45, falarm=8.25, confusion=9.65
#Eval for threshold 0.45 DER=20.77, miss=4.49, falarm=6.36, confusion=9.92
#Eval for threshold 0.5 DER=21.28, miss=6.44, falarm=5.37, confusion=9.47
#Eval for threshold 0.55 DER=22.47, miss=8.99, falarm=4.85, confusion=8.63
#Eval for threshold 0.6 DER=23.89, miss=11.73, falarm=4.32, confusion=7.83
#Eval for threshold 0.7 DER=27.76, miss=18.21, falarm=3.29, confusion=6.26
#Eval for threshold 0.8 DER=34.26, miss=27.42, falarm=2.30, confusion=4.54
#Eval for threshold 0.9 DER=46.32, miss=42.47, falarm=1.29, confusion=2.56

# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=43.43, miss=4.91, falarm=33.33, confusion=5.19
#Eval for threshold 0.3 DER=35.98, miss=7.41, falarm=21.31, confusion=7.26
#Eval for threshold 0.35 DER=33.23, miss=8.93, falarm=15.96, confusion=8.33
#Eval for threshold 0.4 DER=31.33, miss=10.82, falarm=11.41, confusion=9.10
#Eval for threshold 0.45 DER=30.58, miss=13.45, falarm=7.84, confusion=9.29
#Eval for threshold 0.5 DER=31.93, miss=17.83, falarm=6.22, confusion=7.88
#Eval for threshold 0.55 DER=34.11, miss=22.67, falarm=5.24, confusion=6.21
#Eval for threshold 0.6 DER=36.55, miss=27.39, falarm=4.33, confusion=4.83
#Eval for threshold 0.7 DER=42.16, miss=36.68, falarm=2.82, confusion=2.66
#Eval for threshold 0.8 DER=49.16, miss=46.15, falarm=1.68, confusion=1.33
#Eval for threshold 0.9 DER=59.26, miss=57.94, falarm=0.77, confusion=0.54

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=26.95, miss=0.20, falarm=22.51, confusion=4.24
#Eval for threshold 0.3 DER=21.42, miss=0.44, falarm=14.92, confusion=6.06
#Eval for threshold 0.35 DER=18.76, miss=0.67, falarm=10.88, confusion=7.21
#Eval for threshold 0.4 DER=16.23, miss=1.07, falarm=6.64, confusion=8.52
#Eval for threshold 0.45 DER=14.52, miss=1.92, falarm=2.64, confusion=9.96
#Eval for threshold 0.5 DER=15.58, miss=5.19, falarm=1.20, confusion=9.20
#Eval for threshold 0.55 DER=18.13, miss=9.83, falarm=1.09, confusion=7.22
#Eval for threshold 0.6 DER=20.80, miss=14.10, falarm=0.97, confusion=5.73
#Eval for threshold 0.7 DER=26.41, miss=22.01, falarm=0.77, confusion=3.63
#Eval for threshold 0.8 DER=32.47, miss=29.77, falarm=0.57, confusion=2.13
#Eval for threshold 0.9 DER=42.01, miss=40.64, falarm=0.37, confusion=1.01

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=19.94, miss=0.42, falarm=13.17, confusion=6.35
#Eval for threshold 0.3 DER=16.41, miss=0.83, falarm=7.80, confusion=7.78
#Eval for threshold 0.35 DER=15.28, miss=1.12, falarm=5.83, confusion=8.33
#Eval for threshold 0.4 DER=14.45, miss=1.49, falarm=4.08, confusion=8.88
#Eval for threshold 0.45 DER=13.95, miss=2.06, falarm=2.54, confusion=9.35
#Eval for threshold 0.5 DER=14.50, miss=3.57, falarm=1.91, confusion=9.02
#Eval for threshold 0.55 DER=15.76, miss=5.70, falarm=1.79, confusion=8.27
#Eval for threshold 0.6 DER=17.25, miss=8.00, falarm=1.67, confusion=7.58
#Eval for threshold 0.7 DER=21.09, miss=13.50, falarm=1.41, confusion=6.18
#Eval for threshold 0.8 DER=27.52, miss=21.72, falarm=1.13, confusion=4.67
#Eval for threshold 0.9 DER=39.80, miss=36.32, falarm=0.72, confusion=2.75

# cssd_testset of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=25.06, miss=1.65, falarm=20.81, confusion=2.60
#Eval for threshold 0.3 DER=19.31, miss=2.87, falarm=11.79, confusion=4.65
#Eval for threshold 0.35 DER=17.23, miss=3.68, falarm=7.75, confusion=5.80
#Eval for threshold 0.4 DER=15.78, miss=4.73, falarm=4.20, confusion=6.85
#Eval for threshold 0.45 DER=15.19, miss=6.40, falarm=1.37, confusion=7.42
#Eval for threshold 0.5 DER=16.90, miss=9.94, falarm=0.51, confusion=6.44
#Eval for threshold 0.55 DER=19.58, miss=14.20, falarm=0.36, confusion=5.02
#Eval for threshold 0.6 DER=22.50, miss=18.47, falarm=0.23, confusion=3.80
#Eval for threshold 0.7 DER=29.06, miss=27.02, falarm=0.12, confusion=1.92
#Eval for threshold 0.8 DER=37.29, miss=36.41, falarm=0.04, confusion=0.84
#Eval for threshold 0.9 DER=48.82, miss=48.51, falarm=0.02, confusion=0.29

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_speech_encoder_cam++_speaker_encoder_cam++_per_fusion_type_fusion_att_per_speaker_wo_utt_emb/magicdata-ramc_collar0.0
   oracle_rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi

# grep -r Avg logs/run_ts_vad3_stage9-10_att_wo_linear_cam++_per_wo_utt_emb.log
# dev of magicata-ramc, collar=0.0
#Avg CDER : 1.094
#Avg CDER : 0.982
#Avg CDER : 0.908
#Avg CDER : 0.813
#Avg CDER : 0.670
#Avg CDER : 0.402
#Avg CDER : 0.337
#Avg CDER : 0.297
#Avg CDER : 0.275
#Avg CDER : 0.246
#Avg CDER : 0.308
# test of magicata-ramc, collar=0.0
#Avg CDER : 0.900
#Avg CDER : 0.667
#Avg CDER : 0.579
#Avg CDER : 0.501
#Avg CDER : 0.437
#Avg CDER : 0.351
#Avg CDER : 0.309
#Avg CDER : 0.281
#Avg CDER : 0.281
#Avg CDER : 0.276
#Avg CDER : 0.318
# cssd_test of magicdata-ramc, collar=0.0
#Avg CDER : 0.350
#Avg CDER : 0.288
#Avg CDER : 0.245
#Avg CDER : 0.199
#Avg CDER : 0.143
#Avg CDER : 0.128
#Avg CDER : 0.115
#Avg CDER : 0.133
#Avg CDER : 0.160
#Avg CDER : 0.213
#Avg CDER : Error!
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    dataset_name="magicdata-ramc" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    speaker_encoder_type="CAM++"
    speaker_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    fusion_type="att_w_linear"
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_speech_encoder_cam++_speaker_encoder_cam++_fusion_type_att_w_linear

    data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   batch_size=32
   num_gpus=2
   gradient_accumulation_steps=2
  #CUDA_VISIABLE_DEVICES=0,1 \
  #TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --config_file "ts_vad3/fsdp_config.yaml" --num_processes 2 --main_process_port 12715 \
  CUDA_VISIABLE_DEVICES=0,1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch  --main_process_port 12715 \
  ts_vad3/train_accelerate_ddp.py \
    --world-size $num_gpus \
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
    --fusion-type $fusion_type\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --batch-size $batch_size\
    --gradient-accumulation-steps $gradient_accumulation_steps
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_speech_encoder_cam++_speaker_encoder_cam++_fusion_type_att_w_linear
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 batch_size=64
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test cssd_testset"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 speaker_encoder_type="CAM++"
 speaker_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"                                                                                                                                                                  fusion_type="att_w_linear"
 data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path # for dev test cssd_testset

 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad3/infer.py \
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
    --fusion-type $fusion_type\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --batch-size $batch_size
  done
 done
fi

# grep -r Eval  logs/run_ts_vad3_stage11-12_att_w_linear.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=23.43, miss=0.31, falarm=17.67, confusion=5.45
#Eval for threshold 0.3 DER=19.95, miss=0.63, falarm=12.91, confusion=6.41
#Eval for threshold 0.35 DER=18.61, miss=0.83, falarm=10.89, confusion=6.88
#Eval for threshold 0.4 DER=17.48, miss=1.14, falarm=9.01, confusion=7.32
#Eval for threshold 0.45 DER=16.63, miss=1.56, falarm=7.35, confusion=7.72
#Eval for threshold 0.5 DER=16.07, miss=2.19, falarm=5.95, confusion=7.93
#Eval for threshold 0.55 DER=16.44, miss=3.62, falarm=5.38, confusion=7.44
#Eval for threshold 0.6 DER=17.04, miss=5.26, falarm=4.94, confusion=6.84
#Eval for threshold 0.7 DER=18.74, miss=8.84, falarm=4.09, confusion=5.80
#Eval for threshold 0.8 DER=21.41, miss=13.42, falarm=3.16, confusion=4.83
#Eval for threshold 0.9 DER=26.34, miss=20.23, falarm=2.22, confusion=3.89

# test of magicdta-ramc, collar=0.0
#Eval for threshold 0.2 DER=26.16, miss=0.42, falarm=22.42, confusion=3.32
#Eval for threshold 0.3 DER=21.80, miss=0.91, falarm=16.29, confusion=4.59
#Eval for threshold 0.35 DER=20.03, miss=1.23, falarm=13.52, confusion=5.27
#Eval for threshold 0.4 DER=18.53, miss=1.68, falarm=10.83, confusion=6.02
#Eval for threshold 0.45 DER=17.32, miss=2.26, falarm=8.38, confusion=6.68
#Eval for threshold 0.5 DER=16.66, miss=3.20, falarm=6.33, confusion=7.13
#Eval for threshold 0.55 DER=17.35, miss=5.43, falarm=5.62, confusion=6.30
#Eval for threshold 0.6 DER=18.39, miss=7.91, falarm=5.11, confusion=5.38
#Eval for threshold 0.7 DER=21.07, miss=13.14, falarm=4.18, confusion=3.75
#Eval for threshold 0.8 DER=24.69, miss=18.90, falarm=3.29, confusion=2.51
#Eval for threshold 0.9 DER=30.58, miss=26.73, falarm=2.34, confusion=1.51

# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=34.29, miss=3.85, falarm=25.69, confusion=4.74
#Eval for threshold 0.3 DER=29.16, miss=5.52, falarm=17.53, confusion=6.10
#Eval for threshold 0.35 DER=27.42, miss=6.51, falarm=14.22, confusion=6.69
#Eval for threshold 0.4 DER=26.19, miss=7.65, falarm=11.30, confusion=7.25
#Eval for threshold 0.45 DER=25.46, miss=9.07, falarm=8.84, confusion=7.55
#Eval for threshold 0.5 DER=25.65, miss=11.22, falarm=7.13, confusion=7.29
#Eval for threshold 0.55 DER=26.81, miss=14.15, falarm=6.23, confusion=6.43
#Eval for threshold 0.6 DER=28.26, miss=17.32, falarm=5.41, confusion=5.53
#Eval for threshold 0.7 DER=32.40, miss=24.51, falarm=3.88, confusion=4.01
#Eval for threshold 0.8 DER=39.03, miss=33.83, falarm=2.54, confusion=2.66
#Eval for threshold 0.9 DER=50.40, miss=47.75, falarm=1.29, confusion=1.36

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=14.76, miss=0.07, falarm=9.65, confusion=5.04
#Eval for threshold 0.3 DER=12.36, miss=0.16, falarm=6.29, confusion=5.91
#Eval for threshold 0.35 DER=11.43, miss=0.23, falarm=4.88, confusion=6.32
#Eval for threshold 0.4 DER=10.68, miss=0.35, falarm=3.58, confusion=6.76
#Eval for threshold 0.45 DER=10.07, miss=0.50, falarm=2.41, confusion=7.16
#Eval for threshold 0.5 DER=9.67, miss=0.78, falarm=1.41, confusion=7.48
#Eval for threshold 0.55 DER=10.13, miss=1.84, falarm=1.19, confusion=7.10
#Eval for threshold 0.6 DER=10.82, miss=3.10, falarm=1.12, confusion=6.60
#Eval for threshold 0.7 DER=12.57, miss=5.91, falarm=0.98, confusion=5.68
#Eval for threshold 0.8 DER=15.25, miss=9.53, falarm=0.84, confusion=4.88
#Eval for threshold 0.9 DER=20.14, miss=15.39, falarm=0.71, confusion=4.04

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=17.82, miss=0.13, falarm=15.04, confusion=2.65
#Eval for threshold 0.3 DER=14.58, miss=0.34, falarm=10.39, confusion=3.84
#Eval for threshold 0.35 DER=13.22, miss=0.49, falarm=8.26, confusion=4.47
#Eval for threshold 0.4 DER=12.05, miss=0.70, falarm=6.11, confusion=5.23
#Eval for threshold 0.45 DER=11.10, miss=1.01, falarm=4.13, confusion=5.95
#Eval for threshold 0.5 DER=10.55, miss=1.52, falarm=2.46, confusion=6.56
#Eval for threshold 0.55 DER=11.36, miss=3.41, falarm=2.11, confusion=5.84
#Eval for threshold 0.6 DER=12.51, miss=5.56, falarm=1.98, confusion=4.97
#Eval for threshold 0.7 DER=15.26, miss=10.09, falarm=1.72, confusion=3.45
#Eval for threshold 0.8 DER=18.74, miss=14.96, falarm=1.47, confusion=2.31
#Eval for threshold 0.9 DER=24.28, miss=21.72, falarm=1.14, confusion=1.42

# cssd_testset of magicdata0-ramc, collar=0.25
#Eval for threshold 0.2 DER=16.26, miss=1.31, falarm=11.64, confusion=3.31
#Eval for threshold 0.3 DER=12.88, miss=1.92, falarm=6.61, confusion=4.34
#Eval for threshold 0.35 DER=11.80, miss=2.28, falarm=4.63, confusion=4.88
#Eval for threshold 0.4 DER=11.11, miss=2.74, falarm=2.90, confusion=5.47
#Eval for threshold 0.45 DER=10.71, miss=3.36, falarm=1.38, confusion=5.96
#Eval for threshold 0.5 DER=11.01, miss=4.60, falarm=0.39, confusion=6.02
#Eval for threshold 0.55 DER=12.40, miss=6.81, falarm=0.28, confusion=5.30
#Eval for threshold 0.6 DER=14.03, miss=9.14, falarm=0.23, confusion=4.66
#Eval for threshold 0.7 DER=18.39, miss=14.74, falarm=0.13, confusion=3.51
#Eval for threshold 0.8 DER=25.38, miss=22.87, falarm=0.08, confusion=2.43
#Eval for threshold 0.9 DER=37.80, miss=36.41, falarm=0.03, confusion=1.35


if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_speech_encoder_cam++_speaker_encoder_cam++_fusion_type_att_w_linear/magicdata-ramc_collar0.0
   oracle_rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi

#grep -r Avg logs/run_ts_vad3_stage13.log
# dev of magicdata-ramc
#Avg CDER : 0.860
#Avg CDER : 0.721
#Avg CDER : 0.660
#Avg CDER : 0.586
#Avg CDER : 0.493
#Avg CDER : 0.431
#Avg CDER : 0.330
#Avg CDER : 0.267
#Avg CDER : 0.193
#Avg CDER : 0.176
#Avg CDER : 0.190

# test of magicdata-ramc
#Avg CDER : 0.909
#Avg CDER : 0.919
#Avg CDER : 0.890
#Avg CDER : 0.841
#Avg CDER : 0.774
#Avg CDER : 0.677
#Avg CDER : 0.482
#Avg CDER : 0.396
#Avg CDER : 0.256
#Avg CDER : 0.199
#Avg CDER : 0.140

# cssd_testset of magicdata-ramc
#Avg CDER : 0.359
#Avg CDER : 0.268
#Avg CDER : 0.232
#Avg CDER : 0.199
#Avg CDER : 0.172
#Avg CDER : 0.147
#Avg CDER : 0.146
#Avg CDER : 0.138
#Avg CDER : 0.125 as report
#Avg CDER : 0.130
#Avg CDER : 0.163




if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    dataset_name="magicdata-ramc" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    speaker_encoder_type="CAM++_per"
    speaker_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    fusion_type="fusion_att_per_speaker_wo_utt_emb"
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_speech_encoder_cam++_speaker_encoder_cam++_per_fusion_type_fusion_att_per_speaker_wo_utt_emb_fusion_case_fusion_embed_as_mix_embed_w_utt_embed
    data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   batch_size=32
   gradient_accumulation_steps=2
   without_speaker_utt_embed=False
   fusion_case="fusion_embed_as_mix_embed_w_utt_embed"
  #CUDA_VISIABLE_DEVICES=0,1 \
  #TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --config_file "ts_vad3/fsdp_config.yaml" --num_processes 2  --main_process_port 12815 \
  CUDA_VISIABLE_DEVICES=0,1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch  --main_process_port 11515 \
  ts_vad3/train_accelerate_ddp.py \
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
    --fusion-type $fusion_type\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --batch-size $batch_size\
    --gradient-accumulation-steps $gradient_accumulation_steps\
    --without-speaker-utt-embed $without_speaker_utt_embed\
    --fusion-case $fusion_case
fi

if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_speech_encoder_cam++_speaker_encoder_cam++_per_fusion_type_fusion_att_per_speaker_wo_utt_emb_fusion_case_fusion_embed_as_mix_embed_w_utt_embed
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 batch_size=32
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test cssd_testset"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 speaker_encoder_type="CAM++_per"
 speaker_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 fusion_type="fusion_att_per_speaker_wo_utt_emb"


 data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path # for dev test cssd_testset
 without_speaker_utt_embed=False
 fusion_case="fusion_embed_as_mix_embed_w_utt_embed"
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad3/infer.py \
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
    --fusion-type $fusion_type\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --batch-size $batch_size\
    --without-speaker-utt-embed $without_speaker_utt_embed\
    --fusion-case $fusion_case
  done
 done
fi


if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_speech_encoder_cam++_speaker_encoder_cam++_per_fusion_type_fusion_att_per_speaker_wo_utt_emb_fusion_case_fusion_embed_as_mix_embed_w_utt_embed/magicdata-ramc_collar0.0
   oracle_rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi
