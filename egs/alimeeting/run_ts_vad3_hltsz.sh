#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
#set -u
set -o pipefail


stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_speaker_diarization_hltsz.sh

# it is same as stage2 and stage4 of run_ts_vad2.sh,
# here,
# 1. I will extract utt_speaker embedding online.
# 2. I will fuse frame speaker embedding on tsvad
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] ;then
   # it adds noise to train tsvad model , no grad-clip and no freeze update.
   # # speech encoder is cam++ , oracle target speaker embedding is from cam++ pretrain model
   # this cam++ pretrain model is trained on cn-cnceleb and voxceleb dataset, 
   # checkpoint is from https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/files
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"

    speaker_encoder_type=$speech_encoder_type
    speaker_encoder_path=$speech_encoder_path 
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_two_gpus_unfreeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_frame_speaker_embedding
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12815 \
   ts_vad3/train_accelerate_onegpu.py\
     --verbose 1 \
     --batch-size 128\
     --world-size 1 \
     --num-epochs 20\
     --start-epoch 1\
     --keep-last-k 1\
     --keep-last-epoch 1\
     --freeze-speech-encoder-updates 0\
     --freeze-speaker-encoder-updates 62600\
     --fuse-fbank-feat false\
     --fuse-speaker-embedding-feat true\
     --ts-len 4\
     --grad-clip false\
     --lr 2e-4\
     --musan-path $musan_path \
     --rir-path $rir_path \
     --speech-encoder-type $speech_encoder_type\
     --speech-encoder-path $speech_encoder_path\
     --speaker-encoder-type $speaker_encoder_type\
     --speaker-encoder-path $speaker_encoder_path\
     --select-encoder-layer-nums 6\
     --exp-dir $exp_dir\
     --data-dir $data_dir 
fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_two_gpus_unfreeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_frame_speaker_embedding
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
 speaker_encoder_type=$speech_encoder_type
 speaker_encoder_path=$speech_encoder_path
 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
     results_path=$exp_dir
  python3 ts_vad3/infer2.py \
    --model-file $model_file\
    --ts-len 4\
    --rs-len $rs_len\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat true\
    --data-dir $data_dir
done
#sbatch --nodes 1  --gres=gpu:1  --cpus-per-gpu=8  --ntasks-per-node 1  -p speech -o logs/run_ts_vad3_hltsz_stage2_onegpu.log run_ts_vad3_hltsz.sh --stage 2 --stop-stage 2
#logs/run_ts_vad3_hltsz_stage2_onegpu.log
#Eval set
#Model DER:  0.14954542541824004
#Model ACC:  0.9428796884148488
#100%|██████████| 25/25 [00:27<00:00,  1.09s/it]
#Eval for threshold 0.20: DER 10.54%, MS 1.17%, FA 7.99%, SC 1.38%
#
#Eval for threshold 0.30: DER 8.01%, MS 1.78%, FA 4.67%, SC 1.55%
#
#Eval for threshold 0.35: DER 7.30%, MS 2.13%, FA 3.53%, SC 1.63%
#
#Eval for threshold 0.40: DER 7.00%, MS 2.63%, FA 2.67%, SC 1.69%
#
#Eval for threshold 0.45: DER 6.78%, MS 3.14%, FA 1.94%, SC 1.69%
#
#Eval for threshold 0.50: DER 6.85%, MS 3.66%, FA 1.47%, SC 1.72%
#
#Eval for threshold 0.55: DER 7.12%, MS 4.34%, FA 1.11%, SC 1.68%
#
#Eval for threshold 0.60: DER 7.66%, MS 5.23%, FA 0.85%, SC 1.58%
#
#Eval for threshold 0.70: DER 9.22%, MS 7.47%, FA 0.51%, SC 1.23%
#
#Eval for threshold 0.80: DER 11.98%, MS 10.76%, FA 0.40%, SC 0.82%

# Test set
#Model DER:  0.15783674965328107
#Model ACC:  0.9360397486569937
#100%|██████████| 60/60 [01:14<00:00,  1.24s/it]
#Eval for threshold 0.20: DER 13.22%, MS 1.46%, FA 9.82%, SC 1.93%
#
#Eval for threshold 0.30: DER 10.71%, MS 2.26%, FA 5.72%, SC 2.73%
#
#Eval for threshold 0.35: DER 9.98%, MS 2.73%, FA 4.22%, SC 3.03%
#
#Eval for threshold 0.40: DER 9.58%, MS 3.25%, FA 3.01%, SC 3.32%
#
#Eval for threshold 0.45: DER 9.44%, MS 3.91%, FA 2.02%, SC 3.50%
#
#Eval for threshold 0.50: DER 9.50%, MS 4.80%, FA 1.21%, SC 3.50%
#
#Eval for threshold 0.55: DER 9.94%, MS 5.90%, FA 0.80%, SC 3.24%
#
#Eval for threshold 0.60: DER 10.64%, MS 7.21%, FA 0.53%, SC 2.89%
#
#Eval for threshold 0.70: DER 12.52%, MS 10.17%, FA 0.28%, SC 2.08%
#
#Eval for threshold 0.80: DER 15.40%, MS 14.01%, FA 0.15%, SC 1.24%

fi




if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] ;then
   # it adds noise to train tsvad model , no grad-clip and no freeze update.
   # # speech encoder is cam++ , oracle target speaker embedding is from cam++ pretrain model
   # this cam++ pretrain model is trained on cn-cnceleb and voxceleb dataset,
   # checkpoint is from https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/files
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"

    speaker_encoder_type=$speech_encoder_type
    speaker_encoder_path=$speech_encoder_path
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_one_gpu_unfreeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_fbank_feat
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12915 \
   ts_vad3/train_accelerate_onegpu.py\
     --verbose 1 \
     --batch-size 128\
     --world-size 1 \
     --num-epochs 20\
     --start-epoch 1\
     --keep-last-k 1\
     --keep-last-epoch 1\
     --freeze-speech-encoder-updates 0\
     --freeze-speaker-encoder-updates 62600\
     --fuse-fbank-feat true\
     --fuse-speaker-embedding-feat false\
     --ts-len 4\
     --grad-clip false\
     --lr 2e-4\
     --musan-path $musan_path \
     --rir-path $rir_path \
     --speech-encoder-type $speech_encoder_type\
     --speech-encoder-path $speech_encoder_path\
     --speaker-encoder-type $speaker_encoder_type\
     --speaker-encoder-path $speaker_encoder_path\
     --select-encoder-layer-nums 6\
     --exp-dir $exp_dir\
     --data-dir $data_dir
fi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_one_gpu_unfreeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_fbank_feat
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
 speaker_encoder_type=$speech_encoder_type
 speaker_encoder_path=$speech_encoder_path
 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
     results_path=$exp_dir
  python3 ts_vad3/infer2.py \
    --model-file $model_file\
    --ts-len 4\
    --rs-len $rs_len\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat true\
    --fuse-speaker-embedding-feat false\
    --data-dir $data_dir
done

#sbatch --nodes 1  --gres=gpu:1  --cpus-per-gpu=8  --ntasks-per-node 1  -p speech -o logs/run_ts_vad3_hltsz_stage3-4_onegpu.log run_ts_vad3_hltsz.sh --stage 3 --stop-stage 4
#cat logs/run_ts_vad3_hltsz_stage3-4_onegpu.log
## Eval set
#Model DER:  0.20533144740978868
#Model ACC:  0.9214418615304415
#100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
#Eval for threshold 0.20: DER 19.73%, MS 1.36%, FA 16.41%, SC 1.96%
#
#Eval for threshold 0.30: DER 14.76%, MS 2.43%, FA 9.71%, SC 2.62%
#
#Eval for threshold 0.35: DER 13.26%, MS 3.10%, FA 7.29%, SC 2.88%
#
#Eval for threshold 0.40: DER 12.44%, MS 3.86%, FA 5.44%, SC 3.14%
#
#Eval for threshold 0.45: DER 11.92%, MS 4.78%, FA 3.81%, SC 3.33%
#
#Eval for threshold 0.50: DER 11.92%, MS 5.92%, FA 2.61%, SC 3.40%
#
#Eval for threshold 0.55: DER 12.29%, MS 7.32%, FA 1.72%, SC 3.25%
#
#Eval for threshold 0.60: DER 13.14%, MS 9.03%, FA 1.14%, SC 2.97%
#
#Eval for threshold 0.70: DER 15.82%, MS 13.16%, FA 0.58%, SC 2.08%
#
#Eval for threshold 0.80: DER 20.34%, MS 18.71%, FA 0.41%, SC 1.22%
#
## Test set
#Model DER:  0.21618574478975064
#Model ACC:  0.9101463766939186
#100%|██████████| 60/60 [01:07<00:00,  1.12s/it]
#Eval for threshold 0.20: DER 22.93%, MS 1.80%, FA 17.59%, SC 3.54%
#
#Eval for threshold 0.30: DER 17.91%, MS 2.93%, FA 10.50%, SC 4.48%
#
#Eval for threshold 0.35: DER 16.44%, MS 3.58%, FA 7.95%, SC 4.91%
#
#Eval for threshold 0.40: DER 15.41%, MS 4.33%, FA 5.72%, SC 5.36%
#
#Eval for threshold 0.45: DER 14.80%, MS 5.20%, FA 3.83%, SC 5.77%
#
#Eval for threshold 0.50: DER 14.64%, MS 6.44%, FA 2.14%, SC 6.06%
#
#Eval for threshold 0.55: DER 15.30%, MS 8.32%, FA 1.29%, SC 5.69%
#
#Eval for threshold 0.60: DER 16.40%, MS 10.49%, FA 0.84%, SC 5.07%
#
#Eval for threshold 0.70: DER 19.71%, MS 15.63%, FA 0.34%, SC 3.73%
#
#Eval for threshold 0.80: DER 24.76%, MS 22.29%, FA 0.13%, SC 2.34%
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ] ;then
   # it adds noise to train tsvad model , no grad-clip and no freeze update.
   # # speech encoder is cam++ , oracle target speaker embedding is from cam++ pretrain model
   # this cam++ pretrain model is trained on cn-cnceleb and voxceleb dataset,
   # checkpoint is from https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/files
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"

    speaker_encoder_type=$speech_encoder_type
    speaker_encoder_path=$speech_encoder_path
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_one_gpu_unfreeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12815 \
   ts_vad3/train_accelerate_onegpu.py\
     --verbose 1 \
     --batch-size 128\
     --world-size 1 \
     --num-epochs 20\
     --start-epoch 1\
     --keep-last-k 1\
     --keep-last-epoch 1\
     --freeze-speech-encoder-updates 0\
     --freeze-speaker-encoder-updates 62600\
     --fuse-fbank-feat false\
     --fuse-speaker-embedding-feat false\
     --ts-len 4\
     --grad-clip false\
     --lr 2e-4\
     --musan-path $musan_path \
     --rir-path $rir_path \
     --speech-encoder-type $speech_encoder_type\
     --speech-encoder-path $speech_encoder_path\
     --speaker-encoder-type $speaker_encoder_type\
     --speaker-encoder-path $speaker_encoder_path\
     --select-encoder-layer-nums 6\
     --exp-dir $exp_dir\
     --data-dir $data_dir
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_one_gpu_unfreeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
 speaker_encoder_type=$speech_encoder_type
 speaker_encoder_path=$speech_encoder_path
 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
     results_path=$exp_dir
  python3 ts_vad3/infer2.py \
    --model-file $model_file\
    --ts-len 4\
    --rs-len $rs_len\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
     --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat false\
    --data-dir $data_dir
done
#sbatch --nodes 1  --gres=gpu:1  --cpus-per-gpu=8  --ntasks-per-node 1  -p speech -o logs/run_ts_vad3_hltsz_stage5-6_onegpu.log run_ts_vad3_hltsz.sh --stage 5 --stop-stage 6
#cat logs/run_ts_vad3_hltsz_stage5-6_onegpu.log
## Eval set
## Model DER:  0.154508567148308
#Model ACC:  0.9413698475538377
#100%|██████████| 25/25 [00:26<00:00,  1.05s/it]
#Eval for threshold 0.20: DER 11.33%, MS 1.11%, FA 9.21%, SC 1.01%
#
#Eval for threshold 0.30: DER 8.66%, MS 1.81%, FA 5.47%, SC 1.38%
#
#Eval for threshold 0.35: DER 7.92%, MS 2.19%, FA 4.13%, SC 1.60%
#
#Eval for threshold 0.40: DER 7.49%, MS 2.61%, FA 3.14%, SC 1.74%
#
#Eval for threshold 0.45: DER 7.30%, MS 3.08%, FA 2.37%, SC 1.85%
#
#Eval for threshold 0.50: DER 7.27%, MS 3.68%, FA 1.69%, SC 1.90%
#
#Eval for threshold 0.55: DER 7.57%, MS 4.51%, FA 1.24%, SC 1.82%
#
#Eval for threshold 0.60: DER 8.07%, MS 5.49%, FA 0.91%, SC 1.67%
#
#Eval for threshold 0.70: DER 9.81%, MS 7.97%, FA 0.57%, SC 1.27%
#
#Eval for threshold 0.80: DER 12.68%, MS 11.46%, FA 0.40%, SC 0.82%
#
## Test set
#Model DER:  0.1614904663338059
#Model ACC:  0.9347507521434627
#100%|██████████| 60/60 [01:05<00:00,  1.09s/it]
#Eval for threshold 0.20: DER 13.33%, MS 1.26%, FA 9.66%, SC 2.41%
#
#Eval for threshold 0.30: DER 10.54%, MS 2.03%, FA 5.50%, SC 3.02%
#
#Eval for threshold 0.35: DER 9.86%, MS 2.46%, FA 4.15%, SC 3.25%
#
#Eval for threshold 0.40: DER 9.42%, MS 2.95%, FA 3.04%, SC 3.43%
#
#Eval for threshold 0.45: DER 9.25%, MS 3.58%, FA 2.14%, SC 3.53%
#
#Eval for threshold 0.50: DER 9.29%, MS 4.31%, FA 1.45%, SC 3.53%
#
#Eval for threshold 0.55: DER 9.71%, MS 5.25%, FA 1.03%, SC 3.43%
#
#Eval for threshold 0.60: DER 10.40%, MS 6.46%, FA 0.76%, SC 3.19%
#
#Eval for threshold 0.70: DER 12.26%, MS 9.20%, FA 0.41%, SC 2.65%
#
#Eval for threshold 0.80: DER 15.16%, MS 12.97%, FA 0.20%, SC 1.99%
fi


## debug
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ] ;then
   # it adds noise to train tsvad model , no grad-clip and no freeze update.
   # # speech encoder is cam++ , oracle target speaker embedding is from cam++ pretrain model
   # this cam++ pretrain model is trained on cn-cnceleb and voxceleb dataset,
   # checkpoint is from https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/files
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"

    speaker_encoder_type=$speech_encoder_type
    speaker_encoder_path=$speech_encoder_path
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_two_gpus_unfreeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_frame_speaker_embedding_debug
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 13815 \
   ts_vad3/train_accelerate_ddp2_debug_hltsz.py\
     --verbose 1 \
     --batch-size 64\
     --world-size 2 \
     --num-epochs 20\
     --start-epoch 1\
     --keep-last-k 1\
     --keep-last-epoch 1\
     --freeze-speech-encoder-updates 4000\
     --freeze-speaker-encoder-updates 65000\
     --fuse-fbank-feat false\
     --fuse-speaker-embedding-feat true\
     --ts-len 4\
     --grad-clip true\
     --lr 2e-4\
     --musan-path $musan_path \
     --rir-path $rir_path \
     --speech-encoder-type $speech_encoder_type\
     --speech-encoder-path $speech_encoder_path\
     --speaker-encoder-type $speaker_encoder_type\
     --speaker-encoder-path $speaker_encoder_path\
     --select-encoder-layer-nums 6\
     --exp-dir $exp_dir\
     --data-dir $data_dir
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_two_gpus_unfreeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_frame_speaker_embedding_debug
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
 speaker_encoder_type=$speech_encoder_type
 speaker_encoder_path=$speech_encoder_path
 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
     results_path=$exp_dir
  python3 ts_vad3/infer2_hltsz.py \
    --model-file $model_file\
    --ts-len 4\
    --rs-len $rs_len\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat true\
    --data-dir $data_dir
done
#sbatch --nodes 1  --gres=gpu:1  --cpus-per-gpu=8  --ntasks-per-node 1  -p speech -o logs/run_ts_vad3_hltsz_stage11_debug.log run_ts_vad3_hltsz.sh --stage 11 --stop-stage 11
## cat logs/run_ts_vad3_hltsz_stage11_debug.log 
#Eval set
#Model DER:  0.14416208514546325
#Model ACC:  0.9482958151166204
#100%|██████████| 25/25 [00:25<00:00,  1.03s/it]
#Eval for threshold 0.20: DER 10.34%, MS 1.02%, FA 8.74%, SC 0.58%
#
#Eval for threshold 0.30: DER 7.35%, MS 1.63%, FA 5.03%, SC 0.69%
#
#Eval for threshold 0.35: DER 6.54%, MS 1.97%, FA 3.85%, SC 0.72%
#
#Eval for threshold 0.40: DER 6.06%, MS 2.37%, FA 2.94%, SC 0.76%
#
#Eval for threshold 0.45: DER 5.84%, MS 2.83%, FA 2.22%, SC 0.80%
#
#Eval for threshold 0.50: DER 5.85%, MS 3.36%, FA 1.66%, SC 0.83%
#
#Eval for threshold 0.55: DER 6.11%, MS 4.05%, FA 1.30%, SC 0.77%
#
#Eval for threshold 0.60: DER 6.54%, MS 4.91%, FA 0.96%, SC 0.68%
#
#Eval for threshold 0.70: DER 7.91%, MS 6.87%, FA 0.59%, SC 0.46%
#
#Eval for threshold 0.80: DER 10.54%, MS 9.90%, FA 0.39%, SC 0.25%
#
#Test set
#Model DER:  0.1485346198755985
#Model ACC:  0.9420814485035405
#100%|██████████| 60/60 [01:04<00:00,  1.08s/it]
#Eval for threshold 0.20: DER 12.33%, MS 1.30%, FA 9.55%, SC 1.48%
#
#Eval for threshold 0.30: DER 9.41%, MS 2.04%, FA 5.35%, SC 2.01%
#
#Eval for threshold 0.35: DER 8.65%, MS 2.48%, FA 3.97%, SC 2.21%
#
#Eval for threshold 0.40: DER 8.23%, MS 2.99%, FA 2.89%, SC 2.35%
#
#Eval for threshold 0.45: DER 8.05%, MS 3.58%, FA 2.04%, SC 2.43%
#
#Eval for threshold 0.50: DER 8.20%, MS 4.36%, FA 1.42%, SC 2.41%
#
#Eval for threshold 0.55: DER 8.60%, MS 5.31%, FA 1.01%, SC 2.28%
#
#Eval for threshold 0.60: DER 9.22%, MS 6.44%, FA 0.69%, SC 2.08%
#
#Eval for threshold 0.70: DER 11.07%, MS 9.22%, FA 0.31%, SC 1.54%
#
#Eval for threshold 0.80: DER 13.98%, MS 12.81%, FA 0.14%, SC 1.04%
fi

## 
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ] ;then
   # it adds noise to train tsvad model , no grad-clip and no freeze update.
   # # speech encoder is cam++ , oracle target speaker embedding is from cam++ pretrain model
   # this cam++ pretrain model is trained on cn-cnceleb and voxceleb dataset,
   # checkpoint is from https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/files
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"

    speaker_encoder_type=$speech_encoder_type
    speaker_encoder_path=$speech_encoder_path
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_two_gpus_freeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 13815 \
   ts_vad3/train_accelerate_ddp2_debug_hltsz.py\
     --verbose 1 \
     --batch-size 64\
     --world-size 2 \
     --num-epochs 20\
     --start-epoch 1\
     --keep-last-k 1\
     --keep-last-epoch 1\
     --freeze-speech-encoder-updates 4000\
     --freeze-speaker-encoder-updates 65000\
     --fuse-fbank-feat false\
     --fuse-speaker-embedding-feat false\
     --ts-len 4\
     --grad-clip true\
     --lr 2e-4\
     --musan-path $musan_path \
     --rir-path $rir_path \
     --speech-encoder-type $speech_encoder_type\
     --speech-encoder-path $speech_encoder_path\
     --speaker-encoder-type $speaker_encoder_type\
     --speaker-encoder-path $speaker_encoder_path\
     --select-encoder-layer-nums 6\
     --exp-dir $exp_dir\
     --data-dir $data_dir
  fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_two_gpus_freeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
 speaker_encoder_type=$speech_encoder_type
 speaker_encoder_path=$speech_encoder_path
 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
     results_path=$exp_dir
  python3 ts_vad3/infer2_hltsz.py \
    --model-file $model_file\
    --ts-len 4\
    --rs-len $rs_len\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
     --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat false\
    --data-dir $data_dir
done
#sbatch --nodes 1  --gres=gpu:2  --cpus-per-gpu=8  --ntasks-per-node 2  -p speech -o logs/run_ts_vad3_hltsz_stage12_13.log run_ts_vad3_hltsz.sh --stage 12 --stop-stage 13
#cat logs/run_ts_vad3_hltsz_stage12_13.log
#Eval set
#Model DER:  0.14453844061761284
#Model ACC:  0.9473025927335725
#100%|██████████| 25/25 [00:27<00:00,  1.10s/it]
#Eval for threshold 0.20: DER 9.84%, MS 1.10%, FA 7.99%, SC 0.75%
#
#Eval for threshold 0.30: DER 7.35%, MS 1.76%, FA 4.65%, SC 0.94%
#
#Eval for threshold 0.35: DER 6.74%, MS 2.14%, FA 3.57%, SC 1.03%
#
#Eval for threshold 0.40: DER 6.44%, MS 2.60%, FA 2.71%, SC 1.12%
#
#Eval for threshold 0.45: DER 6.29%, MS 3.06%, FA 2.07%, SC 1.16%
#
#Eval for threshold 0.50: DER 6.32%, MS 3.64%, FA 1.55%, SC 1.13%
#
#Eval for threshold 0.55: DER 6.51%, MS 4.30%, FA 1.16%, SC 1.04%
#
#Eval for threshold 0.60: DER 6.92%, MS 5.09%, FA 0.93%, SC 0.91%
#
#Eval for threshold 0.70: DER 8.18%, MS 6.93%, FA 0.59%, SC 0.66%
#
#Eval for threshold 0.80: DER 10.68%, MS 9.83%, FA 0.42%, SC 0.43%
#
#Test set
#Model DER:  0.14661972304454005
#Model ACC:  0.9430303847736495
#100%|██████████| 60/60 [01:07<00:00,  1.12s/it]
#Eval for threshold 0.20: DER 11.50%, MS 1.32%, FA 8.92%, SC 1.25%
#
#Eval for threshold 0.30: DER 8.83%, MS 2.13%, FA 5.06%, SC 1.64%
#
#Eval for threshold 0.35: DER 8.20%, MS 2.60%, FA 3.80%, SC 1.80%
#
#Eval for threshold 0.40: DER 7.87%, MS 3.15%, FA 2.82%, SC 1.90%
#
#Eval for threshold 0.45: DER 7.83%, MS 3.83%, FA 2.05%, SC 1.95%
#
#Eval for threshold 0.50: DER 7.99%, MS 4.65%, FA 1.43%, SC 1.91%
#
#Eval for threshold 0.55: DER 8.40%, MS 5.61%, FA 1.01%, SC 1.79%
#
#Eval for threshold 0.60: DER 9.03%, MS 6.73%, FA 0.70%, SC 1.60%
#
#Eval for threshold 0.70: DER 10.81%, MS 9.26%, FA 0.36%, SC 1.20%
#
#Eval for threshold 0.80: DER 13.80%, MS 12.88%, FA 0.18%, SC 0.74%

fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ] ;then
   # it adds noise to train tsvad model , no grad-clip and no freeze update.
   # # speech encoder is cam++ , oracle target speaker embedding is from cam++ pretrain model
   # this cam++ pretrain model is trained on cn-cnceleb and voxceleb dataset,
   # checkpoint is from https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/files
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"

    speaker_encoder_type=$speech_encoder_type
    speaker_encoder_path=$speech_encoder_path
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_two_gpus_freeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_fbank_feat
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 14815 \
   ts_vad3/train_accelerate_ddp2_debug_hltsz.py\
     --verbose 1 \
     --batch-size 64\
     --world-size 2 \
     --num-epochs 20\
     --start-epoch 1\
     --keep-last-k 1\
     --keep-last-epoch 1\
     --freeze-speech-encoder-updates 4000\
     --freeze-speaker-encoder-updates 65000\
     --fuse-fbank-feat true\
     --fuse-speaker-embedding-feat false\
     --ts-len 4\
     --grad-clip true\
     --lr 2e-4\
     --musan-path $musan_path \
     --rir-path $rir_path \
     --speech-encoder-type $speech_encoder_type\
     --speech-encoder-path $speech_encoder_path\
     --speaker-encoder-type $speaker_encoder_type\
     --speaker-encoder-path $speaker_encoder_path\
     --select-encoder-layer-nums 6\
     --exp-dir $exp_dir\
     --data-dir $data_dir
  fi


 if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_two_gpus_freeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_fbank_feat
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
 speaker_encoder_type=$speech_encoder_type
 speaker_encoder_path=$speech_encoder_path
 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
     results_path=$exp_dir
  python3 ts_vad3/infer2_hltsz.py \
    --model-file $model_file\
    --ts-len 4\
    --rs-len $rs_len\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat true\
    --fuse-speaker-embedding-feat false\
    --data-dir $data_dir
done
#sbatch --nodes 1  --gres=gpu:2  --cpus-per-gpu=8  --ntasks-per-node 2  -p speech -o logs/run_ts_vad3_hltsz_stage14_15.log run_ts_vad3_hltsz.sh --stage 14 --stop-stage 15
#cat logs/run_ts_vad3_hltsz_stage14_15.log
#Eval set
#Model DER:  0.23299124623682432
#Model ACC:  0.906049787239833
#100%|██████████| 25/25 [00:28<00:00,  1.14s/it]
#Eval for threshold 0.20: DER 22.13%, MS 1.93%, FA 14.39%, SC 5.81%
#
#Eval for threshold 0.30: DER 17.30%, MS 3.01%, FA 7.89%, SC 6.40%
#
#Eval for threshold 0.35: DER 16.12%, MS 3.65%, FA 5.84%, SC 6.63%
#
#Eval for threshold 0.40: DER 15.31%, MS 4.34%, FA 4.18%, SC 6.79%
#
#Eval for threshold 0.45: DER 15.11%, MS 5.16%, FA 3.04%, SC 6.91%
#
#Eval for threshold 0.50: DER 15.05%, MS 6.13%, FA 2.02%, SC 6.90%
#
#Eval for threshold 0.55: DER 15.56%, MS 7.45%, FA 1.41%, SC 6.71%
#
#Eval for threshold 0.60: DER 16.39%, MS 8.92%, FA 1.03%, SC 6.45%
#
#Eval for threshold 0.70: DER 18.69%, MS 12.51%, FA 0.53%, SC 5.65%
#
#Eval for threshold 0.80: DER 22.63%, MS 17.49%, FA 0.33%, SC 4.80%
#
#Test set
#Model DER:  0.21790938549920874
#Model ACC:  0.9112571213029962
#100%|██████████| 60/60 [01:08<00:00,  1.14s/it]
#Eval for threshold 0.20: DER 24.16%, MS 1.72%, FA 19.43%, SC 3.00%
#
#Eval for threshold 0.30: DER 18.20%, MS 2.90%, FA 11.29%, SC 4.01%
#
#Eval for threshold 0.35: DER 16.39%, MS 3.63%, FA 8.34%, SC 4.42%
#
#Eval for threshold 0.40: DER 15.16%, MS 4.47%, FA 5.89%, SC 4.79%
#
#Eval for threshold 0.45: DER 14.43%, MS 5.49%, FA 3.87%, SC 5.06%
#
#Eval for threshold 0.50: DER 14.35%, MS 6.90%, FA 2.36%, SC 5.08%
#
#Eval for threshold 0.55: DER 15.13%, MS 8.92%, FA 1.48%, SC 4.73%
#
#Eval for threshold 0.60: DER 16.33%, MS 11.25%, FA 0.90%, SC 4.18%
#
#Eval for threshold 0.70: DER 20.02%, MS 16.70%, FA 0.34%, SC 2.98%
#
#Eval for threshold 0.80: DER 25.75%, MS 23.83%, FA 0.09%, SC 1.83%
 
 fi


#compared with stage10-11,stage16-17 will only freeze first 40k steps speaker encoder.
if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ] ;then
   # it adds noise to train tsvad model , no grad-clip and no freeze update.
   # # speech encoder is cam++ , oracle target speaker embedding is from cam++ pretrain model
   # this cam++ pretrain model is trained on cn-cnceleb and voxceleb dataset,
   # checkpoint is from https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/files
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"

    speaker_encoder_type=$speech_encoder_type
    speaker_encoder_path=$speech_encoder_path
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_two_gpus_freeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_speaker_embedding_feat_freeze_40k_steps_speaker_encoder
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15815 \
   ts_vad3/train_accelerate_ddp2_debug_hltsz2.py\
     --verbose 1 \
     --batch-size 64\
     --world-size 2 \
     --num-epochs 20\
     --start-epoch 1\
     --keep-last-k 1\
     --keep-last-epoch 1\
     --freeze-speech-encoder-updates 4000\
     --freeze-speaker-encoder-updates 40000\
     --fuse-fbank-feat false\
     --fuse-speaker-embedding-feat true\
     --ts-len 4\
     --grad-clip true\
     --lr 2e-4\
     --musan-path $musan_path \
     --rir-path $rir_path \
     --speech-encoder-type $speech_encoder_type\
     --speech-encoder-path $speech_encoder_path\
     --speaker-encoder-type $speaker_encoder_type\
     --speaker-encoder-path $speaker_encoder_path\
     --select-encoder-layer-nums 6\
     --exp-dir $exp_dir\
     --data-dir $data_dir
  fi
if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_two_gpus_freeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_speaker_embedding_feat_freeze_40k_steps_speaker_encoder
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
 speaker_encoder_type=$speech_encoder_type
 speaker_encoder_path=$speech_encoder_path
 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
     results_path=$exp_dir
  python3 ts_vad3/infer2_hltsz.py \
    --model-file $model_file\
    --ts-len 4\
    --rs-len $rs_len\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat true\
    --data-dir $data_dir
done
#cat logs/run_ts_vad3_hltsz_stage17.log
#Eval set
#Model DER:  0.14439890546203255
#Model ACC:  0.9482329488531719
#100%|██████████| 25/25 [00:27<00:00,  1.08s/it]
#Eval for threshold 0.20: DER 10.34%, MS 1.02%, FA 8.75%, SC 0.58%
#
#Eval for threshold 0.30: DER 7.35%, MS 1.63%, FA 5.04%, SC 0.68%
#
#Eval for threshold 0.35: DER 6.55%, MS 1.99%, FA 3.85%, SC 0.71%
#
#Eval for threshold 0.40: DER 6.06%, MS 2.37%, FA 2.94%, SC 0.76%
#
#Eval for threshold 0.45: DER 5.85%, MS 2.81%, FA 2.24%, SC 0.80%
#
#Eval for threshold 0.50: DER 5.83%, MS 3.35%, FA 1.65%, SC 0.83%
#
#Eval for threshold 0.55: DER 6.09%, MS 4.06%, FA 1.26%, SC 0.78%
#
#Eval for threshold 0.60: DER 6.53%, MS 4.91%, FA 0.94%, SC 0.67%
#
#Eval for threshold 0.70: DER 7.90%, MS 6.86%, FA 0.58%, SC 0.46%
#
#Eval for threshold 0.80: DER 10.55%, MS 9.91%, FA 0.39%, SC 0.25%
#
#Test set
#Model DER:  0.14861788025128192
#Model ACC:  0.9420796584931201
#100%|██████████| 60/60 [01:04<00:00,  1.08s/it]
#Eval for threshold 0.20: DER 12.31%, MS 1.29%, FA 9.53%, SC 1.49%
#
#Eval for threshold 0.30: DER 9.38%, MS 2.04%, FA 5.34%, SC 2.00%
#
#Eval for threshold 0.35: DER 8.66%, MS 2.47%, FA 3.98%, SC 2.21%
#
#Eval for threshold 0.40: DER 8.22%, MS 2.99%, FA 2.89%, SC 2.34%
#
#Eval for threshold 0.45: DER 8.08%, MS 3.60%, FA 2.04%, SC 2.43%
#
#Eval for threshold 0.50: DER 8.20%, MS 4.39%, FA 1.42%, SC 2.39%
#
#Eval for threshold 0.55: DER 8.59%, MS 5.32%, FA 1.00%, SC 2.27%
#
#Eval for threshold 0.60: DER 9.20%, MS 6.43%, FA 0.69%, SC 2.08%
#
#Eval for threshold 0.70: DER 11.04%, MS 9.20%, FA 0.31%, SC 1.54%
#
#Eval for threshold 0.80: DER 13.97%, MS 12.78%, FA 0.14%, SC 1.05%

fi

#compared with stage10-11,stage18-19 will only freeze first 4k steps speaker encoder.
if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ] ;then
   # it adds noise to train tsvad model , no grad-clip and no freeze update.
   # # speech encoder is cam++ , oracle target speaker embedding is from cam++ pretrain model
   # this cam++ pretrain model is trained on cn-cnceleb and voxceleb dataset,
   # checkpoint is from https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/files
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"

    speaker_encoder_type=$speech_encoder_type
    speaker_encoder_path=$speech_encoder_path
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_two_gpus_freeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_speaker_embedding_feat_freeze_4k_steps_speaker_encoder
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15115 \
   ts_vad3/train_accelerate_ddp2_debug_hltsz2.py\
     --verbose 1 \
     --batch-size 64\
     --world-size 2 \
     --num-epochs 20\
     --start-epoch 1\
     --keep-last-k 1\
     --keep-last-epoch 1\
     --freeze-speech-encoder-updates 4000\
     --freeze-speaker-encoder-updates 4000\
     --fuse-fbank-feat false\
     --fuse-speaker-embedding-feat true\
     --ts-len 4\
     --grad-clip true\
     --lr 2e-4\
     --musan-path $musan_path \
     --rir-path $rir_path \
     --speech-encoder-type $speech_encoder_type\
     --speech-encoder-path $speech_encoder_path\
     --speaker-encoder-type $speaker_encoder_type\
     --speaker-encoder-path $speaker_encoder_path\
     --select-encoder-layer-nums 6\
     --exp-dir $exp_dir\
     --data-dir $data_dir
fi
if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_two_gpus_freeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_speaker_embedding_feat_freeze_4k_steps_speaker_encoder
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
 speaker_encoder_type=$speech_encoder_type
 speaker_encoder_path=$speech_encoder_path
 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
     results_path=$exp_dir
  python3 ts_vad3/infer2_hltsz.py \
    --model-file $model_file\
    --ts-len 4\
    --rs-len $rs_len\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat true\
    --data-dir $data_dir
done
#cat logs/run_ts_vad3_hltsz_stage19.log
## Eval set
#Model DER:  0.1323866540168438
#Model ACC:  0.9532206825304566
#100%|██████████| 25/25 [00:26<00:00,  1.07s/it]
#Eval for threshold 0.20: DER 9.21%, MS 0.96%, FA 7.74%, SC 0.50%
#
#Eval for threshold 0.30: DER 6.74%, MS 1.50%, FA 4.60%, SC 0.64%
#
#Eval for threshold 0.35: DER 6.10%, MS 1.81%, FA 3.61%, SC 0.67%
#
#Eval for threshold 0.40: DER 5.72%, MS 2.15%, FA 2.90%, SC 0.67%
#
#Eval for threshold 0.45: DER 5.47%, MS 2.54%, FA 2.25%, SC 0.68%
#
#Eval for threshold 0.50: DER 5.40%, MS 3.01%, FA 1.74%, SC 0.65%
#
#Eval for threshold 0.55: DER 5.58%, MS 3.59%, FA 1.39%, SC 0.60%
#
#Eval for threshold 0.60: DER 5.86%, MS 4.26%, FA 1.07%, SC 0.53%
#
#Eval for threshold 0.70: DER 7.00%, MS 5.86%, FA 0.71%, SC 0.44%
#
#Eval for threshold 0.80: DER 9.09%, MS 8.35%, FA 0.47%, SC 0.27%
#
## Test set
#Model DER:  0.13730100646488053
#Model ACC:  0.9479467755684706
#100%|██████████| 60/60 [01:02<00:00,  1.04s/it]
#Eval for threshold 0.20: DER 11.19%, MS 1.22%, FA 9.11%, SC 0.86%
#
#Eval for threshold 0.30: DER 8.49%, MS 1.88%, FA 5.46%, SC 1.16%
#
#Eval for threshold 0.35: DER 7.78%, MS 2.27%, FA 4.24%, SC 1.28%
#
#Eval for threshold 0.40: DER 7.32%, MS 2.70%, FA 3.25%, SC 1.38%
#
#Eval for threshold 0.45: DER 7.04%, MS 3.16%, FA 2.40%, SC 1.48%
#
#Eval for threshold 0.50: DER 7.00%, MS 3.74%, FA 1.74%, SC 1.53%
#
#Eval for threshold 0.55: DER 7.21%, MS 4.48%, FA 1.25%, SC 1.48%
#
#Eval for threshold 0.60: DER 7.69%, MS 5.40%, FA 0.91%, SC 1.37%
#
#Eval for threshold 0.70: DER 9.08%, MS 7.59%, FA 0.43%, SC 1.07%
#
#Eval for threshold 0.80: DER 11.66%, MS 10.76%, FA 0.20%, SC 0.70%

fi

# compared with stage10-11, stage20-21 will use fuse_attn_type=attn 
if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ] ;then
   # it adds noise to train tsvad model , no grad-clip and no freeze update.
   # # speech encoder is cam++ , oracle target speaker embedding is from cam++ pretrain model
   # this cam++ pretrain model is trained on cn-cnceleb and voxceleb dataset,
   # checkpoint is from https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/files
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"

    speaker_encoder_type=$speech_encoder_type
    speaker_encoder_path=$speech_encoder_path
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_two_gpus_unfreeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_frame_speaker_embedding_with_attn_type
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 13815 \
   ts_vad3/train_accelerate_ddp2_debug_hltsz2.py\
     --verbose 1 \
     --batch-size 64\
     --world-size 2 \
     --num-epochs 20\
     --start-epoch 1\
     --keep-last-k 1\
     --keep-last-epoch 1\
     --freeze-speech-encoder-updates 4000\
     --freeze-speaker-encoder-updates 65000\
     --fuse-fbank-feat false\
     --fuse-speaker-embedding-feat true\
     --fuse-attn-type "attn"\
     --ts-len 4\
     --grad-clip true\
     --lr 2e-4\
     --musan-path $musan_path \
     --rir-path $rir_path \
     --speech-encoder-type $speech_encoder_type\
     --speech-encoder-path $speech_encoder_path\
     --speaker-encoder-type $speaker_encoder_type\
     --speaker-encoder-path $speaker_encoder_path\
     --select-encoder-layer-nums 6\
     --exp-dir $exp_dir\
     --data-dir $data_dir
fi
if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_two_gpus_unfreeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_frame_speaker_embedding_with_attn_type
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
 speaker_encoder_type=$speech_encoder_type
 speaker_encoder_path=$speech_encoder_path
 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
     results_path=$exp_dir
  python3 ts_vad3/infer2_hltsz.py \
    --model-file $model_file\
    --ts-len 4\
    --rs-len $rs_len\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat true\
    --fuse-attn-type "attn"\
    --data-dir $data_dir
done
#cat  logs/run_ts_vad3_hltsz_stage21.log
#Eval set
#Model DER:  0.14489002810893137
#Model ACC:  0.9491329185544164
#100%|██████████| 25/25 [00:26<00:00,  1.05s/it]
#Eval for threshold 0.20: DER 10.99%, MS 0.95%, FA 9.58%, SC 0.47%
#
#Eval for threshold 0.30: DER 7.53%, MS 1.51%, FA 5.43%, SC 0.59%
#
#Eval for threshold 0.35: DER 6.68%, MS 1.89%, FA 4.13%, SC 0.66%
#
#Eval for threshold 0.40: DER 6.14%, MS 2.32%, FA 3.16%, SC 0.66%
#
#Eval for threshold 0.45: DER 5.84%, MS 2.79%, FA 2.39%, SC 0.66%
#
#Eval for threshold 0.50: DER 5.83%, MS 3.36%, FA 1.81%, SC 0.66%
#
#Eval for threshold 0.55: DER 6.01%, MS 4.02%, FA 1.39%, SC 0.60%
#
#Eval for threshold 0.60: DER 6.36%, MS 4.83%, FA 1.05%, SC 0.48%
#
#Eval for threshold 0.70: DER 7.82%, MS 6.85%, FA 0.62%, SC 0.36%
#
#Eval for threshold 0.80: DER 10.46%, MS 9.80%, FA 0.42%, SC 0.23%
#
#Test set
#Model DER:  0.16216069049774026
#Model ACC:  0.9349973528462656
#100%|██████████| 60/60 [01:02<00:00,  1.04s/it]
#Eval for threshold 0.20: DER 14.74%, MS 1.22%, FA 11.07%, SC 2.45%
#
#Eval for threshold 0.30: DER 11.51%, MS 2.00%, FA 6.36%, SC 3.15%
#
#Eval for threshold 0.35: DER 10.59%, MS 2.47%, FA 4.72%, SC 3.40%
#
#Eval for threshold 0.40: DER 10.08%, MS 3.06%, FA 3.44%, SC 3.58%
#
#Eval for threshold 0.45: DER 9.82%, MS 3.74%, FA 2.39%, SC 3.69%
#
#Eval for threshold 0.50: DER 9.83%, MS 4.58%, FA 1.56%, SC 3.70%
#
#Eval for threshold 0.55: DER 10.25%, MS 5.71%, FA 1.09%, SC 3.45%
#
#Eval for threshold 0.60: DER 10.94%, MS 7.03%, FA 0.74%, SC 3.17%
#
#Eval for threshold 0.70: DER 12.88%, MS 10.08%, FA 0.34%, SC 2.46%
#
#Eval for threshold 0.80: DER 16.04%, MS 14.27%, FA 0.15%, SC 1.62%

fi


# compared with stage10-11, stage22-23 will use fuse_attn_type=attn2
if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ] ;then
   # it adds noise to train tsvad model , no grad-clip and no freeze update.
   # # speech encoder is cam++ , oracle target speaker embedding is from cam++ pretrain model
   # this cam++ pretrain model is trained on cn-cnceleb and voxceleb dataset,
   # checkpoint is from https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/files
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"

    speaker_encoder_type=$speech_encoder_type
    speaker_encoder_path=$speech_encoder_path
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_two_gpus_unfreeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_frame_speaker_embedding_with_attn2_type
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12815 \
   ts_vad3/train_accelerate_ddp2_debug_hltsz2.py\
     --verbose 1 \
     --batch-size 64\
     --world-size 2 \
     --num-epochs 20\
     --start-epoch 1\
     --keep-last-k 1\
     --keep-last-epoch 1\
     --freeze-speech-encoder-updates 4000\
     --freeze-speaker-encoder-updates 65000\
     --fuse-fbank-feat false\
     --fuse-speaker-embedding-feat true\
     --fuse-attn-type "attn2"\
     --ts-len 4\
     --grad-clip true\
     --lr 2e-4\
     --musan-path $musan_path \
     --rir-path $rir_path \
     --speech-encoder-type $speech_encoder_type\
     --speech-encoder-path $speech_encoder_path\
     --speaker-encoder-type $speaker_encoder_type\
     --speaker-encoder-path $speaker_encoder_path\
     --select-encoder-layer-nums 6\
     --exp-dir $exp_dir\
      --data-dir $data_dir
fi
if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_two_gpus_unfreeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_frame_speaker_embedding_with_attn2_type
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
 speaker_encoder_type=$speech_encoder_type
 speaker_encoder_path=$speech_encoder_path
 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
     results_path=$exp_dir
  python3 ts_vad3/infer2_hltsz.py \
    --model-file $model_file\
    --ts-len 4\
    --rs-len $rs_len\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
     --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat true\
    --fuse-attn-type "attn2"\
    --data-dir $data_dir
done
## Model DER:  0.14286480847936261
#Model ACC:  0.9480446881259349
#100%|██████████| 25/25 [00:26<00:00,  1.06s/it]
#Eval for threshold 0.20: DER 9.71%, MS 1.04%, FA 7.97%, SC 0.69%
#
#Eval for threshold 0.30: DER 7.17%, MS 1.67%, FA 4.58%, SC 0.93%
#
#Eval for threshold 0.35: DER 6.52%, MS 2.05%, FA 3.48%, SC 0.98%
#
#Eval for threshold 0.40: DER 6.16%, MS 2.50%, FA 2.68%, SC 0.98%
#
#Eval for threshold 0.45: DER 6.02%, MS 2.98%, FA 2.01%, SC 1.03%
#
#Eval for threshold 0.50: DER 6.12%, MS 3.60%, FA 1.53%, SC 0.99%
#
#Eval for threshold 0.55: DER 6.44%, MS 4.31%, FA 1.17%, SC 0.95%
#
#Eval for threshold 0.60: DER 6.85%, MS 5.06%, FA 0.93%, SC 0.86%
#
#Eval for threshold 0.70: DER 8.26%, MS 7.02%, FA 0.58%, SC 0.66%
#
#Eval for threshold 0.80: DER 10.67%, MS 9.85%, FA 0.39%, SC 0.42%
#
#Test set
#Model DER:  0.1503490812258047
#Model ACC:  0.9402950622123524
#100%|██████████| 60/60 [01:06<00:00,  1.10s/it]
#Eval for threshold 0.20: DER 12.95%, MS 1.29%, FA 10.02%, SC 1.64%
#
#Eval for threshold 0.30: DER 10.17%, MS 2.07%, FA 5.91%, SC 2.19%
#
#Eval for threshold 0.35: DER 9.39%, MS 2.52%, FA 4.44%, SC 2.43%
#
#Eval for threshold 0.40: DER 8.85%, MS 3.06%, FA 3.19%, SC 2.60%
#
#Eval for threshold 0.45: DER 8.64%, MS 3.69%, FA 2.26%, SC 2.70%
#
#Eval for threshold 0.50: DER 8.76%, MS 4.56%, FA 1.54%, SC 2.67%
#
#Eval for threshold 0.55: DER 9.14%, MS 5.58%, FA 1.05%, SC 2.51%
#
#Eval for threshold 0.60: DER 9.76%, MS 6.71%, FA 0.73%, SC 2.32%
#
#Eval for threshold 0.70: DER 11.62%, MS 9.52%, FA 0.35%, SC 1.76%
#
#Eval for threshold 0.80: DER 14.29%, MS 12.99%, FA 0.13%, SC 1.17%

fi

#compared with stage10-11,stage24-25 will only no freeze speaker encoder.
if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ] ;then
   # it adds noise to train tsvad model , no grad-clip and no freeze update.
   # # speech encoder is cam++ , oracle target speaker embedding is from cam++ pretrain model
   # this cam++ pretrain model is trained on cn-cnceleb and voxceleb dataset,
   # checkpoint is from https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/files
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"

    speaker_encoder_type=$speech_encoder_type
    speaker_encoder_path=$speech_encoder_path
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_two_gpus_freeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_speaker_embedding_feat_nofreeze_speaker_encoder
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15115 \
   ts_vad3/train_accelerate_ddp2_debug_hltsz2.py\
     --verbose 1 \
     --batch-size 64\
     --world-size 2 \
     --num-epochs 20\
     --start-epoch 1\
     --keep-last-k 1\
     --keep-last-epoch 1\
     --freeze-speech-encoder-updates 4000\
     --freeze-speaker-encoder-updates 0\
     --fuse-fbank-feat false\
     --fuse-speaker-embedding-feat true\
     --fuse-attn-type "native"\
     --ts-len 4\
     --grad-clip true\
     --lr 2e-4\
     --musan-path $musan_path \
     --rir-path $rir_path \
     --speech-encoder-type $speech_encoder_type\
     --speech-encoder-path $speech_encoder_path\
     --speaker-encoder-type $speaker_encoder_type\
     --speaker-encoder-path $speaker_encoder_path\
     --select-encoder-layer-nums 6\
     --exp-dir $exp_dir\
     --data-dir $data_dir
fi
if [ ${stage} -le 25 ] && [ ${stop_stage} -ge 25 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad3/ts_vad3_two_gpus_freeze_with_musan_rirs_cam++_epoch20_front_fix_seed_lr2e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_speaker_embedding_feat_nofreeze_speaker_encoder
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
 speaker_encoder_type=$speech_encoder_type
 speaker_encoder_path=$speech_encoder_path
 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
     results_path=$exp_dir
  python3 ts_vad3/infer2_hltsz.py \
    --model-file $model_file\
    --ts-len 4\
    --rs-len $rs_len\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
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
    --speaker-encoder-type $speaker_encoder_type\
    --speaker-encoder-path $speaker_encoder_path\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat true\
    --fuse-attn-type "native"\
    --data-dir $data_dir
done
#sbatch --nodes 1  --gres=gpu:1  --cpus-per-gpu=8  --ntasks-per-node 1  -p speech -o logs/run_ts_vad3_hltsz_stage25.log run_ts_vad3_hltsz.sh --stage 25 --stop-stage 25
#cat logs/run_ts_vad3_hltsz_stage25.log
#Eval set
#Model DER:  0.1403190143274699
#Model ACC:  0.95016776615961
#100%|██████████| 25/25 [00:27<00:00,  1.10s/it]
#Eval for threshold 0.20: DER 10.39%, MS 0.91%, FA 8.90%, SC 0.59%
#
#Eval for threshold 0.30: DER 7.29%, MS 1.45%, FA 5.11%, SC 0.73%
#
#Eval for threshold 0.35: DER 6.56%, MS 1.83%, FA 3.94%, SC 0.78%
#
#Eval for threshold 0.40: DER 6.11%, MS 2.28%, FA 3.03%, SC 0.81%
#
#Eval for threshold 0.45: DER 5.91%, MS 2.75%, FA 2.33%, SC 0.83%
#
#Eval for threshold 0.50: DER 5.99%, MS 3.35%, FA 1.81%, SC 0.83%
#
#Eval for threshold 0.55: DER 6.19%, MS 4.08%, FA 1.39%, SC 0.72%
#
#Eval for threshold 0.60: DER 6.54%, MS 4.80%, FA 1.10%, SC 0.64%
#
#Eval for threshold 0.70: DER 7.72%, MS 6.55%, FA 0.67%, SC 0.50%
#
#Eval for threshold 0.80: DER 10.08%, MS 9.32%, FA 0.43%, SC 0.32%
#
#Test set
#Model DER:  0.1382330138096191
#Model ACC:  0.9482536524318965
#100%|██████████| 60/60 [01:05<00:00,  1.08s/it]
#Eval for threshold 0.20: DER 11.67%, MS 1.02%, FA 9.80%, SC 0.85%
#
#Eval for threshold 0.30: DER 8.48%, MS 1.70%, FA 5.64%, SC 1.14%
#
#Eval for threshold 0.35: DER 7.68%, MS 2.09%, FA 4.33%, SC 1.26%
#
#Eval for threshold 0.40: DER 7.19%, MS 2.54%, FA 3.27%, SC 1.38%
#
#Eval for threshold 0.45: DER 6.90%, MS 3.04%, FA 2.41%, SC 1.46%
#
#Eval for threshold 0.50: DER 6.83%, MS 3.62%, FA 1.71%, SC 1.49%
#
#Eval for threshold 0.55: DER 6.98%, MS 4.31%, FA 1.23%, SC 1.45%
#
#Eval for threshold 0.60: DER 7.40%, MS 5.15%, FA 0.90%, SC 1.35%
#
#Eval for threshold 0.70: DER 8.83%, MS 7.34%, FA 0.43%, SC 1.06%
#
#Eval for threshold 0.80: DER 11.36%, MS 10.48%, FA 0.19%, SC 0.69%

fi
