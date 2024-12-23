#!/usr/bin/env bash
stage=0
stop_stage=1000
#split="Eval" # Eval , Train
. utils/parse_options.sh
. path_for_dia_pt2.4.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -o ... 'error in pipeline', -x 'print commands',
set -e
set -o pipefail
## prepared target speaker embedding
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "prepare eval set target audio list"
   dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/200h_fixed_4spks6s
   #for name in Eval Train;do
   for name in Test;do
    if [ $name = "Train" ];then
      input_dir=$dest_dir/${name}_Ali_far/target_audio
    else
      input_dir=$dest_dir/${name}_Ali/${name}_Ali_far/target_audio
    fi
    file=$input_dir/wavs.txt
   python3 ts_vad2_simu/prepare_alimeeting_target_audio_list.py \
        $input_dir $file
   done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "generate eval(dev) and train speaker embedding"
   dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/200h_fixed_4spks6s
   feature_name=cam++_zh-cn_200k_feature_dir
   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
   model_id=iic/speech_campplus_sv_zh-cn_16k-common
   #subsets="Eval Train"
   subsets="Test"
   for name in $subsets;do
    if [ $name = "Train" ];then
     echo "extract train target speaker embedding"
     # 提取embedding
     input_dir=$dest_dir/${name}_Ali_far/target_audio/
     wav_path=$input_dir/wavs.txt
    else
     echo "extract $name target speaker embedding"
     # 提取embedding
     input_dir=$dest_dir/${name}_Ali/${name}_Ali_far/target_audio/
     wav_path=$input_dir/wavs.txt
    fi
     save_dir=$dest_dir/ts_vad/spk_embed/alimeeting/${feat_type}SpeakerEmbedding/$name/$feature_name
     python3 ts_vad2_simu/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size 96
   done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    simu_data_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/200h_fixed_4spks6s
    spk_path=$simu_data_dir/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spks6s
    data_dir="$simu_data_dir" # oracle target audio , mix audio and labels path

    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12915 \
    ts_vad2_simu/train_accelerate_ddp2_debug2.py \
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
        --speech-encoder-type $speech_encoder_type\
        --speech-encoder-path $speech_encoder_path\
        --select-encoder-layer-nums 6\
        --spk-path $spk_path\
        --speaker-embedding-name-dir $speaker_embedding_name_dir\
        --exp-dir $exp_dir\
         --data-dir $data_dir
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spks6s
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 #infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # for loading speaker embedding file(oracle)
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 #data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 # simulation data(for debug)
 infer_sets="Test"
 spk_path=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/200h_fixed_4spks6s/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/200h_fixed_4spks6s" # target audio , mix audio and labels path
 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad2_simu/infer2.py \
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
    --data-dir $data_dir
done
#sbatch --nodes 1 --gres=gpu:2  --cpus-per-gpu=6  --ntasks-per-node 2  -p p-A100 -A t00120220002 -o logs/run_ts_vad2_simu_stage1-4.log  run_ts_vad2_simu.sh --stage 1 --stop-stage 4
#cat logs/run_ts_vad2_simu_stage1-4.log
#Eval set
#Model DER:  0.20029664489416713
#Model ACC:  0.9324902393479155
#100%|██████████| 25/25 [00:15<00:00,  1.62it/s]
#Eval for threshold 0.20: DER 10.96%, MS 1.81%, FA 8.53%, SC 0.61%
#
#Eval for threshold 0.30: DER 9.38%, MS 2.42%, FA 6.28%, SC 0.68%
#
#Eval for threshold 0.35: DER 8.98%, MS 2.74%, FA 5.55%, SC 0.69%
#
#Eval for threshold 0.40: DER 8.71%, MS 3.08%, FA 4.90%, SC 0.73%
#
#Eval for threshold 0.45: DER 8.62%, MS 3.51%, FA 4.40%, SC 0.71%
#
#Eval for threshold 0.50: DER 8.60%, MS 3.96%, FA 3.98%, SC 0.66%
#
#Eval for threshold 0.55: DER 8.74%, MS 4.51%, FA 3.62%, SC 0.61%
#
#Eval for threshold 0.60: DER 8.98%, MS 5.11%, FA 3.36%, SC 0.51%
#
#Eval for threshold 0.70: DER 9.73%, MS 6.48%, FA 2.89%, SC 0.35%
#
#Eval for threshold 0.80: DER 11.10%, MS 8.61%, FA 2.32%, SC 0.17%
#
#Test set
#Model DER:  0.20089510513835257
#Model ACC:  0.9284977389595609
#100%|██████████| 60/60 [00:37<00:00,  1.60it/s]
#Eval for threshold 0.20: DER 12.61%, MS 2.35%, FA 9.17%, SC 1.09%
#
#Eval for threshold 0.30: DER 10.87%, MS 3.15%, FA 6.57%, SC 1.15%
#
#Eval for threshold 0.35: DER 10.42%, MS 3.56%, FA 5.72%, SC 1.14%
#
#Eval for threshold 0.40: DER 10.10%, MS 3.99%, FA 5.00%, SC 1.12%
#
#Eval for threshold 0.45: DER 9.93%, MS 4.43%, FA 4.41%, SC 1.10%
#
#Eval for threshold 0.50: DER 9.85%, MS 4.90%, FA 3.89%, SC 1.06%
#
#Eval for threshold 0.55: DER 9.93%, MS 5.47%, FA 3.47%, SC 0.98%
#
#Eval for threshold 0.60: DER 10.14%, MS 6.13%, FA 3.13%, SC 0.88%
#
#Eval for threshold 0.70: DER 10.79%, MS 7.65%, FA 2.51%, SC 0.63%
#
#Eval for threshold 0.80: DER 12.26%, MS 9.97%, FA 1.90%, SC 0.39%

# simulation Test data
#Model DER:  0.2007620962678762
#Model ACC:  0.9285541903645077
#100%|██████████| 60/60 [00:37<00:00,  1.59it/s]
#Eval for threshold 0.20: DER 12.59%, MS 2.35%, FA 9.15%, SC 1.09%
#
#Eval for threshold 0.30: DER 10.84%, MS 3.15%, FA 6.54%, SC 1.15%
#
#Eval for threshold 0.35: DER 10.39%, MS 3.55%, FA 5.70%, SC 1.14%
#
#Eval for threshold 0.40: DER 10.10%, MS 3.99%, FA 4.99%, SC 1.12%
#
#Eval for threshold 0.45: DER 9.93%, MS 4.42%, FA 4.41%, SC 1.10%
#
#Eval for threshold 0.50: DER 9.86%, MS 4.91%, FA 3.90%, SC 1.06%
#
#Eval for threshold 0.55: DER 9.93%, MS 5.47%, FA 3.47%, SC 0.98%
#
#Eval for threshold 0.60: DER 10.16%, MS 6.15%, FA 3.13%, SC 0.88%
#
#Eval for threshold 0.70: DER 10.77%, MS 7.63%, FA 2.50%, SC 0.63%
#
#Eval for threshold 0.80: DER 12.26%, MS 9.97%, FA 1.91%, SC 0.38

fi


## prepared target speaker embedding
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "prepare eval set target audio list"
   dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/200h_fixed_4spkscat_16s
   for name in Eval Train;do
    if [ $name = "Train" ];then
      input_dir=$dest_dir/${name}_Ali_far/target_audio
    else
      input_dir=$dest_dir/${name}_Ali/${name}_Ali_far/target_audio
    fi
    file=$input_dir/wavs.txt
   python3 ts_vad2_simu/prepare_alimeeting_target_audio_list.py \
        $input_dir $file
   done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "generate eval(dev) and train speaker embedding"
   dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/200h_fixed_4spkscat_16s
   feature_name=cam++_zh-cn_200k_feature_dir
   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
   model_id=iic/speech_campplus_sv_zh-cn_16k-common
   subsets="Eval Train"
   for name in $subsets;do
    if [ $name = "Train" ];then
     echo "extract train target speaker embedding"
     # 提取embedding
     input_dir=$dest_dir/${name}_Ali_far/target_audio/
     wav_path=$input_dir/wavs.txt
    else
     echo "extract $name target speaker embedding"
     # 提取embedding
     input_dir=$dest_dir/${name}_Ali/${name}_Ali_far/target_audio/
     wav_path=$input_dir/wavs.txt
    fi
     save_dir=$dest_dir/ts_vad/spk_embed/alimeeting/${feat_type}SpeakerEmbedding/$name/$feature_name
     python3 ts_vad2_simu/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size 96
   done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    simu_data_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/200h_fixed_4spkscat_16s
    spk_path=$simu_data_dir/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spkscat_16s
    data_dir="$simu_data_dir" # oracle target audio , mix audio and labels path

    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12615 \
    ts_vad2_simu/train_accelerate_ddp2_debug2.py \
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
        --speech-encoder-type $speech_encoder_type\
        --speech-encoder-path $speech_encoder_path\
        --select-encoder-layer-nums 6\
        --spk-path $spk_path\
        --speaker-embedding-name-dir $speaker_embedding_name_dir\
        --exp-dir $exp_dir\
        --data-dir $data_dir
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spkscat_16s
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad2_simu/infer2.py \
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
    --data-dir $data_dir
done
# sbatch --nodes 1 --gres=gpu:2  --cpus-per-gpu=6  --ntasks-per-node 2  -p p-A100 -A t00120220002 -o logs/run_ts_vad2_simu_stage7-8.log  run_ts_vad2_simu.sh --stage 7 --stop-stage 8
# Eval set
# Model DER:  0.21015868594236267
#Model ACC:  0.9285698821310417
#100%|██████████| 25/25 [00:15<00:00,  1.57it/s]
#Eval for threshold 0.20: DER 10.91%, MS 2.53%, FA 7.51%, SC 0.87%
#
#Eval for threshold 0.30: DER 9.86%, MS 3.49%, FA 5.49%, SC 0.88%
#
#Eval for threshold 0.35: DER 9.68%, MS 4.00%, FA 4.82%, SC 0.86%
#
#Eval for threshold 0.40: DER 9.66%, MS 4.50%, FA 4.33%, SC 0.82%
#
#Eval for threshold 0.45: DER 9.71%, MS 5.02%, FA 3.86%, SC 0.84%
#
#Eval for threshold 0.50: DER 9.91%, MS 5.66%, FA 3.48%, SC 0.77%
#
#Eval for threshold 0.55: DER 10.20%, MS 6.31%, FA 3.18%, SC 0.70%
#
#Eval for threshold 0.60: DER 10.48%, MS 6.95%, FA 2.94%, SC 0.60%
#
#Eval for threshold 0.70: DER 11.43%, MS 8.56%, FA 2.51%, SC 0.36%
#
#Eval for threshold 0.80: DER 13.04%, MS 10.81%, FA 2.04%, SC 0.19%
#Test set
#Model DER:  0.21391318460865533
#Model ACC:  0.9230602765971457
#100%|██████████| 60/60 [00:38<00:00,  1.56it/s]
#Eval for threshold 0.20: DER 13.74%, MS 3.25%, FA 9.32%, SC 1.17%
#
#Eval for threshold 0.30: DER 12.17%, MS 4.43%, FA 6.47%, SC 1.27%
#
#Eval for threshold 0.35: DER 11.76%, MS 5.01%, FA 5.48%, SC 1.28%
#
#Eval for threshold 0.40: DER 11.57%, MS 5.63%, FA 4.71%, SC 1.23%
#
#Eval for threshold 0.45: DER 11.47%, MS 6.27%, FA 3.99%, SC 1.22%
#
#Eval for threshold 0.50: DER 11.51%, MS 6.96%, FA 3.38%, SC 1.17%
#
#Eval for threshold 0.55: DER 11.77%, MS 7.77%, FA 2.98%, SC 1.02%
#
#Eval for threshold 0.60: DER 12.15%, MS 8.61%, FA 2.67%, SC 0.86%
#
#Eval for threshold 0.70: DER 13.39%, MS 10.70%, FA 2.14%, SC 0.54%
#
#Eval for threshold 0.80: DER 15.39%, MS 13.41%, FA 1.69%, SC 0.29%

fi

# compared with stage3-4, stage9-10 will use real alimeeting Eval set as eval set in train simulation data.
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    simu_data_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/200h_fixed_4spks6s
    spk_path=$simu_data_dir/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
    eval_test_spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spks6s_real_evalset
    data_dir="$simu_data_dir" # oracle target audio , mix audio and labels path
    eval_test_data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12915 \
    ts_vad2_simu/train_accelerate_ddp2_debug2.py \
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
        --speech-encoder-type $speech_encoder_type\
        --speech-encoder-path $speech_encoder_path\
        --select-encoder-layer-nums 6\
        --spk-path $spk_path\
        --eval-test-spk-path $eval_test_spk_path\
        --speaker-embedding-name-dir $speaker_embedding_name_dir\
        --exp-dir $exp_dir\
        --data-dir $data_dir\
        --eval-test-data-dir $eval_test_data_dir
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spks6s_real_evalset
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 eval_test_data_dir=$data_dir
 eval_test_spk_path=$spk_path
 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad2_simu/infer2.py \
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
    --eval-test-spk-path $eval_test_spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --eval-test-data-dir $eval_test_data_dir
  done
# cat logs/run_ts_vad2_simu_stage10.log
#Eval set
#Model DER:  0.19868167091046943
#Model ACC:  0.9329986148658959
#100%|██████████| 25/25 [00:15<00:00,  1.60it/s]
#Eval for threshold 0.20: DER 10.94%, MS 1.81%, FA 8.51%, SC 0.62%
#
#Eval for threshold 0.30: DER 9.38%, MS 2.42%, FA 6.25%, SC 0.71%
#
#Eval for threshold 0.35: DER 9.03%, MS 2.76%, FA 5.54%, SC 0.73%
#
#Eval for threshold 0.40: DER 8.74%, MS 3.09%, FA 4.91%, SC 0.75%
#
#Eval for threshold 0.45: DER 8.57%, MS 3.48%, FA 4.38%, SC 0.72%
#
#Eval for threshold 0.50: DER 8.53%, MS 3.88%, FA 3.97%, SC 0.68%
#
#Eval for threshold 0.55: DER 8.61%, MS 4.39%, FA 3.60%, SC 0.62%
#
#Eval for threshold 0.60: DER 8.85%, MS 4.97%, FA 3.34%, SC 0.54%
#
#Eval for threshold 0.70: DER 9.48%, MS 6.24%, FA 2.82%, SC 0.42%
#
#Eval for threshold 0.80: DER 10.90%, MS 8.41%, FA 2.28%, SC 0.21%
#
#Test set
#Model DER:  0.20067300122900925
#Model ACC:  0.9284548526346302
#100%|██████████| 60/60 [00:37<00:00,  1.60it/s]
#Eval for threshold 0.20: DER 12.75%, MS 2.32%, FA 9.44%, SC 0.98%
#
#Eval for threshold 0.30: DER 10.87%, MS 3.11%, FA 6.67%, SC 1.10%
#
#Eval for threshold 0.35: DER 10.38%, MS 3.52%, FA 5.77%, SC 1.09%
#
#Eval for threshold 0.40: DER 10.06%, MS 3.94%, FA 5.03%, SC 1.09%
#
#Eval for threshold 0.45: DER 9.86%, MS 4.39%, FA 4.38%, SC 1.08%
#
#Eval for threshold 0.50: DER 9.81%, MS 4.93%, FA 3.83%, SC 1.05%
#
#Eval for threshold 0.55: DER 9.89%, MS 5.52%, FA 3.40%, SC 0.96%
#
#Eval for threshold 0.60: DER 10.07%, MS 6.18%, FA 3.04%, SC 0.85%
#
#Eval for threshold 0.70: DER 10.77%, MS 7.72%, FA 2.43%, SC 0.62%
#
#Eval for threshold 0.80: DER 12.40%, MS 10.17%, FA 1.84%, SC 0.39%

fi


# compared with stage3-4, stage11-12 will finetune it via using real alimeeting data.
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
    eval_test_spk_path=$spk_path # store speaker embedding directory

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spks6s_ft_on_real_alimeeting
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spks6s
    mkdir -p $exp_dir
    cp -r $dest_dir/best-valid-der.pt $exp_dir/
    mv $exp_dir/best-valid-der.pt $exp_dir/epoch-0.pt

    data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    eval_test_data_dir=$data_dir # oracle target audio , mix audio and labels path
    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 13915 \
    ts_vad2_simu/train_accelerate_ddp2_debug2.py \
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
        --speech-encoder-type $speech_encoder_type\
        --speech-encoder-path $speech_encoder_path\
        --select-encoder-layer-nums 6\
        --spk-path $spk_path\
        --eval-test-spk-path $eval_test_spk_path\
         --speaker-embedding-name-dir $speaker_embedding_name_dir\
        --exp-dir $exp_dir\
        --data-dir $data_dir\
        --eval-test-data-dir $eval_test_data_dir
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spks6s_ft_on_real_alimeeting
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 eval_test_data_dir=$data_dir
 eval_test_spk_path=$spk_path
 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad2_simu/infer2.py \
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
    --eval-test-spk-path $eval_test_spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --eval-test-data-dir $eval_test_data_dir
  done
#
#cat logs/run_ts_vad2_simu_stage12.log
#Eval set
#Model DER:  0.14272098618702117
#Model ACC:  0.950979128877034
#100%|██████████| 25/25 [00:15<00:00,  1.60it/s]
#Eval for threshold 0.20: DER 12.56%, MS 0.70%, FA 11.49%, SC 0.37%
#
#Eval for threshold 0.30: DER 8.53%, MS 1.33%, FA 6.83%, SC 0.37%
#
#Eval for threshold 0.35: DER 7.25%, MS 1.68%, FA 5.16%, SC 0.42%
#
#Eval for threshold 0.40: DER 6.50%, MS 2.14%, FA 3.92%, SC 0.43%
#
#Eval for threshold 0.45: DER 6.01%, MS 2.57%, FA 3.01%, SC 0.43%
#
#Eval for threshold 0.50: DER 5.88%, MS 3.11%, FA 2.35%, SC 0.42%
#
#Eval for threshold 0.55: DER 5.85%, MS 3.67%, FA 1.79%, SC 0.39%
#
#Eval for threshold 0.60: DER 6.04%, MS 4.32%, FA 1.35%, SC 0.37%
#
#Eval for threshold 0.70: DER 7.05%, MS 6.02%, FA 0.73%, SC 0.30%
#
#Eval for threshold 0.80: DER 9.07%, MS 8.46%, FA 0.43%, SC 0.18%
#
#Test set
#Model DER:  0.12887843392363205
#Model ACC:  0.9527833984268673
#100%|██████████| 60/60 [00:37<00:00,  1.59it/s]
#Eval for threshold 0.20: DER 11.73%, MS 0.89%, FA 10.52%, SC 0.33%
#
#Eval for threshold 0.30: DER 7.90%, MS 1.56%, FA 5.90%, SC 0.44%
#
#Eval for threshold 0.35: DER 6.94%, MS 1.95%, FA 4.50%, SC 0.49%
#
#Eval for threshold 0.40: DER 6.32%, MS 2.41%, FA 3.38%, SC 0.53%
#
#Eval for threshold 0.45: DER 5.94%, MS 2.93%, FA 2.47%, SC 0.55%
#
#Eval for threshold 0.50: DER 5.79%, MS 3.50%, FA 1.77%, SC 0.53%
#
#Eval for threshold 0.55: DER 5.89%, MS 4.11%, FA 1.26%, SC 0.53%
#
#Eval for threshold 0.60: DER 6.23%, MS 4.86%, FA 0.89%, SC 0.48%
#
#Eval for threshold 0.70: DER 7.53%, MS 6.78%, FA 0.40%, SC 0.35%
#
#Eval for th/mntcephfs/lab_data/maduo/datasets/alimeetingreshold 0.80: DER 9.82%, MS 9.48%, FA 0.15%, SC 0.20%
fi
# compared with stage9-10, stage13-14 will finetune it via using real alimeeting data.
if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
    eval_test_spk_path=$spk_path # store speaker embedding directory

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spks6s_real_evalset_ft_on_real_alimeeting
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spks6s_real_evalset
    mkdir -p $exp_dir
    cp -r $dest_dir/best-valid-der.pt $exp_dir/
    mv $exp_dir/best-valid-der.pt $exp_dir/epoch-0.pt

    data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    eval_test_data_dir=$data_dir # oracle target audio , mix audio and labels path
    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 13915 \
    ts_vad2_simu/train_accelerate_ddp2_debug2.py \
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
        --speech-encoder-type $speech_encoder_type\
        --speech-encoder-path $speech_encoder_path\
        --select-encoder-layer-nums 6\
        --spk-path $spk_path\
        --eval-test-spk-path $eval_test_spk_path\
         --speaker-embedding-name-dir $speaker_embedding_name_dir\
        --exp-dir $exp_dir\
        --data-dir $data_dir\
        --eval-test-data-dir $eval_test_data_dir
fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spks6s_real_evalset_ft_on_real_alimeeting
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 eval_test_data_dir=$data_dir
 eval_test_spk_path=$spk_path
 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad2_simu/infer2.py \
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
    --eval-test-spk-path $eval_test_spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --eval-test-data-dir $eval_test_data_dir
  done

#sbatch --nodes 1 --gres=gpu:2  --nodelist=pgpu17 --cpus-per-gpu=6  --ntasks-per-node 2  -p p-A100 -A t00120220002 -o logs/run_ts_vad2_simu_stage13-14_ft.log run_ts_vad2_simu.sh --stage 13 --stop-stage 14
#cat logs/run_ts_vad2_simu_stage13-14_ft.log
#Eval set
#Model DER:  0.13903835093937422
#Model ACC:  0.9523363332890321
#100%|██████████| 25/25 [00:15<00:00,  1.64it/s]
#Eval for threshold 0.20: DER 11.43%, MS 0.74%, FA 10.40%, SC 0.29%
#
#Eval for threshold 0.30: DER 7.60%, MS 1.36%, FA 5.93%, SC 0.31%
#
#Eval for threshold 0.35: DER 6.63%, MS 1.71%, FA 4.61%, SC 0.31%
#
#Eval for threshold 0.40: DER 6.01%, MS 2.13%, FA 3.55%, SC 0.33%
#
#Eval for threshold 0.45: DER 5.75%, MS 2.64%, FA 2.75%, SC 0.35%
#
#Eval for threshold 0.50: DER 5.59%, MS 3.17%, FA 2.07%, SC 0.35%
#
#Eval for threshold 0.55: DER 5.67%, MS 3.74%, FA 1.60%, SC 0.34%
#
#Eval for threshold 0.60: DER 5.94%, MS 4.46%, FA 1.16%, SC 0.33%
#
#Eval for threshold 0.70: DER 7.02%, MS 6.14%, FA 0.66%, SC 0.22%
#
#Eval for threshold 0.80: DER 9.10%, MS 8.57%, FA 0.38%, SC 0.14%
#
#Test set
#Model DER:  0.1313668162331136
#Model ACC:  0.9519454000211791
#100%|██████████| 60/60 [00:36<00:00,  1.64it/s]
#Eval for threshold 0.20: DER 11.64%, MS 0.94%, FA 10.33%, SC 0.37%
#
#Eval for threshold 0.30: DER 7.88%, MS 1.69%, FA 5.70%, SC 0.50%
#
#Eval for threshold 0.35: DER 6.89%, MS 2.12%, FA 4.22%, SC 0.54%
#
#Eval for threshold 0.40: DER 6.32%, MS 2.59%, FA 3.15%, SC 0.59%
#
#Eval for threshold 0.45: DER 6.04%, MS 3.13%, FA 2.30%, SC 0.61%
#
#Eval for threshold 0.50: DER 6.01%, MS 3.74%, FA 1.65%, SC 0.61%
#
#Eval for threshold 0.55: DER 6.20%, MS 4.43%, FA 1.17%, SC 0.59%
#
#Eval for threshold 0.60: DER 6.60%, MS 5.25%, FA 0.80%, SC 0.55%
#
#Eval for threshold 0.70: DER 8.13%, MS 7.38%, FA 0.35%, SC 0.40%
#
#Eval for threshold 0.80: DER 10.76%, MS 10.39%, FA 0.14%, SC 0.23%
fi



## prepared non_overlaps target speaker embedding
if [ ${stage} -le 131 ] && [ ${stop_stage} -ge 131 ];then
   echo "prepare eval set target audio list"
   source_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting

   for name in Eval Train;do
    if [ $name = "Train" ];then
      input_dir=$source_dir/${name}_Ali_far/non_overlaps
      dest_dir=$source_dir/${name}_Ali_far/non_overlaps_spk_emb
      mkdir -p $dest_dir
      file=$dest_dir/wavs.txt
    else
      input_dir=$source_dir/${name}_Ali/${name}_Ali_far/non_overlaps
      dest_dir=$source_dir/${name}_Ali/${name}_Ali_far/non_overlaps_spk_emb
      mkdir -p $dest_dir
      file=$dest_dir/wavs.txt
    fi
    #file=$input_dir/wavs.txt
    #python3 ts_vad2_simu/prepare_alimeeting_target_audio_list.py \
    #     $input_dir $file
    find $input_dir -name "*.wav" | grep -v "all.wav" >$file
    head $file
   done
fi
if [ ${stage} -le 141 ] && [ ${stop_stage} -ge 141 ];then
   echo "generate eval(dev) and train speaker embedding"
   source_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting
   feature_name=cam++_zh-cn_200k_feature_dir
   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
   model_id=iic/speech_campplus_sv_zh-cn_16k-common
   subsets="Eval Train"
   #subsets="Train"
   batch_size=400 # 96*4,I will increase batch size for next experiment, because it is very time consuming.
                  # when batch_size=96, it will consume 19820MB CUDA memeroy
                  # 40GB A100 ,batch_size=200, 39GB CUDA memeroy
                  # 80GB A100, pgpu17,batch_size=400,79GB CUDA memeroy

   for name in $subsets;do
    if [ $name = "Train" ];then
     echo "extract train target speaker embedding"
     # 提取embedding
     dest_dir=$source_dir/${name}_Ali_far/non_overlaps_spk_emb
     wav_path=$dest_dir/wavs.txt
    else
     echo "extract $name target speaker embedding"
     # 提取embedding
     dest_dir=$source_dir/${name}_Ali/${name}_Ali_far/non_overlaps_spk_emb
     wav_path=$dest_dir/wavs.txt
    fi
     save_dir=$dest_dir/alimeeting/SpeakerEmbedding/$name/$feature_name
     python3 ts_vad2_simu/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size $batch_size
   done
fi
# extract target speaker embedding of simulation data is very time consuming.
# So I will use non_overlap target speaker embed softlink target speaker embedding of simulation data.
# It required non_overlap target speaker speech embedding and target audio of simulation data
if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
   echo "prepare eval set target audio embedding using softlink"
   source_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting
   feature_name=cam++_zh-cn_200k_feature_dir
   dest_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/data/200h_fixed_4spks6s_all_seg_softlink
   subsets="Train"
   #subsets="Eval"
   for name in $subsets;do
    if [ $name = "Train" ];then
      input_dir=$dest_dir/${name}_Ali_far/target_audio
      file=$dest_dir/${name}_Ali_far/wavs.txt
      non_overlap_target_spk_emb_dir=$source_dir/${name}_Ali_far/non_overlaps_spk_emb/alimeeting/SpeakerEmbedding/$name/$feature_name
    else
      input_dir=$dest_dir/${name}_Ali/${name}_Ali_far/target_audio
      file=$dest_dir/${name}_Ali/${name}_Ali_far/wavs.txt
      non_overlap_target_spk_emb_dir=$source_dir/${name}_Ali/${name}_Ali_far/non_overlaps_spk_emb/alimeeting/SpeakerEmbedding/$name/$feature_name
    fi
    #python3 ts_vad2_simu/prepare_alimeeting_target_audio_list.py \
    #    $input_dir $file
    find $input_dir -name "*.wav" | grep -v "all.wav" >$file
    head $file

    save_dir=$dest_dir/ts_vad/alimeeting/SpeakerEmbedding/$name/$feature_name
    [ -d "$save_dir" ] && echo "exist old $save_dir, remove it!!! " && rm -r $save_dir
    echo "start softlink target speaker embedding of simulation data ...."
    python3  ts_vad2_simu/prepared_simu_data_target_speaker_embedding_using_softlink.py\
               $non_overlap_target_spk_emb_dir\
               $file\
               $save_dir

   done
fi

# compared with stage9-10, stage15-16-17, target speaker speech of simulation data is from all segment concation in one session (real mix) audio
if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    simu_data_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/data/200h_fixed_4spks6s_all_seg_softlink
    spk_path=$simu_data_dir/ts_vad/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
    eval_test_spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spks6s_all_seg_softlink_real_evalset
    data_dir="$simu_data_dir" # oracle target audio , mix audio and labels path
    eval_test_data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12915 \
    ts_vad2_simu/train_accelerate_ddp2_debug2_simu.py \
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
        --speech-encoder-type $speech_encoder_type\
        --speech-encoder-path $speech_encoder_path\
        --select-encoder-layer-nums 6\
        --spk-path $spk_path\
        --eval-test-spk-path $eval_test_spk_path\
        --speaker-embedding-name-dir $speaker_embedding_name_dir\
        --exp-dir $exp_dir\
        --data-dir $data_dir\
        --eval-test-data-dir $eval_test_data_dir
fi

if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spks6s_all_seg_softlink_real_evalset
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 eval_test_data_dir=$data_dir
 eval_test_spk_path=$spk_path
 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad2_simu/infer2.py \
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
    --eval-test-spk-path $eval_test_spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --eval-test-data-dir $eval_test_data_dir
  done

# sbatch --nodes 1 --gres=gpu:1  --nodelist=pgpu17 --cpus-per-gpu=6  --ntasks-per-node 1  -p p-A100 -A t00120220002 -o logs/run_ts_vad2_simu_stage17.log run_ts_vad2_simu.sh --stage 17 --stop-stage 17
# Eval set
#Model DER:  0.21380513440817653
#Model ACC:  0.9277156333035974
#100%|██████████| 25/25 [00:15<00:00,  1.65it/s]
#Eval for threshold 0.20: DER 11.23%, MS 2.81%, FA 7.59%, SC 0.83%
#
#Eval for threshold 0.30: DER 10.31%, MS 3.83%, FA 5.70%, SC 0.79%
#
#Eval for threshold 0.35: DER 10.16%, MS 4.31%, FA 5.05%, SC 0.80%
#
#Eval for threshold 0.40: DER 10.08%, MS 4.78%, FA 4.49%, SC 0.81%
#
#Eval for threshold 0.45: DER 10.13%, MS 5.29%, FA 4.09%, SC 0.75%
#
#Eval for threshold 0.50: DER 10.27%, MS 5.90%, FA 3.68%, SC 0.69%
#
#Eval for threshold 0.55: DER 10.56%, MS 6.56%, FA 3.39%, SC 0.61%
#
#Eval for threshold 0.60: DER 10.94%, MS 7.26%, FA 3.15%, SC 0.53%
#
#Eval for threshold 0.70: DER 11.95%, MS 8.85%, FA 2.73%, SC 0.36%
#
#Eval for threshold 0.80: DER 13.54%, MS 11.10%, FA 2.25%, SC 0.19%
#
#Test set
#Model DER:  0.21807855191150405
#Model ACC:  0.9226032817828975
#100%|██████████| 60/60 [00:36<00:00,  1.64it/s]
#Eval for threshold 0.20: DER 13.73%, MS 3.38%, FA 9.25%, SC 1.10%
#
#Eval for threshold 0.30: DER 12.40%, MS 4.55%, FA 6.70%, SC 1.15%
#
#Eval for threshold 0.35: DER 12.07%, MS 5.14%, FA 5.80%, SC 1.13%
#
#Eval for threshold 0.40: DER 11.94%, MS 5.77%, FA 5.08%, SC 1.08%
#
#Eval for threshold 0.45: DER 11.92%, MS 6.43%, FA 4.46%, SC 1.03%
#
#Eval for threshold 0.50: DER 12.05%, MS 7.17%, FA 3.90%, SC 0.98%
#
#Eval for threshold 0.55: DER 12.30%, MS 7.94%, FA 3.50%, SC 0.86%
#
#Eval for threshold 0.60: DER 12.64%, MS 8.77%, FA 3.15%, SC 0.72%
#
#Eval for threshold 0.70: DER 13.70%, MS 10.63%, FA 2.60%, SC 0.47%
#
#Eval for threshold 0.80: DER 15.51%, MS 13.25%, FA 2.02%, SC 0.25%

fi

# extract target speaker embedding of simulation data is very time consuming.
# So I will use non_overlap target speaker embed softlink target speaker embedding of simulation data.
# It required non_overlap target speaker speech embedding and target audio of simulation data
# runinng script stage is stage18,stage22-23.
if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ];then
   echo "prepare eval set target audio embedding using softlink"
   source_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting
   feature_name=cam++_zh-cn_200k_feature_dir
   dest_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/data/200h_fixed_4spkscat_16s_all_seg_softlink
   #subsets="Train"
   subsets="Eval Train"
   for name in $subsets;do
    if [ $name = "Train" ];then
      input_dir=$dest_dir/${name}_Ali_far/target_audio
      file=$dest_dir/${name}_Ali_far/wavs.txt
      non_overlap_target_spk_emb_dir=$source_dir/${name}_Ali_far/non_overlaps_spk_emb/alimeeting/SpeakerEmbedding/$name/$feature_name
    else
      input_dir=$dest_dir/${name}_Ali/${name}_Ali_far/target_audio
      file=$dest_dir/${name}_Ali/${name}_Ali_far/wavs.txt
      non_overlap_target_spk_emb_dir=$source_dir/${name}_Ali/${name}_Ali_far/non_overlaps_spk_emb/alimeeting/SpeakerEmbedding/$name/$feature_name
    fi
    #python3 ts_vad2_simu/prepare_alimeeting_target_audio_list.py \
    #    $input_dir $file
    find $input_dir -name "*.wav" | grep -v "all.wav" >$file
    head $file
    save_dir=$dest_dir/ts_vad/alimeeting/SpeakerEmbedding/$name/$feature_name
    [ -d "$save_dir" ] && echo "exist old $save_dir, remove it!!! " && rm -r $save_dir
    echo "start softlink target speaker embedding of simulation data ...."
    python3  ts_vad2_simu/prepared_simu_data_target_speaker_embedding_using_softlink.py\
               $non_overlap_target_spk_emb_dir\
               $file\
               $save_dir

   done
fi

## this is very consuming time. stop it.
## prepared target speaker embedding
#if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
#   echo "prepare eval set target audio list"
#   dest_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/data/200h_fixed_4spkscat_16s_all_seg
#   for name in Eval Train;do
#    if [ $name = "Train" ];then
#      input_dir=$dest_dir/${name}_Ali_far/target_audio
#      file=$dest_dir/${name}_Ali_far/wavs.txt
#    else
#      input_dir=$dest_dir/${name}_Ali/${name}_Ali_far/target_audio
#      file=$dest_dir/${name}_Ali/${name}_Ali_far/wavs.txt
#    fi
#    #python3 ts_vad2_simu/prepare_alimeeting_target_audio_list.py \
#    #    $input_dir $file
#    find $input_dir -name "*.wav" | grep -v "all.wav" >$file
#    head $file
#   done
#fi
#
#
#if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
#   echo "generate eval(dev) and train speaker embedding"
#   dest_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/data/200h_fixed_4spkscat_16s_all_seg
#   feature_name=cam++_zh-cn_200k_feature_dir
#   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
#   model_id=iic/speech_campplus_sv_zh-cn_16k-common
#   #subsets="Eval Train"
#   subsets="Train"
#   batch_size=400 # 96*4,I will increase batch size for next experiment, because it is very time consuming.
#                  # when batch_size=96, it will consume 19820MB CUDA memeroy
#                  # 40GB A100 ,batch_size=200, 39GB CUDA memeroy
#                  # 80GB A100, pgpu17,batch_size=400,79GB CUDA memeroy
#
#   for name in $subsets;do
#    if [ $name = "Train" ];then
#     echo "extract train target speaker embedding"
#     # 提取embedding
#     input_dir=$dest_dir/${name}_Ali_far/target_audio/
#     wav_path=$dest_dir/${name}_Ali_far/wavs.txt
#    else
#     echo "extract $name target speaker embedding"
#     # 提取embedding
#     input_dir=$dest_dir/${name}_Ali/${name}_Ali_far/target_audio/
#     wav_path=$dest_dir/${name}_Ali/${name}_Ali_far/wavs.txt
#    fi
#     save_dir=$dest_dir/ts_vad/spk_embed/alimeeting/${feat_type}SpeakerEmbedding/$name/$feature_name
#     python3 ts_vad2_simu/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
#           --model_id $model_id\
#           --wavs $wav_path\
#           --save_dir $save_dir\
#           --batch_size $batch_size
#   done
#fi
if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    simu_data_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/data/200h_fixed_4spkscat_16s_all_seg_softlink
    spk_path=$simu_data_dir/ts_vad/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
    eval_test_spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spkscat_16s_all_seg_softlink_real_evalset
    data_dir="$simu_data_dir" # oracle target audio , mix audio and labels path
    eval_test_data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12615 \
    ts_vad2_simu/train_accelerate_ddp2_debug2_simu.py \
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
        --speech-encoder-type $speech_encoder_type\
        --speech-encoder-path $speech_encoder_path\
        --select-encoder-layer-nums 6\
        --spk-path $spk_path\
        --eval-test-spk-path $eval_test_spk_path\
        --speaker-embedding-name-dir $speaker_embedding_name_dir\
        --exp-dir $exp_dir\
        --data-dir $data_dir\
        --eval-test-data-dir $eval_test_data_dir
fi

if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ];then
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spkscat_16s_all_seg_softlink_real_evalset
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spkscat_16s_all_seg_softlink_real_evalset/
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 eval_test_spk_path=$spk_path
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 eval_test_data_dir=$data_dir
 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad2_simu/infer2.py \
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
    --eval-test-spk-path $eval_test_spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --eval-test-data-dir $eval_test_data_dir
done
# sbatch --nodes 1 --gres=gpu:2   --nodelist=pgpu17  --cpus-per-gpu=6  --ntasks-per-node 2  -p p-A100 -A t00120220002 -o logs/run_ts_vad2_simu_stage21-23_3.log run_ts_vad2_simu.sh --stage 22 --stop-stage 23
# Eval set
# Model DER:  0.21380513440817653
#Model ACC:  0.9277156333035974
#100%|██████████| 25/25 [00:15<00:00,  1.65it/s]
#Eval for threshold 0.20: DER 11.23%, MS 2.81%, FA 7.59%, SC 0.83%
#
#Eval for threshold 0.30: DER 10.31%, MS 3.83%, FA 5.70%, SC 0.79%
#
#Eval for threshold 0.35: DER 10.16%, MS 4.31%, FA 5.05%, SC 0.80%
#
#Eval for threshold 0.40: DER 10.08%, MS 4.78%, FA 4.49%, SC 0.81%
#
#Eval for threshold 0.45: DER 10.13%, MS 5.29%, FA 4.09%, SC 0.75%
#
#Eval for threshold 0.50: DER 10.27%, MS 5.90%, FA 3.68%, SC 0.69%
#
#Eval for threshold 0.55: DER 10.56%, MS 6.56%, FA 3.39%, SC 0.61%
#
#Eval for threshold 0.60: DER 10.94%, MS 7.26%, FA 3.15%, SC 0.53%
#
#Eval for threshold 0.70: DER 11.95%, MS 8.85%, FA 2.73%, SC 0.36%
#
#Eval for threshold 0.80: DER 13.54%, MS 11.10%, FA 2.25%, SC 0.19%
#
#Test set
#Model DER:  0.21807855191150405
#Model ACC:  0.9226032817828975
#100%|██████████| 60/60 [00:36<00:00,  1.64it/s]
#Eval for threshold 0.20: DER 13.73%, MS 3.38%, FA 9.25%, SC 1.10%
#
#Eval for threshold 0.30: DER 12.40%, MS 4.55%, FA 6.70%, SC 1.15%
#
#Eval for threshold 0.35: DER 12.07%, MS 5.14%, FA 5.80%, SC 1.13%
#
#Eval for threshold 0.40: DER 11.94%, MS 5.77%, FA 5.08%, SC 1.08%
#
#Eval for threshold 0.45: DER 11.92%, MS 6.43%, FA 4.46%, SC 1.03%
#
#Eval for threshold 0.50: DER 12.05%, MS 7.17%, FA 3.90%, SC 0.98%
#
#Eval for threshold 0.55: DER 12.30%, MS 7.94%, FA 3.50%, SC 0.86%
#
#Eval for threshold 0.60: DER 12.64%, MS 8.77%, FA 3.15%, SC 0.72%
#
#Eval for threshold 0.70: DER 13.70%, MS 10.63%, FA 2.60%, SC 0.47%
#
#Eval for threshold 0.80: DER 15.51%, MS 13.25%, FA 2.02%, SC 0.25%
fi

## this is very consuming time. stop
## compared with stage9-10, stage25-28 will use 400 hours simulation train data.
#if [ ${stage} -le 25 ] && [ ${stop_stage} -ge 25 ];then
#   echo "prepare eval set target audio list"
#   dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/400h_fixed_4spks6s
#   for name in Eval Train;do
#    if [ $name = "Train" ];then
#      input_dir=$dest_dir/${name}_Ali_far/target_audio
#    else
#      input_dir=$dest_dir/${name}_Ali/${name}_Ali_far/target_audio
#    fi
#    file=$input_dir/wavs.txt
#   python3 ts_vad2_simu/prepare_alimeeting_target_audio_list.py \
#        $input_dir $file
#   done
#fi
#
#
#if [ ${stage} -le 26 ] && [ ${stop_stage} -ge 26 ];then
#   echo "generate eval(dev) and train speaker embedding"
#   dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/400h_fixed_4spks6s
#   feature_name=cam++_zh-cn_200k_feature_dir
#   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
#   model_id=iic/speech_campplus_sv_zh-cn_16k-common
#   subsets="Eval Train"
#   for name in $subsets;do
#    if [ $name = "Train" ];then
#     echo "extract train target speaker embedding"
#     # 提取embedding
#     input_dir=$dest_dir/${name}_Ali_far/target_audio/
#     wav_path=$input_dir/wavs.txt
#    else
#     echo "extract $name target speaker embedding"
#     # 提取embedding
#     input_dir=$dest_dir/${name}_Ali/${name}_Ali_far/target_audio/
#     wav_path=$input_dir/wavs.txt
#    fi
#     save_dir=$dest_dir/ts_vad/spk_embed/alimeeting/${feat_type}SpeakerEmbedding/$name/$feature_name
#     python3 ts_vad2_simu/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
#           --model_id $model_id\
#           --wavs $wav_path\
#           --save_dir $save_dir\
#           --batch_size 96
#   done
#fi


## compared with stage3-4, stage30-34 will use 400 hours simulation train data and softlink target speaker embedding and don't extract it again.
## prepared non_overlap_segment_6s target speaker embedding
if [ ${stage} -le 30 ] && [ ${stop_stage} -ge 30 ];then
   echo "prepare eval set target audio list"
   source_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting

   for name in Eval Train;do
    if [ $name = "Train" ];then
      input_dir=$source_dir/${name}_Ali_far/non_overlap_segment_6s
      dest_dir=$source_dir/${name}_Ali_far/non_overlap_segment_6s_spk_emb
      mkdir -p $dest_dir
      file=$dest_dir/wavs.txt
    else
      input_dir=$source_dir/${name}_Ali/${name}_Ali_far/non_overlap_segment_6s
      dest_dir=$source_dir/${name}_Ali/${name}_Ali_far/non_overlap_segment_6s_spk_emb
      mkdir -p $dest_dir
      file=$dest_dir/wavs.txt
    fi
    find $input_dir -name "*.wav" | grep -v "all.wav" >$file
    head $file
   done
fi

if [ ${stage} -le 31 ] && [ ${stop_stage} -ge 31 ];then
   echo "generate eval(dev) and train speaker embedding"
   source_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting
   feature_name=cam++_zh-cn_200k_feature_dir
   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
   model_id=iic/speech_campplus_sv_zh-cn_16k-common
   subsets="Eval Train"
   #subsets="Train"
   batch_size=400 # 96*4,I will increase batch size for next experiment, because it is very time consuming.
                  # when batch_size=96, it will consume 19820MB CUDA memeroy
                  # 40GB A100 ,batch_size=200, 39GB CUDA memeroy
                  # 80GB A100, pgpu17,batch_size=400,79GB CUDA memeroy

   for name in $subsets;do
    if [ $name = "Train" ];then
     echo "extract train target speaker embedding"
     # 提取embedding
     dest_dir=$source_dir/${name}_Ali_far/non_overlap_segment_6s_spk_emb
     wav_path=$dest_dir/wavs.txt
    else
     echo "extract $name target speaker embedding"
     # 提取embedding
     dest_dir=$source_dir/${name}_Ali/${name}_Ali_far/non_overlap_segment_6s_spk_emb
     wav_path=$dest_dir/wavs.txt
    fi
     save_dir=$dest_dir/alimeeting/SpeakerEmbedding/$name/$feature_name
     python3 ts_vad2_simu/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size $batch_size
   done
fi

# extract target speaker embedding of simulation data is very time consuming.
# So I will use non_overlap target speaker embed softlink target speaker embedding of simulation data.
# It required non_overlap target speaker speech embedding and target audio of simulation data
if [ ${stage} -le 32 ] && [ ${stop_stage} -ge 32 ];then
   echo "prepare eval set target audio embedding using softlink"
   source_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting
   feature_name=cam++_zh-cn_200k_feature_dir
   dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/400h_fixed_4spks6s_softlink
   subsets="Train"
   #subsets="Eval"
   for name in $subsets;do
    if [ $name = "Train" ];then
      input_dir=$dest_dir/${name}_Ali_far/target_audio
      file=$dest_dir/${name}_Ali_far/wavs.txt # target audio list
      non_overlap_target_spk_emb_dir=$source_dir/${name}_Ali_far/non_overlap_segment_6s_spk_emb/alimeeting/SpeakerEmbedding/$name/$feature_name
    else
      input_dir=$dest_dir/${name}_Ali/${name}_Ali_far/target_audio
      file=$dest_dir/${name}_Ali/${name}_Ali_far/wavs.txt # target audio list
      non_overlap_target_spk_emb_dir=$source_dir/${name}_Ali/${name}_Ali_far/non_overlap_segment_6s_spk_emb/alimeeting/SpeakerEmbedding/$name/$feature_name
    fi
    #python3 ts_vad2_simu/prepare_alimeeting_target_audio_list.py \
    #    $input_dir $file
    find $input_dir -name "*.wav" | grep -v "all.wav" >$file
    head $file

    save_dir=$dest_dir/ts_vad/alimeeting/SpeakerEmbedding/$name/$feature_name
    [ -d "$save_dir" ] && echo "exist old $save_dir, remove it!!! " && rm -r $save_dir
    echo "start softlink target speaker embedding of simulation data ...."
    #python3  ts_vad2_simu/prepared_simu_data_target_speaker_embedding_using_softlink.py\
     python3 ts_vad2_simu/prepared_simu_data_target_segment_speaker_embedding_using_softlink.py\
               $non_overlap_target_spk_emb_dir\
               $file\
               $save_dir

   done
fi



# compared with stage3-4, stage33-34 will use real alimeeting Eval set as eval set in train simulation data.
if [ ${stage} -le 33 ] && [ ${stop_stage} -ge 33 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    simu_data_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/400h_fixed_4spks6s_softlink
    spk_path=$simu_data_dir/ts_vad/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
    eval_test_spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_400h_fixed_4spks6s_softlink_real_evalset
    data_dir="$simu_data_dir" # oracle target audio , mix audio and labels path
    eval_test_data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12915 \
    ts_vad2_simu/train_accelerate_ddp2_debug2_simu.py \
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
        --speech-encoder-type $speech_encoder_type\
        --speech-encoder-path $speech_encoder_path\
        --select-encoder-layer-nums 6\
        --spk-path $spk_path\
        --eval-test-spk-path $eval_test_spk_path\
        --speaker-embedding-name-dir $speaker_embedding_name_dir\
        --exp-dir $exp_dir\
        --data-dir $data_dir\
        --eval-test-data-dir $eval_test_data_dir
fi

if [ ${stage} -le 34 ] && [ ${stop_stage} -ge 34 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_400h_fixed_4spks6s_softlink_real_evalset
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 eval_test_data_dir=$data_dir
 eval_test_spk_path=$spk_path
 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad2_simu/infer2.py \
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
    --eval-test-spk-path $eval_test_spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --eval-test-data-dir $eval_test_data_dir
  done

#Eval set
#Model DER:  0.20158845767498138
#Model ACC:  0.9319293972402515
#100%|██████████| 25/25 [00:15<00:00,  1.66it/s]
#Eval for threshold 0.20: DER 10.89%, MS 1.77%, FA 8.42%, SC 0.70%
#
#Eval for threshold 0.30: DER 9.38%, MS 2.50%, FA 6.18%, SC 0.70%
#
#Eval for threshold 0.35: DER 9.03%, MS 2.84%, FA 5.45%, SC 0.75%
#
#Eval for threshold 0.40: DER 8.78%, MS 3.20%, FA 4.80%, SC 0.78%
#
#Eval for threshold 0.45: DER 8.74%, MS 3.68%, FA 4.33%, SC 0.74%
#
#Eval for threshold 0.50: DER 8.80%, MS 4.18%, FA 3.94%, SC 0.69%
#
#Eval for threshold 0.55: DER 8.91%, MS 4.72%, FA 3.57%, SC 0.62%
#
#Eval for threshold 0.60: DER 9.12%, MS 5.29%, FA 3.27%, SC 0.56%
#
#Eval for threshold 0.70: DER 9.87%, MS 6.70%, FA 2.78%, SC 0.39%
#
#Eval for threshold 0.80: DER 11.31%, MS 8.85%, FA 2.27%, SC 0.19%
#
#Test set
#Model DER:  0.20043871906671115
#Model ACC:  0.9284629222717298
#100%|██████████| 60/60 [00:36<00:00,  1.63it/s]
#Eval for threshold 0.20: DER 12.62%, MS 2.29%, FA 9.27%, SC 1.05%
#
#Eval for threshold 0.30: DER 10.91%, MS 3.15%, FA 6.65%, SC 1.11%
#
#Eval for threshold 0.35: DER 10.46%, MS 3.59%, FA 5.77%, SC 1.10%
#
#Eval for threshold 0.40: DER 10.16%, MS 4.02%, FA 5.03%, SC 1.11%
#
#Eval for threshold 0.45: DER 10.00%, MS 4.49%, FA 4.41%, SC 1.10%
#
#Eval for threshold 0.50: DER 9.97%, MS 5.05%, FA 3.84%, SC 1.09%
#
#Eval for threshold 0.55: DER 10.09%, MS 5.68%, FA 3.40%, SC 1.01%
#
#Eval for threshold 0.60: DER 10.28%, MS 6.31%, FA 3.05%, SC 0.92%
#
#Eval for threshold 0.70: DER 11.04%, MS 7.92%, FA 2.46%, SC 0.66%
#
#Eval for threshold 0.80: DER 12.66%, MS 10.37%, FA 1.89%, SC 0.40%
fi

# extract target speaker embedding of simulation data is very time consuming.
# So I will use non_overlap target speaker embed softlink target speaker embedding of simulation data.
# It required non_overlap target speaker speech embedding and target audio of simulation data
if [ ${stage} -le 40 ] && [ ${stop_stage} -ge 40 ];then
   echo "prepare eval set target audio embedding using softlink"
   source_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting
   feature_name=cam++_zh-cn_200k_feature_dir
   dest_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/data/200h_maximum_4spksall_softlink
   subsets="Eval Train"
   #subsets="Eval"
   for name in $subsets;do
    if [ $name = "Train" ];then
      input_dir=$dest_dir/${name}_Ali_far/target_audio
      file=$dest_dir/${name}_Ali_far/wavs.txt # target audio list
      non_overlap_target_spk_emb_dir=$source_dir/${name}_Ali_far/non_overlaps_spk_emb/alimeeting/SpeakerEmbedding/$name/$feature_name
    else
      input_dir=$dest_dir/${name}_Ali/${name}_Ali_far/target_audio
      file=$dest_dir/${name}_Ali/${name}_Ali_far/wavs.txt # target audio list
      non_overlap_target_spk_emb_dir=$source_dir/${name}_Ali/${name}_Ali_far/non_overlaps_spk_emb/alimeeting/SpeakerEmbedding/$name/$feature_name
    fi
    #python3 ts_vad2_simu/prepare_alimeeting_target_audio_list.py \
    #    $input_dir $file
    find $input_dir -name "*.wav" | grep -v "all.wav" >$file
    head $file

    save_dir=$dest_dir/ts_vad/alimeeting/SpeakerEmbedding/$name/$feature_name
    [ -d "$save_dir" ] && echo "exist old $save_dir, remove it!!! " && rm -r $save_dir
    echo "start softlink target speaker embedding of simulation data ...."
    #python3  ts_vad2_simu/prepared_simu_data_target_speaker_embedding_using_softlink.py\
     python3 ts_vad2_simu/prepared_simu_data_target_speaker_embedding_using_softlink.py\
               $non_overlap_target_spk_emb_dir\
               $file\
               $save_dir

   done
fi


# compared with stage3-4, stage41-42 will use real alimeeting Eval set as eval set on train simulation data(200hours, maximum mode, cat all segment to simulate data)
if [ ${stage} -le 41 ] && [ ${stop_stage} -ge 41 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    simu_data_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/data/200h_maximum_4spksall_softlink
    spk_path=$simu_data_dir/ts_vad/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
    eval_test_spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_maximum_4spksall_softlink_real_evalset
    data_dir="$simu_data_dir" # oracle target audio , mix audio and labels path
    eval_test_data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 13915 \
    ts_vad2_simu/train_accelerate_ddp2_debug2_simu.py \
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
        --speech-encoder-type $speech_encoder_type\
        --speech-encoder-path $speech_encoder_path\
        --select-encoder-layer-nums 6\
        --spk-path $spk_path\
        --eval-test-spk-path $eval_test_spk_path\
        --speaker-embedding-name-dir $speaker_embedding_name_dir\
        --exp-dir $exp_dir\
        --data-dir $data_dir\
        --eval-test-data-dir $eval_test_data_dir
fi

if [ ${stage} -le 42 ] && [ ${stop_stage} -ge 42 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_maximum_4spksall_softlink_real_evalset
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 eval_test_data_dir=$data_dir
 eval_test_spk_path=$spk_path
 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad2_simu/infer2.py \
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
    --eval-test-spk-path $eval_test_spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --eval-test-data-dir $eval_test_data_dir
  done
# cat logs/run_ts_vad2_simu_stage40-42_new.log
# Eval set
# Model DER:  0.33462516508851387
#Model ACC:  0.8740730120690844
#100%|██████████| 25/25 [00:15<00:00,  1.62it/s]
#Eval for threshold 0.20: DER 26.02%, MS 7.58%, FA 15.59%, SC 2.85%
#
#Eval for threshold 0.30: DER 21.25%, MS 9.95%, FA 8.60%, SC 2.71%
#
#Eval for threshold 0.35: DER 20.81%, MS 10.73%, FA 7.45%, SC 2.63%
#
#Eval for threshold 0.40: DER 20.55%, MS 11.53%, FA 6.50%, SC 2.52%
#
#Eval for threshold 0.45: DER 20.49%, MS 12.45%, FA 5.52%, SC 2.53%
#
#Eval for threshold 0.50: DER 20.26%, MS 14.13%, FA 3.52%, SC 2.62%
#
#Eval for threshold 0.55: DER 22.01%, MS 17.07%, FA 3.06%, SC 1.88%
#
#Eval for threshold 0.60: DER 23.09%, MS 18.67%, FA 2.94%, SC 1.48%
#
#Eval for threshold 0.70: DER 25.70%, MS 22.00%, FA 2.71%, SC 0.99%
#
#Eval for threshold 0.80: DER 32.19%, MS 29.48%, FA 2.18%, SC 0.53%
#
#Test set
#Model DER:  0.34148916502463966
#Model ACC:  0.8657710326059891
#100%|██████████| 60/60 [00:37<00:00,  1.60it/s]
#Eval for threshold 0.20: DER 30.53%, MS 7.28%, FA 20.00%, SC 3.25%
#
#Eval for threshold 0.30: DER 24.50%, MS 9.83%, FA 11.42%, SC 3.25%
#
#Eval for threshold 0.35: DER 23.65%, MS 10.81%, FA 9.61%, SC 3.23%
#
#Eval for threshold 0.40: DER 23.11%, MS 11.70%, FA 8.16%, SC 3.26%
#
#Eval for threshold 0.45: DER 22.52%, MS 12.56%, FA 6.61%, SC 3.35%
#
#Eval for threshold 0.50: DER 21.87%, MS 14.28%, FA 4.07%, SC 3.52%
#
#Eval for threshold 0.55: DER 23.65%, MS 17.63%, FA 3.40%, SC 2.62%
#
#Eval for threshold 0.60: DER 25.03%, MS 19.70%, FA 3.17%, SC 2.17%
#
#Eval for threshold 0.70: DER 27.93%, MS 23.66%, FA 2.75%, SC 1.52%
#
#Eval for threshold 0.80: DER 35.02%, MS 32.22%, FA 1.94%, SC 0.85%


fi

# extract target speaker embedding of simulation data is very time consuming.
# So I will use non_overlap target speaker embed softlink target speaker embedding of simulation data.
# It required non_overlap target speaker speech embedding and target audio of simulation data
if [ ${stage} -le 50 ] && [ ${stop_stage} -ge 50 ];then
   echo "prepare eval set target audio embedding using softlink"
   source_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting
   feature_name=cam++_zh-cn_200k_feature_dir
   dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/200h_maximum_4spks6s_softlink
   subsets="Eval Train"
   #subsets="Eval"
   for name in $subsets;do
    if [ $name = "Train" ];then
      input_dir=$dest_dir/${name}_Ali_far/target_audio
      file=$dest_dir/${name}_Ali_far/wavs.txt # target audio list
      non_overlap_target_spk_emb_dir=$source_dir/${name}_Ali_far/non_overlap_segment_6s_spk_emb/alimeeting/SpeakerEmbedding/$name/$feature_name
    else
      input_dir=$dest_dir/${name}_Ali/${name}_Ali_far/target_audio
      file=$dest_dir/${name}_Ali/${name}_Ali_far/wavs.txt # target audio list
      non_overlap_target_spk_emb_dir=$source_dir/${name}_Ali/${name}_Ali_far/non_overlap_segment_6s_spk_emb/alimeeting/SpeakerEmbedding/$name/$feature_name
    fi
    #python3 ts_vad2_simu/prepare_alimeeting_target_audio_list.py \
    #    $input_dir $file
    find $input_dir -name "*.wav" | grep -v "all.wav" >$file
    head $file

    save_dir=$dest_dir/ts_vad/alimeeting/SpeakerEmbedding/$name/$feature_name
    [ -d "$save_dir" ] && echo "exist old $save_dir, remove it!!! " && rm -r $save_dir
    echo "start softlink target speaker embedding of simulation data ...."
    #python3  ts_vad2_simu/prepared_simu_data_target_speaker_embedding_using_softlink.py\
     python3 ts_vad2_simu/prepared_simu_data_target_segment_speaker_embedding_using_softlink.py\
               $non_overlap_target_spk_emb_dir\
               $file\
               $save_dir

   done
fi

# compared with stage41-42, stage51-52 will use real alimeeting Eval set as eval set
# on train simulation data(200hours, maximum mode, more than 6s segment to simulate data)
if [ ${stage} -le 51 ] && [ ${stop_stage} -ge 51 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    simu_data_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/200h_maximum_4spks6s_softlink
    spk_path=$simu_data_dir/ts_vad/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
    eval_test_spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_maximum_4spks6s_softlink_real_evalset
    data_dir="$simu_data_dir" # oracle target audio , mix audio and labels path
    eval_test_data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 14915 \
    ts_vad2_simu/train_accelerate_ddp2_debug2_simu.py \
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
        --speech-encoder-type $speech_encoder_type\
        --speech-encoder-path $speech_encoder_path\
        --select-encoder-layer-nums 6\
        --spk-path $spk_path\
        --eval-test-spk-path $eval_test_spk_path\
        --speaker-embedding-name-dir $speaker_embedding_name_dir\
        --exp-dir $exp_dir\
        --data-dir $data_dir\
        --eval-test-data-dir $eval_test_data_dir
fi

if [ ${stage} -le 52 ] && [ ${stop_stage} -ge 52 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_maximum_4spks6s_softlink_real_evalset
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 eval_test_data_dir=$data_dir
 eval_test_spk_path=$spk_path
 for name in $infer_sets;do
    results_path=$exp_dir
   python3 ts_vad2_simu/infer2_simd2.py \
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
    --eval-test-spk-path $eval_test_spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --eval-test-data-dir $eval_test_data_dir
  done
# cat logs/run_ts_vad2_simu_stage50-52_new.log
#Eval set
#Model DER:  0.20006318716614976
#Model ACC:  0.9320013376362317
#100%|██████████| 25/25 [00:16<00:00,  1.54it/s]
#Eval for threshold 0.20: DER 10.78%, MS 2.07%, FA 8.04%, SC 0.68%
#
#Eval for threshold 0.30: DER 9.29%, MS 2.75%, FA 5.77%, SC 0.77%
#
#Eval for threshold 0.35: DER 9.00%, MS 3.15%, FA 5.09%, SC 0.76%
#
#Eval for threshold 0.40: DER 8.83%, MS 3.55%, FA 4.55%, SC 0.72%
#
#Eval for threshold 0.45: DER 8.71%, MS 3.96%, FA 4.05%, SC 0.70%
#
#Eval for threshold 0.50: DER 8.71%, MS 4.41%, FA 3.64%, SC 0.66%
#
#Eval for threshold 0.55: DER 8.86%, MS 4.93%, FA 3.32%, SC 0.62%
#
#Eval for threshold 0.60: DER 9.09%, MS 5.49%, FA 3.04%, SC 0.56%
#
#Eval for threshold 0.70: DER 9.95%, MS 6.96%, FA 2.63%, SC 0.36%
#
#Eval for threshold 0.80: DER 11.40%, MS 9.11%, FA 2.13%, SC 0.16%
#
#
#Test set
## Model DER:  0.20176313963590797
#Model ACC:  0.9279404524807562
#100%|██████████| 60/60 [00:39<00:00,  1.51it/s]
#Eval for threshold 0.20: DER 12.25%, MS 2.62%, FA 8.58%, SC 1.05%
#
#Eval for threshold 0.30: DER 10.69%, MS 3.48%, FA 6.11%, SC 1.11%
#
#Eval for threshold 0.35: DER 10.33%, MS 3.93%, FA 5.31%, SC 1.09%
#
#Eval for threshold 0.40: DER 10.16%, MS 4.45%, FA 4.66%, SC 1.05%
#
#Eval for threshold 0.45: DER 10.08%, MS 4.94%, FA 4.11%, SC 1.04%
#
#Eval for threshold 0.50: DER 10.09%, MS 5.48%, FA 3.60%, SC 1.01%
#
#Eval for threshold 0.55: DER 10.23%, MS 6.12%, FA 3.20%, SC 0.91%
#
#Eval for threshold 0.60: DER 10.43%, MS 6.76%, FA 2.87%, SC 0.80%
#
#Eval for threshold 0.70: DER 11.24%, MS 8.31%, FA 2.35%, SC 0.58%
#
#Eval for threshold 0.80: DER 12.88%, MS 10.71%, FA 1.83%, SC 0.34%

fi
if [ ${stage} -le 53 ] && [ ${stop_stage} -ge 53 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    # for loading speaker embedding file
    simu_data_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/200h_maximum_4spks6s_softlink
    spk_path=$simu_data_dir/ts_vad/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
    eval_test_spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_maximum_4spks6s_softlink_real_evalset_rs_len15s_seg_shift2s
    data_dir="$simu_data_dir" # oracle target audio , mix audio and labels path
    eval_test_data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=15
    segment_shift=2
    batch_size=64
    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 14915 \
    ts_vad2_simu/train_accelerate_ddp2_debug2_simu.py \
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
        --batch-size $batch_size\
        --rs-len $rs_len\
        --segment-shift $segment_shift\
        --speech-encoder-type $speech_encoder_type\
        --speech-encoder-path $speech_encoder_path\
        --select-encoder-layer-nums 6\
        --spk-path $spk_path\
        --eval-test-spk-path $eval_test_spk_path\
        --speaker-embedding-name-dir $speaker_embedding_name_dir\
        --exp-dir $exp_dir\
        --data-dir $data_dir\
        --eval-test-data-dir $eval_test_data_dir
fi

if [ ${stage} -le 54 ] && [ ${stop_stage} -ge 54 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_maximum_4spks6s_softlink_real_evalset_rs_len15s_seg_shift2s
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 eval_test_data_dir=$data_dir
 eval_test_spk_path=$spk_path
 for name in $infer_sets;do
    results_path=$exp_dir
   python3 ts_vad2_simu/infer2_simu2.py \
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
    --eval-test-spk-path $eval_test_spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --eval-test-data-dir $eval_test_data_dir
  done
fi

if [ ${stage} -le 60 ] && [ ${stop_stage} -ge 60 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    # for loading speaker embedding file
    simu_data_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/200h_maximum_4spks6s_softlink
    spk_path=$simu_data_dir/ts_vad/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
    eval_test_spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_maximum_4spks6s_softlink_real_evalset_rs_len30s_seg_shift2s
    data_dir="$simu_data_dir" # oracle target audio , mix audio and labels path
    eval_test_data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=30
    segment_shift=2
    batch_size=64
    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15915 \
    ts_vad2_simu/train_accelerate_ddp2_debug2_simu.py \
        --verbose 2\
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
        --batch-size $batch_size\
        --rs-len $rs_len\
        --segment-shift $segment_shift\
        --speech-encoder-type $speech_encoder_type\
        --speech-encoder-path $speech_encoder_path\
        --select-encoder-layer-nums 6\
        --spk-path $spk_path\
        --eval-test-spk-path $eval_test_spk_path\
        --speaker-embedding-name-dir $speaker_embedding_name_dir\
        --exp-dir $exp_dir\
        --data-dir $data_dir\
        --eval-test-data-dir $eval_test_data_dir
fi

if [ ${stage} -le 61 ] && [ ${stop_stage} -ge 61 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_maximum_4spks6s_softlink_real_evalset_rs_len30s_seg_shift2s
 model_file=$exp_dir/best-valid-der.pt
 rs_len=30
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 eval_test_data_dir=$data_dir
 eval_test_spk_path=$spk_path
 for name in $infer_sets;do
    results_path=$exp_dir
   python3 ts_vad2_simu/infer2_simu2.py \
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
    --eval-test-spk-path $eval_test_spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --eval-test-data-dir $eval_test_data_dir
  done

# bash run_ts_vad2_simu.sh --stage 61 --stop-stage 61
#Eval set
#Model DER:  0.22589323466468325
#Model ACC:  0.9202297237216607
#2024-12-12 11:16:31,061 (infer2_simu2:84) INFO: frame_len: 0.04!!
#100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25/25 [00:25<00:00,  1.03s/it]
#Eval for threshold 0.20: DER 11.79%, MS 3.21%, FA 7.20%, SC 1.38%
#
#Eval for threshold 0.30: DER 11.27%, MS 4.28%, FA 5.55%, SC 1.44%
#
#Eval for threshold 0.35: DER 11.17%, MS 4.81%, FA 4.93%, SC 1.43%
#
#Eval for threshold 0.40: DER 11.15%, MS 5.35%, FA 4.37%, SC 1.43%
#
#Eval for threshold 0.45: DER 11.28%, MS 5.93%, FA 3.92%, SC 1.42%
#
#Eval for threshold 0.50: DER 11.44%, MS 6.55%, FA 3.52%, SC 1.36%
#
#Eval for threshold 0.55: DER 11.70%, MS 7.15%, FA 3.24%, SC 1.31%
#
#Eval for threshold 0.60: DER 12.06%, MS 7.85%, FA 2.99%, SC 1.22%
#
#Eval for threshold 0.70: DER 13.07%, MS 9.51%, FA 2.61%, SC 0.95%
#
#Eval for threshold 0.80: DER 14.51%, MS 11.64%, FA 2.28%, SC 0.60%
#
#Test set
## Model DER:  0.23091182958815834
#Model ACC:  0.914876557433721
#2024-12-12 17:30:32,370 (infer2_simu2:84) INFO: frame_len: 0.04!!
#100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [01:10<00:00,  1.17s/it]
#Eval for threshold 0.20: DER 13.96%, MS 3.99%, FA 8.17%, SC 1.80%
#
#Eval for threshold 0.30: DER 12.98%, MS 5.20%, FA 5.98%, SC 1.80%
#
#Eval for threshold 0.35: DER 12.74%, MS 5.74%, FA 5.19%, SC 1.81%
#
#Eval for threshold 0.40: DER 12.71%, MS 6.34%, FA 4.58%, SC 1.79%
#
#Eval for threshold 0.45: DER 12.73%, MS 6.98%, FA 4.02%, SC 1.73%
#
#Eval for threshold 0.50: DER 12.85%, MS 7.61%, FA 3.55%, SC 1.69%
#
#Eval for threshold 0.55: DER 13.07%, MS 8.33%, FA 3.15%, SC 1.59%
#
#Eval for threshold 0.60: DER 13.42%, MS 9.17%, FA 2.85%, SC 1.41%
#
#Eval for threshold 0.70: DER 14.51%, MS 11.08%, FA 2.35%, SC 1.07%
#
#Eval for threshold 0.80: DER 16.28%, MS 13.63%, FA 1.91%, SC 0.75%

fi
# compared with stage41-42 stage65-66 will increase rs_len into 15seconds
if [ ${stage} -le 65 ] && [ ${stop_stage} -ge 65 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

    simu_data_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/data/200h_maximum_4spksall_softlink
    spk_path=$simu_data_dir/ts_vad/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
    eval_test_spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_maximum_4spksall_softlink_real_evalset_rs_len15s_seg_shift2s
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_maximum_4spksall_softlink_real_evalset_rs_len30s_seg_shift2s/
    data_dir="$simu_data_dir" # oracle target audio , mix audio and labels path
    eval_test_data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=30
    segment_shift=2
    batch_size=64
    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 16915 \
    ts_vad2_simu/train_accelerate_ddp2_debug2_simu.py \
        --verbose 2\
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
        --batch-size $batch_size\
        --rs-len $rs_len\
        --segment-shift $segment_shift\
        --speech-encoder-type $speech_encoder_type\
        --speech-encoder-path $speech_encoder_path\
        --select-encoder-layer-nums 6\
        --spk-path $spk_path\
        --eval-test-spk-path $eval_test_spk_path\
        --speaker-embedding-name-dir $speaker_embedding_name_dir\
        --exp-dir $exp_dir\
        --data-dir $data_dir\
        --eval-test-data-dir $eval_test_data_dir
fi

if [ ${stage} -le 66 ] && [ ${stop_stage} -ge 66 ];then
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_maximum_4spksall_softlink_real_evalset_rs_len15s_seg_shift2s
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/ts_vad2_simu/ts_vad2_simu_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_maximum_4spksall_softlink_real_evalset_rs_len30s_seg_shift2s/
 model_file=$exp_dir/best-valid-der.pt
 rs_len=30
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 eval_test_data_dir=$data_dir
 eval_test_spk_path=$spk_path
 for name in $infer_sets;do
    results_path=$exp_dir
   python3 ts_vad2_simu/infer2_simu2.py \
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
    --eval-test-spk-path $eval_test_spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --eval-test-data-dir $eval_test_data_dir
  done


# Eval set
# Model DER:  0.36613524106565754
#Model ACC:  0.8631226482075682
#100%|██████████| 25/25 [00:23<00:00,  1.07it/s]
#Eval for threshold 0.20: DER 29.85%, MS 8.18%, FA 17.70%, SC 3.97%
#
#Eval for threshold 0.30: DER 26.86%, MS 9.81%, FA 12.54%, SC 4.52%
#
#Eval for threshold 0.35: DER 25.92%, MS 10.55%, FA 10.60%, SC 4.77%
#
#Eval for threshold 0.40: DER 25.11%, MS 11.33%, FA 8.80%, SC 4.98%
#
#Eval for threshold 0.45: DER 24.63%, MS 12.08%, FA 7.39%, SC 5.16%
#
#Eval for threshold 0.50: DER 24.47%, MS 13.06%, FA 6.17%, SC 5.24%
#
#Eval for threshold 0.55: DER 24.39%, MS 14.03%, FA 5.07%, SC 5.29%
#
#Eval for threshold 0.60: DER 24.51%, MS 15.21%, FA 4.18%, SC 5.12%
#
#Eval for threshold 0.70: DER 26.11%, MS 18.84%, FA 3.25%, SC 4.03%
#
#Eval for threshold 0.80: DER 29.14%, MS 23.87%, FA 2.83%, SC 2.44%
#
#Test set
#Model DER:  0.39957905400728294
#Model ACC:  0.8429998119418366
#100%|██████████| 60/60 [00:59<00:00,  1.02it/s]
#Eval for threshold 0.20: DER 36.78%, MS 8.54%, FA 22.46%, SC 5.78%
#
#Eval for threshold 0.30: DER 33.15%, MS 10.16%, FA 16.16%, SC 6.84%
#
#Eval for threshold 0.35: DER 31.84%, MS 10.98%, FA 13.60%, SC 7.26%
#
#Eval for threshold 0.40: DER 30.76%, MS 11.79%, FA 11.43%, SC 7.54%
#
#Eval for threshold 0.45: DER 29.94%, MS 12.66%, FA 9.46%, SC 7.82%
#
#Eval for threshold 0.50: DER 29.32%, MS 13.62%, FA 7.66%, SC 8.03%
#
#Eval for threshold 0.55: DER 29.06%, MS 14.77%, FA 6.25%, SC 8.04%
#
#Eval for threshold 0.60: DER 29.19%, MS 16.28%, FA 5.14%, SC 7.76%
#
#Eval for threshold 0.70: DER 30.83%, MS 20.66%, FA 3.81%, SC 6.37%
#
#Eval for threshold 0.80: DER 33.95%, MS 26.55%, FA 3.22%, SC 4.17%

fi


# compared with run_ts_vad2.sh stage 122-123, I will add 200h_fixed_4spks6s data into offical alimeeting data. as finally we get this hybrid data "200h_fixed_4spks6s_and_alimeeting"
if [ ${stage} -le 70 ] && [ ${stop_stage} -ge 70 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/200h_fixed_4spks6s_and_alimeeting/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spks6s_and_alimeeting
    data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/200h_fixed_4spks6s_and_alimeeting" # oracle target audio , mix audio and labels path

    #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12915 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
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
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
     --data-dir $data_dir
fi

if [ ${stage} -le 71 ] && [ ${stop_stage} -ge 71 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spks6s_and_alimeeting
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

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
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir
done
# cat logs/run_ts_vad2_simu_stage70-71.log
# Eval set
# Model DER:  0.13433411388704478
#Model ACC:  0.9538564587156093
#100%|██████████| 25/25 [00:15<00:00,  1.60it/s]
#Eval for threshold 0.20: DER 12.05%, MS 0.66%, FA 11.13%, SC 0.25%
#
#Eval for threshold 0.30: DER 7.66%, MS 1.23%, FA 6.07%, SC 0.36%
#
#Eval for threshold 0.35: DER 6.60%, MS 1.57%, FA 4.66%, SC 0.37%
#
#Eval for threshold 0.40: DER 5.88%, MS 1.89%, FA 3.61%, SC 0.37%
#
#Eval for threshold 0.45: DER 5.46%, MS 2.36%, FA 2.70%, SC 0.40%
#
#Eval for threshold 0.50: DER 5.25%, MS 2.83%, FA 2.03%, SC 0.38%
#
#Eval for threshold 0.55: DER 5.30%, MS 3.45%, FA 1.51%, SC 0.34%
#
#Eval for threshold 0.60: DER 5.54%, MS 4.16%, FA 1.09%, SC 0.28%
#
#Eval for threshold 0.70: DER 6.66%, MS 5.84%, FA 0.63%, SC 0.19%
#
#Eval for threshold 0.80: DER 8.94%, MS 8.40%, FA 0.42%, SC 0.13%
#
#Test set
#Model DER:  0.13036463776521195
#Model ACC:  0.9522997354409475
#100%|██████████| 60/60 [00:37<00:00,  1.60it/s]
#Eval for threshold 0.20: DER 13.36%, MS 0.76%, FA 12.24%, SC 0.36%
#
#Eval for threshold 0.30: DER 8.70%, MS 1.38%, FA 6.87%, SC 0.44%
#
#Eval for threshold 0.35: DER 7.47%, MS 1.74%, FA 5.24%, SC 0.49%
#
#Eval for threshold 0.40: DER 6.68%, MS 2.19%, FA 3.96%, SC 0.53%
#
#Eval for threshold 0.45: DER 6.15%, MS 2.68%, FA 2.93%, SC 0.54%
#
#Eval for threshold 0.50: DER 5.90%, MS 3.22%, FA 2.13%, SC 0.55%
#
#Eval for threshold 0.55: DER 5.98%, MS 3.89%, FA 1.55%, SC 0.54%
#
#Eval for threshold 0.60: DER 6.24%, MS 4.67%, FA 1.08%, SC 0.49%
#
#Eval for threshold 0.70: DER 7.42%, MS 6.58%, FA 0.46%, SC 0.38%
#
#Eval for threshold 0.80: DER 9.83%, MS 9.43%, FA 0.18%, SC 0.22%

fi

# compared with run_ts_vad2.sh stage 122-123, I will add 200h_fixed_4spks6s data into offical alimeeting data. as finally we get this hybrid data "200h_fixed_4spks6s_and_alimeeting"
if [ ${stage} -le 72 ] && [ ${stop_stage} -ge 72 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    source_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spks6s_and_alimeeting
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spks6s_and_alimeeting_ft_on_alimeeting
    mkdir -p $exp_dir
    cp -r $source_dir/best-valid-der.pt $exp_dir/
    mv $exp_dir/best-valid-der.pt $exp_dir/epoch-0.pt

    data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

    #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12915 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
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
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
     --data-dir $data_dir
fi

if [ ${stage} -le 73 ] && [ ${stop_stage} -ge 73 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_on_200h_fixed_4spks6s_and_alimeeting_ft_on_alimeeting
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

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
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir
done
# cat logs/run_ts_vad2_simu_stage72-73.log
# Eval set
# Model DER:  0.1290534170029559
#Model ACC:  0.955783717572087
#100%|██████████| 25/25 [00:15<00:00,  1.62it/s]
#Eval for threshold 0.20: DER 9.71%, MS 0.79%, FA 8.66%, SC 0.27%
#
#Eval for threshold 0.30: DER 6.53%, MS 1.33%, FA 4.87%, SC 0.33%
#
#Eval for threshold 0.35: DER 5.80%, MS 1.65%, FA 3.82%, SC 0.33%
#
#Eval for threshold 0.40: DER 5.38%, MS 2.03%, FA 3.01%, SC 0.33%
#
#Eval for threshold 0.45: DER 5.09%, MS 2.43%, FA 2.34%, SC 0.33%
#
#Eval for threshold 0.50: DER 4.97%, MS 2.88%, FA 1.80%, SC 0.28%
#
#Eval for threshold 0.55: DER 5.05%, MS 3.42%, FA 1.38%, SC 0.25%
#
#Eval for threshold 0.60: DER 5.30%, MS 4.02%, FA 1.04%, SC 0.24%
#
#Eval for threshold 0.70: DER 6.34%, MS 5.54%, FA 0.62%, SC 0.17%
#
#Eval for threshold 0.80: DER 8.32%, MS 7.81%, FA 0.43%, SC 0.08%
#
## Test set
## Model DER:  0.12161077938039835
#Model ACC:  0.9555833773267308
#100%|██████████| 60/60 [00:37<00:00,  1.62it/s]
#Eval for threshold 0.20: DER 10.15%, MS 0.89%, FA 8.92%, SC 0.34%
#
#Eval for threshold 0.30: DER 7.00%, MS 1.49%, FA 5.10%, SC 0.41%
#
#Eval for threshold 0.35: DER 6.18%, MS 1.83%, FA 3.92%, SC 0.44%
#
#Eval for threshold 0.40: DER 5.67%, MS 2.22%, FA 3.00%, SC 0.46%
#
#Eval for threshold 0.45: DER 5.43%, MS 2.66%, FA 2.29%, SC 0.47%
#
#Eval for threshold 0.50: DER 5.35%, MS 3.19%, FA 1.70%, SC 0.47%
#
#Eval for threshold 0.55: DER 5.44%, MS 3.77%, FA 1.24%, SC 0.44%
#
#Eval for threshold 0.60: DER 5.70%, MS 4.41%, FA 0.87%, SC 0.41%
#
#Eval for threshold 0.70: DER 6.80%, MS 6.04%, FA 0.43%, SC 0.32%
#
#Eval for threshold 0.80: DER 8.88%, MS 8.52%, FA 0.18%, SC 0.19%
fi
