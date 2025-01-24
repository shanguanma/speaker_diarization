#!/usr/bin/env bash
stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_speaker_diarization_hltsz.sh


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
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    dataset_name="alimeeting_ami_aishell_4" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    # cam++ 200k
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    # cam++ en_zh
    #speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    
    # for loading speaker embedding file
    spk_path=/data/maduo/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4 # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len10
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4" # oracle target audio , mix audio and labels path
   rs_len=10
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
   d_state=256
   num_transformer_layer=2
   max_num_speaker=7
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17115 \
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
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len10
 model_file=$exp_dir/best-valid-der.pt
 rs_len=10
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="alimeeting" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding 
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len10
 model_file=$exp_dir/best-valid-der.pt
 rs_len=10
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/datasets/aishell-4/data_processed/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="aishell_4" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/aishell_4/SpeakerEmbedding/ # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/data/maduo/datasets/aishell-4/data_processed" # oracle target audio , mix audio and labels path
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
    --rttm-name ${name}/${name}.rttm\
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
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len10
 model_file=$exp_dir/best-valid-der.pt
 rs_len=10
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/datasets/ami/ami_version1.6.2/data_processed/data/ami
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="ami" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/ami/SpeakerEmbedding/ # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/data/maduo/datasets/ami/ami_version1.6.2/data_processed/data/ami" # oracle target audio , mix audio and labels path
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
    --rttm-name ${name}/${name}.rttm\
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
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi


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
    dataset_name="alimeeting_ami_aishell_4" # dataset name
    # for loading pretrain model weigt
    #speech_encoder_type="CAM++"
    # cam++ 200k
    #speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    # cam++ en_zh
    #speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/data/maduo/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4 # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len10
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4" # oracle target audio , mix audio and labels path
   rs_len=10
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
   d_state=256
   max_num_speaker=7
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17215 \
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
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker 
fi
