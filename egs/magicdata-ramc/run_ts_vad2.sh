#!/usr/bin/env bash
stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_dia_pt2.4.sh


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "prepared magicdata-ramc kaldi format data"
   ## it has been removed G00000000 utt in rttm file
   # based on the paper "The X-Lance Speaker Diarization System for the Conversational Short-phrase Speaker Diarization Challenge 2022"
   source_data_dir=/mntcephfs/lee_dataset/asr/MagicData-RAMC/
   output_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   python3 magicdata_ramc_prepared_180h_with_g0.py $source_data_dir $output_dir

   data_dir=$output_dir
   ## remove  G00000000
   for name in dev test train;do
       grep -v "G00000000" $data_dir/$name/rttm_debug > $data_dir/$name/rttm_debug_nog0
   done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
  echo "get target audio and json label file from rttm file"
  datasets="dev test train"
  #datasets="dev"
  source_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
  for name in $datasets;do
    oracle_rttm=$source_dir/$name/rttm_debug_nog0
    wavscp=$source_dir/$name/wav.scp
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc
    python3 ts_vad2/oracle_rttm_to_generate_target_speaker_wav_and_label_for_ts_vad.py\
         --oracle_rttm $oracle_rttm\
         --wavscp $wavscp\
         --dest_dir $dest_dir\
         --type $name
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "generate oracle vad speaker embedding"
   dest_dir=/mntcephfs/lab_data/maduo/model_hub
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
     input_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc/${name}/target_audio/
     wav_path=/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc/${name}/wavs.txt
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



# magicdata-ramc and 0.5 prob alimeeting
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
    dataset_name="alimeeting+magicdata-ramc" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store magicdata_ramc speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
    data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

    alimeeting_spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    alimeeting_data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

   exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc_with_prob0.5_alimeeting-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
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
# I will stop it, because its no longer decrease.
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc_with_prob0.5_alimeeting-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
 model_file=$exp_dir/checkpoint-190500.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store magicdata_ramc speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path


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
# collar=0.0
# Eval set
# Eval for threshold 0.5 DER=18.45, miss=1.59, falarm=7.55, confusion=9.31
# Test set
# Eval for threshold 0.5 DER=19.14, miss=1.67, falarm=8.03, confusion=9.43

fi



# magicdata-ramc and 0.8 prob alimeeting
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
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    dataset_name="alimeeting+magicdata-ramc" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store magicdata_ramc speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
    data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

    alimeeting_spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    alimeeting_data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

   exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc_with_prob0.8_alimeeting-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
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
    --save-every-n 500\
    --freeze-updates 16000\
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

if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc_with_prob0.8_alimeeting-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store magicdata_ramc speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
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



# magicdata-ramc and 0.3 prob alimeeting
if [ ${stage} -le 25 ] && [ ${stop_stage} -ge 25 ];then
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
    dataset_name="alimeeting+magicdata-ramc" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store magicdata_ramc speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
    data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

    alimeeting_spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    alimeeting_data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

   exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc_with_prob0.3_alimeeting-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
   mkdir -p $exp_dir
   rs_len=4
   segment_shift=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 14815 \
   ts_vad2/train_accelerate_ddp_multi.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-4\
    --alimeeting-prob 0.3\
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

if [ ${stage} -le 26 ] && [ ${stop_stage} -ge 26 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc_with_prob0.3_alimeeting-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store magicdata_ramc speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
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


# magicdata-ramc and 0.8 prob alimeeting and eval set is only magicdata_ramc dev
if [ ${stage} -le 35 ] && [ ${stop_stage} -ge 35 ];then
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
    dataset_name="alimeeting+magicdata-ramc" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store magicdata_ramc speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
    data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

    alimeeting_spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    alimeeting_data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

   exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc_with_prob0.8_alimeeting-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_magicdata_dev_as_eval
   mkdir -p $exp_dir
   rs_len=4
   segment_shift=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15815 \
   ts_vad2/train_accelerate_ddp_multi2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --save-every-n 500\
    --freeze-updates 16000\
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

if [ ${stage} -le 36 ] && [ ${stop_stage} -ge 36 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc_with_prob0.8_alimeeting-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_magicdata_dev_as_eval
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store magicdata_ramc speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
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

# its loss is nan when training to the seventh epoch. you can see the log: logs/run_ts_vad2_stage52-53.log
# compared with stage40-41, stage52-53 will increase lr_rate from 1e-5 to 15e-5 (note: lr_rate=2e-4, loss will have nan, lr_rate=15e-5, loss will have nan)
if [ ${stage} -le 52 ] && [ ${stop_stage} -ge 52 ];then
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
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr15e5_both_mamba_2layers_mamba
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
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
    --lr 15e-5\
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

if [ ${stage} -le 53 ] && [ ${stop_stage} -ge 53 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr15e5_both_mamba_2layers_mamba
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
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

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


# compared with stage40-41 of run_ts_vad2_hltsz.sh, stage62-63 use single_backend_type="transformer"
if [ ${stage} -le 62 ] && [ ${stop_stage} -ge 62 ];then
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
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_transformer_multi_backend_mamba
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   single_backend_type="transformer"
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

if [ ${stage} -le 63 ] && [ ${stop_stage} -ge 63 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_transformer_multi_backend_mamba
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="mamba"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 #collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

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
#grep -r 'Eval' logs/run_ts_vad2_stage62-63.log
#collar=0.0
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=15.80, miss=0.22, falarm=12.76, confusion=2.82
#Eval for threshold 0.3 DER=13.25, miss=0.42, falarm=9.23, confusion=3.59
#Eval for threshold 0.35 DER=12.56, miss=0.57, falarm=8.20, confusion=3.78
#Eval for threshold 0.4 DER=12.11, miss=0.81, falarm=7.43, confusion=3.87
#Eval for threshold 0.45 DER=11.74, miss=1.12, falarm=6.76, confusion=3.87
#Eval for threshold 0.5 DER=11.60, miss=1.56, falarm=6.28, confusion=3.75
#Eval for threshold 0.55 DER=11.58, miss=2.14, falarm=5.88, confusion=3.56
#Eval for threshold 0.6 DER=11.71, miss=2.88, falarm=5.47, confusion=3.36
#Eval for threshold 0.7 DER=12.64, miss=5.39, falarm=4.60, confusion=2.65
#Eval for threshold 0.8 DER=14.81, miss=9.29, falarm=3.66, confusion=1.87
#
## test of magicdata-ramc
#Eval for threshold 0.2 DER=17.84, miss=0.25, falarm=15.76, confusion=1.82
#Eval for threshold 0.3 DER=15.50, miss=0.54, falarm=12.69, confusion=2.27
#Eval for threshold 0.35 DER=14.70, miss=0.77, falarm=11.53, confusion=2.40
#Eval for threshold 0.4 DER=14.11, miss=1.09, falarm=10.51, confusion=2.51
#Eval for threshold 0.45 DER=13.40, miss=1.56, falarm=9.12, confusion=2.73
#Eval for threshold 0.5 DER=12.49, miss=2.83, falarm=6.63, confusion=3.03
#Eval for threshold 0.55 DER=13.41, miss=5.24, falarm=6.01, confusion=2.16
#Eval for threshold 0.6 DER=13.66, miss=6.14, falarm=5.57, confusion=1.95
#Eval for threshold 0.7 DER=14.48, miss=8.17, falarm=4.73, confusion=1.58
#Eval for threshold 0.8 DER=16.34, miss=11.35, falarm=3.80, confusion=1.18


# grep -r 'Eval' logs/run_ts_vad2_stage63.log
#collar=0.25
# dev of magicdata-ramc
# Eval for threshold 0.2 DER=7.38, miss=0.04, falarm=4.89, confusion=2.45
#Eval for threshold 0.3 DER=5.74, miss=0.08, falarm=2.50, confusion=3.16
#Eval for threshold 0.35 DER=5.43, miss=0.11, falarm=1.99, confusion=3.33
#Eval for threshold 0.4 DER=5.26, miss=0.19, falarm=1.68, confusion=3.39
#Eval for threshold 0.45 DER=5.15, miss=0.30, falarm=1.46, confusion=3.39
#Eval for threshold 0.5 DER=5.19, miss=0.51, falarm=1.35, confusion=3.33
#Eval for threshold 0.55 DER=5.29, miss=0.80, falarm=1.26, confusion=3.22
#Eval for threshold 0.6 DER=5.48, miss=1.23, falarm=1.18, confusion=3.07
#Eval for threshold 0.7 DER=6.56, miss=3.04, falarm=1.04, confusion=2.48
#Eval for threshold 0.8 DER=8.74, miss=6.08, falarm=0.90, confusion=1.76

## test of magicdata-ramc
#Eval for threshold 0.2 DER=9.20, miss=0.04, falarm=7.85, confusion=1.32
#Eval for threshold 0.3 DER=7.93, miss=0.12, falarm=6.20, confusion=1.61
#Eval for threshold 0.35 DER=7.55, miss=0.21, falarm=5.66, confusion=1.69
#Eval for threshold 0.4 DER=7.31, miss=0.35, falarm=5.20, confusion=1.76
#Eval for threshold 0.45 DER=6.94, miss=0.56, falarm=4.39, confusion=1.98
#Eval for threshold 0.5 DER=6.22, miss=1.45, falarm=2.36, confusion=2.41
#Eval for threshold 0.55 DER=7.32, miss=3.59, falarm=2.13, confusion=1.61
#Eval for threshold 0.6 DER=7.62, miss=4.13, falarm=2.00, confusion=1.48
#Eval for threshold 0.7 DER=8.35, miss=5.31, falarm=1.77, confusion=1.27
#Eval for threshold 0.8 DER=9.97, miss=7.45, falarm=1.52, confusion=1.00


fi

# compared with stage40-41 of run_ts_vad2_hltsz.sh, stage62-63 use multi_backend_type="transformer"
if [ ${stage} -le 72 ] && [ ${stop_stage} -ge 72 ];then
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
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba_multi_backend_transformer
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   single_backend_type="mamba"
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

if [ ${stage} -le 73 ] && [ ${stop_stage} -ge 73 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba_multi_backend_transformer
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="mamba"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 #collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

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
#grep -r Eval logs/run_ts_vad2_stage72-73.log
#collar=0.0
#dev of magicdata-ramc
#Eval for threshold 0.2 DER=17.88, miss=0.25, falarm=15.32, confusion=2.31
#Eval for threshold 0.3 DER=14.67, miss=0.44, falarm=11.19, confusion=3.03
#Eval for threshold 0.35 DER=13.58, miss=0.60, falarm=9.63, confusion=3.35
#Eval for threshold 0.4 DER=12.78, miss=0.82, falarm=8.35, confusion=3.61
#Eval for threshold 0.45 DER=12.14, miss=1.10, falarm=7.24, confusion=3.80
#Eval for threshold 0.5 DER=11.73, miss=1.49, falarm=6.36, confusion=3.88
#Eval for threshold 0.55 DER=11.70, miss=2.23, falarm=5.77, confusion=3.70
#Eval for threshold 0.6 DER=11.89, miss=3.19, falarm=5.28, confusion=3.42
#Eval for threshold 0.7 DER=12.78, miss=5.74, falarm=4.29, confusion=2.74
#Eval for threshold 0.8 DER=15.06, miss=9.76, falarm=3.30, confusion=2.00
#
#test of magicdata-ramc
#Eval for threshold 0.2 DER=19.52, miss=0.28, falarm=17.30, confusion=1.94
#Eval for threshold 0.3 DER=16.78, miss=0.61, falarm=13.66, confusion=2.52
#Eval for threshold 0.35 DER=15.82, miss=0.83, falarm=12.21, confusion=2.79
#Eval for threshold 0.4 DER=14.96, miss=1.14, falarm=10.73, confusion=3.09
#Eval for threshold 0.45 DER=14.08, miss=1.57, falarm=9.12, confusion=3.39
#Eval for threshold 0.5 DER=13.04, miss=2.20, falarm=6.95, confusion=3.90
#Eval for threshold 0.55 DER=13.34, miss=4.15, falarm=5.82, confusion=3.37
#Eval for threshold 0.6 DER=14.06, miss=5.99, falarm=5.29, confusion=2.78
#Eval for threshold 0.7 DER=15.22, miss=8.90, falarm=4.31, confusion=2.02
#Eval for threshold 0.8 DER=17.26, miss=12.45, falarm=3.34, confusion=1.47


#grep -r Eval logs/run_ts_vad2_stage73.log
# collar=0.25
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=9.38, miss=0.05, falarm=7.39, confusion=1.95
#Eval for threshold 0.3 DER=7.10, miss=0.09, falarm=4.45, confusion=2.57
#Eval for threshold 0.35 DER=6.38, miss=0.13, falarm=3.39, confusion=2.86
#Eval for threshold 0.4 DER=5.90, miss=0.20, falarm=2.59, confusion=3.10
#Eval for threshold 0.45 DER=5.52, miss=0.29, falarm=1.92, confusion=3.31
#Eval for threshold 0.5 DER=5.31, miss=0.44, falarm=1.44, confusion=3.44
#Eval for threshold 0.55 DER=5.43, miss=0.84, falarm=1.26, confusion=3.33
#Eval for threshold 0.6 DER=5.73, miss=1.44, falarm=1.17, confusion=3.13
#Eval for threshold 0.7 DER=6.78, miss=3.20, falarm=1.01, confusion=2.57
#Eval for threshold 0.8 DER=9.01, miss=6.22, falarm=0.87, confusion=1.92

# test of magicdata-ramc
#Eval for threshold 0.2 DER=10.88, miss=0.05, falarm=9.33, confusion=1.50
#Eval for threshold 0.3 DER=9.26, miss=0.16, falarm=7.16, confusion=1.94
#Eval for threshold 0.35 DER=8.73, miss=0.25, falarm=6.34, confusion=2.14
#Eval for threshold 0.4 DER=8.25, miss=0.37, falarm=5.46, confusion=2.42
#Eval for threshold 0.45 DER=7.70, miss=0.56, falarm=4.45, confusion=2.70
#Eval for threshold 0.5 DER=6.89, miss=0.85, falarm=2.79, confusion=3.25
#Eval for threshold 0.55 DER=7.34, miss=2.42, falarm=2.09, confusion=2.83
#Eval for threshold 0.6 DER=8.17, miss=3.88, falarm=1.95, confusion=2.34
#Eval for threshold 0.7 DER=9.28, miss=5.87, falarm=1.70, confusion=1.71
#Eval for threshold 0.8 DER=10.95, miss=8.19, falarm=1.45, confusion=1.31

fi


if [ ${stage} -le 82 ] && [ ${stop_stage} -ge 82 ];then
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
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_mamba2
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="mamba2"
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 11815 \
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

if [ ${stage} -le 83 ] && [ ${stop_stage} -ge 83 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_mamba2
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="mamba2"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
# collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

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
#grep -r Eval logs/run_ts_vad2_stage82-83.log
#collar=0.0
#dev of magicdata-ramc
#Eval for threshold 0.2 DER=18.17, miss=0.35, falarm=15.20, confusion=2.62
#Eval for threshold 0.3 DER=14.23, miss=0.73, falarm=9.95, confusion=3.56
#Eval for threshold 0.35 DER=13.14, miss=1.01, falarm=8.33, confusion=3.79
#Eval for threshold 0.4 DER=12.45, miss=1.43, falarm=7.11, confusion=3.91
#Eval for threshold 0.45 DER=12.05, miss=1.96, falarm=6.18, confusion=3.92
#Eval for threshold 0.5 DER=11.94, miss=2.63, falarm=5.49, confusion=3.81
#Eval for threshold 0.55 DER=12.07, miss=3.45, falarm=5.00, confusion=3.63
#Eval for threshold 0.6 DER=12.39, miss=4.42, falarm=4.55, confusion=3.42
#Eval for threshold 0.7 DER=13.58, miss=6.84, falarm=3.77, confusion=2.97
#Eval for threshold 0.8 DER=16.22, miss=11.01, falarm=3.00, confusion=2.21

# test of magicdata-ramc
#Eval for threshold 0.2 DER=20.03, miss=0.41, falarm=18.10, confusion=1.52
#Eval for threshold 0.3 DER=16.55, miss=0.88, falarm=13.53, confusion=2.13
#Eval for threshold 0.35 DER=15.47, miss=1.28, falarm=11.80, confusion=2.39
#Eval for threshold 0.4 DER=14.65, miss=1.80, falarm=10.25, confusion=2.60
#Eval for threshold 0.45 DER=13.84, miss=2.51, falarm=8.47, confusion=2.87
#Eval for threshold 0.5 DER=12.98, miss=3.58, falarm=6.24, confusion=3.15
#Eval for threshold 0.55 DER=13.73, miss=6.02, falarm=5.20, confusion=2.51
#Eval for threshold 0.6 DER=14.69, miss=8.02, falarm=4.74, confusion=1.93
#Eval for threshold 0.7 DER=16.32, miss=10.93, falarm=3.96, confusion=1.43
#Eval for threshold 0.8 DER=18.92, miss=14.82, falarm=3.12, confusion=0.98


#grep -r 'Eval' logs/run_ts_vad2_stage83.log
# collar=0.25
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=9.06, miss=0.05, falarm=6.72, confusion=2.30
#Eval for threshold 0.3 DER=6.49, miss=0.11, falarm=3.32, confusion=3.06
#Eval for threshold 0.35 DER=5.84, miss=0.17, falarm=2.44, confusion=3.23
#Eval for threshold 0.4 DER=5.52, miss=0.32, falarm=1.86, confusion=3.33
#Eval for threshold 0.45 DER=5.34, miss=0.51, falarm=1.46, confusion=3.37
#Eval for threshold 0.5 DER=5.34, miss=0.77, falarm=1.23, confusion=3.34
#Eval for threshold 0.55 DER=5.49, miss=1.11, falarm=1.11, confusion=3.27
#Eval for threshold 0.6 DER=5.79, miss=1.59, falarm=1.03, confusion=3.17
#Eval for threshold 0.7 DER=6.76, miss=2.94, falarm=0.92, confusion=2.91
#Eval for threshold 0.8 DER=9.14, miss=6.07, falarm=0.81, confusion=2.26

# test of magicdata-ramc
#Eval for threshold 0.2 DER=10.94, miss=0.06, falarm=9.87, confusion=1.02
#Eval for threshold 0.3 DER=8.77, miss=0.19, falarm=7.16, confusion=1.42
#Eval for threshold 0.35 DER=8.16, miss=0.34, falarm=6.22, confusion=1.61
#Eval for threshold 0.4 DER=7.70, miss=0.56, falarm=5.36, confusion=1.78
#Eval for threshold 0.45 DER=7.17, miss=0.88, falarm=4.26, confusion=2.04
#Eval for threshold 0.5 DER=6.47, miss=1.44, falarm=2.59, confusion=2.44
#Eval for threshold 0.55 DER=7.20, miss=3.32, falarm=1.92, confusion=1.96
#Eval for threshold 0.6 DER=8.11, miss=4.86, falarm=1.78, confusion=1.47
#Eval for threshold 0.7 DER=9.46, miss=6.74, falarm=1.56, confusion=1.16
#Eval for threshold 0.8 DER=11.68, miss=9.51, falarm=1.32, confusion=0.86
fi


if [ ${stage} -le 84 ] && [ ${stop_stage} -ge 84 ];then
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
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 11815 \
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

if [ ${stage} -le 85 ] && [ ${stop_stage} -ge 85 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer
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
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
# collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

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
# grep -r 'Eval' logs/run_ts_vad2_stage84-85.log
# # collar=0.25
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=7.68, miss=0.05, falarm=4.97, confusion=2.66
#Eval for threshold 0.3 DER=6.07, miss=0.11, falarm=2.79, confusion=3.17
#Eval for threshold 0.35 DER=5.67, miss=0.15, falarm=2.24, confusion=3.29
#Eval for threshold 0.4 DER=5.42, miss=0.22, falarm=1.86, confusion=3.35
#Eval for threshold 0.45 DER=5.25, miss=0.32, falarm=1.56, confusion=3.38
#Eval for threshold 0.5 DER=5.19, miss=0.46, falarm=1.33, confusion=3.40
#Eval for threshold 0.55 DER=5.29, miss=0.75, falarm=1.22, confusion=3.32
#Eval for threshold 0.6 DER=5.46, miss=1.09, falarm=1.14, confusion=3.23
#Eval for threshold 0.7 DER=6.11, miss=2.10, falarm=0.99, confusion=3.01
#Eval for threshold 0.8 DER=7.80, miss=4.41, falarm=0.87, confusion=2.52
#
#test of magicdata-ramc
#Eval for threshold 0.2 DER=9.33, miss=0.05, falarm=7.93, confusion=1.35
#Eval for threshold 0.3 DER=8.10, miss=0.17, falarm=6.31, confusion=1.62
#Eval for threshold 0.35 DER=7.66, miss=0.26, falarm=5.63, confusion=1.77
#Eval for threshold 0.4 DER=7.17, miss=0.37, falarm=4.78, confusion=2.02
#Eval for threshold 0.45 DER=6.70, miss=0.57, falarm=3.80, confusion=2.33
#Eval for threshold 0.5 DER=6.14, miss=1.06, falarm=2.26, confusion=2.81
#Eval for threshold 0.55 DER=6.78, miss=2.60, falarm=2.04, confusion=2.14
#Eval for threshold 0.6 DER=7.25, miss=3.49, falarm=1.92, confusion=1.83
#Eval for threshold 0.7 DER=8.22, miss=5.13, falarm=1.68, confusion=1.42
#Eval for threshold 0.8 DER=9.72, miss=7.09, falarm=1.46, confusion=1.17
fi


if [ ${stage} -le 86 ] && [ ${stop_stage} -ge 86 ];then
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
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_transformer_multi_backend_mamba2
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   single_backend_type="transformer"
   multi_backend_type="mamba2"
   num_transformer_layer=2
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

if [ ${stage} -le 87 ] && [ ${stop_stage} -ge 87 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_transformer_multi_backend_mamba2
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
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
# collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

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
#grep -r 'Eval' logs/run_ts_vad2_stage86-87.log
#collar=0.25
#dev of magicdata-ramc
#Eval for threshold 0.2 DER=10.65, miss=0.04, falarm=8.86, confusion=1.74
#Eval for threshold 0.3 DER=8.65, miss=0.08, falarm=6.54, confusion=2.03
#Eval for threshold 0.35 DER=7.80, miss=0.10, falarm=5.45, confusion=2.25
#Eval for threshold 0.4 DER=6.93, miss=0.14, falarm=4.27, confusion=2.52
#Eval for threshold 0.45 DER=6.14, miss=0.21, falarm=3.12, confusion=2.81
#Eval for threshold 0.5 DER=5.58, miss=0.33, falarm=2.18, confusion=3.06
#Eval for threshold 0.55 DER=5.30, miss=0.48, falarm=1.58, confusion=3.24
#Eval for threshold 0.6 DER=5.27, miss=0.69, falarm=1.29, confusion=3.28
#Eval for threshold 0.7 DER=5.81, miss=1.77, falarm=1.07, confusion=2.97
#Eval for threshold 0.8 DER=7.87, miss=4.78, falarm=0.93, confusion=2.16
#
#test of magicdata-ramc
#Eval for threshold 0.2 DER=12.12, miss=0.03, falarm=11.23, confusion=0.86
#Eval for threshold 0.3 DER=9.77, miss=0.10, falarm=8.49, confusion=1.19
#Eval for threshold 0.35 DER=8.96, miss=0.16, falarm=7.48, confusion=1.32
#Eval for threshold 0.4 DER=8.32, miss=0.26, falarm=6.62, confusion=1.45
#Eval for threshold 0.45 DER=7.86, miss=0.39, falarm=5.90, confusion=1.57
#Eval for threshold 0.5 DER=7.50, miss=0.59, falarm=5.22, confusion=1.69
#Eval for threshold 0.55 DER=7.03, miss=0.86, falarm=4.14, confusion=2.04
#Eval for threshold 0.6 DER=6.33, miss=1.34, falarm=2.42, confusion=2.56
#Eval for threshold 0.7 DER=7.89, miss=4.63, falarm=1.81, confusion=1.45
#Eval for threshold 0.8 DER=9.47, miss=6.91, falarm=1.54, confusion=1.02
fi


# compared with stage84-85, stage90-91will increase rs_len from 4 to 6.
if [ ${stage} -le 90 ] && [ ${stop_stage} -ge 90 ];then
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
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_rs_len6
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=6
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
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

if [ ${stage} -le 91 ] && [ ${stop_stage} -ge 91 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_rs_len6
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

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
#grep -r Eval logs/run_ts_vad2_stage90-91.log
#collarr=0.25
#dev of magicdata-ramc
#Eval for threshold 0.2 DER=6.81, miss=0.06, falarm=3.79, confusion=2.96
#Eval for threshold 0.3 DER=5.66, miss=0.12, falarm=2.28, confusion=3.26
#Eval for threshold 0.35 DER=5.39, miss=0.17, falarm=1.90, confusion=3.32
#Eval for threshold 0.4 DER=5.22, miss=0.25, falarm=1.63, confusion=3.34
#Eval for threshold 0.45 DER=5.13, miss=0.34, falarm=1.44, confusion=3.35
#Eval for threshold 0.5 DER=5.09, miss=0.48, falarm=1.27, confusion=3.34
#Eval for threshold 0.55 DER=5.18, miss=0.71, falarm=1.18, confusion=3.28
#Eval for threshold 0.6 DER=5.34, miss=0.98, falarm=1.12, confusion=3.24
#Eval for threshold 0.7 DER=5.84, miss=1.77, falarm=0.99, confusion=3.09
#Eval for threshold 0.8 DER=7.21, miss=3.57, falarm=0.87, confusion=2.78
#test of magicdata-ram
#Eval for threshold 0.2 DER=8.81, miss=0.08, falarm=7.37, confusion=1.35
#Eval for threshold 0.3 DER=7.81, miss=0.25, falarm=6.00, confusion=1.56
#Eval for threshold 0.35 DER=7.55, miss=0.38, falarm=5.56, confusion=1.62
#Eval for threshold 0.4 DER=7.34, miss=0.55, falarm=5.08, confusion=1.71
#Eval for threshold 0.45 DER=6.47, miss=0.78, falarm=3.52, confusion=2.17
#Eval for threshold 0.5 DER=6.03, miss=1.26, falarm=2.14, confusion=2.63
#Eval for threshold 0.55 DER=6.93, miss=3.14, falarm=1.97, confusion=1.83
#Eval for threshold 0.6 DER=7.66, miss=4.32, falarm=1.83, confusion=1.51
#Eval for threshold 0.7 DER=8.56, miss=5.63, falarm=1.61, confusion=1.32
#Eval for threshold 0.8 DER=10.21, miss=7.74, falarm=1.37, confusion=1.10

#grep -r Eval logs/run_ts_vad2_stage91.log
# collar=0.0
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=15.21, miss=0.26, falarm=11.74, confusion=3.21
#Eval for threshold 0.3 DER=13.14, miss=0.51, falarm=9.03, confusion=3.60
#Eval for threshold 0.35 DER=12.48, miss=0.67, falarm=8.11, confusion=3.69
#Eval for threshold 0.4 DER=12.04, miss=0.90, falarm=7.41, confusion=3.73
#Eval for threshold 0.45 DER=11.72, miss=1.22, falarm=6.77, confusion=3.73
#Eval for threshold 0.5 DER=11.48, miss=1.59, falarm=6.17, confusion=3.71
#Eval for threshold 0.55 DER=11.42, miss=2.14, falarm=5.69, confusion=3.59
#Eval for threshold 0.6 DER=11.47, miss=2.74, falarm=5.24, confusion=3.49
#Eval for threshold 0.7 DER=11.89, miss=4.39, falarm=4.30, confusion=3.20
#Eval for threshold 0.8 DER=13.33, miss=7.20, falarm=3.37, confusion=2.76
# test of magicdata-ramc
#Eval for threshold 0.2 DER=17.30, miss=0.35, falarm=15.16, confusion=1.80
#Eval for threshold 0.3 DER=15.28, miss=0.77, falarm=12.35, confusion=2.15
#Eval for threshold 0.35 DER=14.63, miss=1.07, falarm=11.29, confusion=2.27
#Eval for threshold 0.4 DER=14.05, miss=1.46, falarm=10.20, confusion=2.39
#Eval for threshold 0.45 DER=12.87, miss=1.96, falarm=8.05, confusion=2.85
#Eval for threshold 0.5 DER=12.26, miss=2.81, falarm=6.25, confusion=3.20
#Eval for threshold 0.55 DER=13.01, miss=5.03, falarm=5.61, confusion=2.37
#Eval for threshold 0.6 DER=13.66, miss=6.58, falarm=5.11, confusion=1.98
#Eval for threshold 0.7 DER=14.66, miss=8.86, falarm=4.18, confusion=1.61
#Eval for threshold 0.8 DER=16.65, miss=12.11, falarm=3.30, confusion=1.24



fi


# compared with stage84-85, stage92-93will reduce d_state from 64 to 16
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
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    dataset_name="magicdata-ramc" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state16
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
   d_state=16
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15815 \
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
    --d-state $d_state
fi

if [ ${stage} -le 93 ] && [ ${stop_stage} -ge 93 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state16
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=16
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

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
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
done

#grep -r 'Eval' logs/run_ts_vad2_stage92-93_mamba2_d_state16.log
# collar=0.25
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=7.31, miss=0.05, falarm=4.70, confusion=2.56
#Eval for threshold 0.3 DER=5.87, miss=0.09, falarm=2.67, confusion=3.11
#Eval for threshold 0.35 DER=5.54, miss=0.14, falarm=2.14, confusion=3.26
#Eval for threshold 0.4 DER=5.33, miss=0.21, falarm=1.81, confusion=3.30
#Eval for threshold 0.45 DER=5.20, miss=0.32, falarm=1.55, confusion=3.33
#Eval for threshold 0.5 DER=5.13, miss=0.45, falarm=1.33, confusion=3.35
#Eval for threshold 0.55 DER=5.21, miss=0.70, falarm=1.24, confusion=3.28
#Eval for threshold 0.6 DER=5.34, miss=0.97, falarm=1.16, confusion=3.21
#Eval for threshold 0.7 DER=5.90, miss=1.91, falarm=1.02, confusion=2.98
#Eval for threshold 0.8 DER=7.36, miss=4.07, falarm=0.89, confusion=2.40

# test of magicdata-ramc
#Eval for threshold 0.2 DER=9.26, miss=0.05, falarm=7.87, confusion=1.34
#Eval for threshold 0.3 DER=8.09, miss=0.16, falarm=6.35, confusion=1.58
#Eval for threshold 0.35 DER=7.76, miss=0.25, falarm=5.82, confusion=1.69
#Eval for threshold 0.4 DER=7.50, miss=0.37, falarm=5.29, confusion=1.84
#Eval for threshold 0.45 DER=6.98, miss=0.56, falarm=4.22, confusion=2.21
#Eval for threshold 0.5 DER=6.15, miss=0.92, falarm=2.38, confusion=2.85
#Eval for threshold 0.55 DER=6.98, miss=2.77, falarm=2.10, confusion=2.11
#Eval for threshold 0.6 DER=7.52, miss=3.85, falarm=1.99, confusion=1.69
#Eval for threshold 0.7 DER=8.16, miss=5.06, falarm=1.73, confusion=1.37
#Eval for threshold 0.8 DER=9.52, miss=6.88, falarm=1.49, confusion=1.14
fi


# compared with stage84-85, stage94-95will increase d_state from 64 to 128
if [ ${stage} -le 94 ] && [ ${stop_stage} -ge 94 ];then
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
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state128
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
   d_state=128
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 16815 \
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
    --d-state $d_state
fi

if [ ${stage} -le 95 ] && [ ${stop_stage} -ge 95 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state128
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=128
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

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
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
done
# grep -r Eval logs/run_ts_vad2_stage94-95_mamba2_d_state128.log
#collar=0.25
#dev of magicdata-ramc
#Eval for threshold 0.2 DER=7.08, miss=0.05, falarm=4.27, confusion=2.75
#Eval for threshold 0.3 DER=5.76, miss=0.12, falarm=2.41, confusion=3.23
#Eval for threshold 0.35 DER=5.47, miss=0.16, falarm=1.98, confusion=3.32
#Eval for threshold 0.4 DER=5.30, miss=0.26, falarm=1.70, confusion=3.34
#Eval for threshold 0.45 DER=5.17, miss=0.36, falarm=1.46, confusion=3.35
#Eval for threshold 0.5 DER=5.13, miss=0.52, falarm=1.29, confusion=3.33
#Eval for threshold 0.55 DER=5.21, miss=0.76, falarm=1.19, confusion=3.26
#Eval for threshold 0.6 DER=5.39, miss=1.06, falarm=1.12, confusion=3.21
#Eval for threshold 0.7 DER=5.94, miss=1.90, falarm=1.00, confusion=3.04
#Eval for threshold 0.8 DER=7.46, miss=4.08, falarm=0.85, confusion=2.53
#test of magicdata-ramc
#Eval for threshold 0.2 DER=9.34, miss=0.06, falarm=8.02, confusion=1.25
#Eval for threshold 0.3 DER=7.95, miss=0.18, falarm=6.24, confusion=1.53
#Eval for threshold 0.35 DER=7.56, miss=0.29, falarm=5.63, confusion=1.64
#Eval for threshold 0.4 DER=7.07, miss=0.43, falarm=4.70, confusion=1.93
#Eval for threshold 0.45 DER=6.51, miss=0.66, falarm=3.47, confusion=2.38
#Eval for threshold 0.5 DER=5.99, miss=1.01, falarm=2.23, confusion=2.74
#Eval for threshold 0.55 DER=6.58, miss=2.24, falarm=2.02, confusion=2.32
#Eval for threshold 0.6 DER=7.21, miss=3.55, falarm=1.88, confusion=1.78
#Eval for threshold 0.7 DER=8.33, miss=5.40, falarm=1.65, confusion=1.29
#Eval for threshold 0.8 DER=10.09, miss=7.67, falarm=1.38, confusion=1.04

#grep -r Eval logs/run_ts_vad2_stage95_mamba2_d_state128.log
# collar=0.0
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=15.63, miss=0.25, falarm=12.35, confusion=3.03
#Eval for threshold 0.3 DER=13.34, miss=0.49, falarm=9.24, confusion=3.60
#Eval for threshold 0.35 DER=12.68, miss=0.68, falarm=8.27, confusion=3.73
#Eval for threshold 0.4 DER=12.19, miss=0.92, falarm=7.50, confusion=3.77
#Eval for threshold 0.45 DER=11.81, miss=1.23, falarm=6.81, confusion=3.78
#Eval for threshold 0.5 DER=11.59, miss=1.66, falarm=6.22, confusion=3.71
#Eval for threshold 0.55 DER=11.56, miss=2.24, falarm=5.75, confusion=3.57
#Eval for threshold 0.6 DER=11.66, miss=2.92, falarm=5.30, confusion=3.44
#Eval for threshold 0.7 DER=12.18, miss=4.64, falarm=4.41, confusion=3.13
#Eval for threshold 0.8 DER=13.83, miss=7.91, falarm=3.40, confusion=2.51
# test of magicdata-ramc
#Eval for threshold 0.2 DER=18.05, miss=0.32, falarm=16.05, confusion=1.69
#Eval for threshold 0.3 DER=15.55, miss=0.67, falarm=12.77, confusion=2.12
#Eval for threshold 0.35 DER=14.74, miss=0.93, falarm=11.52, confusion=2.29
#Eval for threshold 0.4 DER=13.86, miss=1.29, falarm=9.96, confusion=2.62
#Eval for threshold 0.45 DER=12.98, miss=1.79, falarm=8.11, confusion=3.07
#Eval for threshold 0.5 DER=12.31, miss=2.49, falarm=6.46, confusion=3.36
#Eval for threshold 0.55 DER=12.77, miss=4.12, falarm=5.81, confusion=2.84
#Eval for threshold 0.6 DER=13.35, miss=5.83, falarm=5.29, confusion=2.23
#Eval for threshold 0.7 DER=14.54, miss=8.61, falarm=4.34, confusion=1.59
#Eval for threshold 0.8 DER=16.69, miss=12.13, falarm=3.36, confusion=1.19

fi

# compared with stage84-85, stage96-97will increase rs_len from 4 to 8.
if [ ${stage} -le 96 ] && [ ${stop_stage} -ge 96 ];then
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
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_rs_len8
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=8
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
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

if [ ${stage} -le 97 ] && [ ${stop_stage} -ge 97 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_rs_len8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer
  done
 done

#grep -r Eval logs/run_ts_vad2_stage97_mamba2_rs_len8.log
# collar=0.0
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=15.13, miss=0.25, falarm=11.83, confusion=3.05
#Eval for threshold 0.3 DER=12.97, miss=0.51, falarm=8.90, confusion=3.56
#Eval for threshold 0.35 DER=12.41, miss=0.71, falarm=8.01, confusion=3.69
#Eval for threshold 0.4 DER=12.03, miss=0.97, falarm=7.36, confusion=3.71
#Eval for threshold 0.45 DER=11.71, miss=1.27, falarm=6.73, confusion=3.71
#Eval for threshold 0.5 DER=11.55, miss=1.63, falarm=6.25, confusion=3.67
#Eval for threshold 0.55 DER=11.46, miss=2.11, falarm=5.78, confusion=3.57
#Eval for threshold 0.6 DER=11.47, miss=2.63, falarm=5.36, confusion=3.48
#Eval for threshold 0.7 DER=11.87, miss=4.22, falarm=4.49, confusion=3.17
#Eval for threshold 0.8 DER=13.33, miss=7.14, falarm=3.58, confusion=2.62

# test of magicdata-ramc
#Eval for threshold 0.2 DER=17.55, miss=0.40, falarm=15.36, confusion=1.79
#Eval for threshold 0.3 DER=15.44, miss=0.86, falarm=12.45, confusion=2.13
#Eval for threshold 0.35 DER=14.73, miss=1.19, falarm=11.32, confusion=2.23
#Eval for threshold 0.4 DER=14.18, miss=1.62, falarm=10.25, confusion=2.31
#Eval for threshold 0.45 DER=13.48, miss=2.19, falarm=8.65, confusion=2.65
#Eval for threshold 0.5 DER=12.63, miss=3.24, falarm=6.24, confusion=3.15
#Eval for threshold 0.55 DER=13.44, miss=5.53, falarm=5.59, confusion=2.32
#Eval for threshold 0.6 DER=13.91, miss=6.87, falarm=5.13, confusion=1.91
#Eval for threshold 0.7 DER=15.03, miss=9.21, falarm=4.27, confusion=1.55
#Eval for threshold 0.8 DER=17.12, miss=12.56, falarm=3.37, confusion=1.20


# collar=0.25
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=6.82, miss=0.06, falarm=4.00, confusion=2.76
#Eval for threshold 0.3 DER=5.56, miss=0.14, falarm=2.23, confusion=3.19
#Eval for threshold 0.35 DER=5.31, miss=0.22, falarm=1.80, confusion=3.29
#Eval for threshold 0.4 DER=5.18, miss=0.30, falarm=1.56, confusion=3.31
#Eval for threshold 0.45 DER=5.12, miss=0.41, falarm=1.40, confusion=3.31
#Eval for threshold 0.5 DER=5.11, miss=0.54, falarm=1.27, confusion=3.30
#Eval for threshold 0.55 DER=5.16, miss=0.73, falarm=1.17, confusion=3.26
#Eval for threshold 0.6 DER=5.28, miss=0.96, falarm=1.11, confusion=3.21
#Eval for threshold 0.7 DER=5.79, miss=1.76, falarm=0.99, confusion=3.03
#Eval for threshold 0.8 DER=7.16, miss=3.70, falarm=0.86, confusion=2.59

# test of magicdata-ramc
#Eval for threshold 0.2 DER=9.22, miss=0.12, falarm=7.78, confusion=1.32
#Eval for threshold 0.3 DER=8.08, miss=0.32, falarm=6.24, confusion=1.53
#Eval for threshold 0.35 DER=7.74, miss=0.45, falarm=5.70, confusion=1.58
#Eval for threshold 0.4 DER=7.52, miss=0.66, falarm=5.22, confusion=1.64
#Eval for threshold 0.45 DER=7.12, miss=0.95, falarm=4.18, confusion=1.98
#Eval for threshold 0.5 DER=6.41, miss=1.64, falarm=2.17, confusion=2.60
#Eval for threshold 0.55 DER=7.35, miss=3.61, falarm=1.93, confusion=1.81
#Eval for threshold 0.6 DER=7.86, miss=4.57, falarm=1.81, confusion=1.47
#Eval for threshold 0.7 DER=8.90, miss=6.05, falarm=1.60, confusion=1.26
#Eval for threshold 0.8 DER=10.71, miss=8.33, falarm=1.34, confusion=1.04
fi


# compared with stage84-85, stage98-99will increase rs_len from 4 to 10.
if [ ${stage} -le 98 ] && [ ${stop_stage} -ge 98 ];then
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
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_rs_len10
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=10
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 16815 \
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

if [ ${stage} -le 99 ] && [ ${stop_stage} -ge 99 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_rs_len10
 model_file=$exp_dir/best-valid-der.pt
 rs_len=10
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

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
# grep -r Eval logs/run_ts_vad2_stage98-99_mamba2_rs_len10.log
# collar=0.25
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=6.26, miss=0.06, falarm=3.03, confusion=3.18
#Eval for threshold 0.3 DER=5.46, miss=0.15, falarm=2.03, confusion=3.28
#Eval for threshold 0.35 DER=5.32, miss=0.22, falarm=1.81, confusion=3.29
#Eval for threshold 0.4 DER=5.21, miss=0.31, falarm=1.61, confusion=3.30
#Eval for threshold 0.45 DER=5.15, miss=0.41, falarm=1.43, confusion=3.30
#Eval for threshold 0.5 DER=5.14, miss=0.54, falarm=1.31, confusion=3.29
#Eval for threshold 0.55 DER=5.21, miss=0.73, falarm=1.22, confusion=3.26
#Eval for threshold 0.6 DER=5.29, miss=0.93, falarm=1.14, confusion=3.22
#Eval for threshold 0.7 DER=5.65, miss=1.50, falarm=1.02, confusion=3.14
#Eval for threshold 0.8 DER=6.63, miss=2.76, falarm=0.89, confusion=2.98

# test of magicdata-ramc
#Eval for threshold 0.2 DER=9.02, miss=0.10, falarm=7.58, confusion=1.34
#Eval for threshold 0.3 DER=7.96, miss=0.27, falarm=6.17, confusion=1.52
#Eval for threshold 0.35 DER=7.67, miss=0.38, falarm=5.70, confusion=1.59
#Eval for threshold 0.4 DER=7.44, miss=0.54, falarm=5.24, confusion=1.66
#Eval for threshold 0.45 DER=7.30, miss=0.78, falarm=4.81, confusion=1.72
#Eval for threshold 0.5 DER=6.42, miss=1.33, falarm=2.76, confusion=2.32
#Eval for threshold 0.55 DER=7.39, miss=3.81, falarm=1.95, confusion=1.63
#Eval for threshold 0.6 DER=7.69, miss=4.37, falarm=1.84, confusion=1.48
#Eval for threshold 0.7 DER=8.57, miss=5.66, falarm=1.62, confusion=1.29
#Eval for threshold 0.8 DER=10.31, miss=7.86, falarm=1.36, confusion=1.09

#grep -r Eval logs/run_ts_vad2_stage99_mamba2_rs_len10.log
# collar=0.0
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=14.50, miss=0.26, falarm=10.81, confusion=3.43
#Eval for threshold 0.3 DER=12.90, miss=0.52, falarm=8.74, confusion=3.63
#Eval for threshold 0.35 DER=12.43, miss=0.71, falarm=8.05, confusion=3.68
#Eval for threshold 0.4 DER=12.07, miss=0.93, falarm=7.45, confusion=3.69
#Eval for threshold 0.45 DER=11.78, miss=1.22, falarm=6.86, confusion=3.70
#Eval for threshold 0.5 DER=11.59, miss=1.56, falarm=6.37, confusion=3.66
#Eval for threshold 0.55 DER=11.53, miss=2.02, falarm=5.92, confusion=3.58
#Eval for threshold 0.6 DER=11.51, miss=2.53, falarm=5.50, confusion=3.49
#Eval for threshold 0.7 DER=11.78, miss=3.83, falarm=4.66, confusion=3.28
#Eval for threshold 0.8 DER=12.80, miss=6.03, falarm=3.77, confusion=2.99

# test of magicdata-ramc
#Eval for threshold 0.2 DER=17.34, miss=0.37, falarm=15.15, confusion=1.81
#Eval for threshold 0.3 DER=15.37, miss=0.80, falarm=12.44, confusion=2.13
#Eval for threshold 0.35 DER=14.72, miss=1.09, falarm=11.39, confusion=2.24
#Eval for threshold 0.4 DER=14.16, miss=1.46, falarm=10.37, confusion=2.33
#Eval for threshold 0.45 DER=13.74, miss=1.97, falarm=9.39, confusion=2.39
#Eval for threshold 0.5 DER=12.67, miss=2.89, falarm=6.85, confusion=2.92
#Eval for threshold 0.55 DER=13.47, miss=5.59, falarm=5.70, confusion=2.18
#Eval for threshold 0.6 DER=13.74, miss=6.53, falarm=5.26, confusion=1.96
#Eval for threshold 0.7 DER=14.72, miss=8.70, falarm=4.41, confusion=1.61
#Eval for threshold 0.8 DER=16.66, miss=11.90, falarm=3.50, confusion=1.27


fi


# compared with stage94-95, stage108-109will increase rs_len from 4 to 6
if [ ${stage} -le 108 ] && [ ${stop_stage} -ge 108 ];then
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
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state128_rs_len6
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=6
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
   d_state=128
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
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
fi

if [ ${stage} -le 109 ] && [ ${stop_stage} -ge 109 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state128_rs_len6
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=128
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

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
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
done
#grep -r Eval logs/run_ts_vad2_stage108-109_mamba2_rs_len6_d_state128.log
# collar=0.25
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=6.39, miss=0.06, falarm=3.40, confusion=2.93
#Eval for threshold 0.3 DER=5.44, miss=0.13, falarm=2.08, confusion=3.23
#Eval for threshold 0.35 DER=5.24, miss=0.20, falarm=1.76, confusion=3.29
#Eval for threshold 0.4 DER=5.16, miss=0.28, falarm=1.58, confusion=3.30
#Eval for threshold 0.45 DER=5.09, miss=0.38, falarm=1.41, confusion=3.30
#Eval for threshold 0.5 DER=5.05, miss=0.50, falarm=1.26, confusion=3.29
#Eval for threshold 0.55 DER=5.12, miss=0.66, falarm=1.20, confusion=3.26
#Eval for threshold 0.6 DER=5.24, miss=0.90, falarm=1.14, confusion=3.21
#Eval for threshold 0.7 DER=5.59, miss=1.48, falarm=1.01, confusion=3.10
#Eval for threshold 0.8 DER=6.64, miss=2.96, falarm=0.87, confusion=2.80

# test of magicdata-ramc
#Eval for threshold 0.2 DER=8.83, miss=0.11, falarm=7.35, confusion=1.38
#Eval for threshold 0.3 DER=7.82, miss=0.29, falarm=5.96, confusion=1.57
#Eval for threshold 0.35 DER=7.50, miss=0.43, falarm=5.47, confusion=1.59
#Eval for threshold 0.4 DER=7.30, miss=0.60, falarm=5.06, confusion=1.64
#Eval for threshold 0.45 DER=6.52, miss=0.83, falarm=3.71, confusion=1.98
#Eval for threshold 0.5 DER=5.91, miss=1.26, falarm=2.17, confusion=2.49
#Eval for threshold 0.55 DER=6.89, miss=3.22, falarm=1.96, confusion=1.71
#Eval for threshold 0.6 DER=7.63, miss=4.34, falarm=1.83, confusion=1.47
#Eval for threshold 0.7 DER=8.54, miss=5.61, falarm=1.61, confusion=1.32
#Eval for threshold 0.8 DER=10.33, miss=7.86, falarm=1.35, confusion=1.12


#grep -r Eval logs/run_ts_vad2_stage109_mamba2_rs_len6_d_state128_c0.0.log
# collar=0.0
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=14.75, miss=0.26, falarm=11.29, confusion=3.20
#Eval for threshold 0.3 DER=12.93, miss=0.48, falarm=8.87, confusion=3.58
#Eval for threshold 0.35 DER=12.41, miss=0.66, falarm=8.09, confusion=3.66
#Eval for threshold 0.4 DER=12.04, miss=0.88, falarm=7.47, confusion=3.69
#Eval for threshold 0.45 DER=11.73, miss=1.16, falarm=6.88, confusion=3.69
#Eval for threshold 0.5 DER=11.51, miss=1.50, falarm=6.35, confusion=3.66
#Eval for threshold 0.55 DER=11.43, miss=1.94, falarm=5.92, confusion=3.57
#Eval for threshold 0.6 DER=11.45, miss=2.51, falarm=5.50, confusion=3.45
#Eval for threshold 0.7 DER=11.64, miss=3.84, falarm=4.59, confusion=3.21
#Eval for threshold 0.8 DER=12.80, miss=6.35, falarm=3.66, confusion=2.79

# test of magicdata-ramc
#Eval for threshold 0.2 DER=17.29, miss=0.38, falarm=15.05, confusion=1.86
#Eval for threshold 0.3 DER=15.32, miss=0.81, falarm=12.31, confusion=2.19
#Eval for threshold 0.35 DER=14.63, miss=1.12, falarm=11.24, confusion=2.27
#Eval for threshold 0.4 DER=14.09, miss=1.49, falarm=10.27, confusion=2.33
#Eval for threshold 0.45 DER=13.00, miss=1.95, falarm=8.39, confusion=2.66
#Eval for threshold 0.5 DER=12.24, miss=2.74, falarm=6.43, confusion=3.08
#Eval for threshold 0.55 DER=13.05, miss=5.02, falarm=5.79, confusion=2.25
#Eval for threshold 0.6 DER=13.73, miss=6.48, falarm=5.31, confusion=1.94
#Eval for threshold 0.7 DER=14.69, miss=8.65, falarm=4.40, confusion=1.64
#Eval for threshold 0.8 DER=16.73, miss=12.02, falarm=3.44, confusion=1.27
fi


# compared with stage90-91, stage110-111 will use single_backend_type="transformer"
if [ ${stage} -le 110 ] && [ ${stop_stage} -ge 110 ];then
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
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=6
   segment_shift=2
   single_backend_type="transformer"
   multi_backend_type="transformer"
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

if [ ${stage} -le 111 ] && [ ${stop_stage} -ge 111 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6
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
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer
  done
 done
#grep -r Eval logs/run_ts_vad2_stage111_transformer_rs_len6.log
# collar=0.0
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=14.81, miss=0.25, falarm=11.32, confusion=3.23
#Eval for threshold 0.3 DER=13.02, miss=0.46, falarm=8.93, confusion=3.63
#Eval for threshold 0.35 DER=12.40, miss=0.63, falarm=8.04, confusion=3.73
#Eval for threshold 0.4 DER=11.97, miss=0.86, falarm=7.34, confusion=3.77
#Eval for threshold 0.45 DER=11.63, miss=1.13, falarm=6.73, confusion=3.77
#Eval for threshold 0.5 DER=11.42, miss=1.49, falarm=6.21, confusion=3.72
#Eval for threshold 0.55 DER=11.33, miss=1.93, falarm=5.76, confusion=3.63
#Eval for threshold 0.6 DER=11.36, miss=2.49, falarm=5.37, confusion=3.51
#Eval for threshold 0.7 DER=11.71, miss=3.88, falarm=4.56, confusion=3.26
#Eval for threshold 0.8 DER=12.71, miss=6.16, falarm=3.68, confusion=2.88

# test of magicata-ramc
#Eval for threshold 0.2 DER=17.33, miss=0.31, falarm=15.17, confusion=1.84
#Eval for threshold 0.3 DER=15.38, miss=0.69, falarm=12.53, confusion=2.16
#Eval for threshold 0.35 DER=14.70, miss=0.96, falarm=11.46, confusion=2.27
#Eval for threshold 0.4 DER=14.17, miss=1.29, falarm=10.51, confusion=2.37
#Eval for threshold 0.45 DER=13.72, miss=1.71, falarm=9.57, confusion=2.44
#Eval for threshold 0.5 DER=12.49, miss=2.37, falarm=6.87, confusion=3.25
#Eval for threshold 0.55 DER=13.28, miss=5.28, falarm=5.73, confusion=2.27
#Eval for threshold 0.6 DER=13.49, miss=6.12, falarm=5.30, confusion=2.07
#Eval for threshold 0.7 DER=14.31, miss=8.10, falarm=4.48, confusion=1.73
#Eval for threshold 0.8 DER=15.93, miss=10.95, falarm=3.61, confusion=1.36

# collar=0.25
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=6.56, miss=0.06, falarm=3.56, confusion=2.94
#Eval for threshold 0.3 DER=5.65, miss=0.11, falarm=2.28, confusion=3.26
#Eval for threshold 0.35 DER=5.42, miss=0.16, falarm=1.92, confusion=3.34
#Eval for threshold 0.4 DER=5.27, miss=0.23, falarm=1.68, confusion=3.36
#Eval for threshold 0.45 DER=5.15, miss=0.33, falarm=1.47, confusion=3.36
#Eval for threshold 0.5 DER=5.11, miss=0.44, falarm=1.32, confusion=3.35
#Eval for threshold 0.55 DER=5.15, miss=0.61, falarm=1.23, confusion=3.31
#Eval for threshold 0.6 DER=5.27, miss=0.85, falarm=1.17, confusion=3.25
#Eval for threshold 0.7 DER=5.62, miss=1.45, falarm=1.04, confusion=3.14
#Eval for threshold 0.8 DER=6.52, miss=2.74, falarm=0.92, confusion=2.87

# test of magicdata-ramc
#Eval for threshold 0.2 DER=9.05, miss=0.07, falarm=7.62, confusion=1.37
#Eval for threshold 0.3 DER=8.05, miss=0.20, falarm=6.29, confusion=1.56
#Eval for threshold 0.35 DER=7.74, miss=0.30, falarm=5.79, confusion=1.65
#Eval for threshold 0.4 DER=7.53, miss=0.43, falarm=5.38, confusion=1.72
#Eval for threshold 0.45 DER=7.36, miss=0.61, falarm=4.95, confusion=1.80
#Eval for threshold 0.5 DER=6.40, miss=0.93, falarm=2.82, confusion=2.64
#Eval for threshold 0.55 DER=7.36, miss=3.55, falarm=2.08, confusion=1.73
#Eval for threshold 0.6 DER=7.57, miss=3.99, falarm=1.97, confusion=1.60
#Eval for threshold 0.7 DER=8.25, miss=5.09, falarm=1.76, confusion=1.41
#Eval for threshold 0.8 DER=9.53, miss=6.84, falarm=1.50, confusion=1.18
fi




# compared with stage96-97, stage112-113 will use single_backend_type="transformer"
if [ ${stage} -le 112 ] && [ ${stop_stage} -ge 112 ];then
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
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len8
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=8
   segment_shift=2
   single_backend_type="transformer"
   multi_backend_type="transformer"
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 14715 \
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

if [ ${stage} -le 113 ] && [ ${stop_stage} -ge 113 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer
  done
 done
#grep -r Eval logs/run_ts_vad2_stage113_transformer_rs_len8.log
# collar=0.0
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=14.90, miss=0.23, falarm=11.54, confusion=3.12
#Eval for threshold 0.3 DER=13.05, miss=0.42, falarm=9.03, confusion=3.60
#Eval for threshold 0.35 DER=12.43, miss=0.58, falarm=8.12, confusion=3.72
#Eval for threshold 0.4 DER=12.00, miss=0.80, falarm=7.41, confusion=3.78
#Eval for threshold 0.45 DER=11.70, miss=1.10, falarm=6.80, confusion=3.81
#Eval for threshold 0.5 DER=11.44, miss=1.44, falarm=6.23, confusion=3.77
#Eval for threshold 0.55 DER=11.38, miss=1.93, falarm=5.82, confusion=3.64
#Eval for threshold 0.6 DER=11.40, miss=2.48, falarm=5.44, confusion=3.48
#Eval for threshold 0.7 DER=11.76, miss=3.91, falarm=4.66, confusion=3.19
#Eval for threshold 0.8 DER=12.80, miss=6.19, falarm=3.87, confusion=2.75

#test of  magicdata-ramc
#Eval for threshold 0.2 DER=16.98, miss=0.28, falarm=14.77, confusion=1.93
#Eval for threshold 0.3 DER=15.14, miss=0.61, falarm=12.30, confusion=2.23
#Eval for threshold 0.35 DER=14.49, miss=0.84, falarm=11.31, confusion=2.34
#Eval for threshold 0.4 DER=13.96, miss=1.14, falarm=10.37, confusion=2.45
#Eval for threshold 0.45 DER=13.02, miss=1.53, falarm=8.58, confusion=2.92
#Eval for threshold 0.5 DER=12.20, miss=2.35, falarm=6.45, confusion=3.40
#Eval for threshold 0.55 DER=13.00, miss=4.77, falarm=5.89, confusion=2.34
#Eval for threshold 0.6 DER=13.32, miss=5.81, falarm=5.48, confusion=2.03
#Eval for threshold 0.7 DER=13.93, miss=7.54, falarm=4.66, confusion=1.73
#Eval for threshold 0.8 DER=15.26, miss=10.01, falarm=3.84, confusion=1.41

# collar=0.25
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=6.66, miss=0.05, falarm=3.80, confusion=2.81
#Eval for threshold 0.3 DER=5.66, miss=0.09, falarm=2.37, confusion=3.20
#Eval for threshold 0.35 DER=5.42, miss=0.14, falarm=2.00, confusion=3.28
#Eval for threshold 0.4 DER=5.29, miss=0.21, falarm=1.74, confusion=3.33
#Eval for threshold 0.45 DER=5.21, miss=0.33, falarm=1.51, confusion=3.38
#Eval for threshold 0.5 DER=5.14, miss=0.45, falarm=1.31, confusion=3.37
#Eval for threshold 0.55 DER=5.20, miss=0.67, falarm=1.23, confusion=3.30
#Eval for threshold 0.6 DER=5.30, miss=0.92, falarm=1.16, confusion=3.21
#Eval for threshold 0.7 DER=5.68, miss=1.61, falarm=1.05, confusion=3.03
#Eval for threshold 0.8 DER=6.66, miss=3.04, falarm=0.94, confusion=2.68

# test of magicdata-ramc
#Eval for threshold 0.2 DER=8.67, miss=0.05, falarm=7.19, confusion=1.43
#Eval for threshold 0.3 DER=7.80, miss=0.16, falarm=6.02, confusion=1.62
#Eval for threshold 0.35 DER=7.51, miss=0.26, falarm=5.58, confusion=1.68
#Eval for threshold 0.4 DER=7.33, miss=0.40, falarm=5.18, confusion=1.75
#Eval for threshold 0.45 DER=6.73, miss=0.56, falarm=3.97, confusion=2.20
#Eval for threshold 0.5 DER=6.11, miss=1.07, falarm=2.27, confusion=2.77
#Eval for threshold 0.55 DER=7.04, miss=3.15, falarm=2.11, confusion=1.78
#Eval for threshold 0.6 DER=7.38, miss=3.83, falarm=2.01, confusion=1.54
#Eval for threshold 0.7 DER=7.92, miss=4.75, falarm=1.78, confusion=1.38
#Eval for threshold 0.8 DER=8.98, miss=6.22, falarm=1.57, confusion=1.19

fi

# compared with stage108-109, stage114-115will use w2v-bert2 to replace cam++ speech encoder of tsvad.
if [ ${stage} -le 114 ] && [ ${stop_stage} -ge 114 ];then
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
    #speech_encoder_type="CAM++"
    #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch40_w2v-bert2_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state128_rs_len6

    #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=6
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
   d_state=128
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17815 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 5e-5\
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
fi

if [ ${stage} -le 115 ] && [ ${stop_stage} -ge 115 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch40_w2v-bert2_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state128_rs_len6
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=128
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 #speech_encoder_type="CAM++"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
#
maduo add it after running.
spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
 done
done
fi
#  grep -r Eval logs/run_ts_vad2_stage114-115_w2v_bert2_mamba2_d_state128_rs_len6_5e-5.log
#  dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=15.39, miss=0.22, falarm=12.32, confusion=2.86
#Eval for threshold 0.3 DER=12.94, miss=0.44, falarm=8.96, confusion=3.54
#Eval for threshold 0.35 DER=12.25, miss=0.67, falarm=7.98, confusion=3.61
#Eval for threshold 0.4 DER=11.82, miss=0.93, falarm=7.25, confusion=3.63
#Eval for threshold 0.45 DER=11.50, miss=1.31, falarm=6.59, confusion=3.59
#Eval for threshold 0.5 DER=11.32, miss=1.75, falarm=6.06, confusion=3.51
#Eval for threshold 0.55 DER=11.29, miss=2.34, falarm=5.57, confusion=3.38
#Eval for threshold 0.6 DER=11.39, miss=3.04, falarm=5.10, confusion=3.25
#Eval for threshold 0.7 DER=12.23, miss=5.22, falarm=4.14, confusion=2.87
#Eval for threshold 0.8 DER=14.43, miss=9.34, falarm=3.09, confusion=2.00

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=19.35, miss=0.25, falarm=17.29, confusion=1.80
#Eval for threshold 0.3 DER=16.38, miss=0.61, falarm=13.43, confusion=2.34
#Eval for threshold 0.35 DER=15.36, miss=0.94, falarm=11.88, confusion=2.54
#Eval for threshold 0.4 DER=14.54, miss=1.40, falarm=10.40, confusion=2.74
#Eval for threshold 0.45 DER=13.23, miss=2.10, falarm=7.78, confusion=3.35
#Eval for threshold 0.5 DER=13.18, miss=3.99, falarm=6.00, confusion=3.19
#Eval for threshold 0.55 DER=14.10, miss=6.19, falarm=5.41, confusion=2.49
#Eval for threshold 0.6 DER=14.70, miss=7.66, falarm=4.88, confusion=2.16
#Eval for threshold 0.7 DER=16.64, miss=11.13, falarm=3.85, confusion=1.65
#Eval for threshold 0.8 DER=20.44, miss=16.48, falarm=2.82, confusion=1.14

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=6.87, miss=0.04, falarm=4.20, confusion=2.62
#Eval for threshold 0.3 DER=5.50, miss=0.08, falarm=2.21, confusion=3.20
#Eval for threshold 0.35 DER=5.21, miss=0.15, falarm=1.80, confusion=3.26
#Eval for threshold 0.4 DER=5.07, miss=0.23, falarm=1.57, confusion=3.27
#Eval for threshold 0.45 DER=5.01, miss=0.36, falarm=1.41, confusion=3.24
#Eval for threshold 0.5 DER=5.01, miss=0.51, falarm=1.30, confusion=3.21
#Eval for threshold 0.55 DER=5.06, miss=0.71, falarm=1.18, confusion=3.17
#Eval for threshold 0.6 DER=5.22, miss=0.99, falarm=1.11, confusion=3.11
#Eval for threshold 0.7 DER=6.04, miss=2.21, falarm=0.97, confusion=2.86
#Eval for threshold 0.8 DER=8.04, miss=5.16, falarm=0.82, confusion=2.06

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=10.72, miss=0.04, falarm=9.30, confusion=1.38
#Eval for threshold 0.3 DER=8.92, miss=0.15, falarm=7.03, confusion=1.74
#Eval for threshold 0.35 DER=8.37, miss=0.29, falarm=6.18, confusion=1.90
#Eval for threshold 0.4 DER=7.98, miss=0.47, falarm=5.42, confusion=2.08
#Eval for threshold 0.45 DER=6.95, miss=0.79, falarm=3.42, confusion=2.74
#Eval for threshold 0.5 DER=7.01, miss=2.22, falarm=2.08, confusion=2.70
#Eval for threshold 0.55 DER=8.04, miss=4.07, falarm=1.92, confusion=2.06
#Eval for threshold 0.6 DER=8.67, miss=5.09, falarm=1.78, confusion=1.80
#Eval for threshold 0.7 DER=10.44, miss=7.49, falarm=1.50, confusion=1.44
#Eval for threshold 0.8 DER=13.86, miss=11.60, falarm=1.20, confusion=1.06

if [ ${stage} -le 120 ] && [ ${stop_stage} -ge 120 ];then
   echo "generate oracle vad speaker embedding"
   dest_dir=/mntcephfs/lab_data/maduo/model_hub
   #cam++200k
   #feature_name=cam++_zh-cn_200k_feature_dir
   #model_id=iic/speech_campplus_sv_zh-cn_16k-common
   # cam++_voxceleb and cnceleb
   feature_name=cam++_en_zh_advanced_feature_dir
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced

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
     input_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc/${name}/target_audio/
     wav_path=/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc/${name}/wavs.txt
     find $input_dir -name "*.wav" | grep -v "all.wav" >$wav_path
     head $wav_path
     save_dir=$dest_dir/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding/$name/$feature_name
     python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size 96
   done
fi


# compared with  stage114-115, stage121-122 will use w2v-bert2 to replace cam++ voceleb and cnceleb speaker model.
if [ ${stage} -le 121 ] && [ ${stop_stage} -ge 121 ];then
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
    #speech_encoder_type="CAM++"
    #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_en_zh_advanced_epoch40_w2v-bert2_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state128_rs_len6

    data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=6
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
   d_state=128
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 18815 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 5e-5\
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
fi

if [ ${stage} -le 122 ] && [ ${stop_stage} -ge 122 ];then
 rs_len=6
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=128
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 #speech_encoder_type="CAM++"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory

 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_en_zh_advanced_epoch40_w2v-bert2_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state128_rs_len6
 model_file=$exp_dir/best-valid-der.pt

 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
 done
done
fi
#grep -r 'Eval' logs/run_ts_vad2_stage121-122_w2v_bert2_mamba2_d_state128_rs_len6_cam++_advance_5e-5.log
# collar=0.0
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=16.42, miss=0.27, falarm=13.99, confusion=2.16
#Eval for threshold 0.3 DER=12.97, miss=0.53, falarm=9.29, confusion=3.15
#Eval for threshold 0.35 DER=12.06, miss=0.74, falarm=7.80, confusion=3.52
#Eval for threshold 0.4 DER=11.63, miss=1.00, falarm=7.01, confusion=3.63
#Eval for threshold 0.45 DER=11.37, miss=1.35, falarm=6.42, confusion=3.61
#Eval for threshold 0.5 DER=11.23, miss=1.80, falarm=5.89, confusion=3.53
#Eval for threshold 0.55 DER=11.15, miss=2.36, falarm=5.37, confusion=3.43
#Eval for threshold 0.6 DER=11.20, miss=3.09, falarm=4.81, confusion=3.29
#Eval for threshold 0.7 DER=12.35, miss=5.98, falarm=3.85, confusion=2.52
#Eval for threshold 0.8 DER=15.59, miss=11.29, falarm=2.86, confusion=1.44

# test of magicdata-ramc
#Eval for threshold 0.2 DER=17.54, miss=0.32, falarm=15.52, confusion=1.70
#Eval for threshold 0.3 DER=15.15, miss=0.71, falarm=12.30, confusion=2.15
#Eval for threshold 0.35 DER=14.33, miss=1.03, falarm=11.01, confusion=2.30
#Eval for threshold 0.4 DER=13.65, miss=1.49, falarm=9.68, confusion=2.48
#Eval for threshold 0.45 DER=12.74, miss=2.18, falarm=7.65, confusion=2.91
#Eval for threshold 0.5 DER=12.62, miss=4.10, falarm=5.77, confusion=2.75
#Eval for threshold 0.55 DER=13.26, miss=5.97, falarm=5.17, confusion=2.13
#Eval for threshold 0.6 DER=13.82, miss=7.37, falarm=4.64, confusion=1.81
#Eval for threshold 0.7 DER=15.60, miss=10.57, falarm=3.61, confusion=1.43
#Eval for threshold 0.8 DER=19.32, miss=15.71, falarm=2.62, confusion=1.00

# collar=0.25
# dev of magicdata-ramc
#Eval for threshold 0.2 DER=8.27, miss=0.05, falarm=6.37, confusion=1.85
#Eval for threshold 0.3 DER=5.68, miss=0.09, falarm=2.79, confusion=2.80
#Eval for threshold 0.35 DER=5.13, miss=0.15, falarm=1.81, confusion=3.17
#Eval for threshold 0.4 DER=4.99, miss=0.23, falarm=1.49, confusion=3.27
#Eval for threshold 0.45 DER=4.96, miss=0.34, falarm=1.37, confusion=3.25
#Eval for threshold 0.5 DER=4.97, miss=0.49, falarm=1.26, confusion=3.22
#Eval for threshold 0.55 DER=5.03, miss=0.66, falarm=1.18, confusion=3.19
#Eval for threshold 0.6 DER=5.15, miss=0.94, falarm=1.09, confusion=3.13
#Eval for threshold 0.7 DER=6.26, miss=2.86, falarm=0.94, confusion=2.46
#Eval for threshold 0.8 DER=9.43, miss=7.24, falarm=0.79, confusion=1.40

# test of magicdata-ramc
#Eval for threshold 0.2 DER=9.09, miss=0.04, falarm=7.76, confusion=1.29
#Eval for threshold 0.3 DER=7.83, miss=0.14, falarm=6.10, confusion=1.58
#Eval for threshold 0.35 DER=7.44, miss=0.25, falarm=5.51, confusion=1.67
#Eval for threshold 0.4 DER=7.14, miss=0.43, falarm=4.89, confusion=1.81
#Eval for threshold 0.45 DER=6.51, miss=0.74, falarm=3.50, confusion=2.26
#Eval for threshold 0.5 DER=6.53, miss=2.27, falarm=2.06, confusion=2.21
#Eval for threshold 0.55 DER=7.26, miss=3.71, falarm=1.90, confusion=1.64
#Eval for threshold 0.6 DER=7.76, miss=4.58, falarm=1.75, confusion=1.42
#Eval for threshold 0.7 DER=9.20, miss=6.53, falarm=1.46, confusion=1.21
#Eval for threshold 0.8 DER=12.41, miss=10.33, falarm=1.15, confusion=0.93



# compared with stage108-109, stage125-126 is same. I am worried that rttm may contain G0000000, so I want to run it
if [ ${stage} -le 125 ] && [ ${stop_stage} -ge 125 ];then
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
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
   exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state128_rs_len6_again
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=6
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
   d_state=128
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15815 \
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
    --d-state $d_state
fi

if [ ${stage} -le 126 ] && [ ${stop_stage} -ge 126 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state128_rs_len6_again
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=128
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
 done
done
fi

# compared with stage125-126, stage127-128 d_state will increase from 128 to 256
if [ ${stage} -le 127 ] && [ ${stop_stage} -ge 127 ];then
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
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
   exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len6
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=6
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
   d_state=256
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15515 \
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
    --d-state $d_state
fi

if [ ${stage} -le 128 ] && [ ${stop_stage} -ge 128 ];then
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len6_again
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len6/
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 #collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
 done
done
fi
#grep -r Eval logs/run_ts_vad2_stage128_mamba2_d_state256_rs_len6.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=15.14, miss=0.29, falarm=11.80, confusion=3.05
#Eval for threshold 0.3 DER=13.13, miss=0.56, falarm=9.08, confusion=3.50
#Eval for threshold 0.35 DER=12.52, miss=0.75, falarm=8.15, confusion=3.62
#Eval for threshold 0.4 DER=12.07, miss=0.99, falarm=7.37, confusion=3.71
#Eval for threshold 0.45 DER=11.78, miss=1.31, falarm=6.75, confusion=3.72
#Eval for threshold 0.5 DER=11.59, miss=1.71, falarm=6.19, confusion=3.69
#Eval for threshold 0.55 DER=11.50, miss=2.23, falarm=5.71, confusion=3.57
#Eval for threshold 0.6 DER=11.53, miss=2.85, falarm=5.25, confusion=3.43
#Eval for threshold 0.7 DER=11.98, miss=4.51, falarm=4.38, confusion=3.08
#Eval for threshold 0.8 DER=13.44, miss=7.37, falarm=3.46, confusion=2.60

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=17.28, miss=0.41, falarm=15.03, confusion=1.84
#Eval for threshold 0.3 DER=15.29, miss=0.85, falarm=12.29, confusion=2.14
#Eval for threshold 0.35 DER=14.67, miss=1.18, falarm=11.25, confusion=2.25
#Eval for threshold 0.4 DER=14.19, miss=1.57, falarm=10.30, confusion=2.32
#Eval for threshold 0.45 DER=13.72, miss=2.10, falarm=9.24, confusion=2.38
#Eval for threshold 0.5 DER=12.47, miss=3.10, falarm=6.41, confusion=2.97
#Eval for threshold 0.55 DER=13.49, miss=5.72, falarm=5.62, confusion=2.15
#Eval for threshold 0.6 DER=13.74, miss=6.62, falarm=5.17, confusion=1.95
#Eval for threshold 0.7 DER=14.63, miss=8.79, falarm=4.24, confusion=1.61
#Eval for threshold 0.8 DER=16.66, miss=12.04, falarm=3.32, confusion=1.31
#
#
# grep -r Eval logs/run_ts_vad2_stage128_mamba2_d_state256_rs_len6_collar0.25_1.log
#
# # dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=6.86, miss=0.07, falarm=4.04, confusion=2.76
#Eval for threshold 0.3 DER=5.66, miss=0.16, falarm=2.38, confusion=3.12
#Eval for threshold 0.35 DER=5.39, miss=0.22, falarm=1.96, confusion=3.21
#Eval for threshold 0.4 DER=5.20, miss=0.29, falarm=1.62, confusion=3.30
#Eval for threshold 0.45 DER=5.14, miss=0.40, falarm=1.42, confusion=3.33
#Eval for threshold 0.5 DER=5.14, miss=0.55, falarm=1.26, confusion=3.34
#Eval for threshold 0.55 DER=5.20, miss=0.76, falarm=1.16, confusion=3.27
#Eval for threshold 0.6 DER=5.34, miss=1.05, falarm=1.09, confusion=3.20
#Eval for threshold 0.7 DER=5.86, miss=1.92, falarm=0.97, confusion=2.97
#Eval for threshold 0.8 DER=7.23, miss=3.78, falarm=0.85, confusion=2.60
#
## test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=8.90, miss=0.11, falarm=7.42, confusion=1.37
#Eval for threshold 0.3 DER=7.89, miss=0.30, falarm=6.06, confusion=1.53
#Eval for threshold 0.35 DER=7.62, miss=0.43, falarm=5.59, confusion=1.60
#Eval for threshold 0.4 DER=7.46, miss=0.61, falarm=5.20, confusion=1.64
#Eval for threshold 0.45 DER=7.30, miss=0.85, falarm=4.74, confusion=1.71
#Eval for threshold 0.5 DER=6.20, miss=1.46, falarm=2.36, confusion=2.39
#Eval for threshold 0.55 DER=7.39, miss=3.83, falarm=1.95, confusion=1.61
#Eval for threshold 0.6 DER=7.66, miss=4.33, falarm=1.84, confusion=1.49
#Eval for threshold 0.7 DER=8.46, miss=5.55, falarm=1.60, confusion=1.31
#Eval for threshold 0.8 DER=10.13, miss=7.62, falarm=1.36, confusion=1.14



if [ ${stage} -le 129 ] && [ ${stop_stage} -ge 129 ];then
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
    #speech_encoder_type="ReDimNetB2"
    #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"
    speech_encoder_type="ReDimNetB3"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b3-vox2-ft_lm.pt"

    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_feature_dir"
    #speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_using_fbank_norm_feature_dir"
    #speaker_embedding_name_dir="redimnet_b3-vox2-ft_label_rate100_lm_using_fbank_feature_dir"
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
   #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB2_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_fbank_norm
   #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_label_rate100_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_fbank
   exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_cam++_200k_speaker_emb_lr2e-4
   mkdir -p $exp_dir
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   #data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc"
   rs_len=6
   segment_shift=2
   single_backend_type="transformer"
   multi_backend_type="transformer"
   d_state=128
   label_rate=25
   #label_rate=100
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15815 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 0\
    --grad-clip false\
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
    --d-state $d_state\
    --label-rate $label_rate
fi
# lr=1e-6,lr=5e-6, it is working, but it coverges too slowly and has poor performance.

if [ ${stage} -le 131 ] && [ ${stop_stage} -ge 131 ];then
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_cam++_200k_speaker_embedding_lr5e-6
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_cam++_200k_speaker_emb_lr2e-4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 d_state=128
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
# speech_encoder_type="CAM++"
# speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
# #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
#
# # for loading speaker embedding file
# spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
# speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
#
# data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

speech_encoder_type="ReDimNetB3"
speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b3-vox2-ft_lm.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

# for loading speaker embedding file
#spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_feature_dir"
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_using_fbank_feature_dir"

spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path

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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
 done
done
fi
#grep -r Eval logs/run_ts_vad2_stage131_infer_transformer_rs_len6_b3_label_rate25_lr2e-4_cam++_speaker_emb.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=21.61, miss=0.24, falarm=18.35, confusion=3.02
#Eval for threshold 0.3 DER=17.25, miss=0.45, falarm=12.58, confusion=4.22
#Eval for threshold 0.35 DER=15.67, miss=0.60, falarm=10.44, confusion=4.63
#Eval for threshold 0.4 DER=14.51, miss=0.82, falarm=8.72, confusion=4.97
#Eval for threshold 0.45 DER=13.68, miss=1.20, falarm=7.27, confusion=5.21
#Eval for threshold 0.5 DER=13.35, miss=1.98, falarm=6.19, confusion=5.19
#Eval for threshold 0.55 DER=13.71, miss=3.31, falarm=5.66, confusion=4.74
#Eval for threshold 0.6 DER=14.17, miss=4.59, falarm=5.20, confusion=4.38
#Eval for threshold 0.7 DER=15.80, miss=7.89, falarm=4.25, confusion=3.66
#Eval for threshold 0.8 DER=18.87, miss=12.81, falarm=3.35, confusion=2.71

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=21.31, miss=0.28, falarm=18.82, confusion=2.21
#Eval for threshold 0.3 DER=18.16, miss=0.57, falarm=14.64, confusion=2.96
#Eval for threshold 0.35 DER=17.06, miss=0.82, falarm=12.99, confusion=3.26
#Eval for threshold 0.4 DER=16.16, miss=1.15, falarm=11.48, confusion=3.53
#Eval for threshold 0.45 DER=14.77, miss=1.63, falarm=9.04, confusion=4.11
#Eval for threshold 0.5 DER=14.01, miss=3.19, falarm=6.44, confusion=4.37
#Eval for threshold 0.55 DER=14.91, miss=5.62, falarm=5.83, confusion=3.46
#Eval for threshold 0.6 DER=15.43, miss=7.03, falarm=5.37, confusion=3.03
#Eval for threshold 0.7 DER=16.72, miss=9.94, falarm=4.44, confusion=2.34
#Eval for threshold 0.8 DER=19.23, miss=14.07, falarm=3.51, confusion=1.65

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=12.90, miss=0.04, falarm=10.13, confusion=2.73
#Eval for threshold 0.3 DER=9.63, miss=0.07, falarm=5.74, confusion=3.82
#Eval for threshold 0.35 DER=8.54, miss=0.10, falarm=4.28, confusion=4.16
#Eval for threshold 0.4 DER=7.75, miss=0.18, falarm=3.12, confusion=4.45
#Eval for threshold 0.45 DER=7.21, miss=0.33, falarm=2.18, confusion=4.70
#Eval for threshold 0.5 DER=7.04, miss=0.80, falarm=1.47, confusion=4.77
#Eval for threshold 0.55 DER=7.56, miss=1.80, falarm=1.33, confusion=4.43
#Eval for threshold 0.6 DER=8.10, miss=2.72, falarm=1.24, confusion=4.14
#Eval for threshold 0.7 DER=9.89, miss=5.23, falarm=1.08, confusion=3.58
#Eval for threshold 0.8 DER=12.90, miss=9.27, falarm=0.93, confusion=2.71

# test of magicdata-ramc,collar=0.25
#Eval for threshold 0.2 DER=12.58, miss=0.04, falarm=10.84, confusion=1.70
#Eval for threshold 0.3 DER=10.57, miss=0.10, falarm=8.21, confusion=2.27
#Eval for threshold 0.35 DER=9.91, miss=0.17, falarm=7.22, confusion=2.52
#Eval for threshold 0.4 DER=9.37, miss=0.30, falarm=6.30, confusion=2.76
#Eval for threshold 0.45 DER=8.34, miss=0.52, falarm=4.48, confusion=3.34
#Eval for threshold 0.5 DER=7.82, miss=1.66, falarm=2.43, confusion=3.73
#Eval for threshold 0.55 DER=8.87, miss=3.78, falarm=2.21, confusion=2.88
#Eval for threshold 0.6 DER=9.41, miss=4.78, falarm=2.09, confusion=2.54
#Eval for threshold 0.7 DER=10.62, miss=6.79, falarm=1.83, confusion=2.00
#Eval for threshold 0.8 DER=12.83, miss=9.80, falarm=1.56, confusion=1.47


if [ ${stage} -le 132 ] && [ ${stop_stage} -ge 132 ];then
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
    #speech_encoder_type="CAM++"
    #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    speech_encoder_type="ReDimNetB3"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b3-vox2-ft_lm.pt"
    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
   exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3__epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len6_using_cam++_200k_speaker_emb
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=6
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
   d_state=256
   num_transformer_layer=2
   label_rate=25
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15515 \
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
    --label-rate $label_rate
fi


if [ ${stage} -le 133 ] && [ ${stop_stage} -ge 133 ];then
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_cam++_200k_speaker_embedding_lr5e-6
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_cam++_200k_speaker_emb_lr2e-4
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3__epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len6_using_cam++_200k_speaker_emb
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3__epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len6_using_cam++_200k_speaker_emb
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
# speech_encoder_type="CAM++"
# speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
# #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
#
# # for loading speaker embedding file
# spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
# speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
#
# data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

speech_encoder_type="ReDimNetB3"
speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b3-vox2-ft_lm.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

# for loading speaker embedding file
#spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_feature_dir"
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_using_fbank_feature_dir"

spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path

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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
 done
done

# grep -r Eval logs/run_ts_vad2_stage133_transformer_rs_len6_b3_mamba2_label_rate25_lr1e-4_cam++_speaker_emb_infer.log

# dev of magicdata-ramc, collar=0.0

#Eval for threshold 0.2 DER=28.83, miss=0.26, falarm=26.91, confusion=1.66
#Eval for threshold 0.3 DER=22.04, miss=0.50, falarm=18.14, confusion=3.40
#Eval for threshold 0.35 DER=19.24, miss=0.68, falarm=14.11, confusion=4.45
#Eval for threshold 0.4 DER=16.96, miss=0.95, falarm=10.52, confusion=5.49
#Eval for threshold 0.45 DER=15.38, miss=1.44, falarm=7.61, confusion=6.33
#Eval for threshold 0.5 DER=15.27, miss=3.05, falarm=6.11, confusion=6.11
#Eval for threshold 0.55 DER=16.39, miss=5.77, falarm=5.54, confusion=5.08
#Eval for threshold 0.6 DER=17.82, miss=8.77, falarm=4.99, confusion=4.05
#Eval for threshold 0.7 DER=21.54, miss=15.18, falarm=4.00, confusion=2.36
#Eval for threshold 0.8 DER=26.33, miss=22.21, falarm=2.98, confusion=1.13

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=24.85, miss=0.29, falarm=22.63, confusion=1.93
#Eval for threshold 0.3 DER=20.16, miss=0.57, falarm=16.60, confusion=2.99
#Eval for threshold 0.35 DER=18.44, miss=0.79, falarm=14.14, confusion=3.52
#Eval for threshold 0.4 DER=16.90, miss=1.13, falarm=11.64, confusion=4.13
#Eval for threshold 0.45 DER=15.19, miss=1.75, falarm=8.41, confusion=5.03
#Eval for threshold 0.5 DER=14.95, miss=3.88, falarm=6.35, confusion=4.73
#Eval for threshold 0.55 DER=15.94, miss=6.55, falarm=5.72, confusion=3.68
#Eval for threshold 0.6 DER=16.82, miss=8.65, falarm=5.17, confusion=3.00
#Eval for threshold 0.7 DER=18.97, miss=12.90, falarm=4.14, confusion=1.93
#Eval for threshold 0.8 DER=22.64, miss=18.29, falarm=3.23, confusion=1.12

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=19.63, miss=0.05, falarm=18.38, confusion=1.21
#Eval for threshold 0.3 DER=14.21, miss=0.08, falarm=11.30, confusion=2.83
#Eval for threshold 0.35 DER=11.90, miss=0.12, falarm=7.95, confusion=3.82
#Eval for threshold 0.4 DER=10.04, miss=0.20, falarm=4.99, confusion=4.85
#Eval for threshold 0.45 DER=8.72, miss=0.38, falarm=2.57, confusion=5.77
#Eval for threshold 0.5 DER=8.77, miss=1.61, falarm=1.50, confusion=5.67
#Eval for threshold 0.55 DER=10.08, miss=3.97, falarm=1.39, confusion=4.73
#Eval for threshold 0.6 DER=11.69, miss=6.65, falarm=1.28, confusion=3.75
#Eval for threshold 0.7 DER=15.68, miss=12.49, falarm=1.08, confusion=2.11
#Eval for threshold 0.8 DER=20.64, miss=18.79, falarm=0.90, confusion=0.95

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=15.62, miss=0.03, falarm=14.22, confusion=1.38
#Eval for threshold 0.3 DER=12.31, miss=0.09, falarm=10.01, confusion=2.22
#Eval for threshold 0.35 DER=11.17, miss=0.14, falarm=8.37, confusion=2.66
#Eval for threshold 0.4 DER=10.02, miss=0.24, falarm=6.56, confusion=3.22
#Eval for threshold 0.45 DER=8.65, miss=0.51, falarm=4.01, confusion=4.13
#Eval for threshold 0.5 DER=8.56, miss=2.14, falarm=2.41, confusion=4.01
#Eval for threshold 0.55 DER=9.68, miss=4.44, falarm=2.20, confusion=3.04
#Eval for threshold 0.6 DER=10.59, miss=6.08, falarm=2.03, confusion=2.47
#Eval for threshold 0.7 DER=12.64, miss=9.32, falarm=1.71, confusion=1.60
#Eval for threshold 0.8 DER=16.01, miss=13.65, falarm=1.44, confusion=0.92
fi


if [ ${stage} -le 134 ] && [ ${stop_stage} -ge 134 ];then
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
    #speech_encoder_type="CAM++"
    #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    speech_encoder_type="ReDimNetB3"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b3-vox2-ft_lm.pt"
    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
   exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3__epoch20_front_fix_seed_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len6_using_cam++_200k_speaker_emb_lr2e-4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=6
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
   d_state=256
   num_transformer_layer=2
   label_rate=25
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15615 \
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
    --d-state $d_state\
    --label-rate $label_rate
fi


if [ ${stage} -le 135 ] && [ ${stop_stage} -ge 135 ];then
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_cam++_200k_speaker_embedding_lr5e-6
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_cam++_200k_speaker_emb_lr2e-4
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3__epoch20_front_fix_seed_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len6_using_cam++_200k_speaker_emb_lr2e-4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
# speech_encoder_type="CAM++"
# speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
# #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
#
# # for loading speaker embedding file
# spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
# speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
#
# data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

speech_encoder_type="ReDimNetB3"
speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b3-vox2-ft_lm.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

# for loading speaker embedding file
#spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_feature_dir"
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_using_fbank_feature_dir"

spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path

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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
 done
done
fi
# grep -r Eval logs/run_ts_vad2_stage135_infer_mamba2_rs_len6_b3_label_rate25_lr2e-4_cam++_speaker_emb.log

# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=25.65, miss=0.22, falarm=23.69, confusion=1.74
#Eval for threshold 0.3 DER=20.63, miss=0.39, falarm=17.09, confusion=3.15
#Eval for threshold 0.35 DER=18.50, miss=0.52, falarm=14.01, confusion=3.96
#Eval for threshold 0.4 DER=16.51, miss=0.74, falarm=10.90, confusion=4.87
#Eval for threshold 0.45 DER=14.96, miss=1.08, falarm=8.22, confusion=5.65
#Eval for threshold 0.5 DER=14.39, miss=2.13, falarm=6.55, confusion=5.72
#Eval for threshold 0.55 DER=15.08, miss=4.26, falarm=5.94, confusion=4.88
#Eval for threshold 0.6 DER=16.19, miss=6.68, falarm=5.45, confusion=4.06
#Eval for threshold 0.7 DER=19.00, miss=12.21, falarm=4.37, confusion=2.43
#Eval for threshold 0.8 DER=22.71, miss=18.10, falarm=3.33, confusion=1.28

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=23.53, miss=0.26, falarm=21.54, confusion=1.73
#Eval for threshold 0.3 DER=19.41, miss=0.51, falarm=16.33, confusion=2.57
#Eval for threshold 0.35 DER=17.85, miss=0.69, falarm=14.17, confusion=2.99
#Eval for threshold 0.4 DER=16.32, miss=0.96, falarm=11.80, confusion=3.56
#Eval for threshold 0.45 DER=14.62, miss=1.39, falarm=8.66, confusion=4.57
#Eval for threshold 0.5 DER=13.89, miss=2.35, falarm=6.82, confusion=4.72
#Eval for threshold 0.55 DER=14.74, miss=4.95, falarm=6.10, confusion=3.68
#Eval for threshold 0.6 DER=15.72, miss=7.52, falarm=5.48, confusion=2.71
#Eval for threshold 0.7 DER=17.51, miss=11.26, falarm=4.42, confusion=1.84
#Eval for threshold 0.8 DER=20.53, miss=15.96, falarm=3.45, confusion=1.13

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=16.80, miss=0.04, falarm=15.47, confusion=1.29
#Eval for threshold 0.3 DER=12.90, miss=0.07, falarm=10.27, confusion=2.56
#Eval for threshold 0.35 DER=11.15, miss=0.09, falarm=7.73, confusion=3.32
#Eval for threshold 0.4 DER=9.50, miss=0.16, falarm=5.09, confusion=4.25
#Eval for threshold 0.45 DER=8.23, miss=0.30, falarm=2.85, confusion=5.08
#Eval for threshold 0.5 DER=7.83, miss=0.99, falarm=1.56, confusion=5.28
#Eval for threshold 0.55 DER=8.76, miss=2.85, falarm=1.39, confusion=4.52
#Eval for threshold 0.6 DER=10.08, miss=5.06, falarm=1.29, confusion=3.74
#Eval for threshold 0.7 DER=13.32, miss=10.06, falarm=1.11, confusion=2.15
#Eval for threshold 0.8 DER=17.15, miss=15.14, falarm=0.93, confusion=1.08

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=14.31, miss=0.04, falarm=13.03, confusion=1.24
#Eval for threshold 0.3 DER=11.43, miss=0.09, falarm=9.48, confusion=1.86
#Eval for threshold 0.35 DER=10.39, miss=0.13, falarm=8.05, confusion=2.21
#Eval for threshold 0.4 DER=9.33, miss=0.22, falarm=6.42, confusion=2.68
#Eval for threshold 0.45 DER=8.00, miss=0.40, falarm=3.88, confusion=3.72
#Eval for threshold 0.5 DER=7.44, miss=0.98, falarm=2.48, confusion=3.98
#Eval for threshold 0.55 DER=8.46, miss=3.15, falarm=2.24, confusion=3.07
#Eval for threshold 0.6 DER=9.56, miss=5.32, falarm=2.06, confusion=2.17
#Eval for threshold 0.7 DER=11.36, miss=8.11, falarm=1.75, confusion=1.50
#Eval for threshold 0.8 DER=14.12, miss=11.70, falarm=1.47, confusion=0.94


if [ ${stage} -le 136 ] && [ ${stop_stage} -ge 136 ];then
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
    #speech_encoder_type="ReDimNetB2"
    #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"
    speech_encoder_type="ReDimNetB2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"

    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_feature_dir"
    #speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_using_fbank_norm_feature_dir"
    #speaker_embedding_name_dir="redimnet_b3-vox2-ft_label_rate100_lm_using_fbank_feature_dir"
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
   #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB2_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_fbank_norm
   #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_label_rate100_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_fbank
   exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB2_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_cam++_200k_speaker_emb_lr2e-4
   mkdir -p $exp_dir
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   #data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc"
   rs_len=6
   segment_shift=2
   single_backend_type="transformer"
   multi_backend_type="transformer"
   d_state=128
   label_rate=25
   #label_rate=100
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15715 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 0\
    --grad-clip false\
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
    --d-state $d_state\
    --label-rate $label_rate
fi

if [ ${stage} -le 137 ] && [ ${stop_stage} -ge 137 ];then
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_cam++_200k_speaker_embedding_lr5e-6
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB2_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_cam++_200k_speaker_emb_lr2e-4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 d_state=128
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
# speech_encoder_type="CAM++"
# speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
# #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
#
# # for loading speaker embedding file
# spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
# speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
#
# data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

speech_encoder_type="ReDimNetB2"
speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

# for loading speaker embedding file
#spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_feature_dir"
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_using_fbank_feature_dir"

spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path

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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
 done
done
fi
# grep -r Eval logs/run_ts_vad2_stage136-137_transformer_rs_len6_b2_label_rate25_lr2e-4_cam++_speaker_emb.log
# dev of magicdata-ramc. collar=0.0
#Eval for threshold 0.2 DER=18.12, miss=0.23, falarm=14.50, confusion=3.38
#Eval for threshold 0.3 DER=15.08, miss=0.41, falarm=10.70, confusion=3.97
#Eval for threshold 0.35 DER=14.05, miss=0.56, falarm=9.32, confusion=4.18
#Eval for threshold 0.4 DER=13.21, miss=0.76, falarm=8.13, confusion=4.32
#Eval for threshold 0.45 DER=12.66, miss=1.05, falarm=7.17, confusion=4.44
#Eval for threshold 0.5 DER=12.37, miss=1.52, falarm=6.40, confusion=4.45
#Eval for threshold 0.55 DER=12.41, miss=2.30, falarm=5.90, confusion=4.21
#Eval for threshold 0.6 DER=12.63, miss=3.19, falarm=5.48, confusion=3.96
#Eval for threshold 0.7 DER=13.54, miss=5.48, falarm=4.56, confusion=3.50
#Eval for threshold 0.8 DER=15.51, miss=8.92, falarm=3.61, confusion=2.97

# test of magicdata-ramc. collar=0.0
#Eval for threshold 0.2 DER=21.03, miss=0.25, falarm=18.62, confusion=2.16
#Eval for threshold 0.3 DER=18.09, miss=0.57, falarm=14.69, confusion=2.83
#Eval for threshold 0.35 DER=17.04, miss=0.83, falarm=13.11, confusion=3.10
#Eval for threshold 0.4 DER=16.25, miss=1.18, falarm=11.72, confusion=3.35
#Eval for threshold 0.45 DER=15.42, miss=1.60, falarm=10.16, confusion=3.66
#Eval for threshold 0.5 DER=14.13, miss=3.35, falarm=6.59, confusion=4.18
#Eval for threshold 0.55 DER=14.98, miss=5.65, falarm=5.93, confusion=3.39
#Eval for threshold 0.6 DER=15.39, miss=6.89, falarm=5.49, confusion=3.01
#Eval for threshold 0.7 DER=16.54, miss=9.71, falarm=4.56, confusion=2.27
#Eval for threshold 0.8 DER=18.87, miss=13.63, falarm=3.62, confusion=1.62

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=9.37, miss=0.05, falarm=6.22, confusion=3.10
#Eval for threshold 0.3 DER=7.40, miss=0.09, falarm=3.77, confusion=3.54
#Eval for threshold 0.35 DER=6.79, miss=0.13, falarm=2.97, confusion=3.70
#Eval for threshold 0.4 DER=6.33, miss=0.19, falarm=2.33, confusion=3.81
#Eval for threshold 0.45 DER=6.05, miss=0.31, falarm=1.82, confusion=3.92
#Eval for threshold 0.5 DER=5.93, miss=0.52, falarm=1.42, confusion=3.99
#Eval for threshold 0.55 DER=6.11, miss=0.97, falarm=1.29, confusion=3.84
#Eval for threshold 0.6 DER=6.42, miss=1.54, falarm=1.22, confusion=3.66
#Eval for threshold 0.7 DER=7.45, miss=3.05, falarm=1.05, confusion=3.35
#Eval for threshold 0.8 DER=9.40, miss=5.52, falarm=0.91, confusion=2.96

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=12.49, miss=0.04, falarm=10.84, confusion=1.61
#Eval for threshold 0.3 DER=10.59, miss=0.13, falarm=8.34, confusion=2.13
#Eval for threshold 0.35 DER=9.93, miss=0.23, falarm=7.34, confusion=2.36
#Eval for threshold 0.4 DER=9.48, miss=0.37, falarm=6.50, confusion=2.61
#Eval for threshold 0.45 DER=8.99, miss=0.57, falarm=5.51, confusion=2.91
#Eval for threshold 0.5 DER=7.97, miss=1.94, falarm=2.48, confusion=3.55
#Eval for threshold 0.55 DER=8.97, miss=3.96, falarm=2.17, confusion=2.84
#Eval for threshold 0.6 DER=9.43, miss=4.85, falarm=2.06, confusion=2.53
#Eval for threshold 0.7 DER=10.58, miss=6.86, falarm=1.79, confusion=1.93
#Eval for threshold 0.8 DER=12.65, miss=9.70, falarm=1.52, confusion=1.42


if [ ${stage} -le 138 ] && [ ${stop_stage} -ge 138 ];then
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
    #speech_encoder_type="CAM++"
    #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    speech_encoder_type="ReDimNetB2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"
    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
   exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB2__epoch20_front_fix_seed_single_backend_2layer_mamba2_multi_backend_transformer_d_state128_rs_len6_using_cam++_200k_speaker_emb_lr2e-4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=6
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
   d_state=128
   num_transformer_layer=2
   label_rate=25
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15415 \
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
    --d-state $d_state\
    --label-rate $label_rate
fi


if [ ${stage} -le 139 ] && [ ${stop_stage} -ge 139 ];then
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_cam++_200k_speaker_embedding_lr5e-6
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_cam++_200k_speaker_emb_lr2e-4
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB2__epoch20_front_fix_seed_single_backend_2layer_mamba2_multi_backend_transformer_d_state128_rs_len6_using_cam++_200k_speaker_emb_lr2e-4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=128
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
# speech_encoder_type="CAM++"
# speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
# #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
#
# # for loading speaker embedding file
# spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
# speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
#
# data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

speech_encoder_type="ReDimNetB2"
speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

# for loading speaker embedding file
#spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_feature_dir"
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_using_fbank_feature_dir"

spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path

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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
 done
done
fi
## grep -r Eval logs/run_ts_vad2_stage138-139_mamba2_d_stat128_rs_len6.log

# dev of magicdata-ramc, collar=0.0
## Eval for threshold 0.2 DER=22.95, miss=0.27, falarm=20.50, confusion=2.18
#Eval for threshold 0.3 DER=17.73, miss=0.47, falarm=13.61, confusion=3.65
#Eval for threshold 0.35 DER=16.00, miss=0.61, falarm=11.15, confusion=4.24
#Eval for threshold 0.4 DER=14.66, miss=0.83, falarm=9.06, confusion=4.77
#Eval for threshold 0.45 DER=13.71, miss=1.21, falarm=7.42, confusion=5.09
#Eval for threshold 0.5 DER=13.37, miss=1.99, falarm=6.29, confusion=5.09
#Eval for threshold 0.55 DER=13.81, miss=3.44, falarm=5.78, confusion=4.60
#Eval for threshold 0.6 DER=14.52, miss=5.17, falarm=5.26, confusion=4.09
#Eval for threshold 0.7 DER=16.65, miss=9.41, falarm=4.25, confusion=2.99
#Eval for threshold 0.8 DER=20.50, miss=15.54, falarm=3.23, confusion=1.73

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=22.51, miss=0.30, falarm=20.04, confusion=2.16
#Eval for threshold 0.3 DER=18.93, miss=0.60, falarm=15.35, confusion=2.98
#Eval for threshold 0.35 DER=17.66, miss=0.82, falarm=13.47, confusion=3.37
#Eval for threshold 0.4 DER=16.53, miss=1.14, falarm=11.67, confusion=3.72
#Eval for threshold 0.45 DER=15.19, miss=1.61, falarm=9.29, confusion=4.29
#Eval for threshold 0.5 DER=14.12, miss=2.86, falarm=6.52, confusion=4.74
#Eval for threshold 0.55 DER=15.15, miss=5.59, falarm=5.85, confusion=3.71
#Eval for threshold 0.6 DER=15.90, miss=7.41, falarm=5.35, confusion=3.13
#Eval for threshold 0.7 DER=17.53, miss=10.91, falarm=4.33, confusion=2.29
#Eval for threshold 0.8 DER=20.51, miss=15.61, falarm=3.37, confusion=1.53

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=14.32, miss=0.05, falarm=12.42, confusion=1.84
#Eval for threshold 0.3 DER=10.17, miss=0.09, falarm=6.89, confusion=3.19
#Eval for threshold 0.35 DER=8.87, miss=0.12, falarm=4.99, confusion=3.76
#Eval for threshold 0.4 DER=7.84, miss=0.19, falarm=3.42, confusion=4.24
#Eval for threshold 0.45 DER=7.16, miss=0.34, falarm=2.23, confusion=4.58
#Eval for threshold 0.5 DER=6.99, miss=0.82, falarm=1.49, confusion=4.69
#Eval for threshold 0.55 DER=7.56, miss=1.94, falarm=1.37, confusion=4.25
#Eval for threshold 0.6 DER=8.41, miss=3.32, falarm=1.26, confusion=3.83
#Eval for threshold 0.7 DER=10.73, miss=6.81, falarm=1.07, confusion=2.85
#Eval for threshold 0.8 DER=14.73, miss=12.18, falarm=0.90, confusion=1.65

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=13.69, miss=0.06, falarm=12.00, confusion=1.63
#Eval for threshold 0.3 DER=11.28, miss=0.14, falarm=8.88, confusion=2.27
#Eval for threshold 0.35 DER=10.46, miss=0.21, falarm=7.68, confusion=2.58
#Eval for threshold 0.4 DER=9.70, miss=0.33, falarm=6.47, confusion=2.90
#Eval for threshold 0.45 DER=8.72, miss=0.55, falarm=4.69, confusion=3.48
#Eval for threshold 0.5 DER=7.85, miss=1.37, falarm=2.41, confusion=4.07
#Eval for threshold 0.55 DER=9.02, miss=3.74, falarm=2.16, confusion=3.13
#Eval for threshold 0.6 DER=9.85, miss=5.20, falarm=2.02, confusion=2.62
#Eval for threshold 0.7 DER=11.48, miss=7.76, falarm=1.74, confusion=1.97
#Eval for threshold 0.8 DER=14.19, miss=11.37, falarm=1.47, confusion=1.35



if [ ${stage} -le 140 ] && [ ${stop_stage} -ge 140 ];then
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
    #speech_encoder_type="ReDimNetB2"
    #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"
    speech_encoder_type="ReDimNetB2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"

    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_feature_dir"
    #speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_using_fbank_norm_feature_dir"
    #speaker_embedding_name_dir="redimnet_b3-vox2-ft_label_rate100_lm_using_fbank_feature_dir"
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
   #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB2_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_fbank_norm
   #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_label_rate100_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_fbank
   exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB2_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_cam++_200k_speaker_emb_lr3e-4
   mkdir -p $exp_dir
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   #data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc"
   rs_len=6
   segment_shift=2
   single_backend_type="transformer"
   multi_backend_type="transformer"
   d_state=128
   label_rate=25
   #label_rate=100
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15715 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 0\
    --grad-clip false\
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
    --d-state $d_state\
    --label-rate $label_rate
fi

if [ ${stage} -le 141 ] && [ ${stop_stage} -ge 141 ];then
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_cam++_200k_speaker_embedding_lr5e-6
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB2_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_cam++_200k_speaker_emb_lr3e-4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 d_state=128
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
# speech_encoder_type="CAM++"
# speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
# #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
#
# # for loading speaker embedding file
# spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
# speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
#
# data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

speech_encoder_type="ReDimNetB2"
speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

# for loading speaker embedding file
#spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_feature_dir"
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_using_fbank_feature_dir"

spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path

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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
 done
done
fi


if [ ${stage} -le 142 ] && [ ${stop_stage} -ge 142 ];then
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
    #speech_encoder_type="ReDimNetB2"
    #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"

    # # ReDimNetB6 as speech encoder, 2a100 (80GB) OOM, give up it.
    #speech_encoder_type="ReDimNetB6"
    #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b6-vox2-ft_lm.pt"
    #  ReDimNetB5 as speech encoder, 2a100 (80GB) OOM, give up it.
    #speech_encoder_type="ReDimNetB5"
    #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b5-vox2-ft_lm.pt"

    speech_encoder_type="ReDimNetS"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/S-vb2+vox2+cnc-ft_mix.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_feature_dir"
    #speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_using_fbank_norm_feature_dir"
    #speaker_embedding_name_dir="redimnet_b3-vox2-ft_label_rate100_lm_using_fbank_feature_dir"
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
   #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB2_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_fbank_norm
   #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_label_rate100_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_fbank
   exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetS_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_cam++_200k_speaker_emb_lr2e-4
   mkdir -p $exp_dir
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   #data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc"
   rs_len=6
   segment_shift=2
   single_backend_type="transformer"
   multi_backend_type="transformer"
   d_state=128
   label_rate=25
   #label_rate=100
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15615 \
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
    --d-state $d_state\
    --label-rate $label_rate
fi
# ReDimNetB6, 2a100 (80GB) OOM, give up it.
if [ ${stage} -le 143 ] && [ ${stop_stage} -ge 143 ];then
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_cam++_200k_speaker_embedding_lr5e-6
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetS_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_cam++_200k_speaker_emb_lr2e-4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 d_state=128
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
# speech_encoder_type="CAM++"
# speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
# #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
#
# # for loading speaker embedding file
# spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
# speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
#
# data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
# ReDimNetB6 as speech encoder, 2a100 (80GB) OOM, give up it.
speech_encoder_type="ReDimNetS"
speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/S-vb2+vox2+cnc-ft_mix.pt"

# for loading speaker embedding file
#spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_feature_dir"
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_using_fbank_feature_dir"

spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path

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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
 done
done
fi


if [ ${stage} -le 144 ] && [ ${stop_stage} -ge 144 ];then
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
    #speech_encoder_type="ReDimNetB2"
    #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"
    speech_encoder_type="ReDimNetB2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"

    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_feature_dir"
    #speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_using_fbank_norm_feature_dir"
    #speaker_embedding_name_dir="redimnet_b3-vox2-ft_label_rate100_lm_using_fbank_feature_dir"
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_label_rate_100_feature_dir"
    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
   #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB2_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_fbank_norm
   #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_label_rate100_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_fbank
   exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB2_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_cam++_200k_speaker_emb_lr1e-4_label_rate100
   mkdir -p $exp_dir
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   #data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc"
   rs_len=6
   segment_shift=2
   single_backend_type="transformer"
   multi_backend_type="transformer"
   d_state=128
   label_rate=100
   #label_rate=100
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15515 \
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
    --label-rate $label_rate
fi

if [ ${stage} -le 145 ] && [ ${stop_stage} -ge 145 ];then
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_cam++_200k_speaker_embedding_lr5e-6
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB2_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_cam++_200k_speaker_emb_lr1e-4_label_rate100
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 d_state=128
 num_transformer_layer=2
 label_rate=100
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
# speech_encoder_type="CAM++"
# speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
# #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
#
# # for loading speaker embedding file
# spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
# speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
#
# data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

speech_encoder_type="ReDimNetB2"
speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

# for loading speaker embedding file
#spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_feature_dir"
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_using_fbank_feature_dir"

#spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
#speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
speaker_embedding_name_dir="cam++_zh-cn_200k_label_rate_100_feature_dir"
data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path

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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
 done
done
fi
# grep -r Eval logs/run_ts_vad2_stage144-145_transformer_rs_len6_lr2e4_b2_label_rate100.log
# Eval for threshold 0.2 DER=106.76, miss=74.75, falarm=31.57, confusion=0.44
#Eval for threshold 0.3 DER=104.33, miss=74.75, falarm=28.69, confusion=0.89
#Eval for threshold 0.35 DER=102.82, miss=74.75, falarm=26.98, confusion=1.10
#Eval for threshold 0.4 DER=95.16, miss=74.79, falarm=17.71, confusion=2.67
#Eval for threshold 0.45 DER=88.83, miss=81.39, falarm=3.37, confusion=4.06
#Eval for threshold 0.5 DER=94.68, miss=92.25, falarm=1.17, confusion=1.26
#Eval for threshold 0.55 DER=96.38, miss=94.47, falarm=0.86, confusion=1.06
#Eval for threshold 0.6 DER=96.68, miss=95.06, falarm=0.70, confusion=0.92
#Eval for threshold 0.7 DER=98.17, miss=97.46, falarm=0.25, confusion=0.46
#Eval for threshold 0.8 DER=99.86, miss=99.66, falarm=0.05, confusion=0.16
#Eval for threshold 0.2 DER=108.17, miss=75.18, falarm=32.93, confusion=0.06
#Eval for threshold 0.3 DER=104.39, miss=75.18, falarm=28.47, confusion=0.74
#Eval for threshold 0.35 DER=100.10, miss=75.18, falarm=23.10, confusion=1.83
#Eval for threshold 0.4 DER=92.74, miss=75.18, falarm=12.96, confusion=4.60
#Eval for threshold 0.45 DER=88.61, miss=78.62, falarm=4.38, confusion=5.61
#Eval for threshold 0.5 DER=91.50, miss=86.21, falarm=2.37, confusion=2.92
#Eval for threshold 0.55 DER=94.46, miss=91.98, falarm=1.26, confusion=1.21
#Eval for threshold 0.6 DER=96.64, miss=95.41, falarm=0.69, confusion=0.54
#Eval for threshold 0.7 DER=98.53, miss=98.19, falarm=0.30, confusion=0.04
#Eval for threshold 0.8 DER=99.31, miss=99.17, falarm=0.14, confusion=0.00
#Eval for threshold 0.2 DER=102.14, miss=74.63, falarm=27.10, confusion=0.41
#Eval for threshold 0.3 DER=99.93, miss=74.63, falarm=24.47, confusion=0.82
#Eval for threshold 0.35 DER=98.57, miss=74.63, falarm=22.93, confusion=1.00
#Eval for threshold 0.4 DER=91.47, miss=74.66, falarm=14.35, confusion=2.46
#Eval for threshold 0.45 DER=86.65, miss=80.98, falarm=1.74, confusion=3.93
#Eval for threshold 0.5 DER=93.73, miss=92.03, falarm=0.51, confusion=1.19
#Eval for threshold 0.55 DER=95.72, miss=94.36, falarm=0.36, confusion=1.00
#Eval for threshold 0.6 DER=96.09, miss=94.94, falarm=0.27, confusion=0.88
#Eval for threshold 0.7 DER=97.86, miss=97.33, falarm=0.09, confusion=0.45
#Eval for threshold 0.8 DER=99.83, miss=99.64, falarm=0.02, confusion=0.17
#Eval for threshold 0.2 DER=103.37, miss=75.25, falarm=28.08, confusion=0.04
#Eval for threshold 0.3 DER=99.83, miss=75.25, falarm=23.88, confusion=0.69
#Eval for threshold 0.35 DER=95.97, miss=75.25, falarm=18.96, confusion=1.75
#Eval for threshold 0.4 DER=89.50, miss=75.26, falarm=9.87, confusion=4.37
#Eval for threshold 0.45 DER=86.26, miss=78.40, falarm=2.41, confusion=5.45
#Eval for threshold 0.5 DER=89.92, miss=85.89, falarm=1.17, confusion=2.86
#Eval for threshold 0.55 DER=93.53, miss=91.73, falarm=0.63, confusion=1.17
#Eval for threshold 0.6 DER=96.08, miss=95.18, falarm=0.37, confusion=0.53
#Eval for threshold 0.7 DER=98.29, miss=98.10, falarm=0.17, confusion=0.03
#Eval for threshold 0.8 DER=99.20, miss=99.12, falarm=0.08, confusion=0.00






if [ ${stage} -le 146 ] && [ ${stop_stage} -ge 146 ];then
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
    #speech_encoder_type="ReDimNetB2"
    #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"

    # # ReDimNetB6 as speech encoder, 2a100 (80GB) OOM, give up it.
    #speech_encoder_type="ReDimNetB6"
    #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b6-vox2-ft_lm.pt"
    # ReDimNetB4 as speech encoder, 2a100 (80GB) OOM, give up it.
    #speech_encoder_type="ReDimNetB4"
    #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b4-vox2-ft_lm.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    speech_encoder_type="ReDimNetM"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/M-vb2+vox2+cnc-ft_mix.pt"

    # for loading speaker embedding file
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_feature_dir"
    #speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_using_fbank_norm_feature_dir"
    #speaker_embedding_name_dir="redimnet_b3-vox2-ft_label_rate100_lm_using_fbank_feature_dir"
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
   #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB2_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_fbank_norm
   #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_label_rate100_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_fbank
   exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetM_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_cam++_200k_speaker_emb_lr2e-4
   mkdir -p $exp_dir
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   #data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc"
   rs_len=6
   segment_shift=2
   single_backend_type="transformer"
   multi_backend_type="transformer"
   d_state=128
   label_rate=25
   #label_rate=100
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15415 \
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
    --d-state $d_state\
    --label-rate $label_rate
fi
# ReDimNetB6, 2a100 (80GB) OOM, give up it.
if [ ${stage} -le 147 ] && [ ${stop_stage} -ge 147 ];then
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_cam++_200k_speaker_embedding_lr5e-6
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetM_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_cam++_200k_speaker_emb_lr2e-4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 d_state=128
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
# speech_encoder_type="CAM++"
# speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
# #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
#
# # for loading speaker embedding file
# spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
# speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
speech_encoder_type="ReDimNetM"
speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/M-vb2+vox2+cnc-ft_mix.pt"

# for loading speaker embedding file
#spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_feature_dir"
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_using_fbank_feature_dir"

spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path

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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
 done
done
fi




if [ ${stage} -le 149 ] && [ ${stop_stage} -ge 149 ];then
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
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    #speech_encoder_type="ReDimNetB2"
    #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"
    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
   #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB2__epoch20_front_fix_seed_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len6_using_cam++_200k_speaker_emb_lr2e-4
   exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++__epoch20_front_fix_seed_single_backend_2layer_conformer_multi_backend_transformer_rs_len8_using_cam++_200k_speaker_emb_lr2e-4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=8
   segment_shift=2
   single_backend_type="conformer"
   multi_backend_type="transformer"
   d_state=256
   num_transformer_layer=2
   label_rate=25
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 14415 \
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
    --d-state $d_state\
    --label-rate $label_rate
fi


if [ ${stage} -le 150 ] && [ ${stop_stage} -ge 150 ];then
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_cam++_200k_speaker_embedding_lr5e-6
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_cam++_200k_speaker_emb_lr2e-4
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++__epoch20_front_fix_seed_single_backend_2layer_conformer_multi_backend_transformer_rs_len8_using_cam++_200k_speaker_emb_lr2e-4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=1
 single_backend_type="conformer"
 multi_backend_type="transformer"
 d_state=128
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
speech_encoder_type="CAM++"
speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
#speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
#
# # for loading speaker embedding file
# spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
# speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
#
# data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

#speech_encoder_type="ReDimNetB2"
#speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

# for loading speaker embedding file
#spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_feature_dir"
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_using_fbank_feature_dir"

spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path

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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
 done
done
fi
#grep -r Eval logs/run_ts_vad2_stage149-150_conformer_rs_len8.log
# dev of magicdata-ramc, collar=0.0
# Eval for threshold 0.2 DER=14.38, miss=0.28, falarm=10.75, confusion=3.35
#Eval for threshold 0.3 DER=12.75, miss=0.52, falarm=8.66, confusion=3.56
#Eval for threshold 0.35 DER=12.17, miss=0.67, falarm=7.88, confusion=3.62
#Eval for threshold 0.4 DER=11.75, miss=0.87, falarm=7.23, confusion=3.65
#Eval for threshold 0.45 DER=11.44, miss=1.12, falarm=6.66, confusion=3.66
#Eval for threshold 0.5 DER=11.26, miss=1.47, falarm=6.15, confusion=3.64
#Eval for threshold 0.55 DER=11.19, miss=1.88, falarm=5.74, confusion=3.57
#Eval for threshold 0.6 DER=11.18, miss=2.36, falarm=5.33, confusion=3.49
#Eval for threshold 0.7 DER=11.52, miss=3.68, falarm=4.54, confusion=3.30
#Eval for threshold 0.8 DER=12.42, miss=5.68, falarm=3.72, confusion=3.03

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=18.02, miss=0.36, falarm=15.97, confusion=1.69
#Eval for threshold 0.3 DER=15.65, miss=0.76, falarm=12.83, confusion=2.07
#Eval for threshold 0.35 DER=14.82, miss=1.02, falarm=11.61, confusion=2.19
#Eval for threshold 0.4 DER=14.17, miss=1.38, falarm=10.53, confusion=2.27
#Eval for threshold 0.45 DER=13.71, miss=1.86, falarm=9.49, confusion=2.36
#Eval for threshold 0.5 DER=12.46, miss=3.20, falarm=6.37, confusion=2.89
#Eval for threshold 0.55 DER=13.37, miss=5.47, falarm=5.66, confusion=2.23
#Eval for threshold 0.6 DER=13.66, miss=6.38, falarm=5.22, confusion=2.06
#Eval for threshold 0.7 DER=14.77, miss=8.68, falarm=4.39, confusion=1.70
#Eval for threshold 0.8 DER=16.85, miss=12.02, falarm=3.54, confusion=1.29

# dev of magicdata-ramc,collar=0.25
#Eval for threshold 0.2 DER=6.36, miss=0.08, falarm=3.11, confusion=3.16
#Eval for threshold 0.3 DER=5.59, miss=0.16, falarm=2.14, confusion=3.29
#Eval for threshold 0.35 DER=5.35, miss=0.21, falarm=1.83, confusion=3.31
#Eval for threshold 0.4 DER=5.21, miss=0.28, falarm=1.60, confusion=3.33
#Eval for threshold 0.45 DER=5.14, miss=0.37, falarm=1.44, confusion=3.33
#Eval for threshold 0.5 DER=5.12, miss=0.50, falarm=1.29, confusion=3.33
#Eval for threshold 0.55 DER=5.17, miss=0.67, falarm=1.21, confusion=3.29
#Eval for threshold 0.6 DER=5.26, miss=0.88, falarm=1.13, confusion=3.24
#Eval for threshold 0.7 DER=5.64, miss=1.48, falarm=1.01, confusion=3.16
#Eval for threshold 0.8 DER=6.45, miss=2.57, falarm=0.88, confusion=3.01

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=9.84, miss=0.11, falarm=8.40, confusion=1.32
#Eval for threshold 0.3 DER=8.44, miss=0.28, falarm=6.60, confusion=1.56
#Eval for threshold 0.35 DER=7.96, miss=0.38, falarm=5.94, confusion=1.64
#Eval for threshold 0.4 DER=7.66, miss=0.55, falarm=5.41, confusion=1.70
#Eval for threshold 0.45 DER=7.47, miss=0.79, falarm=4.91, confusion=1.77
#Eval for threshold 0.5 DER=6.50, miss=1.78, falarm=2.36, confusion=2.35
#Eval for threshold 0.55 DER=7.58, miss=3.85, falarm=2.03, confusion=1.71
#Eval for threshold 0.6 DER=7.90, miss=4.39, falarm=1.91, confusion=1.60
#Eval for threshold 0.7 DER=8.92, miss=5.89, falarm=1.66, confusion=1.38
#Eval for threshold 0.8 DER=10.74, miss=8.23, falarm=1.40, confusion=1.11


if [ ${stage} -le 151 ] && [ ${stop_stage} -ge 151 ];then
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
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    #speech_encoder_type="ReDimNetB2"
    #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"
    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
   #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB2__epoch20_front_fix_seed_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len6_using_cam++_200k_speaker_emb_lr2e-4
   exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++__epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len8_using_cam++_200k_speaker_emb_lr2e-4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
   rs_len=8
   segment_shift=2
   single_backend_type="transformer"
   multi_backend_type="transformer"
   d_state=256
   num_transformer_layer=2
   label_rate=25
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 14315 \
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
    --d-state $d_state\
    --label-rate $label_rate
fi


if [ ${stage} -le 152 ] && [ ${stop_stage} -ge 152 ];then
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_cam++_200k_speaker_embedding_lr5e-6
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_ReDimNetB3_epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_using_cam++_200k_speaker_emb_lr2e-4
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++__epoch20_front_fix_seed_single_backend_2layer_transformer_multi_backend_transformer_rs_len8_using_cam++_200k_speaker_emb_lr2e-4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 d_state=128
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
speech_encoder_type="CAM++"
speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
#speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
#
# # for loading speaker embedding file
# spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
# speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
#
# data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path

#speech_encoder_type="ReDimNetB2"
#speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/b2-vox2-ft_lm.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

# for loading speaker embedding file
#spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_feature_dir"
#speaker_embedding_name_dir="redimnet_b2-vox2-ft_lm_using_fbank_feature_dir"

spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path

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
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
 done
done
fi

# grep -r Eval logs/run_ts_vad2_stage151-152_transformer_rs_len8.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=15.35, miss=0.23, falarm=11.84, confusion=3.28
#Eval for threshold 0.3 DER=13.33, miss=0.41, falarm=9.37, confusion=3.55
#Eval for threshold 0.35 DER=12.67, miss=0.53, falarm=8.50, confusion=3.63
#Eval for threshold 0.4 DER=12.18, miss=0.71, falarm=7.80, confusion=3.67
#Eval for threshold 0.45 DER=11.77, miss=0.94, falarm=7.12, confusion=3.71
#Eval for threshold 0.5 DER=11.48, miss=1.23, falarm=6.56, confusion=3.70
#Eval for threshold 0.55 DER=11.32, miss=1.60, falarm=6.08, confusion=3.65
#Eval for threshold 0.6 DER=11.26, miss=2.10, falarm=5.62, confusion=3.54
#Eval for threshold 0.7 DER=11.47, miss=3.40, falarm=4.75, confusion=3.31
#Eval for threshold 0.8 DER=12.42, miss=5.50, falarm=3.86, confusion=3.06

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=18.35, miss=0.25, falarm=16.46, confusion=1.64
#Eval for threshold 0.3 DER=15.96, miss=0.54, falarm=13.42, confusion=2.01
#Eval for threshold 0.35 DER=15.15, miss=0.73, falarm=12.25, confusion=2.16
#Eval for threshold 0.4 DER=14.49, miss=0.96, falarm=11.26, confusion=2.28
#Eval for threshold 0.45 DER=13.91, miss=1.27, falarm=10.24, confusion=2.40
#Eval for threshold 0.5 DER=12.14, miss=1.69, falarm=7.06, confusion=3.38
#Eval for threshold 0.55 DER=11.99, miss=2.43, falarm=6.25, confusion=3.31
#Eval for threshold 0.6 DER=13.17, miss=5.42, falarm=5.62, confusion=2.13
#Eval for threshold 0.7 DER=13.85, miss=7.29, falarm=4.80, confusion=1.75
#Eval for threshold 0.8 DER=15.41, miss=10.21, falarm=3.86, confusion=1.35

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=6.94, miss=0.04, falarm=3.81, confusion=3.08
#Eval for threshold 0.3 DER=5.84, miss=0.10, falarm=2.49, confusion=3.25
#Eval for threshold 0.35 DER=5.54, miss=0.13, falarm=2.11, confusion=3.29
#Eval for threshold 0.4 DER=5.32, miss=0.18, falarm=1.83, confusion=3.31
#Eval for threshold 0.45 DER=5.19, miss=0.25, falarm=1.60, confusion=3.34
#Eval for threshold 0.5 DER=5.13, miss=0.35, falarm=1.43, confusion=3.34
#Eval for threshold 0.55 DER=5.11, miss=0.48, falarm=1.31, confusion=3.32
#Eval for threshold 0.6 DER=5.17, miss=0.68, falarm=1.21, confusion=3.27
#Eval for threshold 0.7 DER=5.45, miss=1.22, falarm=1.06, confusion=3.17
#Eval for threshold 0.8 DER=6.35, miss=2.37, falarm=0.93, confusion=3.05

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=9.73, miss=0.04, falarm=8.44, confusion=1.25
#Eval for threshold 0.3 DER=8.35, miss=0.13, falarm=6.76, confusion=1.47
#Eval for threshold 0.35 DER=7.90, miss=0.18, falarm=6.15, confusion=1.56
#Eval for threshold 0.4 DER=7.59, miss=0.27, falarm=5.68, confusion=1.63
#Eval for threshold 0.45 DER=7.32, miss=0.39, falarm=5.22, confusion=1.71
#Eval for threshold 0.5 DER=5.88, miss=0.56, falarm=2.63, confusion=2.69
#Eval for threshold 0.55 DER=5.89, miss=0.97, falarm=2.24, confusion=2.68
#Eval for threshold 0.6 DER=7.25, miss=3.62, falarm=2.06, confusion=1.57
#Eval for threshold 0.7 DER=7.80, miss=4.60, falarm=1.84, confusion=1.37
#Eval for threshold 0.8 DER=9.10, miss=6.39, falarm=1.55, confusion=1.15
