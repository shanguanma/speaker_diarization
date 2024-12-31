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
# compared with stage40-41, stage52-53 will increase lr_rate from 1e-5 to 15e-5 (note: lr_rate=2e-4, loss will have nan)
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
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
done
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
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
done
fi
