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
   dest_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding/
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

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc_with_prob0.5_alimeeting-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4
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
