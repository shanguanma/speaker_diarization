#!/usr/bin/env bash
stage=0
stop_stage=1000
. utils/parse_options.sh
. path.sh # your tsvad python env
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
 exp_dir=magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state128_rs_len6
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
 infer_sets="dev test" # your testset name
 #infer_sets="cssd_testset"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format # your testset oracle rttm dir
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name, this fix it!!!!!
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin" # your cam++_zh_200k model path
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path of your testset
 #data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/"
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



if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   infer_sets="dev test" # your testset name
   predict_rttm_dir=magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state128_rs_len6/magicdata-ramc_collar0.0/
   oracle_rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
   fi
