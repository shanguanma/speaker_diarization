#!/usr/bin/env bash

stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_speaker_diarization_hltsz.sh

 # I use eend_vc(stage8-9 of run_eend_vc_hltsz.sh) predict rttm as system_vad to generate target audio and json, then
 # feed my best tsvad model(stage108-109 of run_ts_vad2.sh)
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
  # I use eend_vc predict rttm as system_vad to generate target audio and json
  datasets="dev test cssd_testset"
  #datasets="dev"
  source_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
  for name in $datasets;do
    eend_vc_exp=/data/maduo/exp/speaker_diarization/eend_vc/wavlm_updated_conformer_magicdata-ramc/infer_constrained_AHC_segmentation_step_0.1_min_cluster_size_30_AHC_thres_0.70_pyan_max_length_merged50_cam++_zh_200k_common/metric_DER_best/avg_ckpt5
    system_rttm=$eend_vc_exp/$name/all.rttm
    wavscp=$source_dir/$name/wav.scp
    dest_dir=$eend_vc_exp/$name/magicdata-ramc_system_data
    python3 ts_vad2/system_rttm_to_generate_target_speaker_wav_and_label_for_ts_vad.py\
	    --system_rttm $system_rttm\
            --wavscp $wavscp\
            --dest_dir $dest_dir\
            --type $name
  done
fi

## prepared target speaker embedding
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "prepare eval set target audio list"
   for name in dev test cssd_testset;do
    eend_vc_exp=/data/maduo/exp/speaker_diarization/eend_vc/wavlm_updated_conformer_magicdata-ramc/infer_constrained_AHC_segmentation_step_0.1_min_cluster_size_30_AHC_thres_0.70_pyan_max_length_merged50_cam++_zh_200k_common/metric_DER_best/avg_ckpt5
    input_dir=$eend_vc_exp/$name/magicdata-ramc_system_data/${name}/target_audio
    file=$input_dir/wavs.txt
    python3 ts_vad2/prepare_magicdata-ramc_target_audio_list.py \
        $input_dir $file
   done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "generate eval set speaker embedding"
   for name in dev test cssd_testset;do
    eend_vc_exp=/data/maduo/exp/speaker_diarization/eend_vc/wavlm_updated_conformer_magicdata-ramc/infer_constrained_AHC_segmentation_step_0.1_min_cluster_size_30_AHC_thres_0.70_pyan_max_length_merged50_cam++_zh_200k_common/metric_DER_best/avg_ckpt5

    exp_dir=$eend_vc_exp/$name/magicdata-ramc_system_data
    feature_name=cam++_zh_200k_feature_dir
    model_id=iic/speech_campplus_sv_zh-cn_16k-common
    # 提取embedding
    input_dir=$exp_dir/${name}/target_audio
    wav_path=$input_dir/wavs.txt
    save_dir=$exp_dir/SpeakerEmbedding/${name}/$feature_name
    python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
        --model_id $model_id --wavs $wav_path\
        --save_dir $save_dir
   echo "Finish extract target speaker embedding!!!!"
   done
fi
# utils now(2025-3-5) it is sota of magicdata-ramc
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
 #exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/magicdata_ramc/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state128_rs_len6
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state128_rs_len6/
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
 infer_sets="dev test cssd_testset"
 #infer_sets="cssd_testset"
 #infer_sets="Test"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 #collar=0.25
 collar="0.0 0.25"

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/magicdata_ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh_200k_feature_dir"

 #data_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 #data_dir="" 
 eend_vc_exp=/data/maduo/exp/speaker_diarization/eend_vc/wavlm_updated_conformer_magicdata-ramc/infer_constrained_AHC_segmentation_step_0.1_min_cluster_size_30_AHC_thres_0.70_pyan_max_length_merged50_cam++_zh_200k_common/metric_DER_best/avg_ckpt5/
#data_dir="/mntcephfs/lab_data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/"
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_eend_vc_system_collar${c}
    data_dir="$eend_vc_exp/$name/magicdata-ramc_system_data/$name" # system target audio , mix audio and labels path
    spk_path=$eend_vc_exp/$name/magicdata-ramc_system_data/SpeakerEmbedding/$name # store speaker embedding directory
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
