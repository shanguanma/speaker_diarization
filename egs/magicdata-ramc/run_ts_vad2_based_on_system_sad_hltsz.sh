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
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
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
    data_dir="$eend_vc_exp/$name/magicdata-ramc_system_data" # system target audio , mix audio and labels path
    spk_path=$eend_vc_exp/$name/magicdata-ramc_system_data/SpeakerEmbedding # store speaker embedding directory
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

#grep -r Eval logs/run_ts_vad2_based_on_system_sad_stage4.log
# dev of magicdata-ramc, collar=0
#Eval for threshold 0.2 DER=14.47, miss=0.25, falarm=10.72, confusion=3.49
#Eval for threshold 0.3 DER=12.94, miss=0.47, falarm=8.84, confusion=3.64
#Eval for threshold 0.35 DER=12.45, miss=0.64, falarm=8.13, confusion=3.68
#Eval for threshold 0.4 DER=12.07, miss=0.85, falarm=7.51, confusion=3.71
#Eval for threshold 0.45 DER=11.76, miss=1.11, falarm=6.94, confusion=3.71
#Eval for threshold 0.5 DER=11.55, miss=1.44, falarm=6.42, confusion=3.69
#Eval for threshold 0.55 DER=11.49, miss=1.87, falarm=6.01, confusion=3.61
#Eval for threshold 0.6 DER=11.50, miss=2.41, falarm=5.59, confusion=3.50
#Eval for threshold 0.7 DER=11.63, miss=3.64, falarm=4.69, confusion=3.30
#Eval for threshold 0.8 DER=12.46, miss=5.60, falarm=3.75, confusion=3.11
#Eval for threshold 0.9 DER=15.41, miss=9.95, falarm=2.63, confusion=2.83

# test of magicdata-ramc, collar=0
#Eval for threshold 0.2 DER=17.66, miss=0.38, falarm=15.10, confusion=2.18
#Eval for threshold 0.3 DER=15.08, miss=0.82, falarm=11.57, confusion=2.69
#Eval for threshold 0.35 DER=14.14, miss=1.13, falarm=10.03, confusion=2.99
#Eval for threshold 0.4 DER=13.43, miss=1.49, falarm=8.74, confusion=3.19
#Eval for threshold 0.45 DER=12.91, miss=1.92, falarm=7.66, confusion=3.33
#Eval for threshold 0.5 DER=12.55, miss=2.56, falarm=6.54, confusion=3.45
#Eval for threshold 0.55 DER=12.85, miss=3.81, falarm=5.97, confusion=3.07
#Eval for threshold 0.6 DER=13.24, miss=5.00, falarm=5.49, confusion=2.75
#Eval for threshold 0.7 DER=14.65, miss=8.10, falarm=4.51, confusion=2.04
#Eval for threshold 0.8 DER=17.09, miss=12.01, falarm=3.51, confusion=1.57
#Eval for threshold 0.9 DER=22.11, miss=18.53, falarm=2.39, confusion=1.19

# cssd_testset of magicdata-ramc, collar=0
#Eval for threshold 0.2 DER=29.03, miss=4.12, falarm=22.45, confusion=2.45
#Eval for threshold 0.3 DER=24.73, miss=5.97, falarm=15.53, confusion=3.24
#Eval for threshold 0.35 DER=23.42, miss=7.12, falarm=12.79, confusion=3.51
#Eval for threshold 0.4 DER=22.72, miss=8.50, falarm=10.64, confusion=3.59 as report
#Eval for threshold 0.45 DER=22.58, miss=10.14, falarm=8.99, confusion=3.45
#Eval for threshold 0.5 DER=23.09, miss=12.19, falarm=7.83, confusion=3.07
#Eval for threshold 0.55 DER=24.02, miss=14.49, falarm=7.03, confusion=2.51
#Eval for threshold 0.6 DER=25.11, miss=16.87, falarm=6.27, confusion=1.97
#Eval for threshold 0.7 DER=27.77, miss=21.80, falarm=4.76, confusion=1.21
#Eval for threshold 0.8 DER=31.82, miss=27.94, falarm=3.19, confusion=0.69
#Eval for threshold 0.9 DER=40.74, miss=38.81, falarm=1.66, confusion=0.27

# dev of magicdata-rmac, collar=0.25
#Eval for threshold 0.2 DER=6.10, miss=0.06, falarm=2.81, confusion=3.23
#Eval for threshold 0.3 DER=5.44, miss=0.13, falarm=2.02, confusion=3.29
#Eval for threshold 0.35 DER=5.26, miss=0.20, falarm=1.76, confusion=3.30
#Eval for threshold 0.4 DER=5.16, miss=0.28, falarm=1.57, confusion=3.32
#Eval for threshold 0.45 DER=5.09, miss=0.36, falarm=1.41, confusion=3.31
#Eval for threshold 0.5 DER=5.06, miss=0.48, falarm=1.28, confusion=3.30
#Eval for threshold 0.55 DER=5.14, miss=0.65, falarm=1.21, confusion=3.28
#Eval for threshold 0.6 DER=5.25, miss=0.87, falarm=1.15, confusion=3.23
#Eval for threshold 0.7 DER=5.54, miss=1.35, falarm=1.02, confusion=3.16
#Eval for threshold 0.8 DER=6.27, miss=2.30, falarm=0.89, confusion=3.09
#Eval for threshold 0.9 DER=8.87, miss=5.20, falarm=0.73, confusion=2.93

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=9.26, miss=0.11, falarm=7.54, confusion=1.62
#Eval for threshold 0.3 DER=7.65, miss=0.30, falarm=5.35, confusion=2.00
#Eval for threshold 0.35 DER=7.04, miss=0.44, falarm=4.34, confusion=2.27
#Eval for threshold 0.4 DER=6.63, miss=0.61, falarm=3.56, confusion=2.47
#Eval for threshold 0.45 DER=6.36, miss=0.81, falarm=2.92, confusion=2.63
#Eval for threshold 0.5 DER=6.15, miss=1.13, falarm=2.20, confusion=2.82
#Eval for threshold 0.55 DER=6.58, miss=2.05, falarm=2.03, confusion=2.50
#Eval for threshold 0.6 DER=7.03, miss=2.84, falarm=1.92, confusion=2.27
#Eval for threshold 0.7 DER=8.49, miss=5.11, falarm=1.68, confusion=1.70
#Eval for threshold 0.8 DER=10.75, miss=7.96, falarm=1.41, confusion=1.38
#Eval for threshold 0.9 DER=15.20, miss=13.01, falarm=1.05, confusion=1.14

# cssd_testset of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=10.78, miss=1.43, falarm=8.49, confusion=0.86
#Eval for threshold 0.3 DER=8.09, miss=2.26, falarm=4.37, confusion=1.46
#Eval for threshold 0.35 DER=7.34, miss=2.84, falarm=2.79, confusion=1.71
#Eval for threshold 0.4 DER=7.01, miss=3.58, falarm=1.56, confusion=1.87 as report
#Eval for threshold 0.45 DER=7.19, miss=4.52, falarm=0.80, confusion=1.87
#Eval for threshold 0.5 DER=7.90, miss=5.85, falarm=0.36, confusion=1.69
#Eval for threshold 0.55 DER=8.94, miss=7.35, falarm=0.27, confusion=1.32
#Eval for threshold 0.6 DER=10.22, miss=9.02, falarm=0.21, confusion=0.99
#Eval for threshold 0.7 DER=13.16, miss=12.49, falarm=0.12, confusion=0.55
#Eval for threshold 0.8 DER=17.48, miss=17.11, falarm=0.06, confusion=0.31
#Eval for threshold 0.9 DER=26.56, miss=26.39, falarm=0.03, confusion=0.15



if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
   echo ""
   infer_sets="dev test cssd_testset"
   collar=0.0
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   for c in $collar;do
    for name  in $infer_sets;do
      for thr in $threshold;do
       echo "compute CDER on $c mode in $name dataset at $thr threshold"
       ref_rttm=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/$name/rttm_debug_nog0
       sys_rttm=/data/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state128_rs_len6/magicdata-ramc_eend_vc_system_collar${c}/$name/res_rttm_${thr} 
       python3 cder/score.py -r $ref_rttm -s $sys_rttm
     done
    done
  done
fi

grep -r 'Avg CDER' logs/run_ts_vad2_based_on_system_sad_stage5.log
# dev of magicdata-ramc, CDER
Avg CDER : 0.411
Avg CDER : 0.298
Avg CDER : 0.251
Avg CDER : 0.211
Avg CDER : 0.172
Avg CDER : 0.149
Avg CDER : 0.132
Avg CDER : 0.114
Avg CDER : 0.104
Avg CDER : 0.097
Avg CDER : 0.092

# test of magicdata-ramc, CDER
Avg CDER : 0.391
Avg CDER : 0.310
Avg CDER : 0.267
Avg CDER : 0.225
Avg CDER : 0.193
Avg CDER : 0.155
Avg CDER : 0.127
Avg CDER : 0.111
Avg CDER : 0.119
Avg CDER : 0.103
Avg CDER : 0.104

# cssd_testset of magicdata-ramc, CDER
Avg CDER : 0.274
Avg CDER : 0.208
Avg CDER : 0.178
Avg CDER : 0.150
Avg CDER : 0.128
Avg CDER : 0.109
Avg CDER : 0.100
Avg CDER : 0.095
Avg CDER : 0.088
Avg CDER : 0.107
Avg CDER : 0.134
