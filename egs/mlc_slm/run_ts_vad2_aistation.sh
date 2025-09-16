#!/usr/bin/env bash


stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_dia_cuda11.8_py3111_aistation.sh


if [ ${stage} -le -10 ] && [ ${stop_stage} -ge -10 ];then
   echo "prepared train set groundtruth rttm file and softlink audio"
   input_dir=/F00120240032/mlc-slm_task2/MLC-SLM_Workshop-Development_Set/MLC-SLM_Workshop-Development_Set/data
   out_dir=/maduo/datasets/mlc-slm/dev # gen train_rttm.list
     python3  fusion/prepare_train_and_dev_reference_files.py \
           --dataset_path $input_dir \
           --output_path $out_dir\
           --dataset_part dev
fi

if [ ${stage} -le -9 ] && [ ${stop_stage} -ge -9 ];then
   echo "prepared train set groundtruth rttm file and softlink audio"
   input_dir=/F00120240032/mlc-slm_task2/data/English
   out_dir=/maduo/datasets/mlc-slm/train # gen train_rttm.list  
     python3  fusion/prepare_train_and_dev_reference_files.py \
           --dataset_path $input_dir \
           --output_path $out_dir\
           --dataset_part train
fi


if [ ${stage} -le -8 ] && [ ${stop_stage} -ge -8 ];then
   echo "for english devset tsvad format"
   data_dir=/maduo/datasets/mlc-slm/dev

   for lid in American  Australian  British  Filipino  Indian;do
     grep -r $lid $data_dir/dev_rttm.list | xargs cat> $data_dir/English/$lid.rttm
   done
fi

if [ ${stage} -le -7 ] && [ ${stop_stage} -ge -7 ];then
   echo "for english trainset tsvad format"
   data_dir=/maduo/datasets/mlc-slm/train
   for lid in American_English  Australian_English  British_English  Filipino_English  Indian_English;do
     mkdir -p $data_dir/English
     grep -r $lid $data_dir/train_rttm.list | xargs cat> $data_dir/English/$lid.rttm
   done
fi


if [ ${stage} -le -6 ] && [ ${stop_stage} -ge -6 ];then
   data_dir=/maduo/datasets/mlc-slm/dev
   for lid in American  Australian  British  Filipino  Indian;do
     python3 ts_vad2/oracle_rttm_to_generate_target_speaker_wav_and_label_for_ts_vad.py\
             --oracle_rttm $data_dir/English/$lid.rttm\
             --audio_dir $data_dir/dev_wav/$lid\
	     --dest_dir $data_dir/English\
	     --language_name $lid\
	     --type dev
   done 
fi


if [ ${stage} -le -5 ] && [ ${stop_stage} -ge -5 ];then
   echo "for english trainset tsvad format"
   data_dir=/maduo/datasets/mlc-slm/train
   for lid in American_English  Australian_English  British_English  Filipino_English  Indian_English;do
     python3 ts_vad2/oracle_rttm_to_generate_target_speaker_wav_and_label_for_ts_vad.py\
             --oracle_rttm $data_dir/English/$lid.rttm\
             --audio_dir $data_dir/train_wav/$lid\
             --dest_dir $data_dir/English\
             --language_name $lid\
             --type train
   done
fi
if [ ${stage} -le -4 ] && [ ${stop_stage} -ge -4 ];then
   echo "generate oracle vad speaker embedding for devset"
   dest_dir=/maduo/model_hub
   feature_name=cam++_zh-cn_200k_feature_dir
   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
   model_id=iic/speech_campplus_sv_zh-cn_16k-common
   subsets="American  Australian  British  Filipino  Indian"
   #subsets="American"
   #subsets="Australian  British  Filipino  Indian"
   for name in $subsets;do
    #if [ $name = "Train" ];then
     #echo "extract Train settrain target speaker embedding"
     # 提取embedding
     #input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/${name}_Ali_far/target_audio/
     #wav_path=$input_dir/wavs.txt
     #else
     echo "extract $name target speaker embedding"
     # 提取embeddingdev/English
     #input_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc/${name}/target_audio/
     #wav_path=/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc/${name}/wavs.txt
     input_dir=/maduo/datasets/mlc-slm/dev/English/${name}/target_audio
     wav_path=/maduo/datasets/mlc-slm/dev/English/${name}/wavs.txt
     find $input_dir -name "*.wav" | grep -v "all.wav" >$wav_path
     head $wav_path
     save_dir=$dest_dir/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/dev/English/$name/$feature_name
     python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size 96
   done
fi




if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "generate oracle vad speaker embedding for devset"
   dest_dir=/maduo/model_hub 
   feature_name=cam++_en_zh_advanced_feature_dir
   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   #subsets="American  Australian  British  Filipino  Indian"
   #subsets="American"
   subsets="Australian  British  Filipino  Indian"
   for name in $subsets;do
    #if [ $name = "Train" ];then
     #echo "extract Train settrain target speaker embedding"
     # 提取embedding
     #input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/${name}_Ali_far/target_audio/
     #wav_path=$input_dir/wavs.txt
     #else
     echo "extract $name target speaker embedding"
     # 提取embeddingdev/English
     #input_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc/${name}/target_audio/
     #wav_path=/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc/${name}/wavs.txt
     input_dir=/maduo/datasets/mlc-slm/dev/English/${name}/target_audio
     wav_path=/maduo/datasets/mlc-slm/dev/English/${name}/wavs.txt
     find $input_dir -name "*.wav" | grep -v "all.wav" >$wav_path
     head $wav_path
     save_dir=$dest_dir/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/dev/English/$name/$feature_name
     python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size 96
   done
fi



# it is pretrained model from alimeeting dataset,
# Eval: 3.97(11.76)
# Test: 4.40(11.96)
# stage182-183 of run_ts_vad2_aistation.sh in egs/alimeeting
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
 #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8
 # note: here, it is wrong name in  exp_dir name , because _cam++_zh_200k_feature_dir should be cam++_en_zh_advanced_feature_dir
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr2e5_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=0.4
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="American  Australian  British  Filipino  Indian"
 #infer_sets="Test"
 rttm_dir=/maduo/datasets/mlc-slm/dev/English
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 #speech_encoder_type="CAM++"
 #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # w2v-bert2
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"


 dataset_name="mlc_slm" # dataset name
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/dev/English # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/maduo/datasets/mlc-slm/dev/English/" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    #data_dir="/maduo/datasets/mlc-slm/dev/English/" # oracle target audio , mix audio and labels path
    results_path=$exp_dir/${dataset_name}_collar${c}
  python3 ts_vad2/infer2.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
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
fi
#grep -r Eval logs/run_ts_vad2_aistation_stage3_3.log
# American, collar=0.0
#Eval for threshold 0.20: DER 39.92%, MS 19.84%, FA 15.85%, SC 4.22%
#Eval for threshold 0.30: DER 42.88%, MS 26.43%, FA 13.04%, SC 3.41%
#Eval for threshold 0.35: DER 44.22%, MS 29.30%, FA 11.95%, SC 2.97%
#Eval for threshold 0.40: DER 45.52%, MS 31.90%, FA 11.05%, SC 2.57%
#Eval for threshold 0.45: DER 46.83%, MS 34.44%, FA 10.11%, SC 2.29%
#Eval for threshold 0.50: DER 48.20%, MS 37.05%, FA 9.11%, SC 2.05%
#Eval for threshold 0.55: DER 49.78%, MS 39.64%, FA 8.29%, SC 1.85%
#Eval for threshold 0.60: DER 51.50%, MS 42.34%, FA 7.49%, SC 1.67%
#Eval for threshold 0.70: DER 55.87%, MS 48.74%, FA 5.73%, SC 1.40%
#Eval for threshold 0.80: DER 62.48%, MS 57.46%, FA 3.96%, SC 1.05%
# Australian , collar=0.0
#Eval for threshold 0.20: DER 60.67%, MS 53.22%, FA 4.25%, SC 3.21%
#Eval for threshold 0.30: DER 66.24%, MS 60.30%, FA 3.06%, SC 2.88%
#Eval for threshold 0.35: DER 68.57%, MS 63.31%, FA 2.60%, SC 2.67%
#Eval for threshold 0.40: DER 70.54%, MS 65.84%, FA 2.20%, SC 2.51%
#Eval for threshold 0.45: DER 72.19%, MS 67.97%, FA 1.85%, SC 2.36%
#Eval for threshold 0.50: DER 73.83%, MS 70.03%, FA 1.62%, SC 2.18%
#Eval for threshold 0.55: DER 75.32%, MS 71.94%, FA 1.37%, SC 2.00%
#Eval for threshold 0.60: DER 76.82%, MS 73.76%, FA 1.18%, SC 1.88%
#Eval for threshold 0.70: DER 80.01%, MS 77.65%, FA 0.87%, SC 1.48%
#Eval for threshold 0.80: DER 84.52%, MS 82.88%, FA 0.54%, SC 1.10%

# British  , collar=0.0
#Eval for threshold 0.20: DER 60.47%, MS 48.46%, FA 7.81%, SC 4.20%
#Eval for threshold 0.30: DER 62.51%, MS 52.72%, FA 6.42%, SC 3.37%
#Eval for threshold 0.35: DER 63.55%, MS 54.65%, FA 5.95%, SC 2.95%
#Eval for threshold 0.40: DER 64.62%, MS 56.55%, FA 5.51%, SC 2.55%
#Eval for threshold 0.45: DER 65.96%, MS 58.70%, FA 5.17%, SC 2.09%
#Eval for threshold 0.50: DER 67.78%, MS 61.30%, FA 4.78%, SC 1.70%
#Eval for threshold 0.55: DER 69.59%, MS 63.94%, FA 4.26%, SC 1.40%
#Eval for threshold 0.60: DER 71.50%, MS 66.61%, FA 3.78%, SC 1.11%
#Eval for threshold 0.70: DER 76.50%, MS 72.88%, FA 2.85%, SC 0.77%
#Eval for threshold 0.80: DER 84.20%, MS 81.79%, FA 1.89%, SC 0.53%

# Filipino  , collar=0.0
#Eval for threshold 0.20: DER 42.82%, MS 33.12%, FA 5.02%, SC 4.69%
#Eval for threshold 0.30: DER 52.88%, MS 46.81%, FA 2.51%, SC 3.56%
#Eval for threshold 0.35: DER 57.70%, MS 53.14%, FA 1.98%, SC 2.58%
#Eval for threshold 0.40: DER 62.48%, MS 59.10%, FA 1.61%, SC 1.76%
#Eval for threshold 0.45: DER 66.52%, MS 64.11%, FA 1.32%, SC 1.09%
#Eval for threshold 0.50: DER 70.56%, MS 68.91%, FA 1.09%, SC 0.56%
#Eval for threshold 0.55: DER 73.89%, MS 72.64%, FA 0.91%, SC 0.34%
#Eval for threshold 0.60: DER 76.82%, MS 75.85%, FA 0.75%, SC 0.22%
#Eval for threshold 0.70: DER 81.79%, MS 81.20%, FA 0.49%, SC 0.10%
#Eval for threshold 0.80: DER 85.96%, MS 85.61%, FA 0.31%, SC 0.04%

# Indian,, collar=0.0
#Eval for threshold 0.20: DER 22.73%, MS 16.37%, FA 5.30%, SC 1.05%
#Eval for threshold 0.30: DER 26.75%, MS 21.62%, FA 4.30%, SC 0.83%
#Eval for threshold 0.35: DER 28.85%, MS 24.19%, FA 3.91%, SC 0.75%
#Eval for threshold 0.40: DER 30.90%, MS 26.63%, FA 3.57%, SC 0.70%
#Eval for threshold 0.45: DER 33.07%, MS 29.15%, FA 3.27%, SC 0.65%
#Eval for threshold 0.50: DER 35.47%, MS 31.90%, FA 2.94%, SC 0.62%
#Eval for threshold 0.55: DER 38.25%, MS 35.03%, FA 2.66%, SC 0.56%
#Eval for threshold 0.60: DER 41.03%, MS 38.19%, FA 2.34%, SC 0.50%
#Eval for threshold 0.70: DER 47.08%, MS 44.90%, FA 1.75%, SC 0.43%
#Eval for threshold 0.80: DER 56.04%, MS 54.62%, FA 1.12%, SC 0.30%

 # American, collar=0.25
#Eval for threshold 0.20: DER 34.31%, MS 18.07%, FA 12.21%, SC 4.03%
#Eval for threshold 0.30: DER 37.76%, MS 24.50%, FA 10.08%, SC 3.18%
#Eval for threshold 0.35: DER 39.35%, MS 27.31%, FA 9.30%, SC 2.74%
#Eval for threshold 0.40: DER 40.87%, MS 29.85%, FA 8.65%, SC 2.37%
#Eval for threshold 0.45: DER 42.36%, MS 32.30%, FA 7.96%, SC 2.11%
#Eval for threshold 0.50: DER 43.93%, MS 34.85%, FA 7.21%, SC 1.87%
#Eval for threshold 0.55: DER 45.71%, MS 37.43%, FA 6.61%, SC 1.68%
#Eval for threshold 0.60: DER 47.64%, MS 40.10%, FA 6.02%, SC 1.52%
#Eval for threshold 0.70: DER 52.41%, MS 46.34%, FA 4.75%, SC 1.31%
#Eval for threshold 0.80: DER 59.42%, MS 54.99%, FA 3.41%, SC 1.02%

# Australian , collar=0.25
#Eval for threshold 0.20: DER 58.13%, MS 53.51%, FA 2.46%, SC 2.15%
#Eval for threshold 0.30: DER 64.17%, MS 60.49%, FA 1.75%, SC 1.92%
#Eval for threshold 0.35: DER 66.63%, MS 63.38%, FA 1.43%, SC 1.81%
#Eval for threshold 0.40: DER 68.74%, MS 65.85%, FA 1.18%, SC 1.71%
#Eval for threshold 0.45: DER 70.50%, MS 67.91%, FA 0.96%, SC 1.63%
#Eval for threshold 0.50: DER 72.25%, MS 69.91%, FA 0.80%, SC 1.54%
#Eval for threshold 0.55: DER 73.84%, MS 71.76%, FA 0.65%, SC 1.44%
#Eval for threshold 0.60: DER 75.41%, MS 73.49%, FA 0.53%, SC 1.38%
#Eval for threshold 0.70: DER 78.73%, MS 77.24%, FA 0.38%, SC 1.11%
#Eval for threshold 0.80: DER 83.45%, MS 82.38%, FA 0.23%, SC 0.84%

# British  , collar=0.25
#Eval for threshold 0.20: DER 56.86%, MS 48.10%, FA 4.59%, SC 4.17%
#Eval for threshold 0.30: DER 59.27%, MS 52.15%, FA 3.79%, SC 3.32%
#Eval for threshold 0.35: DER 60.49%, MS 54.06%, FA 3.55%, SC 2.88%
#Eval for threshold 0.40: DER 61.76%, MS 55.96%, FA 3.33%, SC 2.48%
#Eval for threshold 0.45: DER 63.27%, MS 58.11%, FA 3.16%, SC 2.00%
#Eval for threshold 0.50: DER 65.22%, MS 60.67%, FA 2.96%, SC 1.59%
#Eval for threshold 0.55: DER 67.21%, MS 63.24%, FA 2.68%, SC 1.28%
#Eval for threshold 0.60: DER 69.27%, MS 65.84%, FA 2.44%, SC 0.99%
#Eval for threshold 0.70: DER 74.78%, MS 72.10%, FA 2.00%, SC 0.67%
#Eval for threshold 0.80: DER 83.10%, MS 81.19%, FA 1.44%, SC 0.47%

# Filipino  , collar=0.25
#Eval for threshold 0.20: DER 39.19%, MS 32.00%, FA 2.46%, SC 4.72%
#Eval for threshold 0.30: DER 50.18%, MS 45.93%, FA 0.69%, SC 3.56%
#Eval for threshold 0.35: DER 55.36%, MS 52.36%, FA 0.46%, SC 2.55%
#Eval for threshold 0.40: DER 60.54%, MS 58.48%, FA 0.34%, SC 1.72%
#Eval for threshold 0.45: DER 64.89%, MS 63.55%, FA 0.27%, SC 1.07%
#Eval for threshold 0.50: DER 69.14%, MS 68.41%, FA 0.21%, SC 0.52%
#Eval for threshold 0.55: DER 72.66%, MS 72.17%, FA 0.17%, SC 0.32%
#Eval for threshold 0.60: DER 75.77%, MS 75.46%, FA 0.11%, SC 0.20%
#Eval for threshold 0.70: DER 80.96%, MS 80.82%, FA 0.05%, SC 0.09%
#Eval for threshold 0.80: DER 85.19%, MS 85.13%, FA 0.01%, SC 0.04%

# Indian,, collar=0.25
#Eval for threshold 0.20: DER 17.91%, MS 15.59%, FA 1.43%, SC 0.89%
#Eval for threshold 0.30: DER 22.39%, MS 20.76%, FA 0.98%, SC 0.65%
#Eval for threshold 0.35: DER 24.66%, MS 23.24%, FA 0.84%, SC 0.58%
#Eval for threshold 0.40: DER 26.87%, MS 25.62%, FA 0.74%, SC 0.51%
#Eval for threshold 0.45: DER 29.20%, MS 28.11%, FA 0.63%, SC 0.47%
#Eval for threshold 0.50: DER 31.78%, MS 30.82%, FA 0.52%, SC 0.45%
#Eval for threshold 0.55: DER 34.76%, MS 33.92%, FA 0.44%, SC 0.40%
#Eval for threshold 0.60: DER 37.69%, MS 36.97%, FA 0.36%, SC 0.36%
#Eval for threshold 0.70: DER 44.11%, MS 43.56%, FA 0.26%, SC 0.29%
#Eval for threshold 0.80: DER 53.57%, MS 53.19%, FA 0.16%, SC 0.21%



# cam++_zh-cn_200k_feature_dir
# this model is from stage128-130 of run_ts_vad2_hltsz_4090.sh in egs/magicdata-ramc
# dev of magicdata-ramc: 5.11(11.34)CDER=9.9
# test of magicdata-ramc: 5.92(12.11)CDER=12.6
# cssd_testset of magicdata-ramc: 5.91(20.78)CDER=8.6
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
 #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_shift0.8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=0.8
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="American  Australian  British  Filipino  Indian"
 #infer_sets="Test"
 rttm_dir=/maduo/datasets/mlc-slm/dev/English
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 #speech_encoder_type="CAM++"
 #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # w2v-bert2
 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # cam++ zh_200k
 speech_encoder_type="CAM++"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 speech_encoder_config=" /maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/configuration.json"


 dataset_name="mlc_slm" # dataset name
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/dev/English # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/maduo/datasets/mlc-slm/dev/English/" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    #data_dir="/maduo/datasets/mlc-slm/dev/English/" # oracle target audio , mix audio and labels path
    results_path=$exp_dir/${dataset_name}_collar${c}
  python3 ts_vad2/infer2.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
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
fi
#grep -r Eval logs/run_ts_vad2_aistation_stage4_1.log
# American,collar=0.0
#Eval for threshold 0.20: DER 26.16%, MS 3.56%, FA 20.66%, SC 1.95%
#Eval for threshold 0.30: DER 21.52%, MS 4.88%, FA 14.26%, SC 2.38%
#Eval for threshold 0.35: DER 20.78%, MS 5.80%, FA 12.50%, SC 2.48%
#Eval for threshold 0.40: DER 20.56%, MS 6.94%, FA 11.06%, SC 2.56% as report
#Eval for threshold 0.45: DER 21.02%, MS 8.66%, FA 9.96%, SC 2.39%
#Eval for threshold 0.50: DER 21.68%, MS 10.54%, FA 9.03%, SC 2.10%
#Eval for threshold 0.55: DER 23.15%, MS 13.04%, FA 8.34%, SC 1.78%
#Eval for threshold 0.60: DER 25.02%, MS 15.83%, FA 7.65%, SC 1.54%
#Eval for threshold 0.70: DER 30.14%, MS 22.69%, FA 6.37%, SC 1.07%
#Eval for threshold 0.80: DER 38.07%, MS 32.62%, FA 4.87%, SC 0.58%

# Australian,collar=0.0
#Eval for threshold 0.20: DER 32.98%, MS 4.39%, FA 26.08%, SC 2.50%
#Eval for threshold 0.30: DER 27.52%, MS 6.65%, FA 17.89%, SC 2.98%
#Eval for threshold 0.35: DER 24.42%, MS 8.46%, FA 12.88%, SC 3.08%
#Eval for threshold 0.40: DER 21.80%, MS 10.57%, FA 7.73%, SC 3.50%
#Eval for threshold 0.45: DER 21.39%, MS 14.28%, FA 3.59%, SC 3.52% as report
#Eval for threshold 0.50: DER 24.44%, MS 19.67%, FA 2.18%, SC 2.60%
#Eval for threshold 0.55: DER 28.40%, MS 24.77%, FA 1.82%, SC 1.80%
#Eval for threshold 0.60: DER 32.11%, MS 29.18%, FA 1.56%, SC 1.37%
#Eval for threshold 0.70: DER 38.95%, MS 36.99%, FA 1.10%, SC 0.86%
#Eval for threshold 0.80: DER 45.98%, MS 44.68%, FA 0.76%, SC 0.54%

# British ,collar=0.0
#Eval for threshold 0.20: DER 30.87%, MS 3.89%, FA 23.82%, SC 3.15%
#Eval for threshold 0.30: DER 24.34%, MS 6.85%, FA 13.94%, SC 3.55%
#Eval for threshold 0.35: DER 22.72%, MS 8.52%, FA 10.39%, SC 3.81%
#Eval for threshold 0.40: DER 22.10%, MS 10.60%, FA 7.40%, SC 4.10% as report
#Eval for threshold 0.45: DER 22.80%, MS 13.31%, FA 5.37%, SC 4.12%
#Eval for threshold 0.50: DER 24.72%, MS 16.80%, FA 4.24%, SC 3.67%
#Eval for threshold 0.55: DER 27.48%, MS 20.95%, FA 3.74%, SC 2.79%
#Eval for threshold 0.60: DER 30.87%, MS 25.33%, FA 3.27%, SC 2.26%
#Eval for threshold 0.70: DER 38.51%, MS 34.58%, FA 2.34%, SC 1.59%
#Eval for threshold 0.80: DER 46.68%, MS 43.94%, FA 1.57%, SC 1.16%

# Filipino  collar=0.0
#Eval for threshold 0.20: DER 25.78%, MS 2.40%, FA 23.03%, SC 0.35%
#Eval for threshold 0.30: DER 21.18%, MS 3.09%, FA 17.09%, SC 1.00%
#Eval for threshold 0.35: DER 18.74%, MS 3.49%, FA 13.23%, SC 2.02%
#Eval for threshold 0.40: DER 16.25%, MS 4.13%, FA 8.78%, SC 3.34%
#Eval for threshold 0.45: DER 14.51%, MS 5.50%, FA 4.58%, SC 4.43% as report
#Eval for threshold 0.50: DER 15.78%, MS 9.90%, FA 2.20%, SC 3.68%
#Eval for threshold 0.55: DER 19.12%, MS 15.18%, FA 1.92%, SC 2.01%
#Eval for threshold 0.60: DER 22.18%, MS 19.40%, FA 1.67%, SC 1.11%
#Eval for threshold 0.70: DER 28.23%, MS 26.68%, FA 1.23%, SC 0.32%
#Eval for threshold 0.80: DER 37.55%, MS 36.50%, FA 0.86%, SC 0.19%

# Indian，collar=0.0
#Eval for threshold 0.20: DER 10.80%, MS 1.65%, FA 8.75%, SC 0.39%
#Eval for threshold 0.30: DER 8.67%, MS 2.31%, FA 5.78%, SC 0.58%
#Eval for threshold 0.35: DER 8.31%, MS 2.72%, FA 4.90%, SC 0.70%
#Eval for threshold 0.40: DER 8.19%, MS 3.22%, FA 4.17%, SC 0.79% as report
#Eval for threshold 0.45: DER 8.28%, MS 3.90%, FA 3.58%, SC 0.80%
#Eval for threshold 0.50: DER 8.67%, MS 4.84%, FA 3.06%, SC 0.77%
#Eval for threshold 0.55: DER 9.33%, MS 5.98%, FA 2.80%, SC 0.55%
#Eval for threshold 0.60: DER 10.48%, MS 7.47%, FA 2.59%, SC 0.43%
#Eval for threshold 0.70: DER 13.67%, MS 11.27%, FA 2.16%, SC 0.25%
#Eval for threshold 0.80: DER 19.84%, MS 18.15%, FA 1.59%, SC 0.09%

 # American,collar=0.25
#Eval for threshold 0.20: DER 21.33%, MS 2.30%, FA 17.37%, SC 1.66%
#Eval for threshold 0.30: DER 17.04%, MS 3.14%, FA 11.79%, SC 2.11%
#Eval for threshold 0.35: DER 16.27%, MS 3.77%, FA 10.29%, SC 2.22%
#Eval for threshold 0.40: DER 16.03%, MS 4.64%, FA 9.06%, SC 2.34% as report
#Eval for threshold 0.45: DER 16.39%, MS 6.08%, FA 8.12%, SC 2.19%
#Eval for threshold 0.50: DER 17.06%, MS 7.70%, FA 7.38%, SC 1.98%
#Eval for threshold 0.55: DER 18.46%, MS 9.95%, FA 6.83%, SC 1.67%
#Eval for threshold 0.60: DER 20.26%, MS 12.50%, FA 6.26%, SC 1.50%
#Eval for threshold 0.70: DER 25.26%, MS 18.86%, FA 5.34%, SC 1.06%
#Eval for threshold 0.80: DER 33.20%, MS 28.47%, FA 4.14%, SC 0.59%

# # Australian,collar=0.25
#Eval for threshold 0.20: DER 28.18%, MS 4.21%, FA 22.07%, SC 1.90%
#Eval for threshold 0.30: DER 23.56%, MS 6.13%, FA 15.18%, SC 2.25%
#Eval for threshold 0.35: DER 20.81%, MS 7.71%, FA 10.79%, SC 2.30%
#Eval for threshold 0.40: DER 18.26%, MS 9.55%, FA 5.93%, SC 2.78%
#Eval for threshold 0.45: DER 17.74%, MS 12.97%, FA 1.82%, SC 2.95% as report
#Eval for threshold 0.50: DER 20.95%, MS 18.25%, FA 0.63%, SC 2.07%
#Eval for threshold 0.55: DER 25.15%, MS 23.33%, FA 0.49%, SC 1.32%
#Eval for threshold 0.60: DER 28.84%, MS 27.43%, FA 0.40%, SC 1.01%
#Eval for threshold 0.70: DER 35.84%, MS 34.92%, FA 0.27%, SC 0.65%
#Eval for threshold 0.80: DER 42.86%, MS 42.26%, FA 0.17%, SC 0.42%

# # British ,collar=0.25
#Eval for threshold 0.20: DER 25.46%, MS 3.48%, FA 19.20%, SC 2.78%
#Eval for threshold 0.30: DER 19.57%, MS 6.14%, FA 10.28%, SC 3.15%
#Eval for threshold 0.35: DER 18.18%, MS 7.65%, FA 7.13%, SC 3.40%
#Eval for threshold 0.40: DER 17.70%, MS 9.57%, FA 4.37%, SC 3.76% as report
#Eval for threshold 0.45: DER 18.60%, MS 12.12%, FA 2.60%, SC 3.88%
#Eval for threshold 0.50: DER 20.67%, MS 15.42%, FA 1.79%, SC 3.46%
#Eval for threshold 0.55: DER 23.61%, MS 19.39%, FA 1.58%, SC 2.64%
#Eval for threshold 0.60: DER 27.22%, MS 23.67%, FA 1.40%, SC 2.15%
#Eval for threshold 0.70: DER 35.21%, MS 32.65%, FA 1.00%, SC 1.55%
#Eval for threshold 0.80: DER 43.67%, MS 41.84%, FA 0.66%, SC 1.17%

# Filipino  collar=0.25
#Eval for threshold 0.20: DER 21.54%, MS 2.27%, FA 19.02%, SC 0.25%
#Eval for threshold 0.30: DER 17.78%, MS 2.81%, FA 14.09%, SC 0.89%
#Eval for threshold 0.35: DER 15.56%, MS 3.13%, FA 10.53%, SC 1.90%
#Eval for threshold 0.40: DER 13.31%, MS 3.63%, FA 6.44%, SC 3.23% 
#Eval for threshold 0.45: DER 11.69%, MS 4.85%, FA 2.48%, SC 4.36% as report
#Eval for threshold 0.50: DER 13.12%, MS 9.13%, FA 0.34%, SC 3.65%
#Eval for threshold 0.55: DER 16.61%, MS 14.36%, FA 0.27%, SC 1.99%
#Eval for threshold 0.60: DER 19.75%, MS 18.46%, FA 0.21%, SC 1.08%
#Eval for threshold 0.70: DER 26.02%, MS 25.57%, FA 0.16%, SC 0.30%
#Eval for threshold 0.80: DER 35.52%, MS 35.26%, FA 0.10%, SC 0.16%

# Indian，collar=0.25
#Eval for threshold 0.20: DER 6.45%, MS 1.53%, FA 4.65%, SC 0.26%
#Eval for threshold 0.30: DER 4.69%, MS 2.00%, FA 2.29%, SC 0.40%
#Eval for threshold 0.35: DER 4.41%, MS 2.28%, FA 1.61%, SC 0.53%
#Eval for threshold 0.40: DER 4.40%, MS 2.69%, FA 1.09%, SC 0.62% as report
#Eval for threshold 0.45: DER 4.58%, MS 3.23%, FA 0.68%, SC 0.67%
#Eval for threshold 0.50: DER 5.06%, MS 4.02%, FA 0.38%, SC 0.66%
#Eval for threshold 0.55: DER 5.82%, MS 5.01%, FA 0.32%, SC 0.48%
#Eval for threshold 0.60: DER 6.99%, MS 6.32%, FA 0.29%, SC 0.38%
#Eval for threshold 0.70: DER 10.33%, MS 9.87%, FA 0.24%, SC 0.23%
#Eval for threshold 0.80: DER 16.63%, MS 16.39%, FA 0.16%, SC 0.08%


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
    echo "meger train english subset"
    python  ts_vad2/merge_audio_and_label.py\
	     --dataset-path /maduo/datasets/mlc-slm/train/English \
	     --output-dir /maduo/datasets/mlc-slm/train/train_english/target_audio \
	     --output-json-dir /maduo/datasets/mlc-slm/train/train_english\
	     --part train

fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "generate oracle vad speaker embedding for english trainset (megered)"
   dest_dir=/maduo/model_hub
   feature_name=cam++_en_zh_advanced_feature_dir
   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   #subsets="American  Australian  British  Filipino  Indian"
   #subsets="American"
   subsets="train_english"
   for name in $subsets;do
    #if [ $name = "Train" ];then
     #echo "extract Train settrain target speaker embedding"
     # 提取embedding
     #input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/${name}_Ali_far/target_audio/
     #wav_path=$input_dir/wavs.txt
     #else
     echo "extract $name target speaker embedding"
     # 提取embeddingdev/English
     #input_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc/${name}/target_audio/
     #wav_path=/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc/${name}/wavs.txt
     input_dir=/maduo/datasets/mlc-slm/train/${name}/target_audio
     wav_path=/maduo/datasets/mlc-slm/train/${name}/wavs.txt
     find $input_dir -name "*.wav" | grep -v "all.wav" >$wav_path
     head $wav_path
     save_dir=$dest_dir/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/train/$name/$feature_name
     python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size 96
   done
fi



if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
    echo "meger dev english subset"
    python  ts_vad2/merge_audio_and_label.py\
             --dataset-path /maduo/datasets//mlc-slm/dev/English \
             --output-dir /maduo/datasets/mlc-slm/dev/dev_english/target_audio \
             --output-json-dir /maduo/datasets/mlc-slm/dev/dev_english\
	     --part dev

fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
    echo "meger dev english subset cam++_en_zh_advanced_feature_dir embedding"
    python  ts_vad2/merge_embedding.py\
             --in-embed-dir  /maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/dev/English \
             --out-embed-dir  /maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/dev/dev_english\
             --embeded-feat-name cam++_en_zh_advanced_feature_dir 
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
    echo "meger dev english subset cam++_en_zh_advanced_feature_dir embedding"
    python  ts_vad2/merge_embedding.py\
             --in-embed-dir  /maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/dev/English \
             --out-embed-dir  /maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/dev/dev_english\
             --embeded-feat-name cam++_zh-cn_200k_feature_dir
fi

# - Epoch 6, batch_idx_train: 262946, num_updates: 225000, {'loss': nan, 'DER': 1.0, 'ACC': np.float64(0.7765625), 'MI': 1.0, 'FA': 0.0, 'CF': 0.0}, batch size: 64, grad_norm: nan, grad_scale: , lr: 1.0886666666666667e-05,
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/maduo/datasets/musan
    rir_path=/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    # cam++_zh_200k
    #speech_encoder_type="CAM++"
    #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    # w2v-bert2
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    dataset_name="mlc_slm" # dataset name

    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/ # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8 #
    #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr5e5_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.4
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_mlc_slm_english_two_gpus_freeze_with_musan_rirs_cam++_en_zh_advanced_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr2e5_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.4
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/mlc-slm" # oracle target audio , mix audio and labels path
    rs_len=8
    segment_shift=0.4
    single_backend_type="transformer"
    multi_backend_type="transformer"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 16115 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --start-batch 160500\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-5\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
 #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_mlc_slm_english_two_gpus_freeze_with_musan_rirs_cam++_en_zh_advanced_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr2e5_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=0.4
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="American  Australian  British  Filipino  Indian"
 #infer_sets="Test"
 rttm_dir=/maduo/datasets/mlc-slm/dev/English
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 #speech_encoder_type="CAM++"
 #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # w2v-bert2
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"


 dataset_name="mlc_slm" # dataset name
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/dev/English # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/maduo/datasets/mlc-slm/dev/English/" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    #data_dir="/maduo/datasets/mlc-slm/dev/English/" # oracle target audio , mix audio and labels path
    results_path=$exp_dir/${dataset_name}_collar${c}
  python3 ts_vad2/infer2.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
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
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
   echo "generate oracle vad speaker embedding for english trainset (megered)"
   dest_dir=/maduo/model_hub
   feature_name=cam++_zh-cn_200k_feature_dir
   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
   model_id=iic/speech_campplus_sv_zh-cn_16k-common
   #subsets="American  Australian  British  Filipino  Indian"
   #subsets="American"
   subsets="train_english"
   for name in $subsets;do
    #if [ $name = "Train" ];then
     #echo "extract Train settrain target speaker embedding"
     # 提取embedding
     #input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/${name}_Ali_far/target_audio/
     #wav_path=$input_dir/wavs.txt
     #else
     echo "extract $name target speaker embedding"
     # 提取embeddingdev/English
     #input_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc/${name}/target_audio/
     #wav_path=/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc/${name}/wavs.txt
     input_dir=/maduo/datasets/mlc-slm/train/${name}/target_audio
     wav_path=/maduo/datasets/mlc-slm/train/${name}/wavs.txt
     find $input_dir -name "*.wav" | grep -v "all.wav" >$wav_path
     head $wav_path
     save_dir=$dest_dir/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/train/$name/$feature_name
     python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size 96
   done
fi

# stage13-14 is same as setting of stage 4 , however,stage13-14 will use mlc-slm  english data to train new model from scratch.
if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/maduo/datasets/musan
    rir_path=/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    # w2v-bert2
    #speech_encoder_type="w2v-bert2"
    #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    #speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    
    # cam++ zh_200k
    speech_encoder_type="CAM++"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    speech_encoder_config=" /maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/configuration.json"
    
    dataset_name="mlc_slm" # dataset name

    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/ # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_mlc_slm_english_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e-4_single_backend_transformer_multi_backend_transformer_rs_len6_shift0.8
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/mlc-slm" # oracle target audio , mix audio and labels path
    rs_len=6
    segment_shift=0.8
    single_backend_type="transformer"
    multi_backend_type="transformer"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 16215 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --start-batch 301500\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_mlc_slm_english_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e-4_single_backend_transformer_multi_backend_transformer_rs_len6_shift0.8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=0.8
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="American  Australian  British  Filipino  Indian"
 #infer_sets="Test"
 rttm_dir=/maduo/datasets/mlc-slm/dev/English
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 #speech_encoder_type="CAM++"
 #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # w2v-bert2
 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # cam++ zh_200k
 speech_encoder_type="CAM++"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 speech_encoder_config=" /maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/configuration.json"


 dataset_name="mlc_slm" # dataset name
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/dev/English # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/maduo/datasets/mlc-slm/dev/English/" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    #data_dir="/maduo/datasets/mlc-slm/dev/English/" # oracle target audio , mix audio and labels path
    results_path=$exp_dir/${dataset_name}_collar${c}
  python3 ts_vad2/infer2.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
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
fi

# grep -r Eval logs/run_ts_vad2_aistation_stage13_14_4.log
# American, collar=0.0
#Eval for threshold 0.20: DER 26.08%, MS 0.20%, FA 24.05%, SC 1.83%
#Eval for threshold 0.30: DER 21.56%, MS 0.57%, FA 18.88%, SC 2.11%
#Eval for threshold 0.35: DER 20.15%, MS 0.85%, FA 17.06%, SC 2.24%
#Eval for threshold 0.40: DER 18.91%, MS 1.18%, FA 15.37%, SC 2.35%
#Eval for threshold 0.45: DER 18.04%, MS 1.63%, FA 13.95%, SC 2.45%
#Eval for threshold 0.50: DER 17.46%, MS 2.24%, FA 12.83%, SC 2.39%
#Eval for threshold 0.55: DER 17.18%, MS 3.04%, FA 11.87%, SC 2.26%
#Eval for threshold 0.60: DER 16.96%, MS 4.03%, FA 10.90%, SC 2.03% as report
#Eval for threshold 0.70: DER 17.25%, MS 6.54%, FA 8.98%, SC 1.73%
#Eval for threshold 0.80: DER 18.99%, MS 10.59%, FA 6.86%, SC 1.53%

# Australian, collar=0.0
#Eval for threshold 0.20: DER 14.41%, MS 1.76%, FA 10.69%, SC 1.96%
#Eval for threshold 0.30: DER 12.53%, MS 3.16%, FA 7.26%, SC 2.11%
#Eval for threshold 0.35: DER 12.23%, MS 3.94%, FA 6.20%, SC 2.09%
#Eval for threshold 0.40: DER 12.06%, MS 4.87%, FA 5.17%, SC 2.01% as report
#Eval for threshold 0.45: DER 12.32%, MS 5.90%, FA 4.56%, SC 1.85%
#Eval for threshold 0.50: DER 12.74%, MS 6.98%, FA 4.09%, SC 1.67%
#Eval for threshold 0.55: DER 13.50%, MS 8.32%, FA 3.72%, SC 1.46%
#Eval for threshold 0.60: DER 14.23%, MS 9.60%, FA 3.36%, SC 1.27%
#Eval for threshold 0.70: DER 16.25%, MS 12.64%, FA 2.58%, SC 1.03%
#Eval for threshold 0.80: DER 19.84%, MS 17.06%, FA 1.97%, SC 0.81%

#British  ,collar=0.0
#Eval for threshold 0.20: DER 28.45%, MS 0.22%, FA 25.60%, SC 2.63%
#Eval for threshold 0.30: DER 22.96%, MS 0.56%, FA 19.42%, SC 2.98%
#Eval for threshold 0.35: DER 20.28%, MS 0.74%, FA 16.44%, SC 3.10%
#Eval for threshold 0.40: DER 17.14%, MS 1.04%, FA 12.86%, SC 3.24%
#Eval for threshold 0.45: DER 14.45%, MS 1.39%, FA 9.48%, SC 3.57%
#Eval for threshold 0.50: DER 13.46%, MS 2.04%, FA 7.38%, SC 4.04% as report
#Eval for threshold 0.55: DER 14.34%, MS 4.29%, FA 6.72%, SC 3.34%
#Eval for threshold 0.60: DER 16.39%, MS 7.33%, FA 6.15%, SC 2.91%
#Eval for threshold 0.70: DER 20.72%, MS 13.18%, FA 5.04%, SC 2.50%
#Eval for threshold 0.80: DER 24.16%, MS 18.01%, FA 3.96%, SC 2.19%

# Filipino ,collar=0.0
#Eval for threshold 0.20: DER 28.81%, MS 0.17%, FA 28.44%, SC 0.20%
#Eval for threshold 0.30: DER 23.25%, MS 0.35%, FA 22.62%, SC 0.27%
#Eval for threshold 0.35: DER 20.92%, MS 0.51%, FA 19.93%, SC 0.48%
#Eval for threshold 0.40: DER 17.23%, MS 0.72%, FA 14.81%, SC 1.69%
#Eval for threshold 0.45: DER 13.85%, MS 0.90%, FA 8.77%, SC 4.18%
#Eval for threshold 0.50: DER 11.70%, MS 1.15%, FA 4.60%, SC 5.95% as report
#Eval for threshold 0.55: DER 12.47%, MS 3.60%, FA 3.84%, SC 5.02%
#Eval for threshold 0.60: DER 14.78%, MS 8.80%, FA 3.32%, SC 2.66%
#Eval for threshold 0.70: DER 20.67%, MS 17.88%, FA 2.56%, SC 0.23%
#Eval for threshold 0.80: DER 25.46%, MS 23.42%, FA 1.88%, SC 0.16%

# Indian, collar=0.0
#Eval for threshold 0.20: DER 10.77%, MS 0.19%, FA 10.34%, SC 0.24%
#Eval for threshold 0.30: DER 8.38%, MS 0.40%, FA 7.68%, SC 0.30%
#Eval for threshold 0.35: DER 7.45%, MS 0.54%, FA 6.58%, SC 0.33%
#Eval for threshold 0.40: DER 6.96%, MS 0.76%, FA 5.88%, SC 0.32%
#Eval for threshold 0.45: DER 6.66%, MS 0.99%, FA 5.35%, SC 0.32%
#Eval for threshold 0.50: DER 6.54%, MS 1.30%, FA 4.94%, SC 0.30% as report
#Eval for threshold 0.55: DER 6.61%, MS 1.78%, FA 4.59%, SC 0.24%
#Eval for threshold 0.60: DER 6.79%, MS 2.37%, FA 4.22%, SC 0.19%
#Eval for threshold 0.70: DER 7.81%, MS 4.17%, FA 3.54%, SC 0.11%
#Eval for threshold 0.80: DER 9.78%, MS 6.88%, FA 2.82%, SC 0.07%

# American , collar=0.25
#Eval for threshold 0.20: DER 19.35%, MS 0.07%, FA 17.61%, SC 1.67%
#Eval for threshold 0.30: DER 15.60%, MS 0.28%, FA 13.47%, SC 1.85%
#Eval for threshold 0.35: DER 14.36%, MS 0.41%, FA 11.98%, SC 1.97%
#Eval for threshold 0.40: DER 13.44%, MS 0.61%, FA 10.74%, SC 2.09%
#Eval for threshold 0.45: DER 12.77%, MS 0.89%, FA 9.67%, SC 2.20%
#Eval for threshold 0.50: DER 12.36%, MS 1.30%, FA 8.87%, SC 2.19%
#Eval for threshold 0.55: DER 12.23%, MS 1.90%, FA 8.24%, SC 2.09%
#Eval for threshold 0.60: DER 12.15%, MS 2.63%, FA 7.59%, SC 1.92%
#Eval for threshold 0.70: DER 12.62%, MS 4.57%, FA 6.35%, SC 1.71%
#Eval for threshold 0.80: DER 14.26%, MS 7.79%, FA 4.91%, SC 1.57%

# Australian,  collar=0.25
#Eval for threshold 0.20: DER 7.91%, MS 1.72%, FA 4.84%, SC 1.36%
#Eval for threshold 0.30: DER 7.06%, MS 2.96%, FA 2.70%, SC 1.40%
#Eval for threshold 0.35: DER 7.12%, MS 3.64%, FA 2.12%, SC 1.36%
#Eval for threshold 0.40: DER 7.22%, MS 4.41%, FA 1.51%, SC 1.30%
#Eval for threshold 0.45: DER 7.66%, MS 5.25%, FA 1.23%, SC 1.18%
#Eval for threshold 0.50: DER 8.25%, MS 6.13%, FA 1.05%, SC 1.07%
#Eval for threshold 0.55: DER 9.03%, MS 7.15%, FA 0.93%, SC 0.95%
#Eval for threshold 0.60: DER 9.84%, MS 8.20%, FA 0.81%, SC 0.83%
#Eval for threshold 0.70: DER 11.94%, MS 10.65%, FA 0.56%, SC 0.73%
#Eval for threshold 0.80: DER 15.42%, MS 14.38%, FA 0.41%, SC 0.64%

# British,  collar=0.25
#Eval for threshold 0.20: DER 21.10%, MS 0.19%, FA 18.51%, SC 2.40%
#Eval for threshold 0.30: DER 16.57%, MS 0.47%, FA 13.42%, SC 2.68%
#Eval for threshold 0.35: DER 14.33%, MS 0.62%, FA 10.95%, SC 2.76%
#Eval for threshold 0.40: DER 11.52%, MS 0.85%, FA 7.77%, SC 2.89%
#Eval for threshold 0.45: DER 9.08%, MS 1.14%, FA 4.73%, SC 3.21%
#Eval for threshold 0.50: DER 8.28%, MS 1.65%, FA 2.91%, SC 3.72%
#Eval for threshold 0.55: DER 9.36%, MS 3.72%, FA 2.58%, SC 3.07%
#Eval for threshold 0.60: DER 11.65%, MS 6.63%, FA 2.31%, SC 2.70%
#Eval for threshold 0.70: DER 16.40%, MS 12.25%, FA 1.80%, SC 2.35%
#Eval for threshold 0.80: DER 20.24%, MS 16.76%, FA 1.36%, SC 2.12%

# Filipino, collar=0.25
#Eval for threshold 0.20: DER 22.15%, MS 0.15%, FA 21.82%, SC 0.18%
#Eval for threshold 0.30: DER 17.47%, MS 0.32%, FA 16.94%, SC 0.22%
#Eval for threshold 0.35: DER 15.51%, MS 0.47%, FA 14.62%, SC 0.42%
#Eval for threshold 0.40: DER 12.30%, MS 0.65%, FA 10.01%, SC 1.64%
#Eval for threshold 0.45: DER 9.46%, MS 0.83%, FA 4.51%, SC 4.12%
#Eval for threshold 0.50: DER 7.76%, MS 1.04%, FA 0.83%, SC 5.89%
#Eval for threshold 0.55: DER 8.81%, MS 3.44%, FA 0.39%, SC 4.98%
#Eval for threshold 0.60: DER 11.44%, MS 8.49%, FA 0.28%, SC 2.67%
#Eval for threshold 0.70: DER 17.85%, MS 17.49%, FA 0.16%, SC 0.20%
#Eval for threshold 0.80: DER 23.05%, MS 22.79%, FA 0.11%, SC 0.16%

# Indian, collar=0.25
#Eval for threshold 0.20: DER 4.65%, MS 0.14%, FA 4.38%, SC 0.13%
#Eval for threshold 0.30: DER 2.80%, MS 0.31%, FA 2.33%, SC 0.17%
#Eval for threshold 0.35: DER 2.16%, MS 0.41%, FA 1.55%, SC 0.20%
#Eval for threshold 0.40: DER 1.87%, MS 0.57%, FA 1.11%, SC 0.19%
#Eval for threshold 0.45: DER 1.76%, MS 0.75%, FA 0.81%, SC 0.20%
#Eval for threshold 0.50: DER 1.82%, MS 0.98%, FA 0.65%, SC 0.19%
#Eval for threshold 0.55: DER 2.06%, MS 1.35%, FA 0.56%, SC 0.15%
#Eval for threshold 0.60: DER 2.42%, MS 1.84%, FA 0.46%, SC 0.12%
#Eval for threshold 0.70: DER 3.76%, MS 3.37%, FA 0.32%, SC 0.07%
#Eval for threshold 0.80: DER 6.04%, MS 5.77%, FA 0.22%, SC 0.05%


# step=117000, loss is nan
if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/maduo/datasets/musan
    rir_path=/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    # cam++_zh_200k
    #speech_encoder_type="CAM++"
    #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    # w2v-bert2
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    dataset_name="mlc_slm" # dataset name

    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/ # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8 #
    #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr5e5_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.4
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_mlc_slm_english_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr2e5_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.4
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/mlc-slm" # oracle target audio , mix audio and labels path
    rs_len=8
    segment_shift=0.4
    single_backend_type="transformer"
    multi_backend_type="transformer"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 16315 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --start-batch 106500\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-5\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ];then
 #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_mlc_slm_english_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr2e5_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=0.4
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="American  Australian  British  Filipino  Indian"
 #infer_sets="Test"
 rttm_dir=/maduo/datasets/mlc-slm/dev/English
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 #speech_encoder_type="CAM++"
 #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # w2v-bert2
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"


 dataset_name="mlc_slm" # dataset name
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/dev/English # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/maduo/datasets/mlc-slm/dev/English/" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    #data_dir="/maduo/datasets/mlc-slm/dev/English/" # oracle target audio , mix audio and labels path
    results_path=$exp_dir/${dataset_name}_collar${c}
  python3 ts_vad2/infer2.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
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
fi



#- Epoch 9, batch_idx_train: 411000, num_updates: 341500, {'loss': nan, 'DER': 1.0, 'ACC': np.float64(0.7760546875), 'MI': 1.0, 'FA': 0.0, 'CF': 0.0}, batch size: 64, grad_norm: nan ,stop
if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/maduo/datasets/musan
    rir_path=/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    # w2v-bert2
    #speech_encoder_type="w2v-bert2"
    #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    #speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # cam++ zh_200k
    speech_encoder_type="CAM++"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    speech_encoder_config=" /maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/configuration.json"

    dataset_name="mlc_slm" # dataset name

    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/ # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_mlc_slm_english_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e-4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.4
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/mlc-slm" # oracle target audio , mix audio and labels path
    rs_len=8
    segment_shift=0.4
    single_backend_type="transformer"
    multi_backend_type="transformer"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 16415 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --start-batch 397500\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_mlc_slm_english_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e-4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=0.4
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="American  Australian  British  Filipino  Indian"
 #infer_sets="Test"
 rttm_dir=/maduo/datasets/mlc-slm/dev/English
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 #speech_encoder_type="CAM++"
 #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # w2v-bert2
 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # cam++ zh_200k
 speech_encoder_type="CAM++"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 speech_encoder_config=" /maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/configuration.json"


 dataset_name="mlc_slm" # dataset name
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/dev/English # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/maduo/datasets/mlc-slm/dev/English/" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    #data_dir="/maduo/datasets/mlc-slm/dev/English/" # oracle target audio , mix audio and labels path
    results_path=$exp_dir/${dataset_name}_collar${c}
  python3 ts_vad2/infer2.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
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
fi

if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/maduo/datasets/musan
    rir_path=/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    # w2v-bert2
    #speech_encoder_type="w2v-bert2"
    #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    #speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # cam++ zh_200k
    #speech_encoder_type="CAM++"
    #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config=" /maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/configuration.json"

    #cam++ zh_en
    speech_encoder_type="CAM++"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/configuration.json"
    dataset_name="mlc_slm" # dataset name

    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/ # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_mlc_slm_english_two_gpus_freeze_with_musan_rirs_cam++_en_zh_advanced_feature_dir_epoch20_front_fix_seed_lr2e-4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.4
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/mlc-slm" # oracle target audio , mix audio and labels path
    rs_len=8
    segment_shift=0.4
    single_backend_type="transformer"
    multi_backend_type="transformer"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 16515 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --start-batch 561000\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_mlc_slm_english_two_gpus_freeze_with_musan_rirs_cam++_en_zh_advanced_feature_dir_epoch20_front_fix_seed_lr2e-4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=0.4
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="American  Australian  British  Filipino  Indian"
 #infer_sets="Test"
 rttm_dir=/maduo/datasets/mlc-slm/dev/English
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 #speech_encoder_type="CAM++"
 #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # w2v-bert2
 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # cam++ zh_200k
 speech_encoder_type="CAM++"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 speech_encoder_config=" /maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/configuration.json"


 dataset_name="mlc_slm" # dataset name
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/dev/English # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/maduo/datasets/mlc-slm/dev/English/" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    #data_dir="/maduo/datasets/mlc-slm/dev/English/" # oracle target audio , mix audio and labels path
    results_path=$exp_dir/${dataset_name}_collar${c}
  python3 ts_vad2/infer2.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
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
fi





# stage21-22 is same as setting of stage13-14 , however,stage21-22 will use the mlc_slm data to finetune the model which is from pretrained on magicdata-ramc  
if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/maduo/datasets/musan
    rir_path=/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    # w2v-bert2
    #speech_encoder_type="w2v-bert2"
    #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    #speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # cam++ zh_200k
    speech_encoder_type="CAM++"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    speech_encoder_config=" /maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/configuration.json"

    dataset_name="mlc_slm" # dataset name

    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/ # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    # the model is pretrained from magicdata-ramc, more detail, you can see stage4  in this script.
    #source_dir=/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_shift0.8
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_shift0.8_ft_on_mlc_slm_lr1e-5
    
    #mkdir -p $exp_dir
    #cp -r $source_dir/best-valid-der.pt $exp_dir/
    #mv $exp_dir/best-valid-der.pt $exp_dir/epoch-0.pt

    data_dir="/maduo/datasets/mlc-slm" # oracle target audio , mix audio and labels path
    rs_len=6
    segment_shift=0.8
    single_backend_type="transformer"
    multi_backend_type="transformer"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 16615 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --start-batch 285000\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 1e-5\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_shift0.8_ft_on_mlc_slm_lr1e-5
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=0.8
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="American  Australian  British  Filipino  Indian"
 #infer_sets="Test"
 rttm_dir=/maduo/datasets/mlc-slm/dev/English
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 #speech_encoder_type="CAM++"
 #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # w2v-bert2
 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # cam++ zh_200k
 speech_encoder_type="CAM++"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 speech_encoder_config=" /maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/configuration.json"


 dataset_name="mlc_slm" # dataset name
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/dev/English # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/maduo/datasets/mlc-slm/dev/English/" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    #data_dir="/maduo/datasets/mlc-slm/dev/English/" # oracle target audio , mix audio and labels path
    results_path=$exp_dir/${dataset_name}_collar${c}
  python3 ts_vad2/infer2.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
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
fi
# grep -r Eval logs/run_ts_vad2_aistation_stage21_22_3.log
#Eval for threshold 0.20: DER 26.05%, MS 0.21%, FA 24.09%, SC 1.76%
#Eval for threshold 0.30: DER 21.78%, MS 0.60%, FA 19.19%, SC 1.99%
#Eval for threshold 0.35: DER 20.32%, MS 0.90%, FA 17.34%, SC 2.08%
#Eval for threshold 0.40: DER 19.14%, MS 1.27%, FA 15.71%, SC 2.16%
#Eval for threshold 0.45: DER 18.21%, MS 1.68%, FA 14.36%, SC 2.17%
#Eval for threshold 0.50: DER 17.48%, MS 2.33%, FA 13.03%, SC 2.12%
#Eval for threshold 0.55: DER 17.19%, MS 3.17%, FA 12.02%, SC 2.00%
#Eval for threshold 0.60: DER 17.05%, MS 4.22%, FA 10.95%, SC 1.88%
#Eval for threshold 0.70: DER 17.51%, MS 7.01%, FA 8.86%, SC 1.65%
#Eval for threshold 0.80: DER 19.71%, MS 11.66%, FA 6.68%, SC 1.37%
#Eval for threshold 0.20: DER 15.79%, MS 0.92%, FA 12.88%, SC 1.98%
#Eval for threshold 0.30: DER 12.17%, MS 2.37%, FA 7.73%, SC 2.07%
#Eval for threshold 0.35: DER 11.48%, MS 3.26%, FA 6.19%, SC 2.03%
#Eval for threshold 0.40: DER 11.43%, MS 4.25%, FA 5.26%, SC 1.92%
#Eval for threshold 0.45: DER 11.66%, MS 5.36%, FA 4.57%, SC 1.74%
#Eval for threshold 0.50: DER 12.21%, MS 6.63%, FA 4.02%, SC 1.56%
#Eval for threshold 0.55: DER 12.98%, MS 8.09%, FA 3.56%, SC 1.33%
#Eval for threshold 0.60: DER 13.96%, MS 9.66%, FA 3.13%, SC 1.18%
#Eval for threshold 0.70: DER 16.72%, MS 13.43%, FA 2.37%, SC 0.93%
#Eval for threshold 0.80: DER 21.63%, MS 19.18%, FA 1.70%, SC 0.75%
#Eval for threshold 0.20: DER 29.10%, MS 0.14%, FA 26.36%, SC 2.60%
#Eval for threshold 0.30: DER 20.86%, MS 0.47%, FA 17.39%, SC 3.00%
#Eval for threshold 0.35: DER 17.67%, MS 0.77%, FA 13.71%, SC 3.19%
#Eval for threshold 0.40: DER 15.40%, MS 1.12%, FA 10.87%, SC 3.41%
#Eval for threshold 0.45: DER 14.06%, MS 1.57%, FA 8.85%, SC 3.64%
#Eval for threshold 0.50: DER 13.26%, MS 2.25%, FA 7.09%, SC 3.92%
#Eval for threshold 0.55: DER 13.54%, MS 3.84%, FA 6.28%, SC 3.42%
#Eval for threshold 0.60: DER 14.36%, MS 5.59%, FA 5.68%, SC 3.09%
#Eval for threshold 0.70: DER 18.22%, MS 11.18%, FA 4.59%, SC 2.45%
#Eval for threshold 0.80: DER 24.43%, MS 18.88%, FA 3.43%, SC 2.11%
#Eval for threshold 0.20: DER 28.28%, MS 0.10%, FA 27.97%, SC 0.21%
#Eval for threshold 0.30: DER 22.76%, MS 0.27%, FA 22.16%, SC 0.32%
#Eval for threshold 0.35: DER 19.92%, MS 0.42%, FA 18.94%, SC 0.55%
#Eval for threshold 0.40: DER 17.00%, MS 0.66%, FA 14.10%, SC 2.23%
#Eval for threshold 0.45: DER 14.31%, MS 0.97%, FA 9.56%, SC 3.78%
#Eval for threshold 0.50: DER 11.75%, MS 1.46%, FA 4.72%, SC 5.57%
#Eval for threshold 0.55: DER 13.04%, MS 5.06%, FA 3.59%, SC 4.39%
#Eval for threshold 0.60: DER 15.58%, MS 9.77%, FA 3.22%, SC 2.60%
#Eval for threshold 0.70: DER 20.64%, MS 17.91%, FA 2.48%, SC 0.25%
#Eval for threshold 0.80: DER 25.30%, MS 23.36%, FA 1.77%, SC 0.17%
#Eval for threshold 0.20: DER 9.35%, MS 0.14%, FA 8.93%, SC 0.28%
#Eval for threshold 0.30: DER 7.93%, MS 0.38%, FA 7.25%, SC 0.30%
#Eval for threshold 0.35: DER 7.51%, MS 0.53%, FA 6.68%, SC 0.30%
#Eval for threshold 0.40: DER 7.14%, MS 0.69%, FA 6.15%, SC 0.30%
#Eval for threshold 0.45: DER 6.84%, MS 0.93%, FA 5.62%, SC 0.29%
#Eval for threshold 0.50: DER 6.76%, MS 1.25%, FA 5.25%, SC 0.27%
#Eval for threshold 0.55: DER 6.76%, MS 1.62%, FA 4.91%, SC 0.24%
#Eval for threshold 0.60: DER 6.75%, MS 2.07%, FA 4.49%, SC 0.19%
#Eval for threshold 0.70: DER 7.09%, MS 3.23%, FA 3.73%, SC 0.12%
#Eval for threshold 0.80: DER 8.26%, MS 5.20%, FA 2.98%, SC 0.07%
#Eval for threshold 0.20: DER 19.60%, MS 0.10%, FA 17.94%, SC 1.56%
#Eval for threshold 0.30: DER 15.84%, MS 0.29%, FA 13.79%, SC 1.76%
#Eval for threshold 0.35: DER 14.61%, MS 0.46%, FA 12.32%, SC 1.83%
#Eval for threshold 0.40: DER 13.67%, MS 0.69%, FA 11.07%, SC 1.92%
#Eval for threshold 0.45: DER 12.95%, MS 0.95%, FA 10.05%, SC 1.94%
#Eval for threshold 0.50: DER 12.41%, MS 1.42%, FA 9.06%, SC 1.92%
#Eval for threshold 0.55: DER 12.22%, MS 2.01%, FA 8.35%, SC 1.85%
#Eval for threshold 0.60: DER 12.17%, MS 2.76%, FA 7.63%, SC 1.78%
#Eval for threshold 0.70: DER 12.86%, MS 4.96%, FA 6.30%, SC 1.60%
#Eval for threshold 0.80: DER 15.07%, MS 8.74%, FA 4.94%, SC 1.38%
#Eval for threshold 0.20: DER 9.20%, MS 0.91%, FA 6.82%, SC 1.46%
#Eval for threshold 0.30: DER 6.69%, MS 2.20%, FA 3.05%, SC 1.43%
#Eval for threshold 0.35: DER 6.47%, MS 3.01%, FA 2.11%, SC 1.35%
#Eval for threshold 0.40: DER 6.73%, MS 3.81%, FA 1.66%, SC 1.26%
#Eval for threshold 0.45: DER 7.19%, MS 4.73%, FA 1.35%, SC 1.11%
#Eval for threshold 0.50: DER 7.88%, MS 5.77%, FA 1.10%, SC 1.01%
#Eval for threshold 0.55: DER 8.70%, MS 6.90%, FA 0.93%, SC 0.87%
#Eval for threshold 0.60: DER 9.73%, MS 8.13%, FA 0.78%, SC 0.81%
#Eval for threshold 0.70: DER 12.65%, MS 11.37%, FA 0.58%, SC 0.69%
#Eval for threshold 0.80: DER 17.38%, MS 16.35%, FA 0.41%, SC 0.62%
#Eval for threshold 0.20: DER 21.54%, MS 0.11%, FA 19.01%, SC 2.42%
#Eval for threshold 0.30: DER 14.49%, MS 0.37%, FA 11.43%, SC 2.69%
#Eval for threshold 0.35: DER 11.66%, MS 0.60%, FA 8.21%, SC 2.85%
#Eval for threshold 0.40: DER 9.68%, MS 0.87%, FA 5.74%, SC 3.07%
#Eval for threshold 0.45: DER 8.64%, MS 1.21%, FA 4.14%, SC 3.29%
#Eval for threshold 0.50: DER 8.11%, MS 1.74%, FA 2.78%, SC 3.59%
#Eval for threshold 0.55: DER 8.59%, MS 3.20%, FA 2.24%, SC 3.15%
#Eval for threshold 0.60: DER 9.64%, MS 4.78%, FA 1.99%, SC 2.87%
#Eval for threshold 0.70: DER 13.92%, MS 9.99%, FA 1.62%, SC 2.32%
#Eval for threshold 0.80: DER 20.43%, MS 17.11%, FA 1.23%, SC 2.09%
#Eval for threshold 0.20: DER 21.78%, MS 0.07%, FA 21.53%, SC 0.18%
#Eval for threshold 0.30: DER 17.08%, MS 0.21%, FA 16.58%, SC 0.29%
#Eval for threshold 0.35: DER 14.72%, MS 0.35%, FA 13.88%, SC 0.49%
#Eval for threshold 0.40: DER 12.41%, MS 0.55%, FA 9.69%, SC 2.17%
#Eval for threshold 0.45: DER 10.13%, MS 0.84%, FA 5.58%, SC 3.71%
#Eval for threshold 0.50: DER 7.89%, MS 1.20%, FA 1.13%, SC 5.57%
#Eval for threshold 0.55: DER 9.51%, MS 4.71%, FA 0.40%, SC 4.40%
#Eval for threshold 0.60: DER 12.33%, MS 9.36%, FA 0.35%, SC 2.61%
#Eval for threshold 0.70: DER 17.88%, MS 17.40%, FA 0.24%, SC 0.24%
#Eval for threshold 0.80: DER 22.94%, MS 22.62%, FA 0.16%, SC 0.17%
#Eval for threshold 0.20: DER 3.38%, MS 0.10%, FA 3.13%, SC 0.15%
#Eval for threshold 0.30: DER 2.48%, MS 0.28%, FA 2.04%, SC 0.16%
#Eval for threshold 0.35: DER 2.24%, MS 0.40%, FA 1.67%, SC 0.17%
#Eval for threshold 0.40: DER 2.06%, MS 0.52%, FA 1.38%, SC 0.16%
#Eval for threshold 0.45: DER 1.97%, MS 0.69%, FA 1.11%, SC 0.17%
#Eval for threshold 0.50: DER 2.03%, MS 0.92%, FA 0.94%, SC 0.16%
#Eval for threshold 0.55: DER 2.17%, MS 1.22%, FA 0.81%, SC 0.14%
#Eval for threshold 0.60: DER 2.35%, MS 1.60%, FA 0.64%, SC 0.11%
#Eval for threshold 0.70: DER 3.05%, MS 2.52%, FA 0.45%, SC 0.08%
#Eval for threshold 0.80: DER 4.51%, MS 4.17%, FA 0.29%, SC 0.05%

#if [ ${stage} -le 25 ] && [ ${stop_stage} -ge 25 ];then
#   echo "generate oracle vad speaker embedding for english trainset (megered)"
#   dest_dir=/maduo/model_hub
#   feature_name=cam++_zh-cn_200k_feature_dir_a100_80GB
#   #dest_dir=/mntcephfs/lab_data/maduo/model_hub
#   model_id=iic/speech_campplus_sv_zh-cn_16k-common
#   #subsets="American  Australian  British  Filipino  Indian"
#   #subsets="American"
#   subsets="train_english"
#   for name in $subsets;do
#    #if [ $name = "Train" ];then
#     #echo "extract Train settrain target speaker embedding"
#     # 提取embedding
#     #input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/${name}_Ali_far/target_audio/
#     #wav_path=$input_dir/wavs.txt
#     #else
#     echo "extract $name target speaker embedding"
#     # 提取embeddingdev/English
#     #input_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc/${name}/target_audio/
#     #wav_path=/mntcephfs/lab_data/maduo/exp/speaker_diarization/magicdata_ramc/ts_vad2/data/magicdata-ramc/${name}/wavs.txt
#     input_dir=/maduo/datasets/mlc-slm/train/${name}/target_audio
#     wav_path=/maduo/datasets/mlc-slm/train/${name}/wavs.txt
#     find $input_dir -name "*.wav" | grep -v "all.wav" >$wav_path
#     head $wav_path
#     save_dir=$dest_dir/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/train/$name/$feature_name
#     python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
#           --model_id $model_id\
#           --wavs $wav_path\
#           --save_dir $save_dir\
#           --batch_size 96
#   done
#fi
#
# stage21-22 is same as setting of stage13-14 , however,stage21-22 will use the mlc_slm data to finetune the model which is from pretrained on magicdata-ramc
if [ ${stage} -le 26 ] && [ ${stop_stage} -ge 26 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/maduo/datasets/musan
    rir_path=/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    # w2v-bert2
    #speech_encoder_type="w2v-bert2"
    #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    #speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # cam++ zh_200k
    speech_encoder_type="CAM++"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    speech_encoder_config=" /maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/configuration.json"

    dataset_name="mlc_slm" # dataset name

    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/ # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    # the model is pretrained from magicdata-ramc, more detail, you can see stage4  in this script.
    #source_dir=/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_shift0.8
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_shift0.8_ft_on_mlc_slm_lr5e-5

    #mkdir -p $exp_dir
    #cp -r $source_dir/best-valid-der.pt $exp_dir/
    #mv $exp_dir/best-valid-der.pt $exp_dir/epoch-0.pt

    data_dir="/maduo/datasets/mlc-slm" # oracle target audio , mix audio and labels path
    rs_len=6
    segment_shift=0.8
    single_backend_type="transformer"
    multi_backend_type="transformer"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 16715 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 10\
    --start-epoch 1\
    --start-batch 75000\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 5e-5\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer
    #--do-finetune true\
    #--finetune-ckpt $exp_dir/epoch-0.pt
fi

if [ ${stage} -le 27 ] && [ ${stop_stage} -ge 27 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_shift0.8_ft_on_mlc_slm_lr5e-5
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=0.8
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="American  Australian  British  Filipino  Indian"
 #infer_sets="Test"
 rttm_dir=/maduo/datasets/mlc-slm/dev/English
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 #speech_encoder_type="CAM++"
 #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # w2v-bert2
 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # cam++ zh_200k
 speech_encoder_type="CAM++"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 speech_encoder_config=" /maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/configuration.json"


 dataset_name="mlc_slm" # dataset name
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/mlc_slm/SpeakerEmbedding/dev/English # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/maduo/datasets/mlc-slm/dev/English/" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    #data_dir="/maduo/datasets/mlc-slm/dev/English/" # oracle target audio , mix audio and labels path
    results_path=$exp_dir/${dataset_name}_collar${c}
  python3 ts_vad2/infer2.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
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
fi
#grep -r Eval logs/run_ts_vad2_aistation_stage26_27_1.log
# American  Australian  British  Filipino  Indian ,collar=0.0
#Eval for threshold 0.20: DER 24.55%, MS 0.21%, FA 22.51%, SC 1.83%
#Eval for threshold 0.30: DER 20.92%, MS 0.55%, FA 18.43%, SC 1.95%
#Eval for threshold 0.35: DER 19.55%, MS 0.78%, FA 16.78%, SC 1.99%
#Eval for threshold 0.40: DER 18.47%, MS 1.07%, FA 15.39%, SC 2.02%
#Eval for threshold 0.45: DER 17.67%, MS 1.50%, FA 14.15%, SC 2.03%
#Eval for threshold 0.50: DER 17.05%, MS 2.06%, FA 12.98%, SC 2.01%
#Eval for threshold 0.55: DER 16.60%, MS 2.71%, FA 12.00%, SC 1.89%
#Eval for threshold 0.60: DER 16.36%, MS 3.52%, FA 11.04%, SC 1.79% as report
#Eval for threshold 0.70: DER 16.41%, MS 5.50%, FA 9.24%, SC 1.66%
#Eval for threshold 0.80: DER 17.66%, MS 9.13%, FA 7.04%, SC 1.49%

# Australian  ,collar=0.0
#Eval for threshold 0.20: DER 13.94%, MS 1.11%, FA 10.94%, SC 1.89%
#Eval for threshold 0.30: DER 11.75%, MS 2.33%, FA 7.32%, SC 2.10%
#Eval for threshold 0.35: DER 11.46%, MS 3.04%, FA 6.29%, SC 2.13%
#Eval for threshold 0.40: DER 11.28%, MS 3.86%, FA 5.37%, SC 2.05% as report
#Eval for threshold 0.45: DER 11.42%, MS 4.74%, FA 4.74%, SC 1.94%
#Eval for threshold 0.50: DER 11.82%, MS 5.77%, FA 4.29%, SC 1.77% 
#Eval for threshold 0.55: DER 12.44%, MS 7.00%, FA 3.88%, SC 1.57%
#Eval for threshold 0.60: DER 13.12%, MS 8.25%, FA 3.49%, SC 1.38% 
#Eval for threshold 0.70: DER 14.99%, MS 11.20%, FA 2.75%, SC 1.04%
#Eval for threshold 0.80: DER 18.20%, MS 15.37%, FA 1.99%, SC 0.83%

# British ,collar=0.0
#Eval for threshold 0.20: DER 26.28%, MS 0.16%, FA 23.52%, SC 2.60%
#Eval for threshold 0.30: DER 19.69%, MS 0.38%, FA 16.36%, SC 2.95%
#Eval for threshold 0.35: DER 17.08%, MS 0.56%, FA 13.38%, SC 3.15%
#Eval for threshold 0.40: DER 15.05%, MS 0.82%, FA 10.77%, SC 3.46%
#Eval for threshold 0.45: DER 13.82%, MS 1.21%, FA 8.89%, SC 3.72%
#Eval for threshold 0.50: DER 13.22%, MS 1.82%, FA 7.49%, SC 3.92% as report
#Eval for threshold 0.55: DER 13.44%, MS 3.09%, FA 6.83%, SC 3.53% 
#Eval for threshold 0.60: DER 14.12%, MS 4.73%, FA 6.27%, SC 3.13%
#Eval for threshold 0.70: DER 16.96%, MS 9.33%, FA 5.04%, SC 2.59%
#Eval for threshold 0.80: DER 21.55%, MS 15.48%, FA 3.89%, SC 2.18%

# Filipino  ,collar=0.0
#Eval for threshold 0.20: DER 27.01%, MS 0.14%, FA 26.65%, SC 0.22%
#Eval for threshold 0.30: DER 21.44%, MS 0.34%, FA 20.80%, SC 0.29%
#Eval for threshold 0.35: DER 18.56%, MS 0.46%, FA 17.06%, SC 1.04%
#Eval for threshold 0.40: DER 15.67%, MS 0.72%, FA 12.15%, SC 2.81%
#Eval for threshold 0.45: DER 13.07%, MS 0.97%, FA 7.58%, SC 4.52%
#Eval for threshold 0.50: DER 11.31%, MS 1.36%, FA 4.20%, SC 5.76% as report
#Eval for threshold 0.55: DER 12.44%, MS 4.05%, FA 3.62%, SC 4.76% 
#Eval for threshold 0.60: DER 14.45%, MS 8.44%, FA 3.25%, SC 2.76% 
#Eval for threshold 0.70: DER 19.56%, MS 16.80%, FA 2.53%, SC 0.23%
#Eval for threshold 0.80: DER 24.15%, MS 22.19%, FA 1.80%, SC 0.16%

# Indian, collar=0.0
#Eval for threshold 0.20: DER 9.44%, MS 0.13%, FA 9.06%, SC 0.25%
#Eval for threshold 0.30: DER 7.86%, MS 0.39%, FA 7.18%, SC 0.29%
#Eval for threshold 0.35: DER 7.44%, MS 0.54%, FA 6.60%, SC 0.30%
#Eval for threshold 0.40: DER 7.07%, MS 0.70%, FA 6.07%, SC 0.30%
#Eval for threshold 0.45: DER 6.82%, MS 0.91%, FA 5.63%, SC 0.28% 
#Eval for threshold 0.50: DER 6.72%, MS 1.19%, FA 5.27%, SC 0.26% as report
#Eval for threshold 0.55: DER 6.67%, MS 1.53%, FA 4.92%, SC 0.22%
#Eval for threshold 0.60: DER 6.62%, MS 1.91%, FA 4.55%, SC 0.16%
#Eval for threshold 0.70: DER 7.05%, MS 3.09%, FA 3.85%, SC 0.11%
#Eval for threshold 0.80: DER 8.35%, MS 5.16%, FA 3.11%, SC 0.08%

# American ,collar=0.25
#Eval for threshold 0.20: DER 18.08%, MS 0.07%, FA 16.36%, SC 1.65%
#Eval for threshold 0.30: DER 15.05%, MS 0.25%, FA 13.08%, SC 1.73%
#Eval for threshold 0.35: DER 13.93%, MS 0.36%, FA 11.81%, SC 1.76%
#Eval for threshold 0.40: DER 13.07%, MS 0.54%, FA 10.76%, SC 1.77%
#Eval for threshold 0.45: DER 12.43%, MS 0.78%, FA 9.84%, SC 1.81%
#Eval for threshold 0.50: DER 11.87%, MS 1.15%, FA 8.91%, SC 1.81%
#Eval for threshold 0.55: DER 11.59%, MS 1.59%, FA 8.26%, SC 1.74%
#Eval for threshold 0.60: DER 11.50%, MS 2.17%, FA 7.64%, SC 1.68%
#Eval for threshold 0.70: DER 11.79%, MS 3.58%, FA 6.60%, SC 1.60%
#Eval for threshold 0.80: DER 13.07%, MS 6.42%, FA 5.15%, SC 1.50%

# Australian ,collar=0.25
#Eval for threshold 0.20: DER 7.36%, MS 1.07%, FA 4.91%, SC 1.38%
#Eval for threshold 0.30: DER 6.26%, MS 2.18%, FA 2.65%, SC 1.44%
#Eval for threshold 0.35: DER 6.34%, MS 2.80%, FA 2.10%, SC 1.44%
#Eval for threshold 0.40: DER 6.50%, MS 3.50%, FA 1.64%, SC 1.36% as report
#Eval for threshold 0.45: DER 6.88%, MS 4.26%, FA 1.35%, SC 1.28%
#Eval for threshold 0.50: DER 7.45%, MS 5.09%, FA 1.20%, SC 1.16%
#Eval for threshold 0.55: DER 8.14%, MS 6.07%, FA 1.04%, SC 1.04%
#Eval for threshold 0.60: DER 8.90%, MS 7.05%, FA 0.91%, SC 0.93%
#Eval for threshold 0.70: DER 10.80%, MS 9.38%, FA 0.67%, SC 0.76%
#Eval for threshold 0.80: DER 13.99%, MS 12.85%, FA 0.47%, SC 0.66%

# British ,collar=0.25
#Eval for threshold 0.20: DER 18.89%, MS 0.13%, FA 16.36%, SC 2.40%
#Eval for threshold 0.30: DER 13.23%, MS 0.30%, FA 10.24%, SC 2.69%
#Eval for threshold 0.35: DER 11.04%, MS 0.44%, FA 7.77%, SC 2.83%
#Eval for threshold 0.40: DER 9.34%, MS 0.65%, FA 5.58%, SC 3.11%
#Eval for threshold 0.45: DER 8.36%, MS 0.94%, FA 4.04%, SC 3.39%
#Eval for threshold 0.50: DER 7.99%, MS 1.42%, FA 2.96%, SC 3.61%
#Eval for threshold 0.55: DER 8.42%, MS 2.54%, FA 2.59%, SC 3.28%
#Eval for threshold 0.60: DER 9.25%, MS 4.00%, FA 2.33%, SC 2.91%
#Eval for threshold 0.70: DER 12.59%, MS 8.37%, FA 1.77%, SC 2.45%
#Eval for threshold 0.80: DER 17.59%, MS 14.13%, FA 1.37%, SC 2.09%

# Filipino ,collar=0.25
#Eval for threshold 0.20: DER 20.41%, MS 0.11%, FA 20.12%, SC 0.19%
#Eval for threshold 0.30: DER 15.82%, MS 0.28%, FA 15.29%, SC 0.25%
#Eval for threshold 0.35: DER 13.38%, MS 0.37%, FA 12.00%, SC 1.01%
#Eval for threshold 0.40: DER 11.03%, MS 0.61%, FA 7.68%, SC 2.74%
#Eval for threshold 0.45: DER 8.91%, MS 0.84%, FA 3.59%, SC 4.47%
#Eval for threshold 0.50: DER 7.50%, MS 1.16%, FA 0.60%, SC 5.74%
#Eval for threshold 0.55: DER 8.87%, MS 3.72%, FA 0.37%, SC 4.77%
#Eval for threshold 0.60: DER 11.24%, MS 8.14%, FA 0.34%, SC 2.76%
#Eval for threshold 0.70: DER 16.78%, MS 16.33%, FA 0.24%, SC 0.22%
#Eval for threshold 0.80: DER 21.77%, MS 21.47%, FA 0.14%, SC 0.15%


# Indian, collar=0.25
#Eval for threshold 0.20: DER 3.44%, MS 0.10%, FA 3.21%, SC 0.13%
#Eval for threshold 0.30: DER 2.38%, MS 0.30%, FA 1.93%, SC 0.16%
#Eval for threshold 0.35: DER 2.15%, MS 0.41%, FA 1.57%, SC 0.17%
#Eval for threshold 0.40: DER 1.97%, MS 0.52%, FA 1.28%, SC 0.17%
#Eval for threshold 0.45: DER 1.92%, MS 0.67%, FA 1.09%, SC 0.16%
#Eval for threshold 0.50: DER 1.96%, MS 0.87%, FA 0.93%, SC 0.16%
#Eval for threshold 0.55: DER 2.05%, MS 1.12%, FA 0.80%, SC 0.13%
#Eval for threshold 0.60: DER 2.20%, MS 1.47%, FA 0.64%, SC 0.09%
#Eval for threshold 0.70: DER 2.91%, MS 2.39%, FA 0.45%, SC 0.06%
#Eval for threshold 0.80: DER 4.48%, MS 4.11%, FA 0.32%, SC 0.05%
