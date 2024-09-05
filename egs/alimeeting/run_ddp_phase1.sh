#!/usr/bin/env bash

stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_speaker_diarization.sh



if [ ${stage} -le -10 ] && [ ${stop_stage} -ge -10 ];then
   data_path=/mntcephfs/lab_data/maduo/datasets/alimeeting/
   #fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   echo " Process dataset: Eval dataset, get json files"
   python3 ts_vad_ddp_phase1/prepare_alimeeting_format_data_and_generate_target_audio.py \
    --data_path ${data_path} \
    --type Eval \

fi
if [ ${stage} -le -9 ] && [ ${stop_stage} -ge -9 ];then
   data_path=/mntcephfs/lab_data/maduo/datasets/alimeeting
   #fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
    echo " Process dataset: Train dataset, get json files"
   python3 ts_vad_ddp_phase1/prepare_alimeeting_format_data_and_generate_target_audio.py \
    --data_path ${data_path} \
    --type Train

fi


if [ ${stage} -le -8 ] && [ ${stop_stage} -ge -8 ];then
   data_path=/mntcephfs/lab_data/maduo/datasets/alimeeting
   #fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
    echo " Process dataset: Train dataset, get json files"
   python3 ts_vad_ddp_phase1/prepare_alimeeting_format_data_and_generate_target_audio.py \
    --data_path ${data_path} \
    --type Test

fi

if [ ${stage} -le -7 ] && [ ${stop_stage} -ge -7 ];then
   echo "prepare train target audio list"

   input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Train_Ali_far/target_audio/
   file=$input_dir/wavs.txt
    python3 ts_vad_ddp_phase1/prepare_alimeeting_target_audio_list.py \
        $input_dir $file

fi

if [ ${stage} -le -6 ] && [ ${stop_stage} -ge -6 ];then
   echo "generate train speaker embedding"
   feature_name=cam++_en_zh_advanced_feature_dir
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   # 提取embedding
   input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Train_Ali_far/target_audio/
   file=$input_dir/wavs.txt
   wav_path=$file
   dest_dir=/mntcephfs/lab_data/maduo/model_hub/
   save_dir=$dest_dir/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Train/$feature_name
   python3 ts_vad_ddp_phase1/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
       --model_id $model_id --wavs $wav_path\
       --save_dir $save_dir
fi


if [ ${stage} -le -5 ] && [ ${stop_stage} -ge -5 ];then
   echo "prepare eval(dev) target audio list"

   input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/target_audio/
   file=$input_dir/wavs.txt
    python3 ts_vad_ddp_phase1/prepare_alimeeting_target_audio_list.py \
            $input_dir $file

fi

if [ ${stage} -le -4 ] && [ ${stop_stage} -ge -4 ];then
   echo "generate eval(dev) speaker embedding"
   feature_name=cam++_en_zh_advanced_feature_dir
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   # 提取embedding
   input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/target_audio/
   file=$input_dir/wavs.txt
   wav_path=$file
   dest_dir=/mntcephfs/lab_data/maduo/model_hub/
   save_dir=$dest_dir/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Eval/$feature_name
   python3 ts_vad_ddp_phase1/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id --wavs $wav_path\
           --save_dir $save_dir
fi



if [ ${stage} -le -3 ] && [ ${stop_stage} -ge -3 ];then
   echo "prepare test set target audio list"

   input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Test_Ali/Test_Ali_far/target_audio/
   file=$input_dir/wavs.txt
    python3 ts_vad_ddp_phase1/prepare_alimeeting_target_audio_list.py \
            $input_dir $file

fi

if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ];then
   echo "generate test set speaker embedding"
   feature_name=cam++_en_zh_advanced_feature_dir
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   # 提取embedding

   input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Test_Ali/Test_Ali_far/target_audio/
   file=$input_dir/wavs.txt
   wav_path=$file
   dest_dir=/mntcephfs/lab_data/maduo/model_hub/
   save_dir=$dest_dir/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Test/$feature_name
   python3 ts_vad_ddp_phase1/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id --wavs $wav_path\
           --save_dir $save_dir
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
   splits="Eval Test"
   alimeeting_corpus=/mntcephfs/lab_data/maduo/datasets/alimeeting/
   dest_dir=/path/to/oracle_rttm_dir
   for name in $splits;do
       audio_dir=$alimeeting_corpus/${name}_Ali/${name}_Ali_far/audio_dir/
       textgrid_dir=$alimeeting_corpus/${name}_Ali/${name}_Ali_far/textgrid_dir
       bash ts_vad_ddp_phase1/prepare_rttm_for_ts_vad.sh\
           --stage 0 \
           --dest-dir $dest_dir \
           --split $name\
           --audio-dir $audio_dir\
           --textgrid-dir $textgrid_dir
   done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
 export NCCL_DEBUG=INFO
 export PYTHONFAULTHANDLER=1
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad_ddp_phase1/exp
 CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12683 ts_vad_ddp_phase1/train_accelerate_ddp.py\
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --freeze-updates 0\
    --grad-clip false\
    --exp-dir $exp_dir
fi
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad_ddp_phase1/exp
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 #infer_sets="Eval Test"
 infer_sets="Eval"
 #rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 rttm_dir=/path/to/oracle_rttm_dir
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 results_path=$exp_dir/
 for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path

  python3 ts_vad_ddp_phase1/infer.py \
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
    --split $name
done

# model_file=$exp_dir/epoch-7.pt
# cat logs/run_ddp_phase1_stage10.log
#Model DER:  0.13275177072662736
#Model ACC:  0.9532066907257646
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.06it/s]
#Eval for threshold 0.20: DER 8.04%, MS 0.92%, FA 6.67%, SC 0.45%
#
#Eval for threshold 0.30: DER 6.20%, MS 1.49%, FA 4.20%, SC 0.50%
#
#Eval for threshold 0.35: DER 5.72%, MS 1.82%, FA 3.35%, SC 0.55%
#
#Eval for threshold 0.40: DER 5.38%, MS 2.13%, FA 2.65%, SC 0.60%
#
#Eval for threshold 0.45: DER 5.17%, MS 2.48%, FA 2.07%, SC 0.61%
#
#Eval for threshold 0.50: DER 5.16%, MS 2.94%, FA 1.63%, SC 0.60%
#
#Eval for threshold 0.55: DER 5.35%, MS 3.48%, FA 1.34%, SC 0.54%
#
#Eval for threshold 0.60: DER 5.65%, MS 4.09%, FA 1.08%, SC 0.49%
#
#Eval for threshold 0.70: DER 6.82%, MS 5.75%, FA 0.71%, SC 0.36%
#
#Eval for threshold 0.80: DER 9.10%, MS 8.37%, FA 0.47%, SC 0.26%

# model_file=$exp_dir/epoch-13.pt
# cat logs/run_ddp_phase1_stage10_1.log
# Model DER:  0.13405204603429638
#Model ACC:  0.9538901345254731
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 7.17%, MS 1.23%, FA 5.53%, SC 0.41%
#
#Eval for threshold 0.30: DER 5.62%, MS 1.70%, FA 3.48%, SC 0.44%
#
#Eval for threshold 0.35: DER 5.16%, MS 1.93%, FA 2.77%, SC 0.46%
#
#Eval for threshold 0.40: DER 4.98%, MS 2.21%, FA 2.30%, SC 0.48%
#
#Eval for threshold 0.45: DER 4.92%, MS 2.54%, FA 1.90%, SC 0.48%
#
#Eval for threshold 0.50: DER 4.96%, MS 2.93%, FA 1.56%, SC 0.47%
#
#Eval for threshold 0.55: DER 5.06%, MS 3.38%, FA 1.27%, SC 0.42%
#
#Eval for threshold 0.60: DER 5.29%, MS 3.84%, FA 1.09%, SC 0.37%
#
#Eval for threshold 0.70: DER 6.08%, MS 5.01%, FA 0.78%, SC 0.29%
#
#Eval for threshold 0.80: DER 7.79%, MS 7.04%, FA 0.53%, SC 0.23%

# model_file=$exp_dir/epoch-20.pt
# cat logs/run_ddp_phase1_stage10_2.log
# Model DER:  0.13443398546750784
#Model ACC:  0.953684089475894
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.06it/s]
#Eval for threshold 0.20: DER 7.23%, MS 1.23%, FA 5.57%, SC 0.42%
#
#Eval for threshold 0.30: DER 5.63%, MS 1.67%, FA 3.49%, SC 0.47%
#
#Eval for threshold 0.35: DER 5.30%, MS 1.95%, FA 2.87%, SC 0.48%
#
#Eval for threshold 0.40: DER 5.09%, MS 2.26%, FA 2.35%, SC 0.48%
#
#Eval for threshold 0.45: DER 5.02%, MS 2.59%, FA 1.94%, SC 0.48%
#
#Eval for threshold 0.50: DER 4.99%, MS 2.97%, FA 1.57%, SC 0.45%
#
#Eval for threshold 0.55: DER 5.12%, MS 3.40%, FA 1.30%, SC 0.42%
#
#Eval for threshold 0.60: DER 5.35%, MS 3.85%, FA 1.12%, SC 0.39%
#
#Eval for threshold 0.70: DER 6.19%, MS 5.08%, FA 0.81%, SC 0.30%
#
#Eval for threshold 0.80: DER 7.91%, MS 7.14%, FA 0.54%, SC 0.23%


#  model_file=$exp_dir/best-valid-loss.pt
# logs/run_ddp_phase1_stage10_3.log
# Model DER:  0.13086975012116803
#Model ACC:  0.9547537404020879
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 7.08%, MS 1.10%, FA 5.59%, SC 0.39%
#
#Eval for threshold 0.30: DER 5.47%, MS 1.51%, FA 3.52%, SC 0.45%
#
#Eval for threshold 0.35: DER 5.10%, MS 1.77%, FA 2.86%, SC 0.46%
#
#Eval for threshold 0.40: DER 4.90%, MS 2.07%, FA 2.34%, SC 0.48%
#
#Eval for threshold 0.45: DER 4.79%, MS 2.40%, FA 1.93%, SC 0.45%
#
#Eval for threshold 0.50: DER 4.83%, MS 2.78%, FA 1.57%, SC 0.48%
#
#Eval for threshold 0.55: DER 4.99%, MS 3.24%, FA 1.30%, SC 0.45%
#
#Eval for threshold 0.60: DER 5.23%, MS 3.75%, FA 1.07%, SC 0.40%
#
#Eval for threshold 0.70: DER 6.11%, MS 5.01%, FA 0.77%, SC 0.34%
#
#Eval for threshold 0.80: DER 8.12%, MS 7.38%, FA 0.52%, SC 0.23%

#model_file=$exp_dir/best-valid-der.pt
# cat logs/run_ddp_phase1_stage10_4.log
#Model DER:  0.13087824463709546
#Model ACC:  0.9547311285294958
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.06it/s]
#Eval for threshold 0.20: DER 7.06%, MS 1.10%, FA 5.57%, SC 0.39%
#
#Eval for threshold 0.30: DER 5.47%, MS 1.53%, FA 3.51%, SC 0.43%
#
#Eval for threshold 0.35: DER 5.10%, MS 1.79%, FA 2.85%, SC 0.46%
#
#Eval for threshold 0.40: DER 4.90%, MS 2.07%, FA 2.34%, SC 0.49%
#
#Eval for threshold 0.45: DER 4.78%, MS 2.39%, FA 1.91%, SC 0.47% as report
#
#Eval for threshold 0.50: DER 4.82%, MS 2.79%, FA 1.56%, SC 0.47%
#
#Eval for threshold 0.55: DER 4.95%, MS 3.22%, FA 1.28%, SC 0.45%
#
#Eval for threshold 0.60: DER 5.24%, MS 3.77%, FA 1.08%, SC 0.39%
#
#Eval for threshold 0.70: DER 6.10%, MS 5.00%, FA 0.75%, SC 0.34%
#
#Eval for threshold 0.80: DER 8.14%, MS 7.39%, FA 0.52%, SC 0.22%

fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad_ddp_phase1/exp
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 #infer_sets="Eval Test"
 infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 results_path=$exp_dir/
 for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path

  python3 ts_vad_ddp_phase1/infer.py \
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
    --split $name
done

# model_file=$exp_dir/best-valid-der.pt
# cat logs/run_ddp_phase1_stage11.log
#Model DER:  0.13085100596695015
#Model ACC:  0.9506723245350728
#frame_len: 0.04!!
#100%|██████████| 60/60 [00:58<00:00,  1.02it/s]
#Eval for threshold 0.20: DER 9.58%, MS 1.09%, FA 7.58%, SC 0.91%
#
#Eval for threshold 0.30: DER 7.47%, MS 1.60%, FA 4.73%, SC 1.14%
#
#Eval for threshold 0.35: DER 6.93%, MS 1.87%, FA 3.82%, SC 1.23%
#
#Eval for threshold 0.40: DER 6.60%, MS 2.21%, FA 3.11%, SC 1.28%
#
#Eval for threshold 0.45: DER 6.38%, MS 2.58%, FA 2.47%, SC 1.32% as report
#
#Eval for threshold 0.50: DER 6.37%, MS 3.04%, FA 2.03%, SC 1.30%
#
#Eval for threshold 0.55: DER 6.48%, MS 3.57%, FA 1.66%, SC 1.24%
#
#Eval for threshold 0.60: DER 6.66%, MS 4.15%, FA 1.36%, SC 1.15%
#
#Eval for threshold 0.70: DER 7.54%, MS 5.71%, FA 0.88%, SC 0.95%
#
#Eval for threshold 0.80: DER 9.27%, MS 8.03%, FA 0.51%, SC 0.73%
fi




