#!/usr/bin/env bash
stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_speaker_diarization.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then

 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad_phase1/exp
  python3 ts_vad_phase1/train.py\
    --world-size 1 \
    --num-epochs 20\
    --start-epoch 1\
    --use-fp16 1\
    --exp-dir $exp_dir
fi
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad_phase1/exp2
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 #infer_sets="Eval Test"
 infer_sets="Eval"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 results_path=$exp_dir/
 for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path

  python3 ts_vad_phase1/infer.py \
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
fi
