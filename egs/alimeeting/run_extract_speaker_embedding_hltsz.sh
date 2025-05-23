#!/usr/bin/env bash

stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_speaker_diarization_hltsz.sh

# extract target speaker embedding using eres2netv2w24s4ep4 checkpoint from modescope
# it need GPU memory size is bigger than 32GB(i.e.V100 32GB), So it need A100
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "generate eval(dev) speaker embedding"
   feature_name=eres2netv2w24s4ep4_on_200kspeakers_feature_dir
   dest_dir=/mntcephfs/lab_data/maduo/model_hub/
   # https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common
   model_id=iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common
   subsets="Eval Test Train"
   for name in $subsets;do
    if [ $name = "Train" ];then
     echo "extract train target speaker embedding"
     # 提取embedding
     input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/${name}_Ali_far/target_audio/
     wav_path=$input_dir/wavs.txt
    else
        echo "extract $name target speaker embedding"
        # 提取embedding
     input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/${name}_Ali/${name}_Ali_far/target_audio/
     wav_path=$input_dir/wavs.txt
    fi
    save_dir=$dest_dir/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/$name/$feature_name
    python3 ts_vad2/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size 32
   done
fi

# offline extract fbank feat and frame_feat
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "generate eval(dev) speaker embedding"
   feature_name=cam++_en_zh_advanced_feature_dir
   dest_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   subsets="Eval Test Train"
   for name in $subsets;do
    if [ $name = "Train" ];then
     echo "extract train target speaker embedding"
     # 提取embedding
     input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/${name}_Ali_far/target_audio/
     wav_path=$input_dir/wavs.txt
    else
        echo "extract $name target speaker embedding"
        # 提取embedding
     input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/${name}_Ali/${name}_Ali_far/target_audio/
     wav_path=$input_dir/wavs.txt
    fi
    for feat_type in frame_feat fbank_feat utt_feat;do
     save_dir=$dest_dir/ts_vad/spk_embed/alimeeting/${feat_type}SpeakerEmbedding/$name/$feature_name
     python3 ts_vad3/generate_chunk_multi_level_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size 96\
           --feat_type $feat_type
    done
   done
fi

# debug <6s case
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   feature_name=cam++_en_zh_advanced_feature_dir
   dest_dir=tests
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   wav_path=tests/wavs.txt
   save_dir=tests/
   python3 ts_vad_ddp_phase1/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id\
           --wavs $wav_path\
           --save_dir $save_dir\
           --batch_size 96
fi
