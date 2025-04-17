#!/usr/bin/env bash

stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_dia_cuda11.8_py3111_aistation.sh


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   musan_path=/maduo/datasets/musan
   rir_path=/maduo/datasets/RIRS_NOISES
   data_dir=/maduo/datasets/alimeeting/Train_Ali_far/target_audio/
   speech_enhancement_model_type="modelscope_zipenhancer"
   output_dir=/maduo/datasets/zipenhancer_alimeeting
   python3  ts_vad2/offline_add_noise_and_speech_enhance.py\
		--musan-path $musan_path\
                --rir-path $rir_path\
                --mix-audio-mono-dir $data_dir\
		--speech-enhancement-model-type $speech_enhancement_model_type\
		--output-dir $output_dir

fi
