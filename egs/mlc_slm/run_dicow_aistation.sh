#!/usr/bin/env bash
stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_dia_cuda11.8_py3111_aistation.sh


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/dicow_offical_model_inference
   #subset="American  Australian  British  Filipino  Indian"
   subset="American"
   for name in $subset;do 
    python3   dicow/inference.py \
	      --dicow-model BUT-FIT/DiCoW_v3_2 \
	      --diarization-model BUT-FIT/diarizen-wavlm-large-s80-mlc\
	      --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name

   done

fi
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   subset="American"
   output_dir=/maduo/exp/asr_sd/dicow_offical_model_inference
   for name in $subset;do
       inp_dir=$output_dir/$name
       dest_file=$output_dir/$name/hyp.stm
       python3 dicow/generate_hyp_stm_from_dicow_output.py $inp_dir $dest_file
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/dicow_offical_model_inference
   #subset="American  Australian  British  Filipino  Indian"
   subset="American"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done

#compute tcpwer
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 19891.40 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/American/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/American/hyp_tcpwer.json
#INFO %tcpWER: 31.61% [ 7183 / 22722, 3856 ins, 1080 del, 2247 sub ]
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/dicow_offical_model_inference
   #subset="Australian  British  Filipino  Indian"
   subset="Filipino  Indian"
   for name in $subset;do
    python3   dicow/inference.py \
              --dicow-model BUT-FIT/DiCoW_v3_2 \
              --diarization-model BUT-FIT/diarizen-wavlm-large-s80-mlc\
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name

   done

fi
