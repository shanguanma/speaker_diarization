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


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   #subset="American"
   subset="Australian  British  Filipino  Indian"
   output_dir=/maduo/exp/asr_sd/dicow_offical_model_inference
   for name in $subset;do
       inp_dir=$output_dir/$name
       dest_file=$output_dir/$name/hyp.stm
       python3 dicow/generate_hyp_stm_from_dicow_output.py $inp_dir $dest_file
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/dicow_offical_model_inference
   #subset="American  Australian  British  Filipino  Indian"
   subset="Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
#compute tcpwer
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18640.64 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/Australian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/Australian/hyp_tcpwer.json
#INFO %tcpWER: 24.18% [ 6579 / 27213, 2737 ins, 1272 del, 2570 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18446.00 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/British/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/British/hyp_tcpwer.json
#INFO %tcpWER: 27.06% [ 6623 / 24478, 3238 ins, 1316 del, 2069 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 16923.94 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/Filipino/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/Filipino/hyp_tcpwer.json
#INFO %tcpWER: 26.45% [ 4581 / 17318, 1874 ins, 902 del, 1805 sub ]
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/Indian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/Indian/hyp_tcpwer.json
#INFO %tcpWER: 14.63% [ 3008 / 20567, 854 ins, 949 del, 1205 sub ]
fi
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "compute DER using md_eval_22 "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/dicow_offical_model_inference
   for line in American  Australian  British  Filipino  Indian;do
          meeteval-der md_eval_22 -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar .25
   done
fi

#grep -r "INFO" logs/run_dicow_aistation_stage6.log
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/American/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/American/hyp_md_eval_22.json
#INFO %DER: 12.39% [ 846.10s / 6828.23s, 91.42s missed, 647.97s falarm, 106.72s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/Australian/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/Australian/hyp_md_eval_22.json
#INFO %DER: 6.77% [ 494.87s / 7312.55s, 249.70s missed, 155.35s falarm, 89.82s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/British/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/British/hyp_md_eval_22.json
#INFO %DER: 8.28% [ 583.83s / 7052.82s, 84.66s missed, 278.22s falarm, 220.95s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/Filipino/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/Filipino/hyp_md_eval_22.json
#INFO %DER: 10.00% [ 661.27s / 6615.44s, 64.36s missed, 214.14s falarm, 382.77s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/Indian/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/Indian/hyp_md_eval_22.json
#INFO %DER: 2.77% [ 191.98s / 6940.15s, 63.70s missed, 113.19s falarm, 15.08s spk error ]


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
   echo "compute DER using md_eval_22 "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/dicow_offical_model_inference
   for line in American  Australian  British  Filipino  Indian;do
          meeteval-der md_eval_22 -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar .0
   done
fi

#grep -r "INFO" logs/run_dicow_aistation_stage7.log
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/American/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/American/hyp_md_eval_22.json
#INFO %DER: 16.83% [ 1339.78s / 7962.20s, 180.41s missed, 1029.25s falarm, 130.12s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/Australian/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/Australian/hyp_md_eval_22.json
#INFO %DER: 10.92% [ 916.49s / 8390.17s, 335.22s missed, 434.63s falarm, 146.65s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/British/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/British/hyp_md_eval_22.json
#INFO %DER: 12.55% [ 1003.89s / 7996.33s, 116.84s missed, 618.46s falarm, 268.59s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/Filipino/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/Filipino/hyp_md_eval_22.json
#INFO %DER: 12.94% [ 950.06s / 7339.29s, 87.02s missed, 436.88s falarm, 426.17s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/Indian/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_offical_model_inference/Indian/hyp_md_eval_22.json
#INFO %DER: 6.71% [ 520.52s / 7760.98s, 95.57s missed, 401.73s falarm, 23.23s spk error ]


if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference
   subset="American Australian  British  Filipino  Indian"
   #subset="Filipino  Indian"
   for name in $subset;do
    python3   dicow/inference.py \
              --dicow-model BUT-FIT/DiCoW_v3_MLC \
              --diarization-model BUT-FIT/diarizen-wavlm-large-s80-mlc\
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name

   done

fi


if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
   #subset="American"
   subset="American Australian  British  Filipino  Indian"
   output_dir=/maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference
   for name in $subset;do
       inp_dir=$output_dir/$name
       dest_file=$output_dir/$name/hyp.stm
       python3 dicow/generate_hyp_stm_from_dicow_output.py $inp_dir $dest_file
    done
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference
   subset="American  Australian  British  Filipino  Indian"
   #subset="Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
fi
#compute tcpwer
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 19891.40 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/American/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/American/hyp_tcpwer.json
#INFO %tcpWER: 23.00% [ 5227 / 22722, 2805 ins, 880 del, 1542 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18640.64 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/Australian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/Australian/hyp_tcpwer.json
#INFO %tcpWER: 13.25% [ 3605 / 27213, 1169 ins, 1275 del, 1161 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18446.00 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/British/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/British/hyp_tcpwer.json
#INFO %tcpWER: 18.20% [ 4455 / 24478, 2260 ins, 1110 del, 1085 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 16923.94 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/Filipino/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/Filipino/hyp_tcpwer.json
#INFO %tcpWER: 17.76% [ 3076 / 17318, 1215 ins, 691 del, 1170 sub ]
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/Indian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/Indian/hyp_tcpwer.json
#INFO %tcpWER: 14.80% [ 3043 / 20567, 892 ins, 755 del, 1396 sub ]

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
   echo "compute DER using md_eval_22 "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference
   for line in American  Australian  British  Filipino  Indian;do
          meeteval-der md_eval_22 -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar .25
   done
fi
#grep -r INFO logs/run_dicow_aistation_stage13.log
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/American/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/American/hyp_md_eval_22.json
#INFO %DER: 12.13% [ 828.09s / 6828.23s, 101.94s missed, 619.13s falarm, 107.02s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/Australian/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/Australian/hyp_md_eval_22.json
#INFO %DER: 7.12% [ 520.56s / 7312.55s, 320.02s missed, 123.19s falarm, 77.34s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/British/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/British/hyp_md_eval_22.json
#INFO %DER: 8.33% [ 587.85s / 7052.82s, 85.87s missed, 277.31s falarm, 224.66s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/Filipino/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/Filipino/hyp_md_eval_22.json
#INFO %DER: 10.10% [ 668.08s / 6615.44s, 62.16s missed, 215.78s falarm, 390.14s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/Indian/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/Indian/hyp_md_eval_22.json
#INFO %DER: 3.10% [ 215.02s / 6940.15s, 66.71s missed, 133.72s falarm, 14.60s spk error ]

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
   echo "compute DER using md_eval_22 "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference
   for line in American  Australian  British  Filipino  Indian;do
          meeteval-der md_eval_22 -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar .0
   done
fi

# grep -r INFO logs/run_dicow_aistation_stage14.log
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/American/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/American/hyp_md_eval_22.json
#INFO %DER: 16.62% [ 1323.18s / 7962.20s, 193.64s missed, 998.43s falarm, 131.11s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/Australian/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/Australian/hyp_md_eval_22.json
#INFO %DER: 11.23% [ 942.21s / 8390.17s, 417.57s missed, 392.97s falarm, 131.68s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/British/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/British/hyp_md_eval_22.json
#INFO %DER: 12.73% [ 1017.81s / 7996.33s, 114.55s missed, 628.03s falarm, 275.23s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/Filipino/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/Filipino/hyp_md_eval_22.json
#INFO %DER: 12.99% [ 953.63s / 7339.29s, 80.98s missed, 438.30s falarm, 434.35s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/Indian/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference/Indian/hyp_md_eval_22.json
#INFO %DER: 7.03% [ 545.83s / 7760.98s, 97.75s missed, 425.71s falarm, 22.38s spk error ]


# check
if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference_v2
   subset="Filipino"
   #subset="Filipino  Indian"
   for name in $subset;do
    python3   dicow/inference_v2.py \
              --dicow-model BUT-FIT/DiCoW_v3_MLC \
              --diarization-model BUT-FIT/diarizen-wavlm-large-s80-mlc\
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name

   done

fi


if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ];then
   subset="Filipino"
   #subset="American Australian  British  Filipino  Indian"
   output_dir=/maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference_v2
   for name in $subset;do
       inp_dir=$output_dir/$name
       dest_file=$output_dir/$name/hyp.stm
       python3 dicow/generate_hyp_stm_from_dicow_output.py $inp_dir $dest_file
    done
fi

if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/dicow_v3_mlc_diarizen-wavlm-large-s80-mlc_inference
   #subset="American  Australian  British  Filipino  Indian"
   subset="Filipino"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
fi
