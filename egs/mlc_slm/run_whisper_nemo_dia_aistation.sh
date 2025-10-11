#!/usr/bin/env bash

stage=0
stop_stage=1000
. utils/parse_options.sh
#. path_for_dia_cuda11.8_py3111_aistation.sh
. path_for_diar_cu118_py312_aistation_uv.sh
export HF_ENDPOINT=https://hf-mirror.com

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms_300m_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English
   subset="Australian  British  Filipino  Indian"
   #subset="American"
   export HF_ENDPOINT=https://hf-mirror.com
   for name in $subset;do
      python3 whisper_nemo_dia/diarize_v2.py \
              --whisper-model large-v3 \
              --local-files-only False\
              --diarizer msdd\
              --align-model MahmoudAshraf/mms-300m-1130-forced-aligner\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name
   done
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms_300m_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English
  #subset="American"
  subset="Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms_300m_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English
   #subset="American"
   subset="Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms_300m_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English
   #subset="American"
   subset="Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
 fi
# WARNING Self-overlap detected in reference. Total overlap: 0.00 of 19891.40 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms_300m_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English/American/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms_300m_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English/American/hyp_tcpwer.json
#INFO %tcpWER: 39.52% [ 8979 / 22722, 4402 ins, 3038 del, 1539 sub ] 

# grep -r "%tcpWER:" logs/run_whisper_nemo_dia_aistation_stage1-4.log
#INFO %tcpWER: 18.92% [ 5149 / 27213, 1660 ins, 2318 del, 1171 sub ]
#INFO %tcpWER: 22.22% [ 5439 / 24478, 2260 ins, 2253 del, 926 sub ]
#INFO %tcpWER: 19.99% [ 3462 / 17318, 1327 ins, 1173 del, 962 sub ]
#INFO %tcpWER: 17.88% [ 3677 / 20567, 707 ins, 2025 del, 945 sub ]
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "compute DER using md_eval_22 "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms_300m_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English
   subset="American Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-der md_eval_22 -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar .25
   done

fi
# grep -r "%DER:" logs/run_whisper_nemo_dia_aistation_stage1-4.log
#INFO %DER: 26.65% [ 1819.52s / 6828.23s, 386.31s missed, 859.33s falarm, 573.88s spk error ]
#INFO %DER: 12.95% [ 947.23s / 7312.55s, 561.73s missed, 209.87s falarm, 175.64s spk error ]
#INFO %DER: 14.12% [ 995.56s / 7052.82s, 387.93s missed, 321.95s falarm, 285.67s spk error ]
#INFO %DER: 14.82% [ 980.21s / 6615.44s, 346.75s missed, 221.91s falarm, 411.55s spk error ]
#INFO %DER: 12.17% [ 844.96s / 6940.15s, 523.82s missed, 223.35s falarm, 97.79s spk error ]


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "compute DER using md_eval_22 "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms_300m_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English
   subset="American Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-der md_eval_22 -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 0.0
   done
fi
# grep -r %DER logs/run_whisper_nemo_dia_aistation_stage5.log
#INFO %DER: 33.86% [ 2695.82s / 7962.20s, 791.16s missed, 1238.09s falarm, 666.57s spk error ]
#INFO %DER: 19.61% [ 1645.35s / 8390.17s, 911.31s missed, 439.17s falarm, 294.86s spk error ]
#INFO %DER: 20.42% [ 1632.59s / 7996.33s, 686.27s missed, 573.88s falarm, 372.45s spk error ]
#INFO %DER: 19.65% [ 1442.35s / 7339.29s, 657.98s missed, 328.90s falarm, 455.47s spk error ]
#INFO %DER: 17.63% [ 1368.01s / 7760.98s, 818.76s missed, 418.34s falarm, 130.90s spk error ]


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms_300m_ctc_aligner_without_punc_realign_with_msdd_diarization_infer/English
   #subset="Australian  British  Filipino  Indian"
   subset="American"
   #subset="American Australian  British  Filipino  Indian"
   export HF_ENDPOINT=https://hf-mirror.com
   for name in $subset;do
      python3 whisper_nemo_dia/diarize_v2.py \
              --whisper-model large-v3 \
              --local-files-only False\
              --diarizer msdd\
              --align-model MahmoudAshraf/mms-300m-1130-forced-aligner\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
	      --realign-with-punc false
   done
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms_300m_ctc_aligner_without_punc_realign_with_msdd_diarization_infer/English
  #subset="American"
  #subset="Australian  British  Filipino  Indian"
  subset="American Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms_300m_ctc_aligner_without_punc_realign_with_msdd_diarization_infer/English
   #subset="American"
   subset="American Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms_300m_ctc_aligner_without_punc_realign_with_msdd_diarization_infer/English
   #subset="American"
   #subset="Australian  British  Filipino  Indian"
   subset="American Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
 fi
# grep -r tcpWER logs/run_whisper_nemo_dia_aistation_stage6-10.log
#INFO %tcpWER: 39.96% [ 9079 / 22722, 4431 ins, 3067 del, 1581 sub ]
#INFO %tcpWER: 18.82% [ 5122 / 27213, 1613 ins, 2271 del, 1238 sub ]
#INFO %tcpWER: 24.48% [ 5993 / 24478, 2482 ins, 2475 del, 1036 sub ]
#INFO %tcpWER: 20.16% [ 3492 / 17318, 1338 ins, 1184 del, 970 sub ]
#INFO %tcpWER: 18.09% [ 3721 / 20567, 716 ins, 2034 del, 971 sub ]
 
 if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   echo "compute DER using md_eval_22 "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms_300m_ctc_aligner_without_punc_realign_with_msdd_diarization_infer/English
   subset="American Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-der md_eval_22 -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar .25
   done

fi
#grep -r %DER logs/run_whisper_nemo_dia_aistation_stage6-10.log
#INFO %DER: 25.70% [ 1754.85s / 6828.23s, 301.97s missed, 872.87s falarm, 580.02s spk error ]
#INFO %DER: 12.16% [ 888.86s / 7312.55s, 558.13s missed, 196.92s falarm, 133.82s spk error ]
#INFO %DER: 15.29% [ 1078.61s / 7052.82s, 421.62s missed, 328.15s falarm, 328.84s spk error ]
#INFO %DER: 14.62% [ 967.13s / 6615.44s, 342.56s missed, 226.65s falarm, 397.92s spk error ]
#INFO %DER: 11.44% [ 794.05s / 6940.15s, 494.47s missed, 229.26s falarm, 70.32s spk error ]

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
   echo "compute DER using md_eval_22 "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms_300m_ctc_aligner_without_punc_realign_with_msdd_diarization_infer/English
   subset="American Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-der md_eval_22 -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar .0
   done

fi
# grep -r %DER logs/run_whisper_nemo_dia_aistation_stage11.log
#INFO %DER: 32.73% [ 2605.68s / 7962.20s, 624.39s missed, 1284.59s falarm, 696.69s spk error ]
#INFO %DER: 18.88% [ 1584.23s / 8390.17s, 929.38s missed, 418.28s falarm, 236.57s spk error ]
#INFO %DER: 21.49% [ 1718.78s / 7996.33s, 711.35s missed, 592.75s falarm, 414.68s spk error ]
#INFO %DER: 19.45% [ 1427.49s / 7339.29s, 633.35s missed, 352.49s falarm, 441.66s spk error ]
#INFO %DER: 17.12% [ 1328.82s / 7760.98s, 774.20s missed, 450.14s falarm, 104.48s spk error ]

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English
   subset="American  Australian  British  Filipino  Indian"
   #subset="American"
   export HF_ENDPOINT=https://hf-mirror.com
   for name in $subset;do
    python3 whisper_nemo_dia/diarize_v2.py \
              --whisper-model large-v3 \
              --local-files-only False\
              --diarizer msdd\
              --align-model WAV2VEC2_ASR_LARGE_LV60K_960H\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
              --realign-with-punc false
   done
fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
 fi

#cat logs/run_whisper_nemo_dia_aistation_stage15-16.log
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 19891.40 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English/American/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English/American/hyp_tcpwer.json
#INFO %tcpWER: 41.37% [ 9400 / 22722, 4556 ins, 3192 del, 1652 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18640.64 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English/Australian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English/Australian/hyp_tcpwer.json
#INFO %tcpWER: 20.65% [ 5619 / 27213, 1790 ins, 2448 del, 1381 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18446.00 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English/British/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English/British/hyp_tcpwer.json
#INFO %tcpWER: 25.75% [ 6303 / 24478, 2621 ins, 2614 del, 1068 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 16923.94 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English/Filipino/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English/Filipino/hyp_tcpwer.json
#INFO %tcpWER: 21.48% [ 3720 / 17318, 1456 ins, 1302 del, 962 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English/Indian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English/Indian/hyp_tcpwer.json
#INFO %tcpWER: 20.71% [ 4260 / 20567, 916 ins, 2234 del, 1110 sub ]


 if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
   echo "compute DER using md_eval_22 "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English
   subset="American Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-der md_eval_22 -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar .25
   done

fi
# grep -r DER logs/run_whisper_nemo_dia_aistation_stage17.log
#compute DER using md_eval_22
#INFO %DER: 27.76% [ 1895.57s / 6828.23s, 487.85s missed, 844.62s falarm, 563.10s spk error ]
#INFO %DER: 14.83% [ 1084.33s / 7312.55s, 783.25s missed, 183.51s falarm, 117.56s spk error ]
#INFO %DER: 17.98% [ 1268.17s / 7052.82s, 667.30s missed, 316.71s falarm, 284.16s spk error ]
#INFO %DER: 17.74% [ 1173.79s / 6615.44s, 587.28s missed, 224.20s falarm, 362.31s spk error ]
#INFO %DER: 14.33% [ 994.61s / 6940.15s, 703.73s missed, 229.54s falarm, 61.34s spk error ]


 if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ];then
   echo "compute DER using md_eval_22 "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_with_punc_realign_with_msdd_diarization_infer/English
   subset="American Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-der md_eval_22 -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar .0
   done

fi
#grep -r DER logs/run_whisper_nemo_dia_aistation_stage18.log
#compute DER using md_eval_22
#INFO %DER: 34.77% [ 2768.09s / 7962.20s, 828.14s missed, 1260.45s falarm, 679.51s spk error ]
#INFO %DER: 21.93% [ 1839.56s / 8390.17s, 1255.05s missed, 384.99s falarm, 199.52s spk error ]
#INFO %DER: 24.51% [ 1959.78s / 7996.33s, 1030.39s missed, 579.42s falarm, 349.98s spk error ]
#INFO %DER: 23.12% [ 1697.19s / 7339.29s, 942.58s missed, 359.42s falarm, 395.18s spk error ]
#INFO %DER: 20.46% [ 1588.10s / 7760.98s, 1032.95s missed, 461.51s falarm, 93.64s spk error ]


if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_real_with_punc_realign_with_msdd_diarization_infer/English
   subset="American  Australian  British  Filipino  Indian"
   #subset="American"
   export HF_ENDPOINT=https://hf-mirror.com
   for name in $subset;do
    python3 whisper_nemo_dia/diarize_v2.py \
              --whisper-model large-v3 \
              --local-files-only False\
              --diarizer msdd\
              --align-model WAV2VEC2_ASR_LARGE_LV60K_960H\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
              --realign-with-punc true
   done
fi

if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_real_with_punc_realign_with_msdd_diarization_infer/English
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_real_with_punc_realign_with_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_real_with_punc_realign_with_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
 fi

#grep -r WER logs/run_whisper_nemo_dia_aistation_stage19-23.log
#INFO %tcpWER: 40.13% [ 9119 / 22722, 4459 ins, 3095 del, 1565 sub ]
#INFO %tcpWER: 19.89% [ 5412 / 27213, 1746 ins, 2404 del, 1262 sub ]
#INFO %tcpWER: 23.16% [ 5670 / 24478, 2362 ins, 2355 del, 953 sub ]
#INFO %tcpWER: 20.36% [ 3526 / 17318, 1369 ins, 1215 del, 942 sub ]
#INFO %tcpWER: 18.98% [ 3904 / 20567, 785 ins, 2103 del, 1016 sub ]

if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ];then
   echo "compute DER using md_eval_22 "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_real_with_punc_realign_with_msdd_diarization_infer/English
   subset="American Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-der md_eval_22 -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar .25
   done
fi
# grep -r %DER logs/run_whisper_nemo_dia_aistation_stage19-23.log
#INFO %DER: 29.52% [ 2016.01s / 6828.23s, 517.69s missed, 915.68s falarm, 582.64s spk error ]
#INFO %DER: 15.27% [ 1116.85s / 7312.55s, 670.89s missed, 226.74s falarm, 219.22s spk error ]
#INFO %DER: 17.12% [ 1207.49s / 7052.82s, 558.50s missed, 343.39s falarm, 305.60s spk error ]
#INFO %DER: 18.19% [ 1203.66s / 6615.44s, 530.45s missed, 261.07s falarm, 412.15s spk error ]
#INFO %DER: 15.15% [ 1051.31s / 6940.15s, 623.33s missed, 258.79s falarm, 169.19s spk error ]


if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ];then
   echo "compute DER using md_eval_22 "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_WAV2VEC2_ASR_LARGE_LV60K_960H_ctc_aligner_real_with_punc_realign_with_msdd_diarization_infer/English
   subset="American Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-der md_eval_22 -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar .0
   done
fi

#grep -r %DER logs/run_whisper_nemo_dia_aistation_stage24.log
#INFO %DER: 36.68% [ 2920.24s / 7962.20s, 900.42s missed, 1328.52s falarm, 691.31s spk error ]
#INFO %DER: 22.36% [ 1875.83s / 8390.17s, 1054.18s missed, 471.58s falarm, 350.07s spk error ]
#INFO %DER: 23.92% [ 1912.83s / 7996.33s, 893.26s missed, 625.01s falarm, 394.57s spk error ]
#INFO %DER: 23.87% [ 1751.64s / 7339.29s, 875.21s missed, 411.15s falarm, 465.28s spk error ]
#INFO %DER: 21.38% [ 1659.15s / 7760.98s, 902.57s missed, 520.87s falarm, 235.70s spk error ]



# now, I will prepared mlc-slm english subset to finetune 
if [ ${stage} -le 25 ] && [ ${stop_stage} -ge 25 ];then
   python3 whisper_nemo_dia/pathfiles_to_diarize_manifest.py\
	 --add_duration \
         --paths2audio_files="/maduo/datasets/mlc-slm/train/train_wav.list" \
         --paths2rttm_files="/maduo/datasets/mlc-slm/train/train_rttm.list" \
         --manifest_filepath="/maduo/datasets/mlc-slm/train/train_manifest.json"
fi


if [ ${stage} -le 26 ] && [ ${stop_stage} -ge 26 ];then
   find /maduo/datasets/mlc-slm/dev/dev_wav/ -name "*.wav" > /maduo/datasets/mlc-slm/dev/dev_wav.list
   python3 whisper_nemo_dia/pathfiles_to_diarize_manifest.py\
         --add_duration \
         --paths2audio_files="/maduo/datasets/mlc-slm/dev/dev_wav.list" \
         --paths2rttm_files="/maduo/datasets/mlc-slm/dev/dev_rttm.list" \
         --manifest_filepath="/maduo/datasets/mlc-slm/dev/dev_manifest.json"
fi


if [ ${stage} -le 27 ] && [ ${stop_stage} -ge 27 ];then
mkdir -p /maduo/datasets/mlc-slm/train/rttm_output_folder 
python3  whisper_nemo_dia/create_msdd_train_dataset.py \
  --input_manifest_path='/maduo/datasets/mlc-slm/train/train_manifest.json' \
  --output_manifest_path='/maduo/datasets/mlc-slm/train/train_manifest.50step.json' \
  --pairwise_rttm_output_folder='/maduo/datasets/mlc-slm/train/rttm_output_folder' \
  --window=0.5 \
  --shift=0.25 \
  --step_count=50	
fi

if [ ${stage} -le 28 ] && [ ${stop_stage} -ge 28 ];then
mkdir -p /maduo/datasets/mlc-slm/dev/rttm_output_folder
python3  whisper_nemo_dia/create_msdd_train_dataset.py \
  --input_manifest_path='/maduo/datasets/mlc-slm/dev/dev_manifest.json' \
  --output_manifest_path='/maduo/datasets/mlc-slm/dev/dev_manifest.50step.json' \
  --pairwise_rttm_output_folder='/maduo/datasets/mlc-slm/dev/rttm_output_folder' \
  --window=0.5 \
  --shift=0.25 \
  --step_count=50
fi

# finetune MSDD using mlc-slm
if [ ${stage} -le 29 ] && [ ${stop_stage} -ge 29 ];then
exp_dir=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset
mkdir -p $exp_dir
train_temp_dir=$exp_dir/train_temp
dev_temp_dir=$exp_dir/dev_temp
export HYDRA_FULL_ERROR=1
python  whisper_nemo_dia/multiscale_diar_decoder.py --config-path='conf/neural_diarizer' --config-name='msdd_5scl_15_05_50Povl_256x3x32x2.yaml' \
  trainer.devices=2 \
  trainer.max_epochs=10  \
  num_workers=20\
  trainer.strategy=ddp_find_unused_parameters_true\
  model.diarizer.speaker_embeddings.model_path="titanet_large" \
  model.train_ds.manifest_filepath="/maduo/datasets/mlc-slm/train/train_manifest.50step.json" \
  model.validation_ds.manifest_filepath="/maduo/datasets/mlc-slm/dev/dev_manifest.50step.json" \
  model.train_ds.emb_dir="$train_temp_dir" \
  model.validation_ds.emb_dir="$dev_temp_dir" \
  +model.train_ds.num_workers=4\
  exp_manager.name='sample_train' \
  exp_manager.exp_dir=$exp_dir\
  +exp_manager.resume_from_checkpoint=$exp_dir/sample_train/2025-09-05_13-52-49/checkpoints/sample_train--val_loss0.0112-epoch7-last.ckpt
  #+exp_manager.checkpoint_callback_params.every_n_train_steps=200\
  #+exp_manager.checkpoint_callback_params.save_last=true\
fi

if [ ${stage} -le 30 ] && [ ${stop_stage} -ge 30 ];then
exp_dir=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset_lr1e-4
mkdir -p $exp_dir
train_temp_dir=$exp_dir/train_temp
dev_temp_dir=$exp_dir/dev_temp
export HYDRA_FULL_ERROR=1
python  whisper_nemo_dia/multiscale_diar_decoder.py --config-path='conf/neural_diarizer' --config-name='msdd_5scl_15_05_50Povl_256x3x32x2.yaml' \
  trainer.devices=2 \
  trainer.max_epochs=10  \
  num_workers=20\
  trainer.strategy=ddp_find_unused_parameters_true\
  model.diarizer.speaker_embeddings.model_path="titanet_large" \
  model.train_ds.manifest_filepath="/maduo/datasets/mlc-slm/train/train_manifest.50step.json" \
  model.validation_ds.manifest_filepath="/maduo/datasets/mlc-slm/dev/dev_manifest.50step.json" \
  model.train_ds.emb_dir="$train_temp_dir" \
  model.validation_ds.emb_dir="$dev_temp_dir" \
  +model.train_ds.num_workers=4\
  exp_manager.name='sample_train' \
  exp_manager.exp_dir=$exp_dir\
  model.optim.lr=0.0001\
  +exp_manager.resume_from_checkpoint=$exp_dir/sample_train/2025-09-05_15-21-36/checkpoints/sample_train--val_loss0.0102-epoch6-last.ckpt
  #+exp_manager.checkpoint_callback_params.every_n_train_steps=200\
  #+exp_manager.checkpoint_callback_params.save_last=true\
fi

# 由于stage29 or stage30 生成的chckpoint 不包含speaker model 相关的configure, 
# 因此想使用官方预训练的模型，获得它的完整configure, 之后结合我实际调优的保存的configure 进行修改，作为最后的推理可用的configure
# 手动下载官方预训练的模型，它的格式是.nemo, .nemo 其实是一个.tar 的压缩包，里面包含model_config.yaml 和model_weigth.ckpt
# import nemo.collections.asr as nemo_asr
#  pretrained_model = nemo_asr.models.EncDecDiarLabelModel.from_pretrained(model_name="diar_msdd_telephonic")
 
if [ ${stage} -le 33 ] && [ ${stop_stage} -ge 33 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   subset="American  Australian  British  Filipino  Indian"
   #subset="American"
   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    python3 whisper_nemo_dia/diarize_v2.py \
              --whisper-model large-v3 \
              --local-files-only False\
              --diarizer msdd\
	      --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model MahmoudAshraf/mms-300m-1130-forced-aligner\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
              --realign-with-punc true
   done
fi

if [ ${stage} -le 34 ] && [ ${stop_stage} -ge 34 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 35 ] && [ ${stop_stage} -ge 35 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 36 ] && [ ${stop_stage} -ge 36 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
 fi
#grep -r tcpWER logs/run_whisper_nemo_dia_aistation_stage33_36.log
#INFO %tcpWER: 32.16% [ 7307 / 22722, 3511 ins, 2144 del, 1652 sub ]
#INFO %tcpWER: 19.19% [ 5221 / 27213, 1669 ins, 2337 del, 1215 sub ]
#INFO %tcpWER: 22.29% [ 5455 / 24478, 2276 ins, 2258 del, 921 sub ]
#INFO %tcpWER: 20.02% [ 3467 / 17318, 1331 ins, 1175 del, 961 sub ]
#INFO %tcpWER: 17.81% [ 3663 / 20567, 696 ins, 2018 del, 949 sub ]

if [ ${stage} -le 37 ] && [ ${stop_stage} -ge 37 ];then
   echo "compute DER using md_eval_22 "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   subset="American Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-der md_eval_22 -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar .25
   done
fi

if [ ${stage} -le 38 ] && [ ${stop_stage} -ge 38 ];then
   echo "compute DER using md_eval_22 "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   subset="American Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-der md_eval_22 -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar .0
   done
fi

 if [ ${stage} -le 40 ] && [ ${stop_stage} -ge 40 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_lr1e-4_msdd_diarization_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset_lr1e-4/sample_train/2025-09-08_14-47-13/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   subset="American  Australian  British  Filipino  Indian"
   #subset="American"
   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    python3 whisper_nemo_dia/diarize_v2.py \
              --whisper-model large-v3 \
              --local-files-only False\
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model MahmoudAshraf/mms-300m-1130-forced-aligner\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
              --realign-with-punc true
   done
fi

if [ ${stage} -le 41 ] && [ ${stop_stage} -ge 41 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_lr1e-4_msdd_diarization_infer/English
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 42 ] && [ ${stop_stage} -ge 42 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_lr1e-4_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 43 ] && [ ${stop_stage} -ge 43 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_lr1e-4_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
 fi
#grep -r tcpWER logs/run_whisper_nemo_dia_aistation_stage40_43.log, its align model is WAV2VEC2_ASR_LARGE_LV60K_960H 
#INFO %tcpWER: 33.62% [ 7639 / 22722, 3657 ins, 2290 del, 1692 sub ]
#INFO %tcpWER: 19.95% [ 5430 / 27213, 1760 ins, 2428 del, 1242 sub ]
#INFO %tcpWER: 23.02% [ 5634 / 24478, 2345 ins, 2327 del, 962 sub ]
#INFO %tcpWER: 20.35% [ 3525 / 17318, 1371 ins, 1215 del, 939 sub ]
#INFO %tcpWER: 18.84% [ 3875 / 20567, 760 ins, 2082 del, 1033 sub ]


#grep -rn tcpWER logs/run_whisper_nemo_dia_aistation_stage40_43_1.log, its align model is mms-300m-1130-forced-aligner
#7685:INFO %tcpWER: 32.79% [ 7451 / 22722, 3589 ins, 2222 del, 1640 sub ]
#7689:INFO %tcpWER: 19.01% [ 5174 / 27213, 1646 ins, 2314 del, 1214 sub ]
#7693:INFO %tcpWER: 22.15% [ 5421 / 24478, 2253 ins, 2235 del, 933 sub ]
#7697:INFO %tcpWER: 20.06% [ 3474 / 17318, 1335 ins, 1179 del, 960 sub ]
#7700:INFO %tcpWER: 17.96% [ 3694 / 20567, 713 ins, 2035 del, 946 sub ]


if [ ${stage} -le 50 ] && [ ${stop_stage} -ge 50 ];then
exp_dir=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset_lr1e-3_emb_batch_size100_batch_size15
mkdir -p $exp_dir
train_temp_dir=$exp_dir/train_temp
dev_temp_dir=$exp_dir/dev_temp
export HYDRA_FULL_ERROR=1
python  whisper_nemo_dia/multiscale_diar_decoder.py --config-path='conf/neural_diarizer' --config-name='msdd_5scl_15_05_50Povl_256x3x32x2.yaml' \
  trainer.devices=2 \
  trainer.max_epochs=10  \
  num_workers=20\
  trainer.strategy=ddp_find_unused_parameters_true\
  model.diarizer.speaker_embeddings.model_path="titanet_large" \
  model.train_ds.manifest_filepath="/maduo/datasets/mlc-slm/train/train_manifest.50step.json" \
  model.validation_ds.manifest_filepath="/maduo/datasets/mlc-slm/dev/dev_manifest.50step.json" \
  model.train_ds.emb_dir="$train_temp_dir" \
  model.validation_ds.emb_dir="$dev_temp_dir" \
  +model.train_ds.num_workers=4\
  exp_manager.name='sample_train' \
  exp_manager.exp_dir=$exp_dir\
  model.emb_batch_size=100\
  batch_size=15
  #+exp_manager.resume_from_checkpoint=$exp_dir/sample_train/2025-09-12_10-24-42/checkpoints/'sample_train--val_loss=0.4511-epoch=4-last.ckpt'
  #+exp_manager.checkpoint_callback_params.every_n_train_steps=200\
  #+exp_manager.checkpoint_callback_params.save_last=true\
fi

if [ ${stage} -le 51 ] && [ ${stop_stage} -ge 51 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v2_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_lr1e-3_emb_batch_size100_batch_size15_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset_lr1e-3_emb_batch_size100_batch_size15/sample_train/2025-09-15_10-43-34/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   subset="American  Australian  British  Filipino  Indian"
   #subset="American"
   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    python3 whisper_nemo_dia/diarize_v2.py \
              --whisper-model large-v2 \
              --local-files-only False\
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model MahmoudAshraf/mms-300m-1130-forced-aligner\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
              --realign-with-punc true
   done
fi

if [ ${stage} -le 52 ] && [ ${stop_stage} -ge 52 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v2_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_lr1e-3_emb_batch_size100_batch_size15_infer/English
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 53 ] && [ ${stop_stage} -ge 53 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v2_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_lr1e-3_emb_batch_size100_batch_size15_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 54 ] && [ ${stop_stage} -ge 54 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v2_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_lr1e-3_emb_batch_size100_batch_size15_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
 fi
#grep -r %tcpWER logs/run_whisper_nemo_dia_aistation_stage52-54.log
#INFO %tcpWER: 99.40% [ 22585 / 22722, 11385 ins, 10132 del, 1068 sub ]
#INFO %tcpWER: 71.76% [ 19527 / 27213, 8645 ins, 9675 del, 1207 sub ]
#INFO %tcpWER: 91.82% [ 22476 / 24478, 10866 ins, 11026 del, 584 sub ]
#INFO %tcpWER: 95.66% [ 16567 / 17318, 8182 ins, 8019 del, 366 sub ]
#INFO %tcpWER: 73.48% [ 15113 / 20567, 6423 ins, 7695 del, 995 sub ]

if [ ${stage} -le 55 ] && [ ${stop_stage} -ge 55 ];then
exp_dir=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset_lr1e-3_emb_batch_size0_batch_size15
mkdir -p $exp_dir
train_temp_dir=$exp_dir/train_temp
dev_temp_dir=$exp_dir/dev_temp
export HYDRA_FULL_ERROR=1
python  whisper_nemo_dia/multiscale_diar_decoder.py --config-path='conf/neural_diarizer' --config-name='msdd_5scl_15_05_50Povl_256x3x32x2.yaml' \
  trainer.devices=2 \
  trainer.max_epochs=10  \
  num_workers=20\
  trainer.strategy=ddp_find_unused_parameters_true\
  model.diarizer.speaker_embeddings.model_path="titanet_large" \
  model.train_ds.manifest_filepath="/maduo/datasets/mlc-slm/train/train_manifest.50step.json" \
  model.validation_ds.manifest_filepath="/maduo/datasets/mlc-slm/dev/dev_manifest.50step.json" \
  model.train_ds.emb_dir="$train_temp_dir" \
  model.validation_ds.emb_dir="$dev_temp_dir" \
  +model.train_ds.num_workers=4\
  exp_manager.name='sample_train' \
  exp_manager.exp_dir=$exp_dir\
  model.emb_batch_size=0\
  batch_size=15

fi

if [ ${stage} -le 56 ] && [ ${stop_stage} -ge 56 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_lr1e-3_emb_batch_size0_batch_size15_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset_lr1e-3_emb_batch_size0_batch_size15/sample_train/2025-09-17_14-17-00/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   subset="American  Australian  British  Filipino  Indian"
   #subset="American"
   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    python3 whisper_nemo_dia/diarize_v2.py \
              --whisper-model large-v3 \
              --local-files-only False\
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model MahmoudAshraf/mms-300m-1130-forced-aligner\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
              --realign-with-punc true
   done
fi

if [ ${stage} -le 57 ] && [ ${stop_stage} -ge 57 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_lr1e-3_emb_batch_size0_batch_size15_infer/English
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 58 ] && [ ${stop_stage} -ge 58 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_lr1e-3_emb_batch_size0_batch_size15_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 59 ] && [ ${stop_stage} -ge 59 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_lr1e-3_emb_batch_size100_batch_size15_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
 fi


if [ ${stage} -le 60 ] && [ ${stop_stage} -ge 60 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v2_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   subset="American  Australian  British  Filipino  Indian"
   #subset="American"
   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    python3 whisper_nemo_dia/diarize_v2.py \
              --whisper-model large-v2 \
              --local-files-only False\
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model MahmoudAshraf/mms-300m-1130-forced-aligner\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
              --realign-with-punc true
   done
fi

if [ ${stage} -le 61 ] && [ ${stop_stage} -ge 61 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v2_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 62 ] && [ ${stop_stage} -ge 62 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v2_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 63 ] && [ ${stop_stage} -ge 63 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v2_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
 fi

#  grep -r tcpWER logs/run_whisper_nemo_dia_aistation_stage60-63.log
#INFO %tcpWER: 31.61% [ 7182 / 22722, 3409 ins, 2156 del, 1617 sub ]
#INFO %tcpWER: 20.92% [ 5694 / 27213, 1671 ins, 2701 del, 1322 sub ]
#INFO %tcpWER: 22.47% [ 5501 / 24478, 2241 ins, 2401 del, 859 sub ]
#INFO %tcpWER: 20.12% [ 3484 / 17318, 1359 ins, 1196 del, 929 sub ]
#INFO %tcpWER: 18.00% [ 3702 / 20567, 695 ins, 1967 del, 1040 sub ]



# finetune whisper and reference https://github.com/vasistalodagala/whisper-finetune
if [ ${stage} -le 70 ] && [ ${stop_stage} -ge 70 ];then
   echo "prepared data"
   python3  whisper_nemo_dia/prepared_custom_dataset_for_fine-tune_whisper.py\
	   --source_data /F00120240032/mlc-slm_task2/data/train_asr/multitask.jsonl\
	   --output_data_dir /maduo/datasets/mlc-slm/train/train_asr
fi
if [ ${stage} -le 71 ] && [ ${stop_stage} -ge 71 ];then
   echo "prepared data"
   python3  whisper_nemo_dia/prepared_custom_dataset_for_fine-tune_whisper.py\
           --source_data /F00120240032/mlc-slm_task2/data/dev_asr/multitask.jsonl\
           --output_data_dir /maduo/datasets/mlc-slm/dev/dev_asr
fi

if [ ${stage} -le 72 ] && [ ${stop_stage} -ge 72 ];then
   exp_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english
   ngpu=2  # number of GPUs to perform distributed training on.
   local_files_dir="/maduo/model_hub/whisper-large-v3/"
   torchrun --nproc_per_node=${ngpu} whisper_nemo_dia/fine-tune_whisper_on_custom_dataset.py \
	--model_name $local_files_dir \
	--local_files_only true\
	--language en \
	--sampling_rate 16000 \
	--num_proc 2 \
	--train_strategy epoch \
	--learning_rate 5e-6\
	--warmup 1000 \
	--train_batchsize 16 \
	--gradient_accumulation_steps 2\
	--eval_batchsize 8 \
	--num_epochs 6 \
	--resume_from_ckpt  $exp_dir/checkpoint-33790\
	--output_dir $exp_dir \
	--train_datasets /maduo/datasets/mlc-slm/train/train_asr  \
	--eval_datasets  /maduo/datasets/mlc-slm/dev/dev_asr

fi

if [ ${stage} -le 73 ] && [ ${stop_stage} -ge 73 ];then
   # covert huggingface model format to ctranslate2 format
   #pip install transformers ctranslate2
   
#   finetune_model_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english
#   cp -r $finetune_model_dir/{tokenizer_config.json,vocab.json,normalizer.json,added_tokens.json,merges.txt} $finetune_model_dir/checkpoint-40548
#   ct2-transformers-converter \
#    --model $finetune_model_dir/checkpoint-40548\
#    --output_dir $finetune_model_dir/checkpoint-40548/whisper-large-v3-ct2_ckpt-40548 \
#    --copy_files preprocessor_config.json tokenizer_config.json vocab.json normalizer.json added_tokens.json\
#    --quantization float16
#

   
   finetune_model_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english
   cp -r $finetune_model_dir/{tokenizer_config.json,vocab.json,normalizer.json,added_tokens.json,merges.txt} $finetune_model_dir/checkpoint-6758
   ct2-transformers-converter \
    --model $finetune_model_dir/checkpoint-6758\
    --output_dir $finetune_model_dir/checkpoint-6758/whisper-large-v3-ct2_ckpt-6758 \
    --copy_files preprocessor_config.json tokenizer_config.json vocab.json normalizer.json added_tokens.json\
    --quantization float16
fi

if [ ${stage} -le 74 ] && [ ${stop_stage} -ge 74 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_step6758_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   subset="American  Australian  British  Filipino  Indian"

   #subset="American"
   finetune_model_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english

   asr_model_dir=$finetune_model_dir/checkpoint-6758/whisper-large-v3-ct2_ckpt-6758
   align_model_local_dir=/maduo/model_hub/MahmoudAshraf/mms-300m-1130-forced-aligner/
   punct_model=/maduo/model_hub/models--kredor--punctuate-all/
   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    python3 whisper_nemo_dia/diarize_v2.py \
              --whisper-model $asr_model_dir \
              --local-files-only True\
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model $align_model_local_dir\
              --ctc-model-local-files-only true\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
              --realign-with-punc true\
              --punct-model $punct_model
   done
fi
 if [ ${stage} -le 75 ] && [ ${stop_stage} -ge 75 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_step6758_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
  
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 76 ] && [ ${stop_stage} -ge 76 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_step6758_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 77 ] && [ ${stop_stage} -ge 77 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_step6758_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
 fi



# compared with stage51-54, stage81-84 will use whisper large_v3 .
if [ ${stage} -le 81 ] && [ ${stop_stage} -ge 81 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_lr1e-3_emb_batch_size100_batch_size15_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset_lr1e-3_emb_batch_size100_batch_size15/sample_train/2025-09-15_10-43-34/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   subset="American  Australian  British  Filipino  Indian"
   #subset="American"
   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    python3 whisper_nemo_dia/diarize_v2.py \
              --whisper-model large-v3 \
              --local-files-only False\
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model MahmoudAshraf/mms-300m-1130-forced-aligner\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
              --realign-with-punc true
   done
fi

if [ ${stage} -le 82 ] && [ ${stop_stage} -ge 82 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_lr1e-3_emb_batch_size100_batch_size15_infer/English
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 83 ] && [ ${stop_stage} -ge 83 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_lr1e-3_emb_batch_size100_batch_size15_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 84 ] && [ ${stop_stage} -ge 84 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_lr1e-3_emb_batch_size100_batch_size15_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
 fi



# compared stage33, stage90 will use fireredAsr-llm-l as ASR model
 if [ ${stage} -le 90 ] && [ ${stop_stage} -ge 90 ];then
   #input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_asr/
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_offical_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   asr_model_dir=/maduo/model_hub/FireRedASR-LLM-L
   align_model_local_dir=/maduo/model_hub/MahmoudAshraf/mms-300m-1130-forced-aligner/
   punct_model=/maduo/model_hub/models--kredor--punctuate-all/
   subset="American  Australian  British  Filipino  Indian"
   #subset="American"
   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    python3 whisper_nemo_dia/diarize_with_firered_llm-l.py  \
              --whisper-model $asr_model_dir \
              --local-files-only False\
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model $align_model_local_dir\
	      --ctc-model-local-files-only true\
              --language en \
              --input-folder  $input_audio_dir\
              --output-folder $output_dir\
              --realign-with-punc true\
	      --punct-model $punct_model
   done

 fi

 # compared stage33, stage100 will use  FireRedASR-LLM-L_ft_on_mlc_slm_eng as ASR model
 if [ ${stage} -le 100 ] && [ ${stop_stage} -ge 100 ];then
   #input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_asr/
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_ft_on_mlc_slm_eng_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   asr_model_dir=/maduo/model_hub/FireRedASR-LLM-L_ft_on_mlc_slm_eng
   align_model_local_dir=/maduo/model_hub/MahmoudAshraf/mms-300m-1130-forced-aligner/
   punct_model=/maduo/model_hub/models--kredor--punctuate-all/
   subset="American  Australian  British  Filipino  Indian"
   #subset="American"
   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    python3 whisper_nemo_dia/diarize_with_firered_llm-l.py  \
              --whisper-model $asr_model_dir \
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model $align_model_local_dir\
	      --ctc-model-local-files-only true\
              --language en \
              --input-folder  $input_audio_dir\
              --output-folder $output_dir\
              --realign-with-punc true\
	      --punct-model $punct_model
   done

 fi


 if [ ${stage} -le 110 ] && [ ${stop_stage} -ge 110 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer_debug/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   #subset="American  Australian  British  Filipino  Indian"
   
   subset="American"
   asr_model_dir=/maduo/model_hub/faster-whisper-large-v3
   align_model_local_dir=/maduo/model_hub/MahmoudAshraf/mms-300m-1130-forced-aligner/
   punct_model=/maduo/model_hub/models--kredor--punctuate-all/
   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    python3 whisper_nemo_dia/diarize_v2.py \
              --whisper-model $asr_model_dir \
              --local-files-only True\
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model $align_model_local_dir\
	      --ctc-model-local-files-only true\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
              --realign-with-punc true\
	      --punct-model $punct_model
   done
fi
# use vad model to get speech activate segments 
if [ ${stage} -le 111 ] && [ ${stop_stage} -ge 111 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   #input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_asr/
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_ft_on_mlc_slm_eng_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   asr_model_dir=/maduo/model_hub/FireRedASR-LLM-L_ft_on_mlc_slm_eng
   align_model_local_dir=/maduo/model_hub/MahmoudAshraf/mms-300m-1130-forced-aligner/
   punct_model=/maduo/model_hub/models--kredor--punctuate-all/
   subset="American Australian  British  Filipino  Indian"
   #subset="American"
   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    python3 whisper_nemo_dia/diarize_with_firered_llm-l.py  \
              --whisper-model $asr_model_dir \
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model $align_model_local_dir\
              --ctc-model-local-files-only true\
              --language en \
              --input-folder  $input_audio_dir/$name\
	      --use-vad true \
              --vad-chunk-length 30.0 \
              --vad-threshold 0.5\
	      --output-folder $output_dir/$name\
              --realign-with-punc true\
              --punct-model $punct_model

   done

 fi

 if [ ${stage} -le 112 ] && [ ${stop_stage} -ge 112 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_ft_on_mlc_slm_eng_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 113 ] && [ ${stop_stage} -ge 113 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_ft_on_mlc_slm_eng_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 114 ] && [ ${stop_stage} -ge 114 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_ft_on_mlc_slm_eng_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
 fi
#cat logs/run_whisper_nemo_dia_aistation_stage111-114.log
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 19891.40 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_ft_on_mlc_slm_eng_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/American/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_ft_on_mlc_slm_eng_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/American/hyp_tcpwer.json
#INFO %tcpWER: 32.88% [ 7472 / 22722, 4051 ins, 1782 del, 1639 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18640.64 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_ft_on_mlc_slm_eng_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_ft_on_mlc_slm_eng_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer.json
#INFO %tcpWER: 21.32% [ 5803 / 27213, 2216 ins, 2225 del, 1362 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18446.00 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_ft_on_mlc_slm_eng_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_ft_on_mlc_slm_eng_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer.json
#INFO %tcpWER: 24.03% [ 5883 / 24478, 2800 ins, 1848 del, 1235 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 16923.94 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_ft_on_mlc_slm_eng_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_ft_on_mlc_slm_eng_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer.json
#INFO %tcpWER: 22.54% [ 3903 / 17318, 1800 ins, 1026 del, 1077 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_ft_on_mlc_slm_eng_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_ft_on_mlc_slm_eng_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer.json
#INFO %tcpWER: 20.34% [ 4183 / 20567, 1392 ins, 1380 del, 1411 sub ]



 # use vad model to get speech activate segments
if [ ${stage} -le 121 ] && [ ${stop_stage} -ge 121 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   #input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_asr/
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   asr_model_dir=/maduo/model_hub/FireRedASR-LLM-L
   align_model_local_dir=/maduo/model_hub/MahmoudAshraf/mms-300m-1130-forced-aligner/
   punct_model=/maduo/model_hub/models--kredor--punctuate-all/
   subset="American Australian  British  Filipino  Indian"
   #subset="American"
   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    python3 whisper_nemo_dia/diarize_with_firered_llm-l.py  \
              --whisper-model $asr_model_dir \
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model $align_model_local_dir\
              --ctc-model-local-files-only true\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --use-vad true \
              --vad-chunk-length 30.0 \
              --vad-threshold 0.5\
              --output-folder $output_dir/$name\
              --realign-with-punc true\
              --punct-model $punct_model

   done

 fi

 if [ ${stage} -le 122 ] && [ ${stop_stage} -ge 122 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 123 ] && [ ${stop_stage} -ge 123 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English

   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 124 ] && [ ${stop_stage} -ge 124 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
 fi
#  logs/run_whisper_nemo_dia_aistation_stage121-124.log
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 19891.40 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/American/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/American/hyp_tcpwer.json
#INFO %tcpWER: 33.96% [ 7717 / 22722, 4067 ins, 1995 del, 1655 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18640.64 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer.json
#INFO %tcpWER: 24.82% [ 6754 / 27213, 2593 ins, 2416 del, 1745 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18446.00 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer.json
#INFO %tcpWER: 26.35% [ 6451 / 24478, 3015 ins, 2087 del, 1349 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 16923.94 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer.json
#INFO %tcpWER: 24.41% [ 4228 / 17318, 1852 ins, 1165 del, 1211 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr_llm_l_and_silero-vad_v5_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer.json
#INFO %tcpWER: 22.16% [ 4557 / 20567, 1423 ins, 1518 del, 1616 sub ]



 if [ ${stage} -le 132 ] && [ ${stop_stage} -ge 132 ];then
   exp_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_1e-6_max_step5000
   ngpu=2  # number of GPUs to perform distributed training on.
   local_files_dir="/maduo/model_hub/whisper-large-v3/"
   export HF_ENDPOINT=https://hf-mirror.com
   torchrun --nproc_per_node=${ngpu} whisper_nemo_dia/fine-tune_whisper_on_custom_dataset.py \
        --model_name $local_files_dir \
        --local_files_only true\
        --language en \
        --sampling_rate 16000 \
        --num_proc 2 \
        --train_strategy steps \
        --learning_rate 1e-6\
        --warmup 1000 \
        --num_steps 5000\
        --train_batchsize 16 \
        --gradient_accumulation_steps 2\
        --eval_batchsize 8 \
        --num_epochs 1 \
        --output_dir $exp_dir \
        --train_datasets /maduo/datasets/mlc-slm/train/train_asr  \
        --eval_datasets  /maduo/datasets/mlc-slm/dev/dev_asr\
	#--resume_from_ckpt  $exp_dir/checkpoint-33790

fi

if [ ${stage} -le 133 ] && [ ${stop_stage} -ge 133 ];then

   finetune_model_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_1e-6_max_step5000
   cp -r $finetune_model_dir/{tokenizer_config.json,vocab.json,normalizer.json,added_tokens.json,merges.txt} $finetune_model_dir/checkpoint-1000
   ct2-transformers-converter \
    --model $finetune_model_dir/checkpoint-1000\
    --output_dir $finetune_model_dir/checkpoint-1000/whisper-large-v3-ct2_ckpt-1000 \
    --quantization float16

   cp -r /maduo/model_hub/faster-whisper-large-v3/{config.json,configuration.json,preprocessor_config.json,tokenizer.json} $finetune_model_dir/checkpoint-1000/whisper-large-v3-ct2_ckpt-1000
fi

if [ ${stage} -le 134 ] && [ ${stop_stage} -ge 134 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   subset="American  Australian  British  Filipino  Indian"

   #subset="American"
   finetune_model_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_1e-6_max_step5000

   asr_model_dir=$finetune_model_dir/checkpoint-1000/whisper-large-v3-ct2_ckpt-1000
   align_model_local_dir=/maduo/model_hub/MahmoudAshraf/mms-300m-1130-forced-aligner
   punct_model=/maduo/model_hub/models--kredor--punctuate-all

   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    #python3 whisper_nemo_dia/diarize_v2.py \
     python3 whisper_nemo_dia/diarize_v2_ft_whisper.py\
              --whisper-model $asr_model_dir \
              --local-files-only True\
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model $align_model_local_dir\
              --ctc-model-local-files-only true\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
              --realign-with-punc true\
              --punct-model $punct_model
   done
fi


if [ ${stage} -le 135 ] && [ ${stop_stage} -ge 135 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 136 ] && [ ${stop_stage} -ge 136 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English

   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 137 ] && [ ${stop_stage} -ge 137 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
 fi

# WARNING Self-overlap detected in reference. Total overlap: 0.00 of 19891.40 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/American/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/American/hyp_tcpwer.json
#INFO %tcpWER: 49.47% [ 11241 / 22722, 3629 ins, 3909 del, 3703 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18640.64 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer.json
#INFO %tcpWER: 40.25% [ 10954 / 27213, 2055 ins, 5087 del, 3812 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18446.00 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer.json
#INFO %tcpWER: 43.19% [ 10572 / 24478, 2782 ins, 4700 del, 3090 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 16923.94 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer.json
#INFO %tcpWER: 39.20% [ 6788 / 17318, 1853 ins, 2777 del, 2158 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer.json
#INFO %tcpWER: 39.84% [ 8193 / 20567, 1672 ins, 3871 del, 2650 sub ]


if [ ${stage} -le 140 ] && [ ${stop_stage} -ge 140 ];then

   finetune_model_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_1e-6_max_step5000
   cp -r $finetune_model_dir/{tokenizer_config.json,vocab.json,normalizer.json,added_tokens.json,merges.txt} $finetune_model_dir/checkpoint-2000
   ct2-transformers-converter \
    --model $finetune_model_dir/checkpoint-2000\
    --output_dir $finetune_model_dir/checkpoint-2000/whisper-large-v3-ct2_ckpt-2000 \
    --quantization float16

   cp -r /maduo/model_hub/faster-whisper-large-v3/{config.json,configuration.json,preprocessor_config.json,tokenizer.json} $finetune_model_dir/checkpoint-2000/whisper-large-v3-ct2_ckpt-2000
fi

if [ ${stage} -le 141 ] && [ ${stop_stage} -ge 141 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_2000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   subset="American  Australian  British  Filipino  Indian"

   #subset="American"
   finetune_model_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_1e-6_max_step5000

   asr_model_dir=$finetune_model_dir/checkpoint-2000/whisper-large-v3-ct2_ckpt-2000
   align_model_local_dir=/maduo/model_hub/MahmoudAshraf/mms-300m-1130-forced-aligner
   punct_model=/maduo/model_hub/models--kredor--punctuate-all

   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    #python3 whisper_nemo_dia/diarize_v2.py \
     python3 whisper_nemo_dia/diarize_v2_ft_whisper.py\
              --whisper-model $asr_model_dir \
              --local-files-only True\
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model $align_model_local_dir\
              --ctc-model-local-files-only true\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
              --realign-with-punc true\
              --punct-model $punct_model
   done
fi

if [ ${stage} -le 142 ] && [ ${stop_stage} -ge 142 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_2000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 143 ] && [ ${stop_stage} -ge 143 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_2000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English

   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 144 ] && [ ${stop_stage} -ge 144 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_2000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
 fi

# WARNING Self-overlap detected in reference. Total overlap: 0.00 of 19891.40 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_2000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/American/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_2000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/American/hyp_tcpwer.json
#INFO %tcpWER: 52.30% [ 11883 / 22722, 3564 ins, 4296 del, 4023 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18640.64 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_2000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_2000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer.json
#INFO %tcpWER: 41.75% [ 11362 / 27213, 1840 ins, 5612 del, 3910 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18446.00 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_2000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_2000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer.json
#INFO %tcpWER: 43.89% [ 10744 / 24478, 2573 ins, 5044 del, 3127 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 16923.94 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_2000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_2000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer.json
#INFO %tcpWER: 40.22% [ 6966 / 17318, 1764 ins, 2964 del, 2238 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_2000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_2000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer.json
#INFO %tcpWER: 40.85% [ 8401 / 20567, 1577 ins, 4042 del, 2782 sub ]


if [ ${stage} -le 145 ] && [ ${stop_stage} -ge 145 ];then

   finetune_model_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_1e-6_max_step5000
   cp -r $finetune_model_dir/{tokenizer_config.json,vocab.json,normalizer.json,added_tokens.json,merges.txt} $finetune_model_dir/checkpoint-5000
   ct2-transformers-converter \
    --model $finetune_model_dir/checkpoint-5000\
    --output_dir $finetune_model_dir/checkpoint-5000/whisper-large-v3-ct2_ckpt-5000 \
    --quantization float16

   cp -r /maduo/model_hub/faster-whisper-large-v3/{config.json,configuration.json,preprocessor_config.json,tokenizer.json} $finetune_model_dir/checkpoint-5000/whisper-large-v3-ct2_ckpt-5000
fi

if [ ${stage} -le 146 ] && [ ${stop_stage} -ge 146 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   subset="American  Australian  British  Filipino  Indian"

   #subset="American"
   finetune_model_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_1e-6_max_step5000

   asr_model_dir=$finetune_model_dir/checkpoint-5000/whisper-large-v3-ct2_ckpt-5000
   align_model_local_dir=/maduo/model_hub/MahmoudAshraf/mms-300m-1130-forced-aligner
   punct_model=/maduo/model_hub/models--kredor--punctuate-all

   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    #python3 whisper_nemo_dia/diarize_v2.py \
     python3 whisper_nemo_dia/diarize_v2_ft_whisper.py\
              --whisper-model $asr_model_dir \
              --local-files-only True\
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model $align_model_local_dir\
              --ctc-model-local-files-only true\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
              --realign-with-punc true\
              --punct-model $punct_model
   done
fi

if [ ${stage} -le 147 ] && [ ${stop_stage} -ge 147 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 148 ] && [ ${stop_stage} -ge 148 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English

   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 149 ] && [ ${stop_stage} -ge 149 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
 fi
# cat logs/run_whisper_nemo_dia_aistation_stage145-149.log
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 19891.40 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/American/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/American/hyp_tcpwer.json
#INFO %tcpWER: 52.75% [ 11986 / 22722, 3510 ins, 4721 del, 3755 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18640.64 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer.json
#INFO %tcpWER: 42.08% [ 11451 / 27213, 1855 ins, 5956 del, 3640 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18446.00 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer.json
#INFO %tcpWER: 43.97% [ 10764 / 24478, 2529 ins, 5326 del, 2909 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 16923.94 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer.json
#INFO %tcpWER: 39.24% [ 6796 / 17318, 1763 ins, 3013 del, 2020 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-6_max_step5000_ckpt_5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer.json
#INFO %tcpWER: 39.55% [ 8135 / 20567, 1488 ins, 4185 del, 2462 sub ]


 if [ ${stage} -le 150 ] && [ ${stop_stage} -ge 150 ];then
   exp_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_1e-7_max_step5000
   ngpu=2  # number of GPUs to perform distributed training on.
   local_files_dir="/maduo/model_hub/whisper-large-v3/"
   export HF_ENDPOINT=https://hf-mirror.com
   torchrun --nproc_per_node=${ngpu} whisper_nemo_dia/fine-tune_whisper_on_custom_dataset.py \
        --model_name $local_files_dir \
        --local_files_only true\
        --language en \
        --sampling_rate 16000 \
        --num_proc 2 \
        --train_strategy steps \
        --learning_rate 1e-7\
        --warmup 1000 \
        --num_steps 5000\
        --train_batchsize 16 \
        --gradient_accumulation_steps 2\
        --eval_batchsize 8 \
        --num_epochs 1 \
        --output_dir $exp_dir \
        --train_datasets /maduo/datasets/mlc-slm/train/train_asr  \
        --eval_datasets  /maduo/datasets/mlc-slm/dev/dev_asr\
        #--resume_from_ckpt  $exp_dir/checkpoint-33790

fi

if [ ${stage} -le 151 ] && [ ${stop_stage} -ge 151 ];then

   finetune_model_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_1e-7_max_step5000
   cp -r $finetune_model_dir/{tokenizer_config.json,vocab.json,normalizer.json,added_tokens.json,merges.txt} $finetune_model_dir/checkpoint-1000
   ct2-transformers-converter \
    --model $finetune_model_dir/checkpoint-1000\
    --output_dir $finetune_model_dir/checkpoint-1000/whisper-large-v3-ct2_ckpt-1000 \
    --quantization float16

   cp -r /maduo/model_hub/faster-whisper-large-v3/{config.json,configuration.json,preprocessor_config.json,tokenizer.json} $finetune_model_dir/checkpoint-1000/whisper-large-v3-ct2_ckpt-1000
fi

if [ ${stage} -le 152 ] && [ ${stop_stage} -ge 152 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   subset="American  Australian  British  Filipino  Indian"

   #subset="American"
   finetune_model_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_1e-7_max_step5000

   asr_model_dir=$finetune_model_dir/checkpoint-1000/whisper-large-v3-ct2_ckpt-1000
   align_model_local_dir=/maduo/model_hub/MahmoudAshraf/mms-300m-1130-forced-aligner
   punct_model=/maduo/model_hub/models--kredor--punctuate-all

   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    #python3 whisper_nemo_dia/diarize_v2.py \
     python3 whisper_nemo_dia/diarize_v2_ft_whisper.py\
              --whisper-model $asr_model_dir \
              --local-files-only True\
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model $align_model_local_dir\
              --ctc-model-local-files-only true\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
              --realign-with-punc true\
              --punct-model $punct_model
   done
fi


if [ ${stage} -le 153 ] && [ ${stop_stage} -ge 153 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 154 ] && [ ${stop_stage} -ge 154 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English

   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 155 ] && [ ${stop_stage} -ge 155 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
 fi
# WARNING Self-overlap detected in reference. Total overlap: 0.00 of 19891.40 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/American/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/American/hyp_tcpwer.json
#INFO %tcpWER: 51.34% [ 11665 / 22722, 3388 ins, 4495 del, 3782 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18640.64 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer.json
#INFO %tcpWER: 42.57% [ 11584 / 27213, 2051 ins, 5502 del, 4031 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18446.00 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer.json
#INFO %tcpWER: 44.43% [ 10876 / 24478, 2577 ins, 5113 del, 3186 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 16923.94 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer.json
#INFO %tcpWER: 44.74% [ 7748 / 17318, 2050 ins, 3473 del, 2225 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer.json
#INFO %tcpWER: 45.59% [ 9377 / 20567, 1589 ins, 4752 del, 3036 sub ]



if [ ${stage} -le 157 ] && [ ${stop_stage} -ge 157 ];then

   finetune_model_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_1e-7_max_step5000
   #cp -r $finetune_model_dir/{tokenizer_config.json,vocab.json,normalizer.json,added_tokens.json,merges.txt} $finetune_model_dir/checkpoint-1000
   ct2-transformers-converter \
    --model $finetune_model_dir/checkpoint-1000\
    --output_dir $finetune_model_dir/checkpoint-1000/whisper-large-v3-ct2_ckpt-1000_debug \
    --copy_files tokenizer_config.json preprocessor_config.json\
    --quantization float16

   #cp -r /maduo/model_hub/faster-whisper-large-v3/{config.json,configuration.json,preprocessor_config.json,tokenizer.json} $finetune_model_dir/checkpoint-1000/whisper-large-v3-ct2_ckpt-1000
fi




if [ ${stage} -le 158 ] && [ ${stop_stage} -ge 158 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   subset="American  Australian  British  Filipino  Indian"

   #subset="American"
   finetune_model_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_1e-7_max_step5000

   asr_model_dir=$finetune_model_dir/checkpoint-1000/whisper-large-v3-ct2_ckpt-1000_debug
   align_model_local_dir=/maduo/model_hub/MahmoudAshraf/mms-300m-1130-forced-aligner
   punct_model=/maduo/model_hub/models--kredor--punctuate-all

   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    #python3 whisper_nemo_dia/diarize_v2.py \
     python3 whisper_nemo_dia/diarize_v2_ft_whisper.py\
              --whisper-model $asr_model_dir \
              --local-files-only True\
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model $align_model_local_dir\
              --ctc-model-local-files-only true\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
              --realign-with-punc true\
              --punct-model $punct_model
   done
fi


if [ ${stage} -le 159 ] && [ ${stop_stage} -ge 159 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 160 ] && [ ${stop_stage} -ge 160 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English

   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 161 ] && [ ${stop_stage} -ge 161 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done

fi
# cat logs/run_whisper_nemo_dia_aistation_stage158-161.log
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/American/hyp_tcpwer.json
#INFO %tcpWER: 62.66% [ 14238 / 22722, 2379 ins, 7668 del, 4191 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18640.64 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer.json
#INFO %tcpWER: 64.69% [ 17604 / 27213, 1559 ins, 12068 del, 3977 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18446.00 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer.json
#INFO %tcpWER: 59.51% [ 14566 / 24478, 2092 ins, 8967 del, 3507 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 16923.94 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer.json
#INFO %tcpWER: 56.42% [ 9771 / 17318, 1958 ins, 5454 del, 2359 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer.json
#INFO %tcpWER: 55.04% [ 11320 / 20567, 1397 ins, 7003 del, 2920 sub ]


if [ ${stage} -le 165 ] && [ ${stop_stage} -ge 165 ];then

   finetune_model_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_1e-7_max_step5000
   mkdir -p $finetune_model_dir/checkpoint-2000_tmp
   cp -r $finetune_model_dir/checkpoint-2000/* $finetune_model_dir/checkpoint-2000_tmp/
   cp -r $finetune_model_dir/{tokenizer_config.json,merges.txt,normalizer.json,added_tokens.json,vocab.json,special_tokens_map.json} $finetune_model_dir/checkpoint-2000_tmp
   ct2-transformers-converter \
    --model $finetune_model_dir/checkpoint-2000_tmp\
    --output_dir $finetune_model_dir/checkpoint-2000_tmp/whisper-large-v3-ct2_ckpt-2000_debug \
    --copy_files tokenizer_config.json preprocessor_config.json\
    --quantization float16

fi



if [ ${stage} -le 166 ] && [ ${stop_stage} -ge 166 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-2000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   subset="American  Australian  British  Filipino  Indian"

   #subset="American"
   finetune_model_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_1e-7_max_step5000

   asr_model_dir=$finetune_model_dir/checkpoint-2000_tmp/whisper-large-v3-ct2_ckpt-2000_debug
   align_model_local_dir=/maduo/model_hub/MahmoudAshraf/mms-300m-1130-forced-aligner
   punct_model=/maduo/model_hub/models--kredor--punctuate-all

   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    #python3 whisper_nemo_dia/diarize_v2.py \
     python3 whisper_nemo_dia/diarize_v2_ft_whisper.py\
              --whisper-model $asr_model_dir \
              --local-files-only True\
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model $align_model_local_dir\
              --ctc-model-local-files-only true\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
              --realign-with-punc true\
              --punct-model $punct_model
   done
fi


if [ ${stage} -le 167 ] && [ ${stop_stage} -ge 167 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-2000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 168 ] && [ ${stop_stage} -ge 168 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-2000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English

   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 169 ] && [ ${stop_stage} -ge 169 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-2000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done

fi
# cat logs/run_whisper_nemo_dia_aistation_stage166-169.log
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 19891.40 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-2000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/American/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-2000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/American/hyp_tcpwer.json
#INFO %tcpWER: 62.13% [ 14117 / 22722, 2654 ins, 7874 del, 3589 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18640.64 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-2000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-2000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer.json
#INFO %tcpWER: 65.36% [ 17786 / 27213, 1546 ins, 12640 del, 3600 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18446.00 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-2000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-2000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer.json
#INFO %tcpWER: 57.24% [ 14010 / 24478, 1996 ins, 8831 del, 3183 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 16923.94 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-2000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-2000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer.json
#INFO %tcpWER: 54.51% [ 9440 / 17318, 1787 ins, 5380 del, 2273 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-2000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-2000_debug_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer.json
#INFO %tcpWER: 57.69% [ 11866 / 20567, 1210 ins, 7911 del, 2745 sub ]



########
# I use the sliding window method officially provided by transformers to use whisper to decode long audio
if [ ${stage} -le 171 ] && [ ${stop_stage} -ge 171 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   subset="American  Australian  British  Filipino  Indian"

   #subset="American"
   finetune_model_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_1e-7_max_step5000

   asr_model_dir=$finetune_model_dir/checkpoint-1000_tmp
   align_model_local_dir=/maduo/model_hub/MahmoudAshraf/mms-300m-1130-forced-aligner
   punct_model=/maduo/model_hub/models--kredor--punctuate-all

   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    #python3 whisper_nemo_dia/diarize_v2.py \
     python3  whisper_nemo_dia/diarize_v2_ft_whisper_hf.py\
              --whisper-model $asr_model_dir \
              --local-files-only True\
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model $align_model_local_dir\
              --ctc-model-local-files-only true\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
              --realign-with-punc true\
              --punct-model $punct_model
   done
fi


if [ ${stage} -le 172 ] && [ ${stop_stage} -ge 172 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 173 ] && [ ${stop_stage} -ge 173 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English

   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 174 ] && [ ${stop_stage} -ge 174 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done

fi

#cat logs/run_whisper_nemo_dia_aistation_stage171-174_3.log
#compute tcpwer
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 19891.40 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/American/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/American/hyp_tcpwer.json
#INFO %tcpWER: 39.54% [ 8984 / 22722, 4311 ins, 2468 del, 2205 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18640.64 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer.json
#INFO %tcpWER: 25.11% [ 6833 / 27213, 2361 ins, 2583 del, 1889 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18446.00 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer.json
#INFO %tcpWER: 29.43% [ 7204 / 24478, 2956 ins, 2644 del, 1604 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 16923.94 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer.json
#INFO %tcpWER: 32.19% [ 5574 / 17318, 2077 ins, 2140 del, 1357 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer.json
#INFO %tcpWER: 28.60% [ 5883 / 20567, 1606 ins, 2611 del, 1666 sub ]


# whisper generate configure is not correct for mlc_slm english, it  will can't decode them.
if [ ${stage} -le 178 ] && [ ${stop_stage} -ge 178 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_v2_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   subset="American  Australian  British  Filipino  Indian"

   #subset="American"
   finetune_model_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_1e-7_max_step5000

   asr_model_dir=$finetune_model_dir/checkpoint-1000_tmp
   align_model_local_dir=/maduo/model_hub/MahmoudAshraf/mms-300m-1130-forced-aligner
   punct_model=/maduo/model_hub/models--kredor--punctuate-all

   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    #python3 whisper_nemo_dia/diarize_v2.py \
     python3  whisper_nemo_dia/diarize_v2_ft_whisper_hf_v2.py\
              --whisper-model $asr_model_dir \
              --local-files-only True\
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model $align_model_local_dir\
              --ctc-model-local-files-only true\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
              --realign-with-punc true\
              --punct-model $punct_model
   done
fi


if [ ${stage} -le 179 ] && [ ${stop_stage} -ge 179 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_v2_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 180 ] && [ ${stop_stage} -ge 180 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_v2_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English

   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 181 ] && [ ${stop_stage} -ge 181 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_v2_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done

fi
#####
# it is correct way to infer until now when i use finetune whisper model. 
#####
# compared with stage171-174, stage185-188 will use vad model(silero_vad_v5) to remove silence segment.
if [ ${stage} -le 185 ] && [ ${stop_stage} -ge 185 ];then
   input_audio_dir=/maduo/datasets/mlc-slm/dev/dev_wav
   output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_vad_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/sample_train.nemo
   #finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/'sample_train--val_loss=0.0117-epoch=10-last.ckpt'
   finetune_model=/maduo/exp/asr_sd/finetune_msdd_model_using_mlc_slm_english_subset/sample_train/2025-09-08_15-26-49/checkpoints/new_sample_train.nemo
   use_speaker_model_from_ckpt=True
   subset="American  Australian  British  Filipino  Indian"

   #subset="American"
   finetune_model_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_1e-7_max_step5000

   asr_model_dir=$finetune_model_dir/checkpoint-1000_tmp
   align_model_local_dir=/maduo/model_hub/MahmoudAshraf/mms-300m-1130-forced-aligner
   punct_model=/maduo/model_hub/models--kredor--punctuate-all

   export HF_ENDPOINT=https://hf-mirror.com
   export OC_CAUSE=1
   for name in $subset;do
    #python3 whisper_nemo_dia/diarize_v2.py \
     python3  whisper_nemo_dia/diarize_v2_ft_whisper_hf_vad.py\
              --whisper-model $asr_model_dir \
              --local-files-only True\
              --diarizer msdd\
              --msdd-model-path $finetune_model\
              --use-speaker-model-from-ckpt $use_speaker_model_from_ckpt\
              --align-model $align_model_local_dir\
              --ctc-model-local-files-only true\
              --language en \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir/$name\
              --realign-with-punc true\
              --punct-model $punct_model\
	      --use-vad true
   done
fi


if [ ${stage} -le 186 ] && [ ${stop_stage} -ge 186 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_vad_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
  #subset="American"
  subset="American  Australian  British  Filipino  Indian"
  for line in $subset;do

          find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done
fi
if [ ${stage} -le 187 ] && [ ${stop_stage} -ge 187 ];then
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_vad_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English

   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 whisper_nemo_dia/generate_hyp_stm_from_whisper_nemo_dia_output.py \
             $input $output
   done
fi

if [ ${stage} -le 188 ] && [ ${stop_stage} -ge 188 ];then
   echo "compute tcpwer "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_vad_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English
   #subset="American"
   subset="American  Australian  British  Filipino  Indian"
   for line in $subset;do
          meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done

fi
#cat logs/run_whisper_nemo_dia_aistation_stage185-188.log
#INFO %tcpWER: 35.33% [ 8028 / 22722, 3940 ins, 2096 del, 1992 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18640.64 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_vad_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_vad_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Australian/hyp_tcpwer.json
#INFO %tcpWER: 24.41% [ 6642 / 27213, 2232 ins, 2470 del, 1940 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18446.00 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_vad_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_vad_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/British/hyp_tcpwer.json
#INFO %tcpWER: 28.99% [ 7096 / 24478, 2959 ins, 2628 del, 1509 sub ]
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 16923.94 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_vad_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_vad_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Filipino/hyp_tcpwer.json
#INFO %tcpWER: 30.14% [ 5220 / 17318, 2010 ins, 1997 del, 1213 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_vad_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/whisper_large_v3_ft_mlc_slm_eng_1epoch_train_strategy_steps_lr_1e-7_max_step5000_ckpt-1000_hf_format_vad_mms-300m-1130-forced-aligner_with_punc_realign_with_ft_msdd_on_mlc_slm_english_msdd_diarization_infer/English/Indian/hyp_tcpwer.json
#INFO %tcpWER: 28.68% [ 5899 / 20567, 1611 ins, 2658 del, 1630 sub ]





if [ ${stage} -le 190 ] && [ ${stop_stage} -ge 190 ];then
   exp_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_1e-5_max_step5000
   ngpu=2  # number of GPUs to perform distributed training on.
   local_files_dir="/maduo/model_hub/whisper-large-v3/"
   export HF_ENDPOINT=https://hf-mirror.com
   torchrun --nproc_per_node=${ngpu} whisper_nemo_dia/fine-tune_whisper_on_custom_dataset.py \
        --model_name $local_files_dir \
        --local_files_only true\
        --language en \
        --sampling_rate 16000 \
        --num_proc 2 \
        --train_strategy steps \
        --learning_rate 1e-5\
        --warmup 500 \
        --num_steps 5000\
        --train_batchsize 16 \
        --gradient_accumulation_steps 8\
        --eval_batchsize 8 \
        --num_epochs 1 \
        --output_dir $exp_dir \
        --train_datasets /maduo/datasets/mlc-slm/train/train_asr  \
        --eval_datasets  /maduo/datasets/mlc-slm/dev/dev_asr\
        #--resume_from_ckpt  $exp_dir/checkpoint-33790
fi

if [ ${stage} -le 200 ] && [ ${stop_stage} -ge 200 ];then
   exp_dir=/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_5e-6_max_step5000
   ngpu=2  # number of GPUs to perform distributed training on.
   local_files_dir="/maduo/model_hub/whisper-large-v3/"
   export HF_ENDPOINT=https://hf-mirror.com
   torchrun --nproc_per_node=${ngpu} whisper_nemo_dia/fine-tune_whisper_on_custom_dataset_v2.py \
        --model_name $local_files_dir \
        --local_files_only true\
        --language en \
        --sampling_rate 16000 \
        --num_proc 2 \
        --train_strategy steps \
        --learning_rate 5e-6\
        --warmup 500 \
        --num_steps 5000\
        --train_batchsize 16 \
        --gradient_accumulation_steps 8\
        --eval_batchsize 8 \
        --num_epochs 1 \
        --output_dir $exp_dir \
        --train_datasets /maduo/datasets/mlc-slm/train/train_asr  \
        --eval_datasets  /maduo/datasets/mlc-slm/dev/dev_asr\
        #--resume_from_ckpt  $exp_dir/checkpoint-33790
fi
