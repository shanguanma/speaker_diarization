#!/usr/bin/env bash
stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_dia_cuda11.8_py3111_aistation.sh

hf_token=$1 # your huggingface token, because github reject displace it at script.
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "using fire-red-asr :FireReadASR-AED-L model"

    /maduo/codebase/sherpa-onnx/build/bin/sherpa-onnx-offline \
     --tokens=/maduo/model_hub/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/tokens.txt \
     --fire-red-asr-encoder=/maduo/model_hub/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/encoder.int8.onnx \
     --fire-red-asr-decoder=/maduo/model_hub/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/decoder.int8.onnx \
     --num-threads=1 \
     /maduo/model_hub/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/test_wavs/0.wav

fi

#if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
#   echo "using fire-red-asr :FireReadASR-AED-L model"
#   python3 /maduo/codebase/sherpa-onnx/python-api-examples/generate-subtitles.py  \
#      --silero-vad-model=/maduo/model_hub/vad/silero_vad.int8.onnx \
#      --tokens=/maduo/model_hub/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/tokens.txt \
#      --fire-red-asr-encoder=/maduo/model_hub/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/encoder.int8.onnx \
#      --fire-red-asr-decoder=/maduo/model_hub/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/decoder.int8.onnx \
#      --num-threads=2 \
#      /maduo/model_hub/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/test_wavs/0.wav
#fi

 if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   . /maduo/package/cuda_11.8_path.sh   
   whisperx \
	   /F00120240032/mlc-slm_task2/MLC-SLM_Workshop-Development_Set/MLC-SLM_Workshop-Development_Set/data/English/American/0517_007.wav \
	   --model large-v2 \
	   --align_model WAV2VEC2_ASR_LARGE_LV60K_960H \
	   --batch_size 4
 fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "for english"
   . /maduo/package/cuda_11.8_path.sh
   whisperx \
           /F00120240032/mlc-slm_task2/MLC-SLM_Workshop-Development_Set/MLC-SLM_Workshop-Development_Set/data/English/American/0517_007.wav \
           --model large-v3 \
           --align_model WAV2VEC2_ASR_LARGE_LV60K_960H \
           --batch_size 4\
	   --diarize \
	   --min_speakers 2 --max_speakers 2\
	   --hf_token $hf_token\
	  --return_char_alignments 
 fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "for vi"
   . /maduo/package/cuda_11.8_path.sh
      	whisperx /F00120240032/mlc-slm_task2/MLC-SLM_Workshop-Development_Set/MLC-SLM_Workshop-Development_Set/data/Vietnamese/0055_006_phone.wav\
	   --model large-v3 \
	   --language vi\
	   --align_model "nguyenvulebinh/wav2vec2-base-vi-vlsp2020"\
           --batch_size 4\
           --diarize \
           --min_speakers 2 --max_speakers 2\
           --hf_token $hf_token

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "prepared devset ground truth stm file"
   input_dir=/F00120240032/mlc-slm_task2/MLC-SLM_Workshop-Development_Set/MLC-SLM_Workshop-Development_Set/data
   out_dir=/maduo/datasets/mlc-slm/dev # gen dev_rttm.list
   python3  fusion/prepare_reference_files.py \
	   --dataset_path $input_dir \
	   --output_path $out_dir\
	   --dataset_part dev
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
  dev_data_dir=/F00120240032/mlc-slm_task2/MLC-SLM_Workshop-Development_Set/MLC-SLM_Workshop-Development_Set/data
  dev_segments_path=/maduo/datasets/mlc-slm/dev/segments
  python fusion/prepare_segments.py --data_dir $dev_data_dir --segments_path $dev_segments_path
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
  dev_segments_path=/maduo/datasets/mlc-slm/dev/segments
  dev_split_log_dir=/maduo/datasets/mlc-slm/dev/logs
  dev_split_dir=$dev_split_log_dir
  dev_data_dir=/maduo/datasets/mlc-slm/dev/
  cmd=fusion/run.pl
  nj=30
  nutt=$(<${dev_segments_path} wc -l)
  nj=$((nj<nutt?nj:nutt))
  mkdir -p $dev_split_log_dir
  split_segments=""
  for n in $(seq ${nj}); do
      split_segments="${split_segments} ${dev_split_log_dir}/segments.${n}"
  done
  perl fusion/split_scp.pl "${dev_segments_path}" ${split_segments}

  # shellcheck disable=SC2046
  ${cmd} "JOB=1:${nj}" "${dev_split_log_dir}/split_wavs.JOB.log" \
      python fusion/split_wav.py \
          "--segments_path=${dev_split_log_dir}/segments.JOB" \
          "--output_dir=${dev_split_dir}/split.JOB"

  cat ${dev_split_dir}/split.*/wav.scp | shuf > $dev_data_dir/wav.scp
  cat ${dev_split_dir}/split.*/text | shuf > $dev_data_dir/text
fi
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
   echo "Simple text normalization(remove punctuations ) for the text file"
   dev_data_dir=/maduo/datasets/mlc-slm/dev/
   python3 fusion/text_normalization.py  --input $dev_data_dir/text  --output $dev_data_dir/text_tn
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
   echo "prepared ref stm for devset"
    dev_data_dir=/maduo/datasets/mlc-slm/dev/
    ref_rttm_list=$dev_data_dir/dev_rttm.list
	python fusion/generate_ref_stm.py --rttm $ref_rttm_list --text $dev_data_dir/text_tn --out_file $dev_data_dir/ref.stm
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
   dev_data_dir=/maduo/datasets/mlc-slm/dev
   mkdir -p $dev_data_dir/English
   for line in American  Australian  British  Filipino  Indian;do
 	   grep -r $line  $dev_data_dir/dev_wav.list > $dev_data_dir/English/${line}_dev_wav.list
   done
fi
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   echo "for english"
   . /maduo/package/cuda_11.8_path.sh # for loading library libcudnn_ops_infer.so.8 
   dev_data_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/fusion/whisper_v3_align_model_wav2vec2_asr_large_lv60k_960h_pyannote_sd_v3.1/English
   for line in American  Australian  British  Filipino  Indian;do
    mkdir -p $dest_dir/English/$line 
    cat $dev_data_dir/English/${line}_dev_wav.list |  while read subline;do
     whisperx $subline \
           --model large-v3 \
           --align_model WAV2VEC2_ASR_LARGE_LV60K_960H \
           --batch_size 4\
	   --language en\
           --diarize \
           --min_speakers 2 --max_speakers 2\
           --hf_token $hf_token\
	   --output_dir  $dest_dir/English/$line
   done
  done
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
  echo "generate srt files"
  dest_dir=/maduo/exp/asr_sd/fusion/whisper_v3_align_model_wav2vec2_asr_large_lv60k_960h_pyannote_sd_v3.1/English
  for line in American  Australian  British  Filipino  Indian;do

	  find $dest_dir/$line -iname "*.srt" > $dest_dir/$line/srt_list.txt
  done 
fi
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
   dest_dir=/maduo/exp/asr_sd/fusion/whisper_v3_align_model_wav2vec2_asr_large_lv60k_960h_pyannote_sd_v3.1/English
   for line in American  Australian  British  Filipino  Indian;do
     input=$dest_dir/$line/srt_list.txt
     output=$dest_dir/$line/hyp.stm
     python3 fusion/generate_hyp_stm_from_whisperx_output.py \
	     $input $output
   done
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
   input_dir=/maduo/datasets/mlc-slm/dev/
   for line in American  Australian  British  Filipino  Indian;do
	 grep -r $line $input_dir/ref.stm > $input_dir/English/${line}_ref.stm
   done
fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
   echo "prepared "
   input_dir=/maduo/datasets/mlc-slm/dev/
   dest_dir=/maduo/exp/asr_sd/fusion/whisper_v3_align_model_wav2vec2_asr_large_lv60k_960h_pyannote_sd_v3.1/English
   for line in American  Australian  British  Filipino  Indian;do
	  meeteval-wer tcpwer -r  $input_dir/English/${line}_ref.stm -h $dest_dir/$line/hyp.stm --collar 5
   done
# result:
# WARNING Self-overlap detected in hypothesis. Total overlap: 0.58 of 23493.31 (0.00%).
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 19891.40 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/fusion/whisper_v3_align_model_wav2vec2_asr_large_lv60k_960h_pyannote_sd_v3.1/English/American/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/fusion/whisper_v3_align_model_wav2vec2_asr_large_lv60k_960h_pyannote_sd_v3.1/English/American/hyp_tcpwer.json
#INFO %tcpWER: 50.44% [ 11462 / 22722, 5647 ins, 4392 del, 1423 sub ]
#WARNING Self-overlap detected in hypothesis. Total overlap: 0.88 of 22748.30 (0.00%).
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18640.64 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/fusion/whisper_v3_align_model_wav2vec2_asr_large_lv60k_960h_pyannote_sd_v3.1/English/Australian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/fusion/whisper_v3_align_model_wav2vec2_asr_large_lv60k_960h_pyannote_sd_v3.1/English/Australian/hyp_tcpwer.json
#INFO %tcpWER: 30.45% [ 8285 / 27213, 3235 ins, 3893 del, 1157 sub ]
#WARNING Self-overlap detected in hypothesis. Total overlap: 1.04 of 20484.76 (0.01%).
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 18446.00 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/fusion/whisper_v3_align_model_wav2vec2_asr_large_lv60k_960h_pyannote_sd_v3.1/English/British/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/fusion/whisper_v3_align_model_wav2vec2_asr_large_lv60k_960h_pyannote_sd_v3.1/English/British/hyp_tcpwer.json
#INFO %tcpWER: 38.13% [ 9333 / 24478, 4142 ins, 4245 del, 946 sub ]
#WARNING Self-overlap detected in hypothesis. Total overlap: 0.26 of 16362.73 (0.00%).
#WARNING Self-overlap detected in reference. Total overlap: 0.00 of 16923.94 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/fusion/whisper_v3_align_model_wav2vec2_asr_large_lv60k_960h_pyannote_sd_v3.1/English/Filipino/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/fusion/whisper_v3_align_model_wav2vec2_asr_large_lv60k_960h_pyannote_sd_v3.1/English/Filipino/hyp_tcpwer.json
#INFO %tcpWER: 26.48% [ 4586 / 17318, 2093 ins, 1981 del, 512 sub ]
#WARNING Self-overlap detected in hypothesis. Total overlap: 0.62 of 19207.32 (0.00%).
#INFO Wrote: /maduo/exp/asr_sd/fusion/whisper_v3_align_model_wav2vec2_asr_large_lv60k_960h_pyannote_sd_v3.1/English/Indian/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/fusion/whisper_v3_align_model_wav2vec2_asr_large_lv60k_960h_pyannote_sd_v3.1/English/Indian/hyp_tcpwer.json
#INFO %tcpWER: 22.53% [ 4633 / 20567, 1143 ins, 2487 del, 1003 sub ]
fi
