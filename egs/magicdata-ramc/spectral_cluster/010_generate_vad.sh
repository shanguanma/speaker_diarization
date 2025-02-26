#!/usr/bin/env bash


stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_nn_vad.sh
#vad_type="oracle"

## note: you need to download this vad method recipe:https://github.com/shanguanma/voice-activity-detection.git
## the recipe offer 1. data prepared for vad model
##                  2. how to train a vad model use the above data
##                  3. predict vad file using the above trained vad model
##                  4. the vad file is used at spectral cluster for diarization task
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
   echo "prepare kaldi format for  magicdata-RAMC data"
   vad_code=/home/maduo/codebase/voice-activity-detection
   python $vad_code/prepare_magicdata_180h.py
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "prepare vad  magicdata-RAMC format data for train vad model "
   data_path=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC/
   vad_code=/home/maduo/codebase/voice-activity-detection
   #for name in train dev test;do
   for name in train  test;do
      python $vad_code/prepared_vad_data_for_magicdata-RAMC.py \
              --data_path $data_path \
              --type $name
   done
fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "train a vad model using magicdata-RAMC dataset "
   vad_code=/home/maduo/codebase/voice-activity-detection

   CUDA_VISIBLE_DEVICES=0 python $vad_code/main.py train $vad_code/tests/configs/vad/train_config_magicdata-RAMC.yaml

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "generate vad json file using pretrain transformer vad model "
   #echo "transformer vad model is from https://github.com/voithru/voice-activity-detection/blob/main/tests/checkpoints/vad/sample.checkpoint"
   echo "I used my pretrain vad model to get testset vad segement of magicdata-RAMC"
   vad_code=/home/maduo/codebase/voice-activity-detection
   vad_model=/home/maduo/codebase/voice-activity-detection/results/tests/vad/magicdata-RAMC-sample/v001/checkpoints
   data=data/magicdata-RAMC/test/
   output_dir=$data/predict_vad
   mkdir -p $output_dir
   for audio_path in `awk '{print $2}' $data/wav.scp`;do
      audio_name=$(basename $audio_path | sed s:.wav$::)
      python  $vad_code/main.py predict \
	        --threshold 0.91 \
	        --output-path $output_dir/${audio_name}.json \
                $audio_path\
	        $vad_model
   done
 fi


 if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "generate vad json file using pretrain transformer vad model "
   #echo "transformer vad model is from https://github.com/voithru/voice-activity-detection/blob/main/tests/checkpoints/vad/sample.checkpoint"
   echo "I used my pretrain vad model to get testset vad segement of magicdata-RAMC"
   vad_code=/home/maduo/codebase/voice-activity-detection
   vad_model=/home/maduo/codebase/voice-activity-detection/results/tests/vad/magicdata-RAMC-sample/v001/checkpoints/vad-magicdata-RAMC-sample-v001-epoch-019-val-acc-0.92249.checkpoint
   data=data/magicdata-RAMC/test/
   #output_dir=$data/predict_vad
   #mkdir -p $output_dir
   vad_threshold="0.5 0.6 0.7 0.8 0.9 0.91 0.92"
   for name in $vad_threshold;do
    for audio_path in `awk '{print $2}' $data/wav.scp`;do
      audio_name=$(basename $audio_path | sed s:.wav$::)
      echo "audio_path : $audio_path"
      output_dir=$data/predict_vad_${name}
      mkdir -p $output_dir
      python  $vad_code/main.py predict \
            --threshold $name \
            --output-path $output_dir/${audio_name}.json \
            $audio_path\
            $vad_model
   done
  done
 fi

 if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "generate vad json file using pretrain transformer vad model "
   #echo "transformer vad model is from https://github.com/voithru/voice-activity-detection/blob/main/tests/checkpoints/vad/sample.checkpoint"
   echo "I used my pretrain vad model to get testset vad segement of magicdata-RAMC"
   vad_code=/home/maduo/codebase/voice-activity-detection
   vad_model=/home/maduo/codebase/voice-activity-detection/results/tests/vad/magicdata-RAMC-sample/v001/checkpoints/vad-magicdata-RAMC-sample-v001-epoch-019-val-acc-0.92249.checkpoint
   data=data/magicdata-RAMC/test/
   #output_dir=$data/predict_vad
   #mkdir -p $output_dir
   vad_threshold="0.1 0.2 0.3 0.4 0.45 0.46 0.47 0.51 0.52 0.53"
   for name in $vad_threshold;do
    for audio_path in `awk '{print $2}' $data/wav.scp`;do
      audio_name=$(basename $audio_path | sed s:.wav$::)
      echo "audio_path : $audio_path"
      output_dir=$data/predict_vad_${name}
      mkdir -p $output_dir
      python  $vad_code/main.py predict \
            --threshold $name \
            --output-path $output_dir/${audio_name}.json \
            $audio_path\
            $vad_model
   done
  done
 fi


  if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "generate vad json file using pretrain transformer vad model "
   #echo "transformer vad model is from https://github.com/voithru/voice-activity-detection/blob/main/tests/checkpoints/vad/sample.checkpoint"
   echo "I used my pretrain vad model to get testset vad segement of magicdata-RAMC"
   vad_code=/home/maduo/codebase/voice-activity-detection
   vad_model=/home/maduo/codebase/voice-activity-detection/results/tests/vad/magicdata-RAMC-sample/v001/checkpoints/vad-magicdata-RAMC-sample-v001-epoch-019-val-acc-0.92249.checkpoint
   data=data/magicdata-RAMC/test/
   #output_dir=$data/predict_vad
   #mkdir -p $output_dir
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   for name in $vad_threshold;do
    for audio_path in `awk '{print $2}' $data/wav.scp`;do
      audio_name=$(basename $audio_path | sed s:.wav$::)
      echo "audio_path : $audio_path"
      output_dir=$data/predict_vad_${name}
      mkdir -p $output_dir
      python  $vad_code/main.py predict \
            --threshold $name \
            --output-path $output_dir/${audio_name}.json \
            $audio_path\
            $vad_model
   done
  done
 fi

