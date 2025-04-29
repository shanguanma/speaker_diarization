#!/usr/bin/env bash

stage=0
stop_stage=1000

sad_type="oracle"
#partition="alimeeting_eval"
#partition="alimeeting_test"
# do cmn on the sub-segment or on the vad segment
subseg_cmn=true
speaker_model="cam++_advanced" # you can start stage 10 and skip stage5-8
model_id="iic/speech_campplus_sv_zh_en_16k-common_advanced"
period_secs=0.75 # 0.25s or 0.50s
. utils/parse_options.sh
. path_for_speaker_diarization.sh


echo " We use sad_type: $sad_type , partition: ${partition}, subseg_cmn: ${subseg_cmn} to do spectral cluster"

# dev test train oracle rttm 
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
   echo "prepared magicdata-ramc kaldi format data"
   ## it has been removed G00000000 utt in rttm file
   # based on the paper "The X-Lance Speaker Diarization System for the Conversational Short-phrase Speaker Diarization Challenge 2022"
   source_data_dir=/data2/shared_datasets/speechdata/18_MagicData-RAMC
   output_dir=/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   python3 magicdata_ramc_prepared_180h_with_g0.py $source_data_dir $output_dir

   data_dir=/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   ## remove  G00000000
   for name in dev test train;do
           grep -v "G00000000" $data_dir/$name/rttm_debug > $data_dir/$name/rttm_debug_nog0
   done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "cssd_testset wav.scp"
   cssd_testset_dir=/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/cssd_testset 
   mkdir -p $cssd_testset_dir
   cssd_testset_audio_dir=/data1/home/maduo/datasets/MagicData-RAMC/cssd_testset/WAV
   ls $cssd_testset_audio_dir/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > $cssd_testset_dir/wav.scp
   cssd_testset_rttm=/data1/home/maduo/datasets/MagicData-RAMC/cssd_testset/ref.rttm
   cat $cssd_testset_rttm>$cssd_testset_dir/rttm_debug_nog0
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "download a speaker model i.e.: ResNet34_LM "
   model_hub_dir=/data1/home/maduo/model_hub/speaker_pretrain_model/zh_cn/wespeaker
   # refer : https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md
   # but it can't wget download the model
   # you need to go to https://hf-mirror.com/
   # search ResNet34_LM model
   git lfs install
   git clone https://hf-mirror.com/Wespeaker/wespeaker-cnceleb-resnet34-LM/
   mv wespeaker-cnceleb-resnet34-LM $model_hub_dir/
fi

# Voice activity detection
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # Set VAD min duration
    min_duration=0.255
    sets="dev test cssd_testset"
    for partition in $sets;do
     oracle_rttm=/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/${partition}/rttm_debug_nog0
     dest_dir=/data1/home/maduo/exp/speaker_diarization/spectral_cluster_wespeaker_magicdata-ramc
     if [ ${sad_type} == "oracle" ]; then
        # Oracle SAD: handling overlapping or too short regions in ground truth RTTM

        echo "Extract oracle sad and store it under ${dest_dir}/data/${partition}/oracle_sad"
        echo "..."
        python3 spectral_cluster_wespeaker/make_oracle_sad.py \
                 --rttm $oracle_rttm \
                 --min-duration $min_duration > ${dest_dir}/data/${partition}/oracle_sad
        echo "Finish oracle sad"
     fi

     if [ ${sad_type} == "system" ]; then
       # System SAD: applying 'silero' VAD
       echo "Extract system sad and store it under ${dest_dir}/data/${partition}/system_sad"
       echo "..."
       python3 spectral_cluster_wespeaker/make_system_sad.py \
               --scp ${dest_dir}/data/${partition}/wav.scp \
               --min-duration $min_duration > ${dest_dir}/data/${partition}/system_sad
       echo "Finish system sad"
     fi
   done
fi


# Extract fbank features
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    sets="dev test cssd_testset"
    for partition in $sets;do
     dest_dir=/data1/home/maduo/exp/speaker_diarization/spectral_cluster_wespeaker_magicdata-ramc
     [ -d "${dest_dir}/exp/${partition}_${sad_type}_sad_fbank" ] && rm -r ${dest_dir}/exp/${partition}_${sad_type}_sad_fbank

     echo "Make Fbank features and store it under ${dest_dir}/exp/${partition}_${sad_type}_sad_fbank"
     echo "..."
     bash spectral_cluster_wespeaker/make_fbank.sh \
            --scp ${dest_dir}/data/${partition}/wav.scp \
            --segments ${dest_dir}/data/${partition}/${sad_type}_sad \
            --store_dir ${dest_dir}/exp/${partition}_${sad_type}_sad_fbank \
            --subseg_cmn ${subseg_cmn} \
            --nj 8
    done
fi



# Extract embeddings
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    dest_dir=/data1/home/maduo/exp/speaker_diarization/spectral_cluster_wespeaker_magicdata-ramc
    pretrained_model=/data1/home/maduo/model_hub/speaker_pretrain_model/zh_cn/wespeaker/wespeaker-cnceleb-resnet34-LM/cnceleb_resnet34_LM.onnx
    sets="dev test cssd_testset"
    for partition in $sets;do
     [ -d "${dest_dir}/exp/${partition}_${sad_type}_sad_embedding" ] && rm -r ${dest_dir}/exp/${partition}_${sad_type}_sad_embedding

     echo "Extract embeddings and store it under ${dest_dir}/exp/${partition}_${sad_type}_sad_embedding"
     echo "..."
     bash spectral_cluster_wespeaker/extract_emb_from_onnx_model.sh \
            --scp ${dest_dir}/exp/${partition}_${sad_type}_sad_fbank/fbank.scp \
            --pretrained_model $pretrained_model \
            --device cuda \
            --store_dir ${dest_dir}/exp/${partition}_${sad_type}_sad_embedding \
            --batch_size 96 \
            --frame_shift 10 \
            --window_secs 1.5 \
            --period_secs 0.75 \
            --subseg_cmn ${subseg_cmn} \
            --nj 1
    done
fi


# Applying spectral clustering algorithm
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    dest_dir=/data1/home/maduo/exp/speaker_diarization/spectral_cluster_wespeaker_magicdata-ramc
    sets="dev test cssd_testset"
    for partition in $sets;do
     [ -f "${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels" ] && rm ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels

     echo "Doing spectral clustering and store the result in ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels"
     echo "..."
     python3 spectral_cluster_wespeaker/spectral_clusterer.py \
            --scp ${dest_dir}/exp/${partition}_${sad_type}_sad_embedding/emb.scp \
            --output ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels
    done
fi


# Convert labels to RTTMs
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    sets="dev test cssd_testset"
    for partition in $sets;do
     dest_dir=/data1/home/maduo/exp/speaker_diarization/spectral_cluster_wespeaker_magicdata-ramc
     python3 spectral_cluster_wespeaker/make_rttm.py \
            --labels ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels \
            --channel 1 > ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_rttm
    done
fi
# # Evaluate the result
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    dest_dir=/data1/home/maduo/exp/speaker_diarization/spectral_cluster_wespeaker_magicdata-ramc
    sets="dev test cssd_testset"
    for partition in $sets;do
     oracle_rttm=/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/${partition}/rttm_debug_nog0
     predict_rttm=${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_rttm
     echo -e "Get the DER results\n..."
     echo -e "DER, MS, FA, SC "
     perl SCTK-2.4.12/src/md-eval/md-eval.pl \
          -c 0.25 \
          -r $oracle_rttm\
          -s $predict_rttm 2>&1 | tee $dest_dir/exp/spectral_cluster/${partition}_${sad_type}_sad_res
   done
fi



# Extract embeddings using better model i.e. cam++ is trained on cn-celeb and voxceleb by modelscope
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    dest_dir=/data1/home/maduo/exp/speaker_diarization/spectral_cluster_wespeaker_magicdata-ramc
    #model_id="iic/speech_campplus_sv_zh_en_16k-common_advanced"
    sets="dev test cssd_testset"
    for partition in $sets;do
     [ -d "${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding" ] && rm -r ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding

     echo "Extract embeddings and store it under ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding"
     echo "..."
     bash spectral_cluster_wespeaker/extract_emb_from_pt_modelscope_model.sh \
            --scp ${dest_dir}/exp/${partition}_${sad_type}_sad_fbank/fbank.scp \
            --model_id $model_id\
            --store_dir ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding \
            --batch_size 96 \
            --frame_shift 10 \
            --window_secs 1.5 \
            --period_secs 0.75 \
            --subseg_cmn ${subseg_cmn} \
            --nj 1
    done
fi

# Applying spectral clustering algorithm
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    dest_dir=/data1/home/maduo/exp/speaker_diarization/spectral_cluster_wespeaker_magicdata-ramc
    sets="dev test cssd_testset"
    for partition in $sets;do
     [ -f "${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}" ] && rm ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}

     echo "Doing spectral clustering and store the result in ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}"
     echo "..."
     python3 spectral_cluster_wespeaker/spectral_clusterer.py \
            --scp ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding/emb.scp \
            --output ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}
    done
fi


# Convert labels to RTTMs
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    dest_dir=/data1/home/maduo/exp/speaker_diarization/spectral_cluster_wespeaker_magicdata-ramc
    sets="dev test cssd_testset"
    for partition in $sets;do
     echo "Doing make rttm and store the result in ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_rttm_${speaker_model}"
     echo "..."
     python3 spectral_cluster_wespeaker/make_rttm.py \
            --labels ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model} \
            --channel 1 > ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_rttm_${speaker_model}
    done
fi
# # Evaluate the result
if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    dest_dir=/data1/home/maduo/exp/speaker_diarization/spectral_cluster_wespeaker_magicdata-ramc
    sets="dev test cssd_testset"
    for partition in $sets;do
     oracle_rttm=/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/${partition}/rttm_debug_nog0
     predict_rttm=${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_rttm_${speaker_model}
     echo -e "Get the DER results\n..."
     echo -e "DER, MS, FA, SC "
     perl SCTK-2.4.12/src/md-eval/md-eval.pl \
          -c 0.25 \
          -r $oracle_rttm\
          -s $predict_rttm 2>&1 | tee $dest_dir/exp/spectral_cluster/${partition}_${sad_type}_sad_${speaker_model}_res
    done
fi



# I want to finetune period_secs, it is from `FROM MODULAR TO END-TO-END SPEAKER DIARIZATION`
if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
    dest_dir=/data1/home/maduo/exp/speaker_diarization/spectral_cluster_wespeaker_magicdata-ramc
    sets="dev test cssd_testset"
    for partition in $sets;do
     [ -d "${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding_shift${period_secs}" ] && rm -r ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding_shift${period_secs}

     echo "Extract embeddings and store it under ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding_shift${period_secs}"
     echo "..."
     bash spectral_cluster_wespeaker/extract_emb_from_pt_modelscope_model.sh \
            --scp ${dest_dir}/exp/${partition}_${sad_type}_sad_fbank/fbank.scp \
            --model_id $model_id\
            --store_dir ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding_shift${period_secs} \
            --batch_size 96 \
            --frame_shift 10 \
            --window_secs 1.5 \
            --period_secs $period_secs \
            --subseg_cmn ${subseg_cmn} \
            --nj 1
    done
fi

# Applying spectral clustering algorithm
if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
    dest_dir=/data1/home/maduo/exp/speaker_diarization/spectral_cluster_wespeaker_magicdata-ramc
    sets="dev test cssd_testset"
    for partition in $sets;do
     [ -f "${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}_shift${period_secs}" ] && rm ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}_shift${period_secs}

     echo "Doing spectral clustering and store the result in ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}_shift${period_secs}"
     echo "..."
     python3 spectral_cluster_wespeaker/spectral_clusterer.py \
            --scp ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding_shift${period_secs}/emb.scp \
            --output ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}_shift${period_secs}
     done
fi


# Convert labels to RTTMs
if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
    dest_dir=/data1/home/maduo/exp/speaker_diarization/spectral_cluster_wespeaker_magicdata-ramc
    sets="dev test cssd_testset"
    for partition in $sets;do
     echo "Doing make rttm and store the result in ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_rttm_${speaker_model}_shift${period_secs}"
     echo "..."
     python3 spectral_cluster/make_rttm.py \
            --labels ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}_shift${period_secs} \
            --channel 1 > ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_rttm_${speaker_model}_shift${period_secs}
    done
fi
# # Evaluate the result
if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ]; then
    dest_dir=/data1/home/maduo/exp/speaker_diarization/spectral_cluster_wespeaker_magicdata-ramc
    sets="dev test cssd_testset"
    for partition in $sets;do
     oracle_rttm=/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/${partition}/rttm_debug_nog0
     predict_rttm=${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_rttm_${speaker_model}_shift${period_secs}
     echo -e "Get the DER results\n..."
     echo -e "DER, MS, FA, SC "
     perl SCTK-2.4.12/src/md-eval/md-eval.pl \
          -c 0.25 \
          -r $oracle_rttm\
          -s $predict_rttm 2>&1 | tee $dest_dir/exp/spectral_cluster/${partition}_${sad_type}_sad_${speaker_model}_shift${period_secs}_res
    done
fi
