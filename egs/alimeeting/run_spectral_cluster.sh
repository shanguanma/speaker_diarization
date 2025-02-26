#!/usr/bin/env bash

stage=0
stop_stage=1000

sad_type="oracle"
partition="alimeeting_eval"
#partition="alimeeting_test"
# do cmn on the sub-segment or on the vad segment
subseg_cmn=true
speaker_model="cam++_advanced" # you can start stage 10 and skip stage5-8
model_id="iic/speech_campplus_sv_zh_en_16k-common_advanced"
period_secs=0.75 # 0.25s or 0.50s
. utils/parse_options.sh
. path_for_speaker_diarization.sh


echo " We use sad_type: $sad_type , partition: ${partition}, subseg_cmn: ${subseg_cmn} to do spectral cluster"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
   echo "prepared oracle rttm"
   splits="Eval Test"
   alimeeting_corpus=/mntcephfs/lab_data/maduo/datasets/alimeeting/
   dest_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad/
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
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "prepared oracle rttm"
   splits="Train"
   alimeeting_corpus=/mntcephfs/lab_data/maduo/datasets/alimeeting/
   dest_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad/
   for name in $splits;do
       audio_dir=$alimeeting_corpus/${name}_Ali_far/audio_dir/
       textgrid_dir=$alimeeting_corpus/${name}_Ali_far/textgrid_dir
       bash ts_vad_ddp_phase1/prepare_rttm_for_ts_vad.sh\
           --stage 0 \
           --dest-dir $dest_dir \
           --split $name\
           --audio-dir $audio_dir\
           --textgrid-dir $textgrid_dir
   done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "download a speaker model i.e.: ResNet34_LM "
   model_hub_dir=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/wespeaker/
   # refer : https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md
   # but it can't wget download the model
   # you need to go to https://hf-mirror.com/
   # search ResNet34_LM model
   git lfs install
   git clone https://hf-mirror.com/Wespeaker/wespeaker-cnceleb-resnet34-LM/
   mv wespeaker-cnceleb-resnet34-LM $model_hub_dir/
fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
    # Create wav.scp for eval audios
    dest_eval_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/data/alimeeting_eval
    mkdir -p $dest_eval_dir
    alimeeting_eval_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/audio_dir/
    ls $alimeeting_eval_dir/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > $dest_eval_dir/wav.scp

    # Create wav.scp for test audios
    dest_test_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/data/alimeeting_test
    mkdir -p $dest_test_dir
    alimeeting_test_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Test_Ali/Test_Ali_far/audio_dir/
    ls $alimeeting_test_dir/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > $dest_test_dir/wav.scp

    #  Create wav.scp for train audios
    dest_train_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/data/alimeeting_train
    mkdir -p $dest_train_dir
    alimeeting_train_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Train_Ali_far/audio_dir/
    ls $alimeeting_train_dir/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > $dest_train_dir/wav.scp
fi



# Voice activity detection
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # Set VAD min duration
    # oracle_rttm is getting from /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/scripts/alimeeting/011_prepare_rttm_for_ts_vad_hltsz.sh
    min_duration=0.255
    oracle_rttm=/mntcephfs/lab_data/maduo/model_hub/ts_vad/${partition}.rttm
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/
    if [ ${sad_type} == "oracle" ]; then
        # Oracle SAD: handling overlapping or too short regions in ground truth RTTM

        echo "Extract oracle sad and store it under ${dest_dir}/data/${partition}/oracle_sad"
        echo "..."
        python3 spectral_cluster/make_oracle_sad.py \
                 --rttm $oracle_rttm \
                 --min-duration $min_duration > ${dest_dir}/data/${partition}/oracle_sad
        echo "Finish oracle sad"
    fi

    if [ ${sad_type} == "system" ]; then
       # System SAD: applying 'silero' VAD
       echo "Extract system sad and store it under ${dest_dir}/data/${partition}/system_sad"
       echo "..."
       python3 spectral_cluster/make_system_sad.py \
               --scp ${dest_dir}/data/${partition}/wav.scp \
               --min-duration $min_duration > ${dest_dir}/data/${partition}/system_sad
       echo "Finish system sad"
    fi
fi


# Extract fbank features
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/
    [ -d "${dest_dir}/exp/${partition}_${sad_type}_sad_fbank" ] && rm -r ${dest_dir}/exp/${partition}_${sad_type}_sad_fbank

    echo "Make Fbank features and store it under ${dest_dir}/exp/${partition}_${sad_type}_sad_fbank"
    echo "..."
    bash spectral_cluster/make_fbank.sh \
            --scp ${dest_dir}/data/${partition}/wav.scp \
            --segments ${dest_dir}/data/${partition}/${sad_type}_sad \
            --store_dir ${dest_dir}/exp/${partition}_${sad_type}_sad_fbank \
            --subseg_cmn ${subseg_cmn} \
            --nj 8
fi



# Extract embeddings
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/
    pretrained_model=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/wespeaker/wespeaker-cnceleb-resnet34-LM/cnceleb_resnet34_LM.onnx
    [ -d "${dest_dir}/exp/${partition}_${sad_type}_sad_embedding" ] && rm -r ${dest_dir}/exp/${partition}_${sad_type}_sad_embedding

    echo "Extract embeddings and store it under ${dest_dir}/exp/${partition}_${sad_type}_sad_embedding"
    echo "..."
    bash spectral_cluster/extract_emb_from_onnx_model.sh \
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
fi


# Applying spectral clustering algorithm
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/
    [ -f "${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels" ] && rm ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels

    echo "Doing spectral clustering and store the result in ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels"
    echo "..."
    python3 spectral_cluster/spectral_clusterer.py \
            --scp ${dest_dir}/exp/${partition}_${sad_type}_sad_embedding/emb.scp \
            --output ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels
fi


# Convert labels to RTTMs
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/
    python3 spectral_cluster/make_rttm.py \
            --labels ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels \
            --channel 1 > ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_rttm
fi
# # Evaluate the result
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/
    oracle_rttm=/mntcephfs/lab_data/maduo/model_hub/ts_vad/${partition}.rttm
    predict_rttm=${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_rttm
    echo -e "Get the DER results\n..."
    echo -e "DER, MS, FA, SC "
    perl SCTK-2.4.12/src/md-eval/md-eval.pl \
          -c 0.25 \
          -r $oracle_rttm\
          -s $predict_rttm 2>&1 | tee $dest_dir/exp/spectral_cluster/${partition}_${sad_type}_sad_res

fi
# cat logs/run_spectral_cluster_stage3-8_alimeeting_eval_oracle.log
# Eval set oracle vad, speaker embedding is from cn-cnceleb resnet34-lm model.
# DER, MS, FA, SC
# 14.57/12.92/0.00/1.65
# cat logs/run_spectral_cluster_stage3-8_alimeeting_test_oracle.log
# Test set oracle vad, speaker embedding is from cn-cnceleb resnet34-lm model.
# DER, MS, FA, SC
# 13.98/12.72/0.00/1.26

# cat logs/run_spectral_cluster_stage3-8_alimeeting_eval_system.log
# Eval set system vad, speaker embedding is from cn-cnceleb resnet34-lm model.
# DER, MS, FA, SC
# 17.91/15.39/0.82/1.70

# cat logs/run_spectral_cluster_stage6-8_alimeeting_test_system.log
# Test set system vad, speaker embedding is from cn-cnceleb resnet34-lm model.
# DER, MS, FA, SC
# 18.44/16.72/0.39/1.33



# Extract embeddings using better model i.e. cam++ is trained on cn-celeb and voxceleb by modelscope
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/
    #model_id="iic/speech_campplus_sv_zh_en_16k-common_advanced"
    [ -d "${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding" ] && rm -r ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding

    echo "Extract embeddings and store it under ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding"
    echo "..."
    bash spectral_cluster/extract_emb_from_pt_modelscope_model.sh \
            --scp ${dest_dir}/exp/${partition}_${sad_type}_sad_fbank/fbank.scp \
            --model_id $model_id\
            --store_dir ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding \
            --batch_size 96 \
            --frame_shift 10 \
            --window_secs 1.5 \
            --period_secs 0.75 \
            --subseg_cmn ${subseg_cmn} \
            --nj 1
fi

# Applying spectral clustering algorithm
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/
    [ -f "${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}" ] && rm ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}

    echo "Doing spectral clustering and store the result in ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}"
    echo "..."
    python3 spectral_cluster/spectral_clusterer.py \
            --scp ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding/emb.scp \
            --output ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}
fi


# Convert labels to RTTMs
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/
    echo "Doing make rttm and store the result in ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_rttm_${speaker_model}"
    echo "..."
    python3 spectral_cluster/make_rttm.py \
            --labels ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model} \
            --channel 1 > ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_rttm_${speaker_model}
fi
# # Evaluate the result
if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/
    oracle_rttm=/mntcephfs/lab_data/maduo/model_hub/ts_vad/${partition}.rttm
    predict_rttm=${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_rttm_${speaker_model}
    echo -e "Get the DER results\n..."
    echo -e "DER, MS, FA, SC "
    perl SCTK-2.4.12/src/md-eval/md-eval.pl \
          -c 0.25 \
          -r $oracle_rttm\
          -s $predict_rttm 2>&1 | tee $dest_dir/exp/spectral_cluster/${partition}_${sad_type}_sad_${speaker_model}_res

fi

#  cat logs/run_spectral_cluster_stage10-13_alimeeting_eval_oracle.log
# Eval set oracle vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# collar=0.25
# DER, MS, FA, SC
# 13.57/12.92/0.00/0.65
# cat logs/run_spectral_cluster_stage10-13_alimeeting_test_oracle.log
# Test set oracle vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 13.56/12.72/0.00/0.85

# only overlap
# Eval and collar=0.25
# spyder -r overlap -c 0.25  /mntcephfs/lab_data/maduo/model_hub/ts_vad/alimeeting_eval.rttm /mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/exp/spectral_cluster/alimeeting_eval_oracle_sad_rttm_cam++_advanced
# ╒═════════════╤════════════════╤═════════╤════════════╤═════════╤════════╕
#│ Recording   │   Duration (s) │   Miss. │   F.Alarm. │   Conf. │    DER │
#╞═════════════╪════════════════╪═════════╪════════════╪═════════╪════════╡
#│ Overall     │        2782.96 │  53.74% │      0.00% │   0.58% │ 54.32% │
#╘═════════════╧════════════════╧═════════╧════════════╧═════════╧════════╛
# Test and collar=0.25
# spyder -r overlap -c 0.25  /mntcephfs/lab_data/maduo/model_hub/ts_vad/alimeeting_test.rttm /mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/exp/spectral_cluster/alimeeting_test_oracle_sad_rttm_cam++_advanced
# ╒═════════════╤════════════════╤═════════╤════════════╤═════════╤════════╕
# │ Recording   │   Duration (s) │   Miss. │   F.Alarm. │   Conf. │    DER │
# ╞═════════════╪════════════════╪═════════╪════════════╪═════════╪════════╡
# │ Overall     │        6848.25 │  54.95% │      0.00% │   0.90% │ 55.85% │
# ╘═════════════╧════════════════╧═════════╧════════════╧═════════╧════════╛

# cat logs/run_spectral_cluster_stage12-13_alimeeting_eval_system.log
# collar=0.0
# Eval set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 16.94/15.39/0.82/0.73

# cat logs/run_spectral_cluster_stage10-13_alimeeting_test_system.log
# Test set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 18.04/16.72/0.39/0.92

# cat logs/run_spectral_cluster_stage13_for_train_set.log
# Train set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 23.50/20.09/1.53/1.89

# 2025-2-26, more specify short term DER
# for example: less than 1s segment
# Eval of alimeeting, collar=0.25
# python3 ../multi_datasets/ts_vad2/short_term_statistics.py 1 /mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/exp/spectral_cluster/alimeeting_eval_oracle_sad_rttm_cam++_advanced /mntcephfs/lab_data/maduo/model_hub/ts_vad/alimeeting_eval.rttm
# der=10.39,miss=0.01, false=0.0,confusion=10.38

# Test of alimeeting, collar=0.25
#  python3 ../multi_datasets/ts_vad2/short_term_statistics.py 1 /mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/exp/spectral_cluster/alimeeting_test_oracle_sad_rttm_cam++_advanced /mntcephfs/lab_data/maduo/model_hub/ts_vad/alimeeting_test.rttm
# der=9.61,miss=0.0, false=0.0,confusion=9.6

# for example: less than 2s segment
# Eval of alimeeting, collar=0.25
#  python3 ../multi_datasets/ts_vad2/short_term_statistics.py 2 /mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/exp/spectral_cluster/alimeeting_eval_oracle_sad_rttm_cam++_advanced /mntcephfs/lab_data/maduo/model_hub/ts_vad/alimeeting_eval.rttm
# der=4.02,miss=0.02, false=0.0,confusion=4.0

# Test of alimeeting, collar=0.25
#  python3 ../multi_datasets/ts_vad2/short_term_statistics.py 1 /mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/exp/spectral_cluster/alimeeting_test_oracle_sad_rttm_cam++_advanced /mntcephfs/lab_data/maduo/model_hub/ts_vad/alimeeting_test.rttm
# der=4.02,miss=0.02, false=0.0,confusion=4.0



# I want to finetune period_secs, it is from `FROM MODULAR TO END-TO-END SPEAKER DIARIZATION`
if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/
    #model_id="iic/speech_campplus_sv_zh_en_16k-common_advanced"
    [ -d "${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding_shift${period_secs}" ] && rm -r ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding_shift${period_secs}

    echo "Extract embeddings and store it under ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding_shift${period_secs}"
    echo "..."
    bash spectral_cluster/extract_emb_from_pt_modelscope_model.sh \
            --scp ${dest_dir}/exp/${partition}_${sad_type}_sad_fbank/fbank.scp \
            --model_id $model_id\
            --store_dir ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding_shift${period_secs} \
            --batch_size 96 \
            --frame_shift 10 \
            --window_secs 1.5 \
            --period_secs $period_secs \
            --subseg_cmn ${subseg_cmn} \
            --nj 1
fi

# Applying spectral clustering algorithm
if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/
    [ -f "${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}_shift${period_secs}" ] && rm ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}_shift${period_secs}

    echo "Doing spectral clustering and store the result in ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}_shift${period_secs}"
    echo "..."
    python3 spectral_cluster/spectral_clusterer.py \
            --scp ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding_shift${period_secs}/emb.scp \
            --output ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}_shift${period_secs}
fi


# Convert labels to RTTMs
if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/
    echo "Doing make rttm and store the result in ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_rttm_${speaker_model}_shift${period_secs}"
    echo "..."
    python3 spectral_cluster/make_rttm.py \
            --labels ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}_shift${period_secs} \
            --channel 1 > ${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_rttm_${speaker_model}_shift${period_secs}
fi
# # Evaluate the result
if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/
    oracle_rttm=/mntcephfs/lab_data/maduo/model_hub/ts_vad/${partition}.rttm
    predict_rttm=${dest_dir}/exp/spectral_cluster/${partition}_${sad_type}_sad_rttm_${speaker_model}_shift${period_secs}
    echo -e "Get the DER results\n..."
    echo -e "DER, MS, FA, SC "
    perl SCTK-2.4.12/src/md-eval/md-eval.pl \
          -c 0.25 \
          -r $oracle_rttm\
          -s $predict_rttm 2>&1 | tee $dest_dir/exp/spectral_cluster/${partition}_${sad_type}_sad_${speaker_model}_shift${period_secs}_res

fi
# # cat logs/run_spectral_cluster_stage14_alimeeting_eval_system_period-secs0.25.log
# Eval set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# period_secs=0.25s,
# DER, MS, FA, SC
# 16.77/15.39/0.82/0.56

# # cat logs/run_spectral_cluster_stage14_alimeeting_test_system_period-secs0.25.log
# Test set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# period_secs=0.25s,
# DER, MS, FA, SC
# 17.86/16.72/0.39/0.74

# # cat logs/run_spectral_cluster_stage14_alimeeting_train_system_period-secs0.25.log
# Train set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# period_secs=0.25s,
# DER, MS, FA, SC
# 23.34/20.09/1.53/1.73

# # cat logs/run_spectral_cluster_stage14_alimeeting_eval_system_period-secs0.5.log
# Eval set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# period_secs=0.5s,
# DER, MS, FA, SC
# 16.85/15.39/0.82/0.64

# # cat logs/run_spectral_cluster_stage14_alimeeting_test_system_period-secs0.5.log
# Test set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# period_secs=0.5s,
# DER, MS, FA, SC
# 17.90/16.72/0.39/0.79

# # cat logs/run_spectral_cluster_stage14_alimeeting_train_system_period-secs0.5.log
# Train set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# period_secs=0.5s,
# DER, MS, FA, SC
# 23.32/20.09/1.53/1.71
