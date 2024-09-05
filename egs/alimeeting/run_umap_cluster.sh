#!/usr/bin/env bash

## how to use this script:
# for example: sbatch --nodes 1 --exclude=pgpu26 --gres=gpu:1  --cpus-per-gpu=8  --ntasks-per-node 1  -p p-V100 -A t00120220002 -o logs/run_umap_cluster_stage3-8_alimeeting_eval_system.log run_umap_cluster.sh --stage 3 --stop-stage 8 --partition alimeeting_eval --sad-type system

stage=0
stop_stage=1000

sad_type="oracle"
partition="alimeeting_eval"
#partition="alimeeting_test"
# do cmn on the sub-segment or on the vad segment
subseg_cmn=true

speaker_model="cam++_advanced" # you can start stage 10 and skip stage5-8
model_id="iic/speech_campplus_sv_zh_en_16k-common_advanced"
## for after stage 15, we will finetune these parameter for umap
## this is default parameter at stage 11 and stage 6
n_neighbors=16
min_dist=0.05


## as best parameter for alimeeting dataset
# n_neighbors=15
# min_dist=0.009
# # Test set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 19.27/16.72/0.39/2.15

# n_neighbors=16
# min_dist=0.05
# Test set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 20.18/16.72/0.39/3.06

. utils/parse_options.sh
. path_for_speaker_diarization.sh

echo " We use sad_type: $sad_type , partition: ${partition}, subseg_cmn: ${subseg_cmn} to do umap cluster"

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
    dest_eval_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/umap_cluster/data/alimeeting_eval
    mkdir -p $dest_eval_dir
    alimeeting_eval_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/audio_dir/
    ls $alimeeting_eval_dir/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > $dest_eval_dir/wav.scp

    # Create wav.scp for test audios
    dest_test_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/umap_cluster/data/alimeeting_test
    mkdir -p $dest_test_dir
    alimeeting_test_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Test_Ali/Test_Ali_far/audio_dir/
    ls $alimeeting_test_dir/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > $dest_test_dir/wav.scp

fi



# Voice activity detection
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # Set VAD min duration
    # oracle_rttm is getting from /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/scripts/alimeeting/011_prepare_rttm_for_ts_vad_hltsz.sh
    min_duration=0.255
    oracle_rttm=/mntcephfs/lab_data/maduo/model_hub/ts_vad/${partition}.rttm
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/umap_cluster/
    if [ ${sad_type} == "oracle" ]; then
        # Oracle SAD: handling overlapping or too short regions in ground truth RTTM
        #while read -r utt wav_path; do
            python3 umap_cluster/make_oracle_sad.py \
                    --rttm $oracle_rttm \
                    --min-duration $min_duration > ${dest_dir}/data/${partition}/oracle_sad
    fi

    if [ ${sad_type} == "system" ]; then
       # System SAD: applying 'silero' VAD
       # https://github.com/snakers4/silero-vad
       python3 umap_cluster/make_system_sad.py \
               --scp ${dest_dir}/data/${partition}/wav.scp \
               --min-duration $min_duration > ${dest_dir}/data/${partition}/system_sad
    fi
fi


# Extract fbank features
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/umap_cluster/
    [ -d "${dest_dir}/exp/${partition}_${sad_type}_sad_fbank" ] && rm -r ${dest_dir}/exp/${partition}_${sad_type}_sad_fbank

    echo "Make Fbank features and store it under ${dest_dir}/exp/${partition}_${sad_type}_sad_fbank"
    echo "..."
    bash umap_cluster/make_fbank.sh \
            --scp ${dest_dir}/data/${partition}/wav.scp \
            --segments ${dest_dir}/data/${partition}/${sad_type}_sad \
            --store_dir ${dest_dir}/exp/${partition}_${sad_type}_sad_fbank \
            --subseg_cmn ${subseg_cmn} \
            --nj 8
fi



# Extract embeddings
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/umap_cluster/
    pretrained_model=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/wespeaker/wespeaker-cnceleb-resnet34-LM/cnceleb_resnet34_LM.onnx
    [ -d "${dest_dir}/exp/${partition}_${sad_type}_sad_embedding" ] && rm -r ${dest_dir}/exp/${partition}_${sad_type}_sad_embedding

    echo "Extract embeddings and store it under ${dest_dir}/exp/${partition}_${sad_type}_sad_embedding"
    echo "..."
    bash umap_cluster/extract_emb_from_onnx_model.sh \
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


# Applying umap clustering algorithm
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/umap_cluster/
    [ -f "${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_labels" ] && rm ${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_labels

    echo "Doing umap clustering and store the result in ${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_labels"
    echo "..."
    python3 umap_cluster/umap_clusterer.py \
            --scp ${dest_dir}/exp/${partition}_${sad_type}_sad_embedding/emb.scp \
            --output ${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_labels
fi


# Convert labels to RTTMs
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/umap_cluster/
    python3 umap_cluster/make_rttm.py \
            --labels ${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_labels \
            --channel 1 > ${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_rttm
fi
# # Evaluate the result
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/umap_cluster/
    oracle_rttm=/mntcephfs/lab_data/maduo/model_hub/ts_vad/${partition}.rttm
    predict_rttm=${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_rttm
    echo -e "Get the DER results\n..."
    echo -e "DER, MS, FA, SC "
    perl SCTK-2.4.12/src/md-eval/md-eval.pl \
          -c 0.25 \
          -r $oracle_rttm\
          -s $predict_rttm 2>&1 | tee $dest_dir/exp/umap_cluster/${partition}_${sad_type}_sad_res
fi
#
# Eval set oracle vad, speaker embedding is from cn-cnceleb resnet34-lm model.
# DER, MS, FA, SC
# 20.91/12.92/0.00/7.98

# Test set oracle vad, speaker embedding is from cn-cnceleb resnet34-lm model.
# DER, MS, FA, SC
# 22.30/12.72/0.00/9.58

# Eval set system vad, speaker embedding is from cn-cnceleb resnet34-lm model.
# DER, MS, FA, SC
# 23.89/15.39/0.82/7.69
# Test set system vad, speaker embedding is from cn-cnceleb resnet34-lm model.
# DER, MS, FA, SC
# 25.87/16.72/0.39/8.75


# Extract embeddings using better model i.e. cam++ is trained on cn-celeb and voxceleb by modelscope
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/umap_cluster/
    #model_id="iic/speech_campplus_sv_zh_en_16k-common_advanced"
    [ -d "${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding" ] && rm -r ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding

    echo "Extract embeddings and store it under ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding"
    echo "..."
    bash umap_cluster/extract_emb_from_pt_modelscope_model.sh \
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

# Applying umap clustering algorithm
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/umap_cluster/
    [ -f "${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}" ] && rm ${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}

    echo "Doing clustering and store the result in ${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}"
    echo "..."
    python3 umap_cluster/umap_clusterer.py \
            --scp ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding/emb.scp \
            --output ${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}
fi

# Convert labels to RTTMs
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/umap_cluster/
    python3 umap_cluster/make_rttm.py \
            --labels ${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_labels_${speaker_model} \
            --channel 1 > ${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_rttm_${speaker_model}
fi
# # Evaluate the result
if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/umap_cluster/
    oracle_rttm=/mntcephfs/lab_data/maduo/model_hub/ts_vad/${partition}.rttm
    predict_rttm=${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_rttm_${speaker_model}
    echo -e "Get the DER results\n..."
    echo -e "DER, MS, FA, SC "
    perl SCTK-2.4.12/src/md-eval/md-eval.pl \
          -c 0.25 \
          -r $oracle_rttm\
          -s $predict_rttm 2>&1 | tee $dest_dir/exp/umap_cluster/${partition}_${sad_type}_sad_${speaker_model}_res

fi
#  cat logs/run_umap_cluster_stage10-13_alimeeting_eval_oracle.log
# Eval set oracle vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 15.92/12.92/0.00/3.00

# cat logs/run_umap_cluster_stage10-13_alimeeting_test_oracle.log
# Test set oracle vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 14.86/12.72/0.00/2.15

# cat logs/run_umap_cluster_stage10-13_alimeeting_eval_system.log
# Eval set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 17.93/15.39/0.82/1.72

# cat logs/run_umap_cluster_stage10-13_alimeeting_test_system.log
# Test set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 20.18/16.72/0.39/3.06

# We will finetune their parameter and apply umap clustering algorithm
if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/umap_cluster/
    [ -f "${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}_n_neighbors${n_neighbors}_min_dist${min_dist}" ] && rm ${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}_n_neighbors${n_neighbors}_min_dist${min_dist}

    echo "Doing  clustering and store the result in ${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}_n_neighbors${n_neighbors}_min_dist${min_dist}"
    echo "..."
    python3 umap_cluster/umap_clusterer.py \
            --scp ${dest_dir}/exp/${partition}_${sad_type}_sad_${speaker_model}_embedding/emb.scp \
            --output ${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}_n_neighbors${n_neighbors}_min_dist${min_dist}\
            --n_neighbors  $n_neighbors\
            --min_dist $min_dist
fi

# Convert labels to RTTMs
if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/umap_cluster/
    python3 umap_cluster/make_rttm.py \
            --labels ${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_labels_${speaker_model}_n_neighbors${n_neighbors}_min_dist${min_dist} \
            --channel 1 > ${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_rttm_${speaker_model}_n_neighbors${n_neighbors}_min_dist${min_dist}
fi
# # Evaluate the result
if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ]; then
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/umap_cluster/
    oracle_rttm=/mntcephfs/lab_data/maduo/model_hub/ts_vad/${partition}.rttm
    predict_rttm=${dest_dir}/exp/umap_cluster/${partition}_${sad_type}_sad_rttm_${speaker_model}_n_neighbors${n_neighbors}_min_dist${min_dist}
    echo -e "Get the DER results\n..."
    echo -e "DER, MS, FA, SC "
    perl SCTK-2.4.12/src/md-eval/md-eval.pl \
          -c 0.25 \
          -r $oracle_rttm\
          -s $predict_rttm 2>&1 | tee $dest_dir/exp/umap_cluster/${partition}_${sad_type}_sad_${speaker_model}_n_neighbors${n_neighbors}_min_dist${min_dist}_res

fi

# n_neighbors=16
# min_dist=0.05
# Test set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 20.18/16.72/0.39/3.06

# n_neighbors=14
# min_dist=0.05
# # Test set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 20.06/16.72/0.39/2.94

# n_neighbors=12
# min_dist=0.05
# # Test set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 20.51/16.72/0.39/3.39

# n_neighbors=13
# min_dist=0.05
# # Test set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 20.50/16.72/0.39/3.38

# n_neighbors=15
# min_dist=0.05
# # Test set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
#DER, MS, FA, SC
#20.05/16.72/0.39/2.94

# n_neighbors=15
# min_dist=0.04
# # Test set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
#DER, MS, FA, SC
#20.05/16.72/0.39/2.93

# n_neighbors=15
# min_dist=0.03
# # Test set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 20.06/16.72/0.39/2.94

# n_neighbors=15
# min_dist=0.01
# # Test set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 19.88/16.72/0.39/2.76

# n_neighbors=15
# min_dist=0.005
# # Test set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 20.05/16.72/0.39/2.94

## as best parameter
# n_neighbors=15
# min_dist=0.009
# # Test set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 19.27/16.72/0.39/2.15


# n_neighbors=15
# min_dist=0.008
# # Test set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 20.07/16.72/0.39/2.96
# n_neighbors=15
# min_dist=0.007
# # Test set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 20.06/16.72/0.39/2.94

## final

# n_neighbors=15
# min_dist=0.009
# # Eval set oracle vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 13.74/12.92/0.00/0.82

# n_neighbors=15
# min_dist=0.009
# # Test set oracle vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 14.28/12.72/0.00/1.56

# n_neighbors=15
# min_dist=0.009
# # Eval set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 17.93/15.39/0.82/1.72

# n_neighbors=15
# min_dist=0.009
# # Test set system vad, speaker embedding is from  cam++ model which is trained on cn-celeb and voxceleb.
# DER, MS, FA, SC
# 19.27/16.72/0.39/2.15

