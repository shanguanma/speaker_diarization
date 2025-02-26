#!/usr/bin/env bash

stage=0
stop_stage=1000
nj=32
. utils/parse_options.sh
#. path_for_fairseq_speechtext.sh
#. path_for_fsq_speechtext.sh
. path_for_fsq_sptt.sh
#. path.sh # with kaldi env and fsq_sptt
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

if [ ${stage} -le 1 ]&&[ ${stop_stage} -ge 1 ];then
    echo "Start model averaging"
    exp_name=eend_eda_base_100epoch_conformer_magicdata-RAMC
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    model_dir=$root_dir/exp/$exp_name
    dst_model=$model_dir/best_avg.th
    avg_num=10
    remove_models='True'
    python eend/eend/bin/model_averaging2.py \
            --dst_model $dst_model\
            --src_path $model_dir\
            --nums $avg_num\
            --remove_models $remove_models

fi


# Inferring
if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Start inferring dev set"
    exp_name=eend_eda_base_100epoch_conformer_magicdata-RAMC
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    infer_conf=$root_dir/eend/conf/base/mini_librispeech_infer.yaml
    testname=dev
    test_dir=$root_dir/data/magicdata-RAMC/$testname
    test_model=$root_dir/exp/$exp_name/best_avg.th
    infer_out_dir=$root_dir/exp/$exp_name/infer_eda_${testname}_3speaker
    model_type="ConformerEda"
    conformer_encoder_n_layers=2
    sampling_rate=16000
    num_speakers=3 # because I have removed G00000000 segment,every utterance actually contains 2 speakers,
                   # at train stage, I haven't removed G00000000 segment, every utterance actually contains 3 speakers,
    python eend/eend/bin/infer_eda.py \
            -c $infer_conf \
            $test_dir \
            $test_model\
            $infer_out_dir \
            --model_type $model_type\
            --transformer-encoder-n-layers $conformer_encoder_n_layers\
	    --sampling-rate $sampling_rate\
            --num-speakers $num_speakers
fi

# Scoring
if [ $stage -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Start scoring"
    exp_name=eend_eda_base_100epoch_conformer_magicdata-RAMC
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    stck_dir=$root_dir/SCTK-2.4.12/src/md-eval/
    testname=dev
    infer_out_dir=$root_dir/exp/$exp_name/infer_eda_${testname}_3speaker
    work=$infer_out_dir/.work
    scoring_dir=$infer_out_dir/score/magicdata-RAMC_${testname}
    test_dir=$root_dir/data/magicdata-RAMC/$testname
    mkdir -p $work
    mkdir -p $scoring_dir
    find $infer_out_dir -iname "*.h5" > $work/file_list
      for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
        python eend/eend/bin/make_rttm.py --median=$med --threshold=$th \
                --frame_shift=80 --subsampling=10 --sampling_rate=8000 \
                $work/file_list $scoring_dir/hyp_${th}_$med.rttm
        $stck_dir/md-eval.pl -c 0.25 \
                -r $test_dir/rttm_openslr_gt_${testname}_nog0 \
                -s $scoring_dir/hyp_${th}_$med.rttm > $scoring_dir/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
        done
      done
   echo "result: DER,MS,FA,SC"
   head  $scoring_dir/result_th0.*
fi

if [ $stage -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Start inferring dev set"
    exp_name=eend_eda_base_100epoch_conformer_magicdata-RAMC
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    infer_conf=$root_dir/eend/conf/base/mini_librispeech_infer.yaml
    testname=dev
    test_dir=$root_dir/data/magicdata-RAMC/$testname
    test_model=$root_dir/exp/$exp_name/best_avg.th
    infer_out_dir=$root_dir/exp/$exp_name/infer_eda_${testname}_2speaker
    model_type="ConformerEda"
    conformer_encoder_n_layers=2
    sampling_rate=16000
    num_speakers=2 # because I have removed G00000000 segment,every utterance actually contains 2 speakers,
                   # at train stage, I haven't removed G00000000 segment, every utterance actually contains 3 speakers,
    python eend/eend/bin/infer_eda.py \
            -c $infer_conf \
            $test_dir \
            $test_model\
            $infer_out_dir \
            --model_type $model_type\
            --transformer-encoder-n-layers $conformer_encoder_n_layers\
            --sampling-rate $sampling_rate\
            --num-speakers $num_speakers
fi

# Scoring
if [ $stage -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Start scoring"
    exp_name=eend_eda_base_100epoch_conformer_magicdata-RAMC
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    stck_dir=$root_dir/SCTK-2.4.12/src/md-eval/
    testname=dev
    infer_out_dir=$root_dir/exp/$exp_name/infer_eda_${testname}_2speaker
    work=$infer_out_dir/.work
    scoring_dir=$infer_out_dir/score/magicdata-RAMC_${testname}
    test_dir=$root_dir/data/magicdata-RAMC/$testname
    mkdir -p $work
    mkdir -p $scoring_dir
    find $infer_out_dir -iname "*.h5" > $work/file_list
      for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
        python eend/eend/bin/make_rttm.py --median=$med --threshold=$th \
                --frame_shift=80 --subsampling=10 --sampling_rate=8000 \
                $work/file_list $scoring_dir/hyp_${th}_$med.rttm
        $stck_dir/md-eval.pl -c 0.25 \
                -r $test_dir/rttm_openslr_gt_${testname}_nog0 \
                -s $scoring_dir/hyp_${th}_$med.rttm > $scoring_dir/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
        done
      done
   echo "result: DER,MS,FA,SC"
   head  $scoring_dir/result_th0.*
fi


if [ ${stage} -le 6 ]&&[ ${stop_stage} -ge 6 ];then
    echo "Start model averaging"
    exp_name=eend_eda_base_100epoch_conformer_magicdata-RAMC_nog0
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    model_dir=$root_dir/exp/$exp_name
    dst_model=$model_dir/best_avg.th
    avg_num=10
    remove_models='True'
    python eend/eend/bin/model_averaging2.py \
            --dst_model $dst_model\
            --src_path $model_dir\
            --nums $avg_num\
            --remove_models $remove_models

fi

if [ $stage -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "Start inferring dev set"
    exp_name=eend_eda_base_100epoch_conformer_magicdata-RAMC_nog0
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    infer_conf=$root_dir/eend/conf/base/mini_librispeech_infer.yaml
    testname=dev
    test_dir=$root_dir/data/magicdata-RAMC_nog0/$testname
    test_model=$root_dir/exp/$exp_name/best_avg.th
    infer_out_dir=$root_dir/exp/$exp_name/infer_eda_${testname}_2speaker
    model_type="ConformerEda"
    conformer_encoder_n_layers=2
    sampling_rate=16000
    num_speakers=2 
    python eend/eend/bin/infer_eda.py \
            -c $infer_conf \
            $test_dir \
            $test_model\
            $infer_out_dir \
            --model_type $model_type\
            --transformer-encoder-n-layers $conformer_encoder_n_layers\
            --sampling-rate $sampling_rate\
            --num-speakers $num_speakers
fi

# Scoring
if [ $stage -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "Start scoring"
    exp_name=eend_eda_base_100epoch_conformer_magicdata-RAMC_nog0
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    stck_dir=$root_dir/SCTK-2.4.12/src/md-eval/
    testname=dev
    infer_out_dir=$root_dir/exp/$exp_name/infer_eda_${testname}_2speaker
    work=$infer_out_dir/.work
    scoring_dir=$infer_out_dir/score/magicdata-RAMC_nog0${testname}
    test_dir=$root_dir/data/magicdata-RAMC_nog0/$testname
    mkdir -p $work
    mkdir -p $scoring_dir
    find $infer_out_dir -iname "*.h5" > $work/file_list
      for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
        python eend/eend/bin/make_rttm.py --median=$med --threshold=$th \
                --frame_shift=80 --subsampling=10 --sampling_rate=8000 \
                $work/file_list $scoring_dir/hyp_${th}_$med.rttm
        $stck_dir/md-eval.pl -c 0.25 \
                -r $test_dir/rttm \
                -s $scoring_dir/hyp_${th}_$med.rttm > $scoring_dir/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
        done
      done
   echo "result: DER,MS,FA,SC"
   head  $scoring_dir/result_th0.*
   #Start scoring
#result: DER,MS,FA,SC
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_magicdata-RAMC_nog0/infer_eda_dev_2speaker/score/magicdata-RAMC_nog0dev/result_th0.3_med11_collar0.25 <==
#61.18/11.62/13.52/36.04
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_magicdata-RAMC_nog0/infer_eda_dev_2speaker/score/magicdata-RAMC_nog0dev/result_th0.3_med1_collar0.25 <==
#61.43/11.99/13.60/35.84
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_magicdata-RAMC_nog0/infer_eda_dev_2speaker/score/magicdata-RAMC_nog0dev/result_th0.4_med11_collar0.25 <==
#60.10/12.76/10.47/36.87
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_magicdata-RAMC_nog0/infer_eda_dev_2speaker/score/magicdata-RAMC_nog0dev/result_th0.4_med1_collar0.25 <==
#60.36/13.16/10.54/36.66
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_magicdata-RAMC_nog0/infer_eda_dev_2speaker/score/magicdata-RAMC_nog0dev/result_th0.5_med11_collar0.25 <==
#60.14/14.73/8.83/36.58
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_magicdata-RAMC_nog0/infer_eda_dev_2speaker/score/magicdata-RAMC_nog0dev/result_th0.5_med1_collar0.25 <==
#60.34/15.12/8.81/36.41
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_magicdata-RAMC_nog0/infer_eda_dev_2speaker/score/magicdata-RAMC_nog0dev/result_th0.6_med11_collar0.25 <==
#61.31/17.83/8.33/35.16
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_magicdata-RAMC_nog0/infer_eda_dev_2speaker/score/magicdata-RAMC_nog0dev/result_th0.6_med1_collar0.25 <==
#61.56/18.28/8.28/35.00
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_magicdata-RAMC_nog0/infer_eda_dev_2speaker/score/magicdata-RAMC_nog0dev/result_th0.7_med11_collar0.25 <==
#63.04/21.70/7.91/33.42
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_magicdata-RAMC_nog0/infer_eda_dev_2speaker/score/magicdata-RAMC_nog0dev/result_th0.7_med1_collar0.25 <==
#63.39/22.26/7.88/33.24
fi
