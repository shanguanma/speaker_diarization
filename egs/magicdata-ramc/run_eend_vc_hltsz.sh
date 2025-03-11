#!/usr/bin/env bash

stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_speaker_diarization_hltsz.sh


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
   echo "prepared uem file"
   data=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   for name in dev test train;do
	   python3  prepared_magicdata-ramc_uem.py $data/$name/wav.scp > $data/$name/all.uem 
   done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
    echo "stage1: use single-opt for model training..."
    train_conf=eend_vc/conf/wavlm_updated_conformer_magicdata-ramc.toml
    CUDA_VISIBLE_DEVICES="0,1" \
    TORCH_DISTRIBUTED_DEBUG=DETAIL\
    accelerate launch --num_processes 2 --main_process_port 14815 \
            eend_vc/run_dual_opt.py -C $train_conf -M train 

fi

if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then

	#recipe_root=/YOUR_PATH/DiariZen/recipes/diar_ssl
	exp_root=/data/maduo/exp/speaker_diarization/eend_vc/ # it is seted by [meta] save_dir in  $conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
	conf_dir=eend_vc/conf

	# training setup
	#use_dual_opt=true  # true for wavlm_updated_conformer.toml; false for the others
	train_conf=$conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
	# train_conf=$conf_dir/wavlm_frozen_conformer.toml
	# train_conf=$conf_dir/fbank_conformer.toml
	# train_conf=$conf_dir/pyannote_baseline.toml

	conf_name=`ls $train_conf | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`

	# inference setup
	#dtype="dev test"
	dtype="cssd_testset"
	data_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format

	pyan_merge_closer=0.5
	pyan_max_length_merged=50
	pyan_inf_max_batch=32

	cluster_threshold=0.70
	segmentation_step=0.1
	min_cluster_size=30
	infer_affix=_constrained_AHC_segmentation_step_${segmentation_step}_min_cluster_size_${min_cluster_size}_AHC_thres_${cluster_threshold}_pyan_max_length_merged${pyan_max_length_merged}

	avg_ckpt_num=5
	val_metric=Loss   # Loss or DER
	#val_mode=prev   # [prev, best, center]
        
	val_mode=best
	# scoring setup
	collar="0 0.25"
	REF_DIR=$data_dir
	dscore_dir=eend_vc/dscore/
	diarization_dir=$exp_root/$conf_name    # can be replaced by our pre-trained models, e.g. diarization_dir=/YOUR_PATH/checkpoints/wavlm_updated_conformer
	#config_dir=`ls $diarization_dir/*.toml | sort -r | head -n 1`
        config_dir=$conf_dir/wavlm_updated_conformer_magicdata-ramc.toml	
	#embedding_model=/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/resnet34-lm/pytorch_model.bin     # it's necessary to have "pyannote" in your directory path
        #embedding_model="/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin"
        embedding_model="/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/voxceleb_resnet34_LM.onnx"
	echo "stage2: model inference..."
    export CUDA_VISIBLE_DEVICES=0

    train_log=`du -h $diarization_dir/*.log | sort -rh | head -n 1 | awk '{print $NF}'`
    cat $train_log | grep 'Loss/DER' | awk -F ']:' '{print $NF}' > $diarization_dir/val_metric_summary.lst
    for dset in $dtype; do
      CUDA_VISIBLE_DEVICES=0\
       python eend_vc/infer_avg.py -C $config_dir \
            -i ${data_dir}/${dset}/wav.scp \
            -o ${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}/${dset} \
            -u ${data_dir}/${dset}/all.uem \
            --avg_ckpt_num $avg_ckpt_num \
            --val_metric $val_metric \
            --val_mode $val_mode \
            --val_metric_summary $diarization_dir/val_metric_summary.lst \
            --segmentation_step $segmentation_step \
            --min_cluster_size $min_cluster_size \
            --cluster_threshold $cluster_threshold \
            --embedding_model $embedding_model \
            --merge_closer $pyan_merge_closer \
            --max_length_merged $pyan_max_length_merged \
            --batch_size $pyan_inf_max_batch
     done
fi

if [ $stage -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    #dtype="dev test"
    dtype="cssd_testset"
    collar="0 0.25"
    exp_root=/data/maduo/exp/speaker_diarization/eend_vc/ # it is seted by [meta] save_dir in  $conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
    conf_dir=eend_vc/conf

    # training setup
    #use_dual_opt=true  # true for wavlm_updated_conformer.toml; false for the others
    train_conf=$conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
        # train_conf=$conf_dir/wavlm_frozen_conformer.toml
        # train_conf=$conf_dir/fbank_conformer.toml
        # train_conf=$conf_dir/pyannote_baseline.toml

    conf_name=`ls $train_conf | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`
    pyan_merge_closer=0.5
    pyan_max_length_merged=50
    pyan_inf_max_batch=32

    cluster_threshold=0.70
    segmentation_step=0.1
    min_cluster_size=30
    infer_affix=_constrained_AHC_segmentation_step_${segmentation_step}_min_cluster_size_${min_cluster_size}_AHC_thres_${cluster_threshold}_pyan_max_length_merged${pyan_max_length_merged}

    avg_ckpt_num=5
    val_metric=Loss   # Loss or DER
    val_mode=best  # [prev, best, center]
    data_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
    REF_DIR=$data_dir
    dscore_dir=eend_vc/dscore/
    diarization_dir=$exp_root/$conf_name
    for c in $collar;do
     for dset in $dtype;do
      echo "stage3: scoring..."
      SYS_DIR=${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}
      OUT_DIR=${SYS_DIR}/${dset}
       python3 ${dscore_dir}/score.py \
            -r ${REF_DIR}/${dset}/rttm_debug_nog0 \
            -s $OUT_DIR/*.rttm --collar ${c} \
            > $OUT_DIR/result_collar${c}
    done
   done
fi
# dev of magicdata-ramc, collar=0.0
# DER=11.60
# test of magicdata-ramc, collar=0.0
# DER=12.33

# dev of magicdata-ramc, collar=0.25
# DER=5.16
# test of magicdata-ramc, collar=0.25
# DER=6.01


# cssd_testset(CSSD_Eval) of magicata-ramc, collar=0.0
# DER=20.90
# cssd_testset(CSSD_Eval) of magicata-ramc, collar=0.25
# DER=5.90



if [ $stage -le 4 ] && [ ${stop_stage} -ge 4 ]; then

        #recipe_root=/YOUR_PATH/DiariZen/recipes/diar_ssl
        exp_root=/data/maduo/exp/speaker_diarization/eend_vc/ # it is seted by [meta] save_dir in  $conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
        conf_dir=eend_vc/conf

        # training setup
        #use_dual_opt=true  # true for wavlm_updated_conformer.toml; false for the others
        train_conf=$conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
        # train_conf=$conf_dir/wavlm_frozen_conformer.toml
        # train_conf=$conf_dir/fbank_conformer.toml
        # train_conf=$conf_dir/pyannote_baseline.toml

        conf_name=`ls $train_conf | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`

        # inference setup
        #dtype="dev test"
        dtype="dev test cssd_testset"
        data_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format

        pyan_merge_closer=0.5
        pyan_max_length_merged=50
        pyan_inf_max_batch=32

        cluster_threshold=0.70
        segmentation_step=0.1
        min_cluster_size=30
        infer_affix=_constrained_AHC_segmentation_step_${segmentation_step}_min_cluster_size_${min_cluster_size}_AHC_thres_${cluster_threshold}_pyan_max_length_merged${pyan_max_length_merged}

        avg_ckpt_num=5
        val_metric=DER   # Loss or DER
        #val_mode=prev   # [prev, best, center]

        val_mode=best
        # scoring setup
        #collar="0 0.25"
        REF_DIR=$data_dir
        dscore_dir=eend_vc/dscore/
        diarization_dir=$exp_root/$conf_name    # can be replaced by our pre-trained models, e.g. diarization_dir=/YOUR_PATH/checkpoints/wavlm_updated_conformer
        #config_dir=`ls $diarization_dir/*.toml | sort -r | head -n 1`
        config_dir=$conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
        #embedding_model=/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/resnet34-lm/pytorch_model.bin     # it's necessary to have "pyannote" in your directory path
        #embedding_model="/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin"
        embedding_model="/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/voxceleb_resnet34_LM.onnx"
        echo "stage2: model inference..."
    export CUDA_VISIBLE_DEVICES=0

    train_log=`du -h $diarization_dir/*.log | sort -rh | head -n 1 | awk '{print $NF}'`
    cat $train_log | grep 'Loss/DER' | awk -F ']:' '{print $NF}' > $diarization_dir/val_metric_summary.lst
    for dset in $dtype; do
      CUDA_VISIBLE_DEVICES=0\
       python eend_vc/infer_avg.py -C $config_dir \
            -i ${data_dir}/${dset}/wav.scp \
            -o ${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}/${dset} \
            -u ${data_dir}/${dset}/all.uem \
            --avg_ckpt_num $avg_ckpt_num \
            --val_metric $val_metric \
            --val_mode $val_mode \
            --val_metric_summary $diarization_dir/val_metric_summary.lst \
            --segmentation_step $segmentation_step \
            --min_cluster_size $min_cluster_size \
            --cluster_threshold $cluster_threshold \
            --embedding_model $embedding_model \
            --merge_closer $pyan_merge_closer \
            --max_length_merged $pyan_max_length_merged \
            --batch_size $pyan_inf_max_batch
     done
fi

if [ $stage -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    #dtype="dev test"
    dtype="dev test cssd_testset"
    collar="0 0.25"
    exp_root=/data/maduo/exp/speaker_diarization/eend_vc/ # it is seted by [meta] save_dir in  $conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
    conf_dir=eend_vc/conf

    # training setup
    #use_dual_opt=true  # true for wavlm_updated_conformer.toml; false for the others
    train_conf=$conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
        # train_conf=$conf_dir/wavlm_frozen_conformer.toml
        # train_conf=$conf_dir/fbank_conformer.toml
        # train_conf=$conf_dir/pyannote_baseline.toml

    conf_name=`ls $train_conf | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`
    pyan_merge_closer=0.5
    pyan_max_length_merged=50
    pyan_inf_max_batch=32

    cluster_threshold=0.70
    segmentation_step=0.1
    min_cluster_size=30
    infer_affix=_constrained_AHC_segmentation_step_${segmentation_step}_min_cluster_size_${min_cluster_size}_AHC_thres_${cluster_threshold}_pyan_max_length_merged${pyan_max_length_merged}

    avg_ckpt_num=5
    val_metric=DER   # Loss or DER
    val_mode=best  # [prev, best, center]
    data_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
    REF_DIR=$data_dir
    dscore_dir=eend_vc/dscore/
    diarization_dir=$exp_root/$conf_name
    for c in $collar;do
     for dset in $dtype;do
      echo "stage3: scoring..."
      SYS_DIR=${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}
      OUT_DIR=${SYS_DIR}/${dset}
       python3 ${dscore_dir}/score.py \
            -r ${REF_DIR}/${dset}/rttm_debug_nog0 \
            -s $OUT_DIR/*.rttm --collar ${c} \
            > $OUT_DIR/result_collar${c}
    done
   done
fi
# dev of magicdata-ramc, DER
# 5.17(11.67)
# test of magicdata-ramc, DER
# 5.92(12.31)
# cssd_testset of magicdata-ramc, DER
# 5.81(20.82)


# dev of magicdata-ramc, CDER
#  python3 cder/score.py -r /data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/dev/rttm_debug_nog0 -s exp/wavlm_updated_conformer_magicdata-ramc/i
#nfer_constrained_AHC_segmentation_step_0.1_min_cluster_size_30_AHC_thres_0.70_pyan_max_length_merged50/metric_DER_best/avg_ckpt5/dev/all.rttm
#CTS-CN-F2F-2019-11-15-1421 CDER = 0.541
#CTS-CN-F2F-2019-11-15-1422 CDER = 0.404
#CTS-CN-F2F-2019-11-15-1423 CDER = 0.138
#CTS-CN-F2F-2019-11-15-1426 CDER = 0.627
#CTS-CN-F2F-2019-11-15-1428 CDER = 0.057
#CTS-CN-F2F-2019-11-15-1434 CDER = 0.452
#CTS-CN-F2F-2019-11-15-1447 CDER = 0.146
#CTS-CN-F2F-2019-11-15-1448 CDER = 0.198
#CTS-CN-F2F-2019-11-15-1449 CDER = 0.322
#CTS-CN-F2F-2019-11-15-1452 CDER = 0.265
#CTS-CN-F2F-2019-11-15-1458 CDER = 0.439
#CTS-CN-F2F-2019-11-15-1461 CDER = 0.526
#CTS-CN-F2F-2019-11-15-1463 CDER = 0.135
#CTS-CN-F2F-2019-11-15-1468 CDER = 0.078
#CTS-CN-F2F-2019-11-15-1469 CDER = 0.300
#CTS-CN-F2F-2019-11-15-1470 CDER = 0.500
#CTS-CN-F2F-2019-11-15-1473 CDER = 0.438
#CTS-CN-F2F-2019-11-15-1475 CDER = 0.371
#CTS-CN-F2F-2019-11-15-1477 CDER = 0.115
#Avg CDER : 0.318

# test of magicdata-ramc, CDER
#  python3 cder/score.py -r /data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/test/rttm_debug_nog0 -s exp/wavlm_updated_conformer_magicdata-ramc
#/infer_constrained_AHC_segmentation_step_0.1_min_cluster_size_30_AHC_thres_0.70_pyan_max_length_merged50/metric_DER_best/avg_ckpt5/test/all.rttm
#CTS-CN-F2F-2019-11-04-387 CDER = 0.631
#CTS-CN-F2F-2019-11-11-193 CDER = 0.308
#CTS-CN-F2F-2019-11-11-198 CDER = 0.289
#CTS-CN-F2F-2019-11-11-390 CDER = 0.589
#CTS-CN-F2F-2019-11-11-423 CDER = 0.059
#CTS-CN-F2F-2019-11-11-431 CDER = 0.312
#CTS-CN-F2F-2019-11-11-716 CDER = 0.231
#CTS-CN-F2F-2019-11-15-3 CDER = 0.080
#CTS-CN-F2F-2019-11-15-8 CDER = 0.115
#CTS-CN-F2F-2019-11-15-9 CDER = 0.238
#CTS-CN-F2F-2019-11-15-17 CDER = 0.389
#CTS-CN-F2F-2019-11-15-26 CDER = 0.215
#CTS-CN-F2F-2019-11-15-27 CDER = 0.269
#CTS-CN-F2F-2019-11-15-37 CDER = 0.111
#CTS-CN-F2F-2019-11-15-39 CDER = 0.040
#CTS-CN-F2F-2019-11-15-45 CDER = 0.199
#CTS-CN-F2F-2019-11-15-52 CDER = 0.098
#CTS-CN-F2F-2019-11-15-53 CDER = 0.137
#CTS-CN-F2F-2019-11-15-56 CDER = 0.327
#CTS-CN-F2F-2019-11-15-65 CDER = 0.249
#CTS-CN-F2F-2019-11-15-68 CDER = 0.236
#CTS-CN-F2F-2019-11-15-73 CDER = 0.026
#CTS-CN-F2F-2019-11-15-75 CDER = 0.160
#CTS-CN-F2F-2019-11-15-77 CDER = 0.105
#CTS-CN-F2F-2019-11-15-81 CDER = 0.388
#CTS-CN-F2F-2019-11-15-82 CDER = 0.107
#CTS-CN-F2F-2019-11-15-88 CDER = 0.290
#CTS-CN-F2F-2019-11-15-91 CDER = 0.277
#CTS-CN-F2F-2019-11-15-94 CDER = 0.265
#CTS-CN-F2F-2019-11-15-100 CDER = 0.068
#CTS-CN-F2F-2019-11-15-101 CDER = 0.235
#CTS-CN-F2F-2019-11-15-105 CDER = 0.000
#CTS-CN-F2F-2019-11-15-121 CDER = 0.023
#CTS-CN-F2F-2019-11-15-123 CDER = 0.760
#CTS-CN-F2F-2019-11-15-126 CDER = 0.046
#CTS-CN-F2F-2019-11-15-128 CDER = 0.201
#CTS-CN-F2F-2019-11-15-135 CDER = 0.318
#CTS-CN-F2F-2019-11-15-138 CDER = 0.377
#CTS-CN-F2F-2019-11-15-142 CDER = 0.257
#CTS-CN-F2F-2019-11-15-144 CDER = 0.686
#CTS-CN-F2F-2019-11-15-150 CDER = 0.253
#CTS-CN-F2F-2019-11-15-151 CDER = 0.301
#CTS-CN-F2F-2019-11-15-153 CDER = 0.200
#Avg CDER : 0.243

# cssd_testset of magicdata-ramc, CDER
## python3 cder/score.py -r /data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/cssd_testset/rttm_debug_nog0 -s exp/wavlm_updated_conformer_magicdata-ramc/infer_constrained_AHC_segmentation_step_0.1_min_cluster_size_30_AHC_thres_0.70_pyan_max_length_merged50/metric_DER_best/avg_ckpt5/cssd_testset/all.rttm
#CTS-CN-F2F-2017-11-14_85 CDER = 0.277
#CTS-CN-F2F-2017-11-14_87 CDER = 0.236
#CTS-CN-F2F-2017-11-14_88 CDER = 0.247
#CTS-CN-F2F-2017-11-29_109 CDER = 0.373
#CTS-CN-F2F-2017-12-15_412 CDER = 0.310
#CTS-CN-F2F-2017-12-15_438 CDER = 0.339
#CTS-CN-F2F-2017-12-15_440 CDER = 0.252
#CTS-CN-F2F-2017-12-20_70 CDER = 0.128
#CTS-CN-F2F-2018-04-16_545 CDER = 0.262
#CTS-CN-F2F-2018-04-16_560 CDER = 0.281
#CTS-CN-F2F-2018-04-16_563 CDER = 0.251
#CTS-CN-F2F-2018-04-16_615 CDER = 0.328
#CTS-CN-F2F-2018-04-16_699 CDER = 0.213
#CTS-CN-F2F-2018-04-16_700 CDER = 0.207
#CTS-CN-F2F-2018-04-16_703 CDER = 0.274
#CTS-CN-F2F-2018-04-16_716 CDER = 0.337
#CTS-CN-F2F-2018-04-16_723 CDER = 0.349
#CTS-CN-F2F-2018-04-16_726 CDER = 0.277
#CTS-CN-F2F-2018-04-16_757 CDER = 0.337
#CTS-CN-F2F-2018-04-16_763 CDER = 0.303
#CTS-CN-F2F-2018-04-23_507 CDER = 0.154
#CTS-CN-F2F-2018-04-23_620 CDER = 0.323
#CTS-CN-F2F-2018-05-08_110 CDER = 0.193
#CTS-CN-F2F-2018-05-08_236 CDER = 0.284
#CTS-CN-F2F-2018-05-08_469 CDER = 0.302
#CTS-CN-F2F-2018-05-08_470 CDER = 0.333
#CTS-CN-F2F-2018-05-08_471 CDER = 0.332
#CTS-CN-F2F-2018-05-08_472 CDER = 0.310
#CTS-CN-F2F-2018-05-08_712 CDER = 0.200
#CTS-CN-F2F-2018-05-08_713 CDER = 0.242
#CTS-CN-F2F-2018-05-08_776 CDER = 0.144
#CTS-CN-F2F-2018-05-08_777 CDER = 0.176
#CTS-CN-F2F-2018-05-08_779 CDER = 0.286
#CTS-CN-F2F-2018-05-08_782 CDER = 0.148
#CTS-CN-F2F-2018-05-08_799 CDER = 0.324
#CTS-CN-F2F-2018-05-08_825 CDER = 0.251
#CTS-CN-F2F-2018-05-08_909 CDER = 0.405
#CTS-CN-F2F-2018-05-08_948 CDER = 0.087
#CTS-CN-F2F-2018-05-08_949 CDER = 0.130
#CTS-CN-F2F-2018-05-08_950 CDER = 0.188
#Avg CDER : 0.260

if [ $stage -le 6 ] && [ ${stop_stage} -ge 6 ]; then

        #recipe_root=/YOUR_PATH/DiariZen/recipes/diar_ssl
        exp_root=/data/maduo/exp/speaker_diarization/eend_vc/ # it is seted by [meta] save_dir in  $conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
        conf_dir=eend_vc/conf

        # training setup
        #use_dual_opt=true  # true for wavlm_updated_conformer.toml; false for the others
        train_conf=$conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
        # train_conf=$conf_dir/wavlm_frozen_conformer.toml
        # train_conf=$conf_dir/fbank_conformer.toml
        # train_conf=$conf_dir/pyannote_baseline.toml

        conf_name=`ls $train_conf | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`

        # inference setup
        #dtype="dev test"
        dtype="dev test cssd_testset"
        data_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format

        pyan_merge_closer=0.5
        pyan_max_length_merged=50
        pyan_inf_max_batch=32

        cluster_threshold=0.70
        segmentation_step=0.1
        min_cluster_size=30
	infer_affix=_constrained_AHC_segmentation_step_${segmentation_step}_min_cluster_size_${min_cluster_size}_AHC_thres_${cluster_threshold}_pyan_max_length_merged${pyan_max_length_merged}_cam++_zh_en_common_advanced

        avg_ckpt_num=5
        val_metric=DER   # Loss or DER
        #val_mode=prev   # [prev, best, center]

        val_mode=best
        # scoring setup
        #collar="0 0.25"
        REF_DIR=$data_dir
        dscore_dir=eend_vc/dscore/
        diarization_dir=$exp_root/$conf_name    # can be replaced by our pre-trained models, e.g. diarization_dir=/YOUR_PATH/checkpoints/wavlm_updated_conformer
        #config_dir=`ls $diarization_dir/*.toml | sort -r | head -n 1`
        config_dir=$conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
        #embedding_model=/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/resnet34-lm/pytorch_model.bin     # it's necessary to have "pyannote" in your directory path
        #embedding_model="/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin"
        #embedding_model="/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/voxceleb_resnet34_LM.onnx"
        embedding_model="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.onnx"
	echo "stage2: model inference..."
    export CUDA_VISIBLE_DEVICES=0

    train_log=`du -h $diarization_dir/*.log | sort -rh | head -n 1 | awk '{print $NF}'`
    cat $train_log | grep 'Loss/DER' | awk -F ']:' '{print $NF}' > $diarization_dir/val_metric_summary.lst
    for dset in $dtype; do
      CUDA_VISIBLE_DEVICES=0\
       python eend_vc/infer_avg.py -C $config_dir \
            -i ${data_dir}/${dset}/wav.scp \
            -o ${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}/${dset} \
            -u ${data_dir}/${dset}/all.uem \
            --avg_ckpt_num $avg_ckpt_num \
            --val_metric $val_metric \
            --val_mode $val_mode \
            --val_metric_summary $diarization_dir/val_metric_summary.lst \
            --segmentation_step $segmentation_step \
            --min_cluster_size $min_cluster_size \
            --cluster_threshold $cluster_threshold \
            --embedding_model $embedding_model \
            --merge_closer $pyan_merge_closer \
            --max_length_merged $pyan_max_length_merged \
            --batch_size $pyan_inf_max_batch
     done
fi

if [ $stage -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    #dtype="dev test"
    dtype="dev test cssd_testset"
    collar="0 0.25"
    exp_root=/data/maduo/exp/speaker_diarization/eend_vc/ # it is seted by [meta] save_dir in  $conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
    conf_dir=eend_vc/conf

    # training setup
    #use_dual_opt=true  # true for wavlm_updated_conformer.toml; false for the others
    train_conf=$conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
        # train_conf=$conf_dir/wavlm_frozen_conformer.toml
        # train_conf=$conf_dir/fbank_conformer.toml
        # train_conf=$conf_dir/pyannote_baseline.toml

    conf_name=`ls $train_conf | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`
    pyan_merge_closer=0.5
    pyan_max_length_merged=50
    pyan_inf_max_batch=32

    cluster_threshold=0.70
    segmentation_step=0.1
    min_cluster_size=30
    infer_affix=_constrained_AHC_segmentation_step_${segmentation_step}_min_cluster_size_${min_cluster_size}_AHC_thres_${cluster_threshold}_pyan_max_length_merged${pyan_max_length_merged}_cam++_zh_en_common_advanced

    avg_ckpt_num=5
    val_metric=DER   # Loss or DER
    val_mode=best  # [prev, best, center]
    data_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
    REF_DIR=$data_dir
    dscore_dir=eend_vc/dscore/
    diarization_dir=$exp_root/$conf_name
    for c in $collar;do
     for dset in $dtype;do
      echo "stage3: scoring..."
      SYS_DIR=${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}
      OUT_DIR=${SYS_DIR}/${dset}
       python3 ${dscore_dir}/score.py \
            -r ${REF_DIR}/${dset}/rttm_debug_nog0 \
            -s $OUT_DIR/*.rttm --collar ${c} \
            > $OUT_DIR/result_collar${c}
    done
   done
fi
# dev of magicdata-ramc, DER
# 5.14(11.35)
# test of magicdata-ramc, DER
# 5.62(11.75)
# cssd_testset of magicdata-ramc, DER
# 5.17(18.82)

# dev of magicdata-ramc, CDER
# python3 cder/score.py  -r /data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/dev/rttm_debug_nog0  -s  exp/wavlm_updated_conformer_magicdata-ramc/infer_constrained_AHC_segmentation_step_0.1_min_cluster_size_30_AHC_thres_0.70_pyan_max_length_merged50_cam++_zh_en_common_advanced/metric_DER_best/avg_ckpt5/dev/all.rttm
#CTS-CN-F2F-2019-11-15-1421 CDER = 0.519
#CTS-CN-F2F-2019-11-15-1422 CDER = 0.404
#CTS-CN-F2F-2019-11-15-1423 CDER = 0.138
#CTS-CN-F2F-2019-11-15-1426 CDER = 0.512
#CTS-CN-F2F-2019-11-15-1428 CDER = 0.061
#CTS-CN-F2F-2019-11-15-1434 CDER = 0.219
#CTS-CN-F2F-2019-11-15-1447 CDER = 0.146
#CTS-CN-F2F-2019-11-15-1448 CDER = 0.155
#CTS-CN-F2F-2019-11-15-1449 CDER = 0.308
#CTS-CN-F2F-2019-11-15-1452 CDER = 0.117
#CTS-CN-F2F-2019-11-15-1458 CDER = 0.462
#CTS-CN-F2F-2019-11-15-1461 CDER = 0.526
#CTS-CN-F2F-2019-11-15-1463 CDER = 0.135
#CTS-CN-F2F-2019-11-15-1468 CDER = 0.082
#CTS-CN-F2F-2019-11-15-1469 CDER = 0.138
#CTS-CN-F2F-2019-11-15-1470 CDER = 0.482
#CTS-CN-F2F-2019-11-15-1473 CDER = 0.279
#CTS-CN-F2F-2019-11-15-1475 CDER = 0.361
#CTS-CN-F2F-2019-11-15-1477 CDER = 0.115
#Avg CDER : 0.272

# test of magicdata-ramc, CDER
# python3 cder/score.py  -r /data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/test/rttm_debug_nog0  -s  exp/wavlm_updated_conformer_magicdata-ra
#mc/infer_constrained_AHC_segmentation_step_0.1_min_cluster_size_30_AHC_thres_0.70_pyan_max_length_merged50_cam++_zh_en_common_advanced/metric_DER_best/avg_ckpt5/test/all.rttm
#CTS-CN-F2F-2019-11-04-387 CDER = 0.561
#CTS-CN-F2F-2019-11-11-193 CDER = 0.246
#CTS-CN-F2F-2019-11-11-198 CDER = 0.263
#CTS-CN-F2F-2019-11-11-390 CDER = 0.322
#CTS-CN-F2F-2019-11-11-423 CDER = 0.059
#CTS-CN-F2F-2019-11-11-431 CDER = 0.312
#CTS-CN-F2F-2019-11-11-716 CDER = 0.154
#CTS-CN-F2F-2019-11-15-3 CDER = 0.080
#CTS-CN-F2F-2019-11-15-8 CDER = 0.230
#CTS-CN-F2F-2019-11-15-9 CDER = 0.231
#CTS-CN-F2F-2019-11-15-17 CDER = 0.326
#CTS-CN-F2F-2019-11-15-26 CDER = 0.148
#CTS-CN-F2F-2019-11-15-27 CDER = 0.134
#CTS-CN-F2F-2019-11-15-37 CDER = 0.111
#CTS-CN-F2F-2019-11-15-39 CDER = 0.048
#CTS-CN-F2F-2019-11-15-45 CDER = 0.197
#CTS-CN-F2F-2019-11-15-52 CDER = 0.098
#CTS-CN-F2F-2019-11-15-53 CDER = 0.074
#CTS-CN-F2F-2019-11-15-56 CDER = 0.302
#CTS-CN-F2F-2019-11-15-65 CDER = 0.197
#CTS-CN-F2F-2019-11-15-68 CDER = 0.148
#CTS-CN-F2F-2019-11-15-73 CDER = 0.026
#CTS-CN-F2F-2019-11-15-75 CDER = 0.160
#CTS-CN-F2F-2019-11-15-77 CDER = 0.105
#CTS-CN-F2F-2019-11-15-81 CDER = 0.281
#CTS-CN-F2F-2019-11-15-82 CDER = 0.107
#CTS-CN-F2F-2019-11-15-88 CDER = 0.278
#CTS-CN-F2F-2019-11-15-91 CDER = 0.229
#CTS-CN-F2F-2019-11-15-94 CDER = 0.211
#CTS-CN-F2F-2019-11-15-100 CDER = 0.068
#CTS-CN-F2F-2019-11-15-101 CDER = 0.235
#CTS-CN-F2F-2019-11-15-105 CDER = 0.000
#CTS-CN-F2F-2019-11-15-121 CDER = 0.011
#CTS-CN-F2F-2019-11-15-123 CDER = 0.748
#CTS-CN-F2F-2019-11-15-126 CDER = 0.046
#CTS-CN-F2F-2019-11-15-128 CDER = 0.194
#CTS-CN-F2F-2019-11-15-135 CDER = 0.207
#CTS-CN-F2F-2019-11-15-138 CDER = 0.285
#CTS-CN-F2F-2019-11-15-142 CDER = 0.229
#CTS-CN-F2F-2019-11-15-144 CDER = 0.631
#CTS-CN-F2F-2019-11-15-150 CDER = 0.192
#CTS-CN-F2F-2019-11-15-151 CDER = 0.263
#CTS-CN-F2F-2019-11-15-153 CDER = 0.208
#Avg CDER : 0.208

# cssd_testset of magicdata-ramc, CDER
#  python3 cder/score.py -r /data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/cssd_testset/rttm_debug_nog0 -s  exp/wavlm_updated_conformer_magicdata-ramc/infer_constrained_AHC_segmentation_step_0.1_min_cluster_size_30_AHC_thres_0.70_pyan_max_length_merged50_cam++_zh_en_common_advanced/metric_DER_best/avg_ckpt5/cssd_testset/all.rttm
#CTS-CN-F2F-2017-11-14_85 CDER = 0.266
#CTS-CN-F2F-2017-11-14_87 CDER = 0.196
#CTS-CN-F2F-2017-11-14_88 CDER = 0.199
#CTS-CN-F2F-2017-11-29_109 CDER = 0.362
#CTS-CN-F2F-2017-12-15_412 CDER = 0.145
#CTS-CN-F2F-2017-12-15_438 CDER = 0.119
#CTS-CN-F2F-2017-12-15_440 CDER = 0.115
#CTS-CN-F2F-2017-12-20_70 CDER = 0.131
#CTS-CN-F2F-2018-04-16_545 CDER = 0.150
#CTS-CN-F2F-2018-04-16_560 CDER = 0.128
#CTS-CN-F2F-2018-04-16_563 CDER = 0.131
#CTS-CN-F2F-2018-04-16_615 CDER = 0.124
#CTS-CN-F2F-2018-04-16_699 CDER = 0.261
#CTS-CN-F2F-2018-04-16_700 CDER = 0.215
#CTS-CN-F2F-2018-04-16_703 CDER = 0.228
#CTS-CN-F2F-2018-04-16_716 CDER = 0.276
#CTS-CN-F2F-2018-04-16_723 CDER = 0.151
#CTS-CN-F2F-2018-04-16_726 CDER = 0.248
#CTS-CN-F2F-2018-04-16_757 CDER = 0.230
#CTS-CN-F2F-2018-04-16_763 CDER = 0.261
#CTS-CN-F2F-2018-04-23_507 CDER = 0.156
#CTS-CN-F2F-2018-04-23_620 CDER = 0.272
#CTS-CN-F2F-2018-05-08_110 CDER = 0.046
#CTS-CN-F2F-2018-05-08_236 CDER = 0.138
#CTS-CN-F2F-2018-05-08_469 CDER = 0.131
#CTS-CN-F2F-2018-05-08_470 CDER = 0.237
#CTS-CN-F2F-2018-05-08_471 CDER = 0.262
#CTS-CN-F2F-2018-05-08_472 CDER = 0.142
#CTS-CN-F2F-2018-05-08_712 CDER = 0.185
#CTS-CN-F2F-2018-05-08_713 CDER = 0.211
#CTS-CN-F2F-2018-05-08_776 CDER = 0.086
#CTS-CN-F2F-2018-05-08_777 CDER = 0.118
#CTS-CN-F2F-2018-05-08_779 CDER = 0.261
#CTS-CN-F2F-2018-05-08_782 CDER = 0.076
#CTS-CN-F2F-2018-05-08_799 CDER = 0.222
#CTS-CN-F2F-2018-05-08_825 CDER = 0.195
#CTS-CN-F2F-2018-05-08_909 CDER = 0.392
#CTS-CN-F2F-2018-05-08_948 CDER = 0.087
#CTS-CN-F2F-2018-05-08_949 CDER = 0.071
#CTS-CN-F2F-2018-05-08_950 CDER = 0.195
#Avg CDER : 0.186 

if [ $stage -le 8 ] && [ ${stop_stage} -ge 8 ]; then

        #recipe_root=/YOUR_PATH/DiariZen/recipes/diar_ssl
        exp_root=/data/maduo/exp/speaker_diarization/eend_vc/ # it is seted by [meta] save_dir in  $conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
        conf_dir=eend_vc/conf

        # training setup
        #use_dual_opt=true  # true for wavlm_updated_conformer.toml; false for the others
        train_conf=$conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
        # train_conf=$conf_dir/wavlm_frozen_conformer.toml
        # train_conf=$conf_dir/fbank_conformer.toml
        # train_conf=$conf_dir/pyannote_baseline.toml

        conf_name=`ls $train_conf | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`

        # inference setup
        #dtype="dev test"
        dtype="dev test cssd_testset"
        data_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format

        pyan_merge_closer=0.5
        pyan_max_length_merged=50
        pyan_inf_max_batch=32

        cluster_threshold=0.70
        segmentation_step=0.1
        min_cluster_size=30
        infer_affix=_constrained_AHC_segmentation_step_${segmentation_step}_min_cluster_size_${min_cluster_size}_AHC_thres_${cluster_threshold}_pyan_max_length_merged${pyan_max_length_merged}_cam++_zh_200k_common

        avg_ckpt_num=5
        val_metric=DER   # Loss or DER
        #val_mode=prev   # [prev, best, center]

        val_mode=best
        # scoring setup
        #collar="0 0.25"
        REF_DIR=$data_dir
        dscore_dir=eend_vc/dscore/
        diarization_dir=$exp_root/$conf_name    # can be replaced by our pre-trained models, e.g. diarization_dir=/YOUR_PATH/checkpoints/wavlm_updated_conformer
        #config_dir=`ls $diarization_dir/*.toml | sort -r | head -n 1`
        config_dir=$conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
        #embedding_model=/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/resnet34-lm/pytorch_model.bin     # it's necessary to have "pyannote" in your directory path
        #embedding_model="/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin"
        #embedding_model="/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/voxceleb_resnet34_LM.onnx"
        #embedding_model="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.onnx"
        embedding_model="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.onnx"
	echo "stage2: model inference..."
    export CUDA_VISIBLE_DEVICES=0

    train_log=`du -h $diarization_dir/*.log | sort -rh | head -n 1 | awk '{print $NF}'`
    cat $train_log | grep 'Loss/DER' | awk -F ']:' '{print $NF}' > $diarization_dir/val_metric_summary.lst
    for dset in $dtype; do
      CUDA_VISIBLE_DEVICES=0\
       python eend_vc/infer_avg.py -C $config_dir \
            -i ${data_dir}/${dset}/wav.scp \
            -o ${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}/${dset} \
            -u ${data_dir}/${dset}/all.uem \
            --avg_ckpt_num $avg_ckpt_num \
            --val_metric $val_metric \
            --val_mode $val_mode \
            --val_metric_summary $diarization_dir/val_metric_summary.lst \
            --segmentation_step $segmentation_step \
            --min_cluster_size $min_cluster_size \
            --cluster_threshold $cluster_threshold \
            --embedding_model $embedding_model \
            --merge_closer $pyan_merge_closer \
            --max_length_merged $pyan_max_length_merged \
            --batch_size $pyan_inf_max_batch
     done
fi

if [ $stage -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    #dtype="dev test"
    dtype="dev test cssd_testset"
    collar="0 0.25"
    exp_root=/data/maduo/exp/speaker_diarization/eend_vc/ # it is seted by [meta] save_dir in  $conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
    conf_dir=eend_vc/conf

    # training setup
    #use_dual_opt=true  # true for wavlm_updated_conformer.toml; false for the others
    train_conf=$conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
        # train_conf=$conf_dir/wavlm_frozen_conformer.toml
        # train_conf=$conf_dir/fbank_conformer.toml
        # train_conf=$conf_dir/pyannote_baseline.toml

    conf_name=`ls $train_conf | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`
    pyan_merge_closer=0.5
    pyan_max_length_merged=50
    pyan_inf_max_batch=32

    cluster_threshold=0.70
    segmentation_step=0.1
    min_cluster_size=30
    infer_affix=_constrained_AHC_segmentation_step_${segmentation_step}_min_cluster_size_${min_cluster_size}_AHC_thres_${cluster_threshold}_pyan_max_length_merged${pyan_max_length_merged}_cam++_zh_200k_common

    avg_ckpt_num=5
    val_metric=DER   # Loss or DER
    val_mode=best  # [prev, best, center]
    data_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
    REF_DIR=$data_dir
    dscore_dir=eend_vc/dscore/
    diarization_dir=$exp_root/$conf_name
    for c in $collar;do
     for dset in $dtype;do
      echo " scoring..."
      SYS_DIR=${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}
      OUT_DIR=${SYS_DIR}/${dset}
       python3 ${dscore_dir}/score.py \
            -r ${REF_DIR}/${dset}/rttm_debug_nog0 \
            -s $OUT_DIR/*.rttm --collar ${c} \
            > $OUT_DIR/result_collar${c}
    done
   done
fi
# dev of magicdata-ramc, DER
# 5.13(11.26)
# test of magicdata-ramc, DER
# 5.61(11.73)
# cssd_testset of magicdata-ramc, DER
# 4.92(18.13)


# dev of magicdata-ramc, CDER
#  python3 cder/score.py -r /data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/dev/rttm_debug_nog0 -s  exp/wavlm_updated_conformer_magicdata-ramc/infer_constrained_AHC_segmentation_step_0.1_min_cluster_size_30_AHC_thres_0.70_pyan_max_length_merged50_cam++_zh_200k_common/metric_DER_best/avg_ckpt5/dev/all.rttm
#CTS-CN-F2F-2019-11-15-1421 CDER = 0.526
#CTS-CN-F2F-2019-11-15-1422 CDER = 0.404
#CTS-CN-F2F-2019-11-15-1423 CDER = 0.138
#CTS-CN-F2F-2019-11-15-1426 CDER = 0.512
#CTS-CN-F2F-2019-11-15-1428 CDER = 0.057
#CTS-CN-F2F-2019-11-15-1434 CDER = 0.216
#CTS-CN-F2F-2019-11-15-1447 CDER = 0.146
#CTS-CN-F2F-2019-11-15-1448 CDER = 0.152
#CTS-CN-F2F-2019-11-15-1449 CDER = 0.210
#CTS-CN-F2F-2019-11-15-1452 CDER = 0.117
#CTS-CN-F2F-2019-11-15-1458 CDER = 0.348
#CTS-CN-F2F-2019-11-15-1461 CDER = 0.526
#CTS-CN-F2F-2019-11-15-1463 CDER = 0.135
#CTS-CN-F2F-2019-11-15-1468 CDER = 0.078
#CTS-CN-F2F-2019-11-15-1469 CDER = 0.138
#CTS-CN-F2F-2019-11-15-1470 CDER = 0.482
#CTS-CN-F2F-2019-11-15-1473 CDER = 0.279
#CTS-CN-F2F-2019-11-15-1475 CDER = 0.371
#CTS-CN-F2F-2019-11-15-1477 CDER = 0.115
#Avg CDER : 0.261

# test of magicdata-rama, CDER
# python3 cder/score.py -r /data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/test/rttm_debug_nog0 -s  exp/wavlm_updated_conformer_magicdata-ramc
#/infer_constrained_AHC_segmentation_step_0.1_min_cluster_size_30_AHC_thres_0.70_pyan_max_length_merged50_cam++_zh_200k_common/metric_DER_best/avg_ckpt5/test/all.rttm
#CTS-CN-F2F-2019-11-04-387 CDER = 0.561
#CTS-CN-F2F-2019-11-11-193 CDER = 0.246
#CTS-CN-F2F-2019-11-11-198 CDER = 0.289
#CTS-CN-F2F-2019-11-11-390 CDER = 0.302
#CTS-CN-F2F-2019-11-11-423 CDER = 0.059
#CTS-CN-F2F-2019-11-11-431 CDER = 0.312
#CTS-CN-F2F-2019-11-11-716 CDER = 0.115
#CTS-CN-F2F-2019-11-15-3 CDER = 0.080
#CTS-CN-F2F-2019-11-15-8 CDER = 0.344
#CTS-CN-F2F-2019-11-15-9 CDER = 0.218
#CTS-CN-F2F-2019-11-15-17 CDER = 0.326
#CTS-CN-F2F-2019-11-15-26 CDER = 0.151
#CTS-CN-F2F-2019-11-15-27 CDER = 0.134
#CTS-CN-F2F-2019-11-15-37 CDER = 0.111
#CTS-CN-F2F-2019-11-15-39 CDER = 0.040
#CTS-CN-F2F-2019-11-15-45 CDER = 0.207
#CTS-CN-F2F-2019-11-15-52 CDER = 0.098
#CTS-CN-F2F-2019-11-15-53 CDER = 0.074
#CTS-CN-F2F-2019-11-15-56 CDER = 0.310
#CTS-CN-F2F-2019-11-15-65 CDER = 0.197
#CTS-CN-F2F-2019-11-15-68 CDER = 0.152
#CTS-CN-F2F-2019-11-15-73 CDER = 0.026
#CTS-CN-F2F-2019-11-15-75 CDER = 0.160
#CTS-CN-F2F-2019-11-15-77 CDER = 0.105
#CTS-CN-F2F-2019-11-15-81 CDER = 0.281
#CTS-CN-F2F-2019-11-15-82 CDER = 0.071
#CTS-CN-F2F-2019-11-15-88 CDER = 0.284
#CTS-CN-F2F-2019-11-15-91 CDER = 0.231
#CTS-CN-F2F-2019-11-15-94 CDER = 0.211
#CTS-CN-F2F-2019-11-15-100 CDER = 0.068
#CTS-CN-F2F-2019-11-15-101 CDER = 0.235
#CTS-CN-F2F-2019-11-15-105 CDER = 0.000
#CTS-CN-F2F-2019-11-15-121 CDER = 0.023
#CTS-CN-F2F-2019-11-15-123 CDER = 0.732
#CTS-CN-F2F-2019-11-15-126 CDER = 0.046
#CTS-CN-F2F-2019-11-15-128 CDER = 0.201
#CTS-CN-F2F-2019-11-15-135 CDER = 0.207
#CTS-CN-F2F-2019-11-15-138 CDER = 0.279
#CTS-CN-F2F-2019-11-15-142 CDER = 0.230
#CTS-CN-F2F-2019-11-15-144 CDER = 0.647
#CTS-CN-F2F-2019-11-15-150 CDER = 0.182
#CTS-CN-F2F-2019-11-15-151 CDER = 0.215
#CTS-CN-F2F-2019-11-15-153 CDER = 0.208
#Avg CDER : 0.209

# cssd_testset(CSSD_Eval) of magicdata-rama, CDER
##python3 cder/score.py -r /data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/cssd_testset/rttm_debug_nog0 -s  exp/wavlm_updated_conformer_magicdata-ramc/infer_constrained_AHC_segmentation_step_0.1_min_cluster_size_30_AHC_thres_0.70_pyan_max_length_merged50_cam++_zh_200k_common/metric_DER_best/avg_ckpt5//cssd_testset/all.rttm
#CTS-CN-F2F-2017-11-14_85 CDER = 0.263
#CTS-CN-F2F-2017-11-14_87 CDER = 0.215
#CTS-CN-F2F-2017-11-14_88 CDER = 0.111
#CTS-CN-F2F-2017-11-29_109 CDER = 0.124
#CTS-CN-F2F-2017-12-15_412 CDER = 0.144
#CTS-CN-F2F-2017-12-15_438 CDER = 0.119
#CTS-CN-F2F-2017-12-15_440 CDER = 0.115
#CTS-CN-F2F-2017-12-20_70 CDER = 0.135
#CTS-CN-F2F-2018-04-16_545 CDER = 0.181
#CTS-CN-F2F-2018-04-16_560 CDER = 0.128
#CTS-CN-F2F-2018-04-16_563 CDER = 0.131
#CTS-CN-F2F-2018-04-16_615 CDER = 0.262
#CTS-CN-F2F-2018-04-16_699 CDER = 0.205
#CTS-CN-F2F-2018-04-16_700 CDER = 0.205
#CTS-CN-F2F-2018-04-16_703 CDER = 0.229
#CTS-CN-F2F-2018-04-16_716 CDER = 0.196
#CTS-CN-F2F-2018-04-16_723 CDER = 0.151
#CTS-CN-F2F-2018-04-16_726 CDER = 0.217
#CTS-CN-F2F-2018-04-16_757 CDER = 0.218
#CTS-CN-F2F-2018-04-16_763 CDER = 0.130
#CTS-CN-F2F-2018-04-23_507 CDER = 0.156
#CTS-CN-F2F-2018-04-23_620 CDER = 0.272
#CTS-CN-F2F-2018-05-08_110 CDER = 0.046
#CTS-CN-F2F-2018-05-08_236 CDER = 0.111
#CTS-CN-F2F-2018-05-08_469 CDER = 0.129
#CTS-CN-F2F-2018-05-08_470 CDER = 0.241
#CTS-CN-F2F-2018-05-08_471 CDER = 0.133
#CTS-CN-F2F-2018-05-08_472 CDER = 0.142
#CTS-CN-F2F-2018-05-08_712 CDER = 0.108
#CTS-CN-F2F-2018-05-08_713 CDER = 0.088
#CTS-CN-F2F-2018-05-08_776 CDER = 0.086
#CTS-CN-F2F-2018-05-08_777 CDER = 0.120
#CTS-CN-F2F-2018-05-08_779 CDER = 0.277
#CTS-CN-F2F-2018-05-08_782 CDER = 0.074
#CTS-CN-F2F-2018-05-08_799 CDER = 0.228
#CTS-CN-F2F-2018-05-08_825 CDER = 0.221
#CTS-CN-F2F-2018-05-08_909 CDER = 0.167
#CTS-CN-F2F-2018-05-08_948 CDER = 0.093
#CTS-CN-F2F-2018-05-08_949 CDER = 0.068
#CTS-CN-F2F-2018-05-08_950 CDER = 0.217
#Avg CDER : 0.161



if [ $stage -le 10 ] && [ ${stop_stage} -ge 10 ]; then

        #recipe_root=/YOUR_PATH/DiariZen/recipes/diar_ssl
        #exp_root=/data/maduo/exp/speaker_diarization/eend_vc/ # it is seted by [meta] save_dir in  $conf_dir/wavlm_updated_conformer_magicdata-ramc.toml  
	exp_root=exp
	conf_dir=eend_vc/conf
        # training setup
        #use_dual_opt=true  # true for wavlm_updated_conformer.toml; false for the others
        train_conf=$conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
        # train_conf=$conf_dir/wavlm_frozen_conformer.toml
        # train_conf=$conf_dir/fbank_conformer.toml
        # train_conf=$conf_dir/pyannote_baseline.toml

        conf_name=`ls $train_conf | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`

        # inference setup
        #dtype="dev test"
        dtype="dev test cssd_testset"
        data_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format

        pyan_merge_closer=0.5
        pyan_max_length_merged=50
        pyan_inf_max_batch=32

        cluster_threshold=0.70
        segmentation_step=0.1
        min_cluster_size=30
        infer_affix=_constrained_AHC_segmentation_step_${segmentation_step}_min_cluster_size_${min_cluster_size}_AHC_thres_${cluster_threshold}_pyan_max_length_merged${pyan_max_length_merged}_eres2netv2_sv_zh-cn_16k-common_200k

        avg_ckpt_num=5
        val_metric=DER   # Loss or DER
        #val_mode=prev   # [prev, best, center]

        val_mode=best
        # scoring setup
        #collar="0 0.25"
        REF_DIR=$data_dir
        dscore_dir=eend_vc/dscore/
        diarization_dir=$exp_root/$conf_name    # can be replaced by our pre-trained models, e.g. diarization_dir=/YOUR_PATH/checkpoints/wavlm_updated_conformer
        #config_dir=`ls $diarization_dir/*.toml | sort -r | head -n 1`
        config_dir=$conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
        #embedding_model=/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/resnet34-lm/pytorch_model.bin     # it's necessary to have "pyannote" in your directory path
        #embedding_model="/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin"
        #embedding_model="/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/voxceleb_resnet34_LM.onnx"
        #embedding_model="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.onnx"
        #embedding_model="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.onnx"
        embedding_model="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_eres2netv2_sv_zh-cn_16k-common/pretrained_eres2netv2.onnx"        
	echo "stage2: model inference..."
    export CUDA_VISIBLE_DEVICES=0

    train_log=`du -h $diarization_dir/*.log | sort -rh | head -n 1 | awk '{print $NF}'`
    cat $train_log | grep 'Loss/DER' | awk -F ']:' '{print $NF}' > $diarization_dir/val_metric_summary.lst
    for dset in $dtype; do
      CUDA_VISIBLE_DEVICES=0\
       python eend_vc/infer_avg.py -C $config_dir \
            -i ${data_dir}/${dset}/wav.scp \
            -o ${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}/${dset} \
            -u ${data_dir}/${dset}/all.uem \
            --avg_ckpt_num $avg_ckpt_num \
            --val_metric $val_metric \
            --val_mode $val_mode \
            --val_metric_summary $diarization_dir/val_metric_summary.lst \
            --segmentation_step $segmentation_step \
            --min_cluster_size $min_cluster_size \
            --cluster_threshold $cluster_threshold \
            --embedding_model $embedding_model \
            --merge_closer $pyan_merge_closer \
            --max_length_merged $pyan_max_length_merged \
            --batch_size $pyan_inf_max_batch
     done
fi

if [ $stage -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    #dtype="dev test"
    dtype="dev test cssd_testset"
    collar="0 0.25"
    #exp_root=/data/maduo/exp/speaker_diarization/eend_vc/ # it is seted by [meta] save_dir in  $conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
    exp_root=exp
    conf_dir=eend_vc/conf

    # training setup
    #use_dual_opt=true  # true for wavlm_updated_conformer.toml; false for the others
    train_conf=$conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
        # train_conf=$conf_dir/wavlm_frozen_conformer.toml
        # train_conf=$conf_dir/fbank_conformer.toml
        # train_conf=$conf_dir/pyannote_baseline.toml

    conf_name=`ls $train_conf | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`
    pyan_merge_closer=0.5
    pyan_max_length_merged=50
    pyan_inf_max_batch=32

    cluster_threshold=0.70
    segmentation_step=0.1
    min_cluster_size=30
    infer_affix=_constrained_AHC_segmentation_step_${segmentation_step}_min_cluster_size_${min_cluster_size}_AHC_thres_${cluster_threshold}_pyan_max_length_merged${pyan_max_length_merged}_eres2netv2_sv_zh-cn_16k-common_200k

    avg_ckpt_num=5
    val_metric=DER   # Loss or DER
    val_mode=best  # [prev, best, center]
    data_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
    REF_DIR=$data_dir
    dscore_dir=eend_vc/dscore/
    diarization_dir=$exp_root/$conf_name
    for c in $collar;do
     for dset in $dtype;do
      echo " scoring..."
      SYS_DIR=${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}
      OUT_DIR=${SYS_DIR}/${dset}
       python3 ${dscore_dir}/score.py \
            -r ${REF_DIR}/${dset}/rttm_debug_nog0 \
            -s $OUT_DIR/*.rttm --collar ${c} \
            > $OUT_DIR/result_collar${c}
    done
   done
fi

if [ $stage -le 12 ] && [ ${stop_stage} -ge 12 ]; then

        #recipe_root=/YOUR_PATH/DiariZen/recipes/diar_ssl
        #exp_root=/data/maduo/exp/speaker_diarization/eend_vc/ # it is seted by [meta] save_dir in  $conf_dir/wavlm_updated_conformer_magicdata-ramc.toml 
        
        exp_root=exp
        conf_dir=eend_vc/conf
	# training setup
        #use_dual_opt=true  # true for wavlm_updated_conformer.toml; false for the others
        train_conf=$conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
        # train_conf=$conf_dir/wavlm_frozen_conformer.toml
        # train_conf=$conf_dir/fbank_conformer.toml
        # train_conf=$conf_dir/pyannote_baseline.toml

        conf_name=`ls $train_conf | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`

        # inference setup
        #dtype="dev test"
        dtype="dev test cssd_testset"
        data_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format

        pyan_merge_closer=0.5
        pyan_max_length_merged=50
        pyan_inf_max_batch=32

        cluster_threshold=0.70
        segmentation_step=0.1
        min_cluster_size=30
        infer_affix=_constrained_AHC_segmentation_step_${segmentation_step}_min_cluster_size_${min_cluster_size}_AHC_thres_${cluster_threshold}_pyan_max_length_merged${pyan_max_length_merged}_eres2netv2w24s4ep4_sv_zh-cn_16k-common_200k

        avg_ckpt_num=5
        val_metric=DER   # Loss or DER
        #val_mode=prev   # [prev, best, center]

        val_mode=best
        # scoring setup
        #collar="0 0.25"
        REF_DIR=$data_dir
        dscore_dir=eend_vc/dscore/
        diarization_dir=$exp_root/$conf_name    # can be replaced by our pre-trained models, e.g. diarization_dir=/YOUR_PATH/checkpoints/wavlm_updated_conformer
        #config_dir=`ls $diarization_dir/*.toml | sort -r | head -n 1`
        config_dir=$conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
        #embedding_model=/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/resnet34-lm/pytorch_model.bin     # it's necessary to have "pyannote" in your directory path
        #embedding_model="/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin"
        #embedding_model="/data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/voxceleb_resnet34_LM.onnx"
        #embedding_model="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.onnx"
        #embedding_model="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.onnx"
        #embedding_model="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_eres2netv2_sv_zh-cn_16k-common/pretrained_eres2netv2.onnx"
        embedding_model="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common/pretrained_eres2netv2w24s4ep4.onnx"
	echo "stage2: model inference..."
    export CUDA_VISIBLE_DEVICES=0

    train_log=`du -h $diarization_dir/*.log | sort -rh | head -n 1 | awk '{print $NF}'`
    cat $train_log | grep 'Loss/DER' | awk -F ']:' '{print $NF}' > $diarization_dir/val_metric_summary.lst
    for dset in $dtype; do
      CUDA_VISIBLE_DEVICES=0\
       python eend_vc/infer_avg.py -C $config_dir \
            -i ${data_dir}/${dset}/wav.scp \
            -o ${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}/${dset} \
            -u ${data_dir}/${dset}/all.uem \
            --avg_ckpt_num $avg_ckpt_num \
            --val_metric $val_metric \
            --val_mode $val_mode \
            --val_metric_summary $diarization_dir/val_metric_summary.lst \
            --segmentation_step $segmentation_step \
            --min_cluster_size $min_cluster_size \
            --cluster_threshold $cluster_threshold \
            --embedding_model $embedding_model \
            --merge_closer $pyan_merge_closer \
            --max_length_merged $pyan_max_length_merged \
            --batch_size $pyan_inf_max_batch
     done
fi

if [ $stage -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    #dtype="dev test"
    dtype="dev test cssd_testset"
    collar="0 0.25"
    #exp_root=/data/maduo/exp/speaker_diarization/eend_vc/ # it is seted by [meta] save_dir in  $conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
    exp_root=exp
    conf_dir=eend_vc/conf

    # training setup
    #use_dual_opt=true  # true for wavlm_updated_conformer.toml; false for the others
    train_conf=$conf_dir/wavlm_updated_conformer_magicdata-ramc.toml
        # train_conf=$conf_dir/wavlm_frozen_conformer.toml
        # train_conf=$conf_dir/fbank_conformer.toml
        # train_conf=$conf_dir/pyannote_baseline.toml

    conf_name=`ls $train_conf | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`
    pyan_merge_closer=0.5
    pyan_max_length_merged=50
    pyan_inf_max_batch=32

    cluster_threshold=0.70
    segmentation_step=0.1
    min_cluster_size=30
    infer_affix=_constrained_AHC_segmentation_step_${segmentation_step}_min_cluster_size_${min_cluster_size}_AHC_thres_${cluster_threshold}_pyan_max_length_merged${pyan_max_length_merged}_eres2netv2w24s4ep4_sv_zh-cn_16k-common_200k

    avg_ckpt_num=5
    val_metric=DER   # Loss or DER
    val_mode=best  # [prev, best, center]
    data_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
    REF_DIR=$data_dir
    dscore_dir=eend_vc/dscore/
    diarization_dir=$exp_root/$conf_name
    for c in $collar;do
     for dset in $dtype;do
      echo " scoring..."
      SYS_DIR=${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}
      OUT_DIR=${SYS_DIR}/${dset}
       python3 ${dscore_dir}/score.py \
            -r ${REF_DIR}/${dset}/rttm_debug_nog0 \
            -s $OUT_DIR/*.rttm --collar ${c} \
            > $OUT_DIR/result_collar${c}
    done
   done
fi
