#!/usr/bin/env bash

stage=0
stop_stage=1000
duration=2 # per non overlap target speaker speech duration is more than 2 seconds.
target_wavs_name_dir="non_overlap_segment"
audio_amount=200 # simu 200h mix train data
determine_spks="fixed" # choice from "fixed, maximum"
num_spks=4 # per conversation contains speaker numbers
sampling_rate=16000 # simu mix audio sample rate
dataset_dir=/mntcephfs/lab_data/maduo/datasets
suffix="" # It will indicate the folder name of the simulation dataset
continue_cat=false #
#datasub="Eval Train"
datasub="Test"
. utils/parse_options.sh
. path.sh
#dataset_dir=$1
alimeeting_dataset_dir=$dataset_dir/alimeeting
#simu_alimeeting_dataset_dir=$dataset_dir/simu_alimeeting
simu_alimeeting_dataset_dir=$1
mkdir -p $simu_alimeeting_dataset_dir
## how to use this script
## for example: sbatch  --nodes 1 --ntasks-per-node=6 --cpus-per-task=1 -p p-cpu -A t00120220002 -o logs/run_simu_mix_on_alimeeting_kaldi_format_sribd_stage0-9_duration2s.log run_simu_mix_on_alimeeting_kaldi_format_sribd.sh --stage 0 --stop-stage 9 --duration 2 --target-wavs-name-dir non_overlap_segment
# or
# sbatch  --nodes 1 --ntasks-per-node=6 --cpus-per-task=1 -p p-cpu -A t00120220002 -o logs/run_simu_mix_on_alimeeting_kaldi_format_sribd_stage0-10_duration6s.log run_simu_mix_on_alimeeting_kaldi_format_sribd.sh --stage 0 --stop-stage 10 --duration 6 --target-wavs-name-dir non_overlap_segment_6s --suffix 6s
# or
# sbatch  --nodes 1 --ntasks-per-node=6 --cpus-per-task=1 -p p-cpu -A t00120220002 -o logs/run_simu_mix_on_alimeeting_kaldi_format_sribd_stage0-10_duration16s.log run_simu_mix_on_alimeeting_kaldi_format_sribd.sh --stage 0 --stop-stage 10 --duration 16 --continue-cat true --target-wavs-name-dir non_overlap_segment_cat_16s --suffix cat_16s

# or
#  sbatch  --nodes 1 --ntasks-per-node=6 --cpus-per-task=1 -p p-cpu -A t00120220002 -o logs/run_simu_mix_on_alimeeting_kaldi_format_sribd_stage-1.log run_simu_mix_on_alimeeting_kaldi_format_sribd.sh --stage -1 --stop-stage -1 --target-wavs-name-dir non_overlaps

## This script will output RTTM of simulation data finally.
## This script counts pauses and overlaps in the training set of alimeeting,
# and constructs conversational simulation data(i.e. method is from https://arxiv.org/pdf/2204.00890) based on this information and sentences with non-overlap alimeeting.

## 6s segments and simu-data= 400hours
# sbatch  --nodes 1 --ntasks-per-node=6 --cpus-per-task=1 -p p-cpu -A t00120220002 -o logs/run_simu_mix_on_alimeeting_kaldi_format_sribd_stage4-10_duration6s_400h.log run_simu_mix_on_alimeeting_kaldi_format_sribd.sh --stage 4 --stop-stage 10 --duration 6 --target-wavs-name-dir non_overlap_segment_6s --suffix 6s --audio-amount 400


## 6s segments and simu-data= 800hours
# sbatch  --nodes 1 --ntasks-per-node=6 --cpus-per-task=1 -p p-cpu -A t00120220002 -o logs/run_simu_mix_on_alimeeting_kaldi_format_sribd_stage4-10_duration6s_800h.log run_simu_mix_on_alimeeting_kaldi_format_sribd.sh --stage 4 --stop-stage 10 --duration 6 --target-wavs-name-dir non_overlap_segment_6s --suffix 6s --audio-amount 800

## 6s segments and simu-data= 200hours, determine_spks="maximum"
# sbatch  --nodes 1 --ntasks-per-node=6 --cpus-per-task=1 -p p-cpu -A t00120220002 -o logs/run_simu_mix_on_alimeeting_kaldi_format_sribd_stage4-10_duration6s_200h_determine_spks_maximum.log run_simu_mix_on_alimeeting_kaldi_format_sribd.sh --stage 4 --stop-stage 10 --duration 6 --target-wavs-name-dir non_overlap_segment_6s --suffix 6s --audio-amount 200 --determine-spks maximum  /mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/simu_alimeeting

##cat all segments and simu-data= 200hours, determine_spks="maximum"
# sbatch  --nodes 1 --ntasks-per-node=6 --cpus-per-task=1 -p p-cpu -A t00120220002 -o logs/run_simu_mix_on_alimeeting_kaldi_format_sribd_stage2-10_duration_all_200h_determine_spks_maximum.log run_simu_mix_on_alimeeting_kaldi_format_sribd.sh --stage 2 --stop-stage 10 --target-wavs-name-dir non_overlaps --suffix all --audio-amount 200 --determine-spks maximum /mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/simu_alimeeting


# 2024-12-19
#sbatch  --nodes 1 --ntasks-per-node=6 --cpus-per-task=1 -p p-cpu -A t00120220002 -o logs/run_simu_mix_on_alimeeting_kaldi_format_sribd_stage0-4_duration6s.log run_simu_mix_on_alimeeting_kaldi_format_sribd.sh --stage 0 --stop-stage 4 --duration 6 --target-wavs-name-dir non_overlap_segment_6s --suffix 6s --audio-amount 200 /mntcephfs/lab_data/maduo/datasets/simu_alimeeting

## Processing reference (oracle ) RTTMs(i.e. Alimeeting train set) to obtain statistics about turns
## It will offer some usefull information for simulation conversation style diarization dataset.
if [ ${stage} -le -10 ] && [ ${stop_stage} -ge -10 ];then
   RTTMS_FILE=/mntcephfs/lab_data/maduo/model_hub/ts_vad/alimeeting_train.rttm
   STATS_DIR=${alimeeting_dataset_dir}/Train_Ali_far/stats_truns
   mkdir -p $STATS_DIR
   # awk determines the type of overlap between pairs of consecutive segments.
   # For each case, the length of the overlap is calculated. The categories are:
   # 'takeover', when, after the overlap, the second speaker keeps speaking
   # 'interrupt', when, after the overlap, the second speaker stops
   # 'both', when both speakers end the overlap simultaneously
   cat $RTTMS_FILE | awk '{if(NR==1 || !match($2,file)){endt=$4+$5;file=$2}else{if(endt>$4){if($4+$5>endt){print "takeover",-(endt-$4);endt=$4+$5}else{if(endt>$4+$5){print "interrupt",-$5}else{print "both",-$5}}}else{endt=$4+$5}}}' > $STATS_DIR/overlaps_info.txt

   DISTSNAME=newspk_samespk_pause_distribution_overlap_distribution
   # awk determines the nature of consecutive segments and calculates lengths
   # 'same spk', when the segments belong to the same speaker (length of the pause)
   # 'new spk', when the segments belong to different speakers (length of the pause)
   # 'overlap', when segments overlap (length of the overlap)
   cat $RTTMS_FILE | awk '{if(NR==1 ||!match($2,file)){file=$2;spk=$8;init=$4;end=$4+$5}else{if(match($8,spk)){print "same spk, pause",$4-end;init=$4;end=$4+$5}else{if($4>end){print "new spk, pause",$4-end;end=$4+$5;init=$4;spk=$8}else{if($4+$5>end){print "overlap",end-$4;end=$4+$5;init=$4;spk=$8}else{print "overlap",$5}}}}}' > $STATS_DIR/${DISTSNAME}.txt
   grep "same spk, pause" $STATS_DIR/${DISTSNAME}.txt | awk '{printf "%.2f\n", $NF}' | \
        sort -n | uniq -c > $STATS_DIR/same_spk_pause.txt
   grep "new spk, pause"  $STATS_DIR/${DISTSNAME}.txt | awk '{printf "%.2f\n", $NF}' | \
        sort -n | uniq -c > $STATS_DIR/diff_spk_pause.txt
   grep "same spk, pause" $STATS_DIR/${DISTSNAME}.txt | awk '{$NF=""; print}' | \
        sort -n | uniq -c > $STATS_DIR/diff_spk_pause_vs_overlap.txt
   cat $STATS_DIR/overlaps_info.txt | awk '{printf "%.2f\n", -$2}' | \
        sort -n | uniq -c > $STATS_DIR/diff_spk_overlap.txt
fi
if [ ${stage} -le -4 ] && [ ${stop_stage} -ge -4 ];then
    echo "get non_overlap speaker speech(cat all segment in one mix audio) from textgrid(oracle vad) "
    . path_for_dia_pt2.4.sh
    alimeeting_corpus=${alimeeting_dataset_dir}
    #for name in Eval;do # debug
    #for name in $datasub;do
    for name in Train;do
      torchrun --nproc_per_node=16 --master_port=12345 ts_vad2_simu/prepare_non_overlapped_all_single_speaker_speech_from_alimeeting_parallel.py\
              --data_path $alimeeting_corpus\
              --type $name\
              --duration $duration\
              --dest_name_dir ${target_wavs_name_dir}\
              --nj 16
   done
fi

if [ ${stage} -le -3 ] && [ ${stop_stage} -ge -3 ];then
    echo "get non_overlap speaker speech(cat all segment in one mix audio) from textgrid(oracle vad) "
    . path_for_dia_pt2.4.sh
    alimeeting_corpus=${alimeeting_dataset_dir}
    #for name in Eval;do # debug
    #for name in $datasub;do
    nj=8
    for name in Eval;do
      torchrun --nproc_per_node=$nj --master_port=12456 ts_vad2_simu/prepare_non_overlapped_all_single_speaker_speech_from_alimeeting_parallel.py\
              --data_path $alimeeting_corpus\
              --type $name\
              --duration $duration\
              --dest_name_dir ${target_wavs_name_dir}\
              --nj $nj
   done
fi
if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ];then
    echo "get non_overlap speaker speech(cat all segment in one mix audio) from textgrid(oracle vad) "
    . path_for_dia_pt2.4.sh
    alimeeting_corpus=${alimeeting_dataset_dir}
    #for name in Eval;do # debug
    #for name in $datasub;do
    nj=8
    for name in Test;do
      torchrun --nproc_per_node=$nj --master_port=12456 ts_vad2_simu/prepare_non_overlapped_all_single_speaker_speech_from_alimeeting_parallel.py\
              --data_path $alimeeting_corpus\
              --type $name\
              --duration $duration\
              --dest_name_dir ${target_wavs_name_dir}\
              --nj $nj
   done
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
    echo "prepared kaldi format for non_overlap speaker speech(cat all segment in one mix audio)"
    dataset_dir=${alimeeting_dataset_dir}/
    echo "prapred wav.scp"
    #for name in Eval Test Train;do
    for name in $datasub;do
    #for name in Train;do
     if [ $name = "Train" ];then
      indir=$dataset_dir/${name}_Ali_far/${target_wavs_name_dir}
     else
      indir=$dataset_dir/${name}_Ali/${name}_Ali_far/${target_wavs_name_dir}
     fi
     # uttid is session+spkid + segment_id (i.e. SPK8013-R8001_M8004_MS801_SPK8013_0), session is R8001_M8004_MS801, spkid is SPK8013, segment_id is 0
     #find -L $indir -iname "*.wav" | sort |  awk '{n=split($1,A,/[\/\.]/); print A[n-2]"_"A[n-1], $1}' | sort > $indir/wav.scp
     python3 ts_vad2_simu/prepare_nonoverlap_all_seg_alimeeting_wavscp.py $indir | sort > $indir/wav.scp
   done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
    echo "get non_overlap speaker speech from textgrid(oracle vad) "
    alimeeting_corpus=${alimeeting_dataset_dir}
    #for name in Eval;do # debug
    for name in $datasub;do
    #for name in Train;do
    if $continue_cat;then
       python3 ts_vad2_simu/prepare_non_overlapped_fixed_single_speaker_speech_from_alimeeting.py\
              --data_path $alimeeting_corpus\
              --type $name\
              --duration $duration\
              --cat_cut true\
              --dest_name_dir ${target_wavs_name_dir}
   else
       python3 ts_vad2_simu/prepare_non_overlapped_single_speaker_speech_from_alimeeting.py\
              --data_path $alimeeting_corpus\
              --type $name\
              --duration $duration\
              --dest_name_dir ${target_wavs_name_dir}
    fi
   done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
    echo "prepared kaldi format for non_overlap speaker speech"
    dataset_dir=${alimeeting_dataset_dir}/
    echo "prapred wav.scp"
    #for name in Eval Test Train;do
    for name in $datasub;do
    #for name in Train;do
     if [ $name = "Train" ];then
      indir=$dataset_dir/${name}_Ali_far/${target_wavs_name_dir}
     else
      indir=$dataset_dir/${name}_Ali/${name}_Ali_far/${target_wavs_name_dir}
     fi
     # uttid is session+spkid + segment_id (i.e. SPK8013-R8001_M8004_MS801_SPK8013_0), session is R8001_M8004_MS801, spkid is SPK8013, segment_id is 0
     #find -L $indir -iname "*.wav" | sort |  awk '{n=split($1,A,/[\/\.]/); print A[n-2]"_"A[n-1], $1}' | sort > $indir/wav.scp
     python3 ts_vad2_simu/prepare_nonoverlap_alimeeting_wavscp.py $indir | sort > $indir/wav.scp
   done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
    echo "prepred utt2spk and spk2utt"
    dataset_dir=${alimeeting_dataset_dir}/
    for name in $datasub;do
    #for name in Train;do
     if [ $name = "Train" ];then
      indir=$dataset_dir/${name}_Ali_far/${target_wavs_name_dir}
     else
      indir=$dataset_dir/${name}_Ali/${name}_Ali_far/${target_wavs_name_dir}
     fi
     #
     awk '{n=split($1,A,"-"); print $1, A[1]}' $indir/wav.scp | sort > $indir/utt2spk
     utils/utt2spk_to_spk2utt.pl <$indir/utt2spk >$indir/spk2utt || exit 1
     echo "prepared utt2dur"
     [[ -f "$indir/utt2dur" ]] && rm $indir/utt2dur
     utils/data/get_utt2dur.sh $indir 1>&2 || exit 1
     utils/data/get_segments_for_data.sh $indir > $indir/segments
     utils/validate_data_dir.sh --no-feats --no-text $indir || exit 1;
     echo "$0: successfully prepared data in $indir"
   done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
  echo "Generating conversations for the test set"
  CONV_VAL_DIR=${simu_alimeeting_dataset_dir}/${audio_amount}h_${determine_spks}_${num_spks}spks${suffix}/test/conv_val_dir
  mkdir -p $CONV_VAL_DIR
  STATS_DIR=${alimeeting_dataset_dir}/Train_Ali_far/stats_truns
  val_seg=${alimeeting_dataset_dir}/Test_Ali/Test_Ali_far/${target_wavs_name_dir}/segments
  python3 ts_vad2_simu/conv_generator.py --stats-dir $STATS_DIR \
        --seg-list-file $val_seg --out-conv-dir $CONV_VAL_DIR \
        --determine-spks "$determine_spks" --num-spks $num_spks \
        --audio-amount -1 --sampling-frequency $sampling_rate
  echo "----- Test conversation generated -----"
fi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
  echo "Generating conversations for the validation set"
  CONV_VAL_DIR=${simu_alimeeting_dataset_dir}/${audio_amount}h_${determine_spks}_${num_spks}spks${suffix}/eval/conv_val_dir
  mkdir -p $CONV_VAL_DIR
  STATS_DIR=${alimeeting_dataset_dir}/Train_Ali_far/stats_truns
  val_seg=${alimeeting_dataset_dir}/Eval_Ali/Eval_Ali_far/${target_wavs_name_dir}/segments
  python3 ts_vad2_simu/conv_generator.py --stats-dir $STATS_DIR \
		--seg-list-file $val_seg --out-conv-dir $CONV_VAL_DIR \
		--determine-spks "$determine_spks" --num-spks $num_spks \
		--audio-amount -1 --sampling-frequency $sampling_rate
  echo "----- Validation conversation generated -----"
fi
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
  echo "Generating conversations for the train set"
  CONV_Train_DIR=${simu_alimeeting_dataset_dir}/${audio_amount}h_${determine_spks}_${num_spks}spks${suffix}/train/conv_train_dir
  mkdir -p $CONV_Train_DIR
  STATS_DIR=${alimeeting_dataset_dir}/Train_Ali_far/stats_truns
  train_seg=${alimeeting_dataset_dir}/Train_Ali_far/${target_wavs_name_dir}/segments
  python3 ts_vad2_simu/conv_generator.py --stats-dir $STATS_DIR \
        --seg-list-file $train_seg --out-conv-dir $CONV_Train_DIR \
        --determine-spks "$determine_spks" --num-spks $num_spks \
        --audio-amount $audio_amount --sampling-frequency $sampling_rate
  echo "----- Train conversation generated -----"
fi


# generate validation conversation mix audio
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   val_dir=${simu_alimeeting_dataset_dir}/${audio_amount}h_${determine_spks}_${num_spks}spks${suffix}/eval
   CONV_VAL_DIR=$val_dir/conv_val_dir
   CONV_VAL_LIST=$val_dir/conv_val_list
   find $CONV_VAL_DIR -iname "*.conv" | \
			xargs -n 1 basename -s .conv > $CONV_VAL_LIST
   WAV_SCP=${alimeeting_dataset_dir}/Eval_Ali/Eval_Ali_far/${target_wavs_name_dir}/wav.scp
   WAVS_VAL_DIR=$val_dir/wavs
   mkdir -p $WAVS_VAL_DIR
   python3 ts_vad2_simu/conv2wav.py --conversations-list-filename $CONV_VAL_LIST \
			--input-wav-scp $WAV_SCP --in-conv-dir $CONV_VAL_DIR \
			--out-wav-dir $WAVS_VAL_DIR \
			--sampling-frequency $sampling_rate

fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
    train_dir=${simu_alimeeting_dataset_dir}/${audio_amount}h_${determine_spks}_${num_spks}spks${suffix}/train/
   CONV_train_DIR=${train_dir}/conv_train_dir
   CONV_train_LIST=${train_dir}/conv_train_list
   find $CONV_train_DIR -iname "*.conv" | \
            xargs -n 1 basename -s .conv > $CONV_train_LIST
   WAV_SCP=${alimeeting_dataset_dir}/Train_Ali_far/${target_wavs_name_dir}/wav.scp
   WAVS_TR_DIR=$train_dir/wavs
   mkdir -p $WAVS_TR_DIR
   python3 ts_vad2_simu/conv2wav.py --conversations-list-filename $CONV_train_LIST \
            --input-wav-scp $WAV_SCP --in-conv-dir $CONV_train_DIR \
            --out-wav-dir $WAVS_TR_DIR \
            --sampling-frequency $sampling_rate

fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
   echo "Generating general Kaldi-style data directory for validation"
   val_dir=${simu_alimeeting_dataset_dir}/${audio_amount}h_${determine_spks}_${num_spks}spks${suffix}/eval
   CONV_VAL_DIR=$val_dir/conv_val_dir
   CONV_VAL_LIST=$val_dir/conv_val_list
   data_val_dir=$val_dir/data
   mkdir -p $data_val_dir

   for f in $(<$CONV_VAL_LIST); do
	   awk -v sampling_frequency=$sampling_rate '{printf "SPEAKER %s 1 %.3f %.3f <NA> <NA> %s <NA> <NA>\n", FILENAME, $5/sampling_frequency, ($4-$3)/sampling_frequency, $1}' $CONV_VAL_DIR/$f.conv
   done > $data_val_dir/rttm



   awk -F"/" '{print $NF}' $data_val_dir/rttm | sed 's/.conv//g' | \
   		awk '{printf "%s_%s_%06d_%06d %s %f %f\n", $7, $1, $3*100, ($3+$4)*100, $1, $3, $3+$4}' > $data_val_dir/segments
   awk -F"/" '{print $NF}' $data_val_dir/rttm | sed 's/.conv//g' | \
   		awk '{printf "%s_%s_%06d_%06d %s\n", $7, $1, $3*100, ($3+$4)*100, $7}' > $data_val_dir/utt2spk
   utils/utt2spk_to_spk2utt.pl $data_val_dir/utt2spk > $data_val_dir/spk2utt
   WAVS_VAL_DIR=$val_dir/wavs
   ls $WAVS_VAL_DIR | awk -F'.' -v path=$WAVS_VAL_DIR '{print $1" "path"/"$1".wav"}' > $data_val_dir/wav.scp
   echo "----- VALIDATION  KALDI DIR GENERATED it is stored at $data_val_dir-----"
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
   echo "Generating general Kaldi-style data directory for train"
   train_dir=${simu_alimeeting_dataset_dir}/${audio_amount}h_${determine_spks}_${num_spks}spks${suffix}/train
   CONV_train_DIR=$train_dir/conv_train_dir
   CONV_train_LIST=$train_dir/conv_train_list
   data_train_dir=$train_dir/data
   mkdir -p $data_train_dir
   for f in $(<$CONV_train_LIST); do
       awk -v sampling_frequency=$sampling_rate '{printf "SPEAKER %s 1 %.3f %.3f <NA> <NA> %s <NA> <NA>\n", FILENAME, $5/sampling_frequency, ($4-$3)/sampling_frequency, $1}' $CONV_train_DIR/$f.conv
   done > $data_train_dir/rttm


   awk -F"/" '{print $NF}' $data_train_dir/rttm | sed 's/.conv//g' | \
        awk '{printf "%s_%s_%06d_%06d %s %f %f\n", $7, $1, $3*100, ($3+$4)*100, $1, $3, $3+$4}' > $data_train_dir/segments
   awk -F"/" '{print $NF}' $data_train_dir/rttm | sed 's/.conv//g' | \
        awk '{printf "%s_%s_%06d_%06d %s\n", $7, $1, $3*100, ($3+$4)*100, $7}' > $data_train_dir/utt2spk
   utils/utt2spk_to_spk2utt.pl $data_train_dir/utt2spk > $data_train_dir/spk2utt
   WAVS_train_DIR=$train_dir/wavs
   ls $WAVS_train_DIR | awk -F'.' -v path=$WAVS_train_DIR '{print $1" "path"/"$1".wav"}' > $data_train_dir/wav.scp
   echo "----- Train KALDI DIR GENERATED it is stored at $data_train_dir -----"
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   echo "remove .conv from rttm filename"
   train_dir=${simu_alimeeting_dataset_dir}/${audio_amount}h_${determine_spks}_${num_spks}spks${suffix}/train
   data_train_dir=$train_dir/data
   CONV_train_DIR=$train_dir/conv_train_dir
   sed -i "s $CONV_train_DIR/  g" $data_train_dir/rttm
   sed -i 's .conv  g' $data_train_dir/rttm
   val_dir=${simu_alimeeting_dataset_dir}/${audio_amount}h_${determine_spks}_${num_spks}spks${suffix}/eval
   data_val_dir=$val_dir/data
   CONV_VAL_DIR=$val_dir/conv_val_dir
   sed -i "s $CONV_VAL_DIR/  g" $data_val_dir/rttm
   sed -i 's .conv  g' $data_val_dir/rttm
fi


if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
   val_dir=${simu_alimeeting_dataset_dir}/${audio_amount}h_${determine_spks}_${num_spks}spks${suffix}/test
   CONV_VAL_DIR=$val_dir/conv_val_dir
   CONV_VAL_LIST=$val_dir/conv_val_list
   find $CONV_VAL_DIR -iname "*.conv" | \
            xargs -n 1 basename -s .conv > $CONV_VAL_LIST
   WAV_SCP=${alimeeting_dataset_dir}/Test_Ali/Test_Ali_far/${target_wavs_name_dir}/wav.scp
   WAVS_VAL_DIR=$val_dir/wavs
   mkdir -p $WAVS_VAL_DIR
   python3 ts_vad2_simu/conv2wav.py --conversations-list-filename $CONV_VAL_LIST \
            --input-wav-scp $WAV_SCP --in-conv-dir $CONV_VAL_DIR \
            --out-wav-dir $WAVS_VAL_DIR \
            --sampling-frequency $sampling_rate

fi
if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
   echo "Generating general Kaldi-style data directory for validation"
   val_dir=${simu_alimeeting_dataset_dir}/${audio_amount}h_${determine_spks}_${num_spks}spks${suffix}/test
   CONV_VAL_DIR=$val_dir/conv_val_dir
   CONV_VAL_LIST=$val_dir/conv_val_list
   data_val_dir=$val_dir/data
   mkdir -p $data_val_dir

   for f in $(<$CONV_VAL_LIST); do
       awk -v sampling_frequency=$sampling_rate '{printf "SPEAKER %s 1 %.3f %.3f <NA> <NA> %s <NA> <NA>\n", FILENAME, $5/sampling_frequency, ($4-$3)/sampling_frequency, $1}' $CONV_VAL_DIR/$f.conv
   done > $data_val_dir/rttm



   awk -F"/" '{print $NF}' $data_val_dir/rttm | sed 's/.conv//g' | \
        awk '{printf "%s_%s_%06d_%06d %s %f %f\n", $7, $1, $3*100, ($3+$4)*100, $1, $3, $3+$4}' > $data_val_dir/segments
   awk -F"/" '{print $NF}' $data_val_dir/rttm | sed 's/.conv//g' | \
        awk '{printf "%s_%s_%06d_%06d %s\n", $7, $1, $3*100, ($3+$4)*100, $7}' > $data_val_dir/utt2spk
   utils/utt2spk_to_spk2utt.pl $data_val_dir/utt2spk > $data_val_dir/spk2utt
   WAVS_VAL_DIR=$val_dir/wavs
   ls $WAVS_VAL_DIR | awk -F'.' -v path=$WAVS_VAL_DIR '{print $1" "path"/"$1".wav"}' > $data_val_dir/wav.scp
   echo "----- Testset  KALDI DIR GENERATED it is stored at $data_val_dir-----"
fi


if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
   echo "remove .conv from rttm filename"
   val_dir=${simu_alimeeting_dataset_dir}/${audio_amount}h_${determine_spks}_${num_spks}spks${suffix}/test
   data_val_dir=$val_dir/data
   CONV_VAL_DIR=$val_dir/conv_val_dir
   sed -i "s $CONV_VAL_DIR/  g" $data_val_dir/rttm
   sed -i 's .conv  g' $data_val_dir/rttm
fi
