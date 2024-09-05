#!/usr/bin/env bash


stage=0
stop_stage=1000
audio_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/audio_dir/
textgrid_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/textgrid_dir
dest_dir=""
split="Eval"
. utils/parse_options.sh
. path_for_speaker_diarization.sh


if [ $stage -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Process textgrid to obtain rttm label"
    work_dir=$dest_dir/tmp1
    mkdir -p $work_dir
    find -L $audio_dir -name "*.wav" > $work_dir/wavlist
    sort  $work_dir/wavlist > $work_dir/tmp
    cp $work_dir/tmp $work_dir/wavlist
    awk -F '/' '{print $NF}' $work_dir/wavlist | awk -F '.' '{print $1}' > $work_dir/uttid

    find -L $textgrid_dir -iname "*.TextGrid" >  $work_dir/textgrid.flist
    sort  $work_dir/textgrid.flist  > $work_dir/tmp
    cp $work_dir/tmp $work_dir/textgrid.flist
    paste $work_dir/uttid $work_dir/textgrid.flist > $work_dir/uttid_textgrid.flist
    while read line;do
    #for line in $(cat $work_dir/uttid_textgrid.flist) do
        text_grid=`echo $line | awk '{print $1}'`
        text_grid_path=`echo $line | awk '{print $2}'`
    echo "text_grid: $text_grid"
    echo "text_grid_path: ${text_grid_path}"
        python3 ts_vad_ddp_phase1/make_textgrid_rttm.py\
          --input_textgrid_file $text_grid_path \
          --uttid $text_grid \
          --output_rttm_file $work_dir/${text_grid}.rttm
    done < $work_dir/uttid_textgrid.flist
    cat $work_dir/*.rttm > $work_dir/all.rttm1
    name=`echo "$split" | tr "[:upper:]" "[:lower:]"`
    mv $work_dir/all.rttm1  $dest_dir/alimeeting_${name}.rttm

    head $work_dir/*
    echo "remove unused files"
    rm -rf $work_dir
fi


