#!/usr/bin/env bash

stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_speaker_diarization.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then

 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp
  python3 ts_vad1/train.py\
    --world-size 1 \
    --num-epochs 20\
    --start-epoch 1\
    --use-fp16 1\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
 echo "add musan noise"
 musan_path=/mntcephfs/lee_dataset/asr/musan
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_with_musan
  python3 ts_vad1/train.py\
    --world-size 1 \
    --num-epochs 20\
    --start-epoch 1\
    --use-fp16 1\
    --exp-dir $exp_dir\
    --musan-path $musan_path
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then

 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_2gpu
 #CUDA_LAUNCH_BLOCKING=1  python3 ts_vad1/train_multi_gpus.py\
 TORCH_DISTRIBUTED_DEBUG=DETAIL  python3 ts_vad1/train_multi_gpus.py\
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --use-fp16 1\
    --master-port 12350\
    --exp-dir $exp_dir
fi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_debug
  python3 ts_vad1/train.py\
    --world-size 1 \
    --num-epochs 20\
    --start-epoch 1\
    --use-fp16 1\
    --exp-dir $exp_dir
fi
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then

 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_2gpu_3
 #CUDA_LAUNCH_BLOCKING=1  python3 ts_vad1/train_multi_gpus.py\
 #TORCH_DISTRIBUTED_DEBUG=DETAIL  python3 ts_vad1/train_multi_gpus3.py\
  python3 ts_vad1/train_multi_gpus4.py\
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --use-fp16 1\
    --master-port 12355\
    --exp-dir $exp_dir
fi
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then

 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_2gpu_4
 #CUDA_LAUNCH_BLOCKING=1  python3 ts_vad1/train_multi_gpus.py\
 #TORCH_DISTRIBUTED_DEBUG=DETAIL  python3 ts_vad1/train_multi_gpus3.py\

 TORCH_DISTRIBUTED_DEBUG=DETAIL python3 ts_vad1/train_multi_gpus4.py\
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --use-fp16 1\
    --master-port 12345\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then

 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_2gpu_5
 #CUDA_LAUNCH_BLOCKING=1  python3 ts_vad1/train_multi_gpus.py\
 #TORCH_DISTRIBUTED_DEBUG=DETAIL  python3 ts_vad1/train_multi_gpus3.py\
  python3 ts_vad1/train_multi_gpus5.py\
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --use-fp16 1\
    --master-port 12347\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then

 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_2gpu_4_1
 #CUDA_LAUNCH_BLOCKING=1  python3 ts_vad1/train_multi_gpus.py\
 #TORCH_DISTRIBUTED_DEBUG=DETAIL  python3 ts_vad1/train_multi_gpus3.py\

 TORCH_DISTRIBUTED_DEBUG=DETAIL python3 ts_vad1/train_multi_gpus4_1.py\
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --use-fp16 1\
    --master-port 12349\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_2gpu_fabric
   TORCH_DISTRIBUTED_DEBUG=DETAIL python3 ts_vad1/train_fabric.py\
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --use-fp16 1\
    --master-port 12351\
    --exp-dir $exp_dir
fi
# srun
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_2gpu_accelerate
   CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12351 ts_vad1/train_accelerate.py\
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --use-fp16 1\
    --master-port 12351\
    --exp-dir $exp_dir
fi


# sbatch
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_2gpu_accelerate2
   CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12553 ts_vad1/train_accelerate2.py\
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --use-fp16 1\
    --master-port 12553\
    --exp-dir $exp_dir
fi
# compared with stage=11, stage12 freeze speech encoder before num_update=4000
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_2gpu_accelerate3_debug
   CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12563 ts_vad1/train_accelerate3.py\
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --use-fp16 1\
    --master-port 12563\
    --exp-dir $exp_dir
fi
# compared with stage=12, stage13 add clip grad_norm
if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/ts_vad_ddp_phase1_two_gpus
    # refer:https://huggingface.co/docs/accelerate/basic_tutorials/launch
    #accelerate launch --main_process_port 12573 --multi_gpu --mixed_precision=fp16 --num_processes=2 ts_vad_ddp_phase1/train_accelerate_ddp.py\
   CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12563 ts_vad_ddp_phase1/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --master-port 12563\
    --exp-dir $exp_dir
fi

# compared with stage=13, stage14 don't freeze speech encoder
if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/ts_vad_ddp_phase1_two_gpus_unfreeze
    # refer:https://huggingface.co/docs/accelerate/basic_tutorials/launch
    #accelerate launch --main_process_port 12573 --multi_gpu --mixed_precision=fp16 --num_processes=2 ts_vad_ddp_phase1/train_accelerate_ddp.py\
   CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad_ddp_phase1/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --freeze-updates 0\
    --master-port 12673\
    --exp-dir $exp_dir
fi

