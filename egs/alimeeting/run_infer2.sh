#!/usr/bin/env bash
stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_speaker_diarization.sh
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
exp_dir=ts_vad/exp2
rs_len=4
segment_shift=1
label_rate=25
min_silence=0.32
min_speech=0.0
#infer_sets="Eval Test"
infer_sets="Eval"
rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
sctk_tool_path="./SCTK-2.4.12"
collar=0.25
results_path=$exp_dir/
#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7/best-valid-loss.pt
# cat logs/run_infer2_1.log          Eval for threshold 0.50: DER 8.97%, MS 4.24%, FA 3.57%, SC 1.16%
#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7/checkpoint-85500.pt
# cat logs/run_infer2_2.log          Eval for threshold 0.55: DER 9.21%, MS 5.06%, FA 2.85%, SC 1.30%

#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7/best-valid-loss.pt
# cat logs/run_infer2_3.log          Eval for threshold 0.55: DER 8.66%, MS 5.16%, FA 2.53%, SC 0.97%

#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7/epoch-30.pt
#                                    Eval for threshold 0.55: DER 8.63%, MS 5.09%, FA 2.52%, SC 1.02%
#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7_1/epoch-29.pt
# cat  logs/run_infer2_6.log         Eval for threshold 0.55: DER 8.86%, MS 5.20%, FA 2.46%, SC 1.20%
#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7_1/epoch-30.pt
# cat logs/run_infer2_7.log          Eval for threshold 0.55: DER 8.54%, MS 5.02%, FA 2.48%, SC 1.04%
#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7_1/best-valid-loss.pt
# cat logs/run_infer2_8.log  Eval for threshold 0.55: DER 8.84%, MS 5.19%, FA 2.46%, SC 1.20%
#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp6_3/epoch-13.pt
# cat logs/run_infer2_9.log
# Model DER:  0.12599087457029542
#Model ACC:  0.9571437840213997
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:24<00:00,  1.04it/s]
#Eval for threshold 0.20: DER 7.60%, MS 0.97%, FA 6.35%, SC 0.28%
#
#Eval for threshold 0.30: DER 5.71%, MS 1.47%, FA 3.94%, SC 0.31%
#
#Eval for threshold 0.35: DER 5.19%, MS 1.73%, FA 3.13%, SC 0.33%
#
#Eval for threshold 0.40: DER 4.81%, MS 2.01%, FA 2.45%, SC 0.35%
#
#Eval for threshold 0.45: DER 4.65%, MS 2.35%, FA 1.96%, SC 0.34%
#
#Eval for threshold 0.50: DER 4.64%, MS 2.75%, FA 1.55%, SC 0.34%
#
#Eval for threshold 0.55: DER 4.78%, MS 3.19%, FA 1.26%, SC 0.33%
#
#Eval for threshold 0.60: DER 5.12%, MS 3.78%, FA 1.04%, SC 0.30%
#
#Eval for threshold 0.70: DER 5.92%, MS 5.02%, FA 0.67%, SC 0.23%
#
#Eval for threshold 0.80: DER 7.74%, MS 7.15%, FA 0.46%, SC 0.13%

#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_with_musan/epoch-13.pt
# cat logs/run_infer2_10.log
# Model DER:  0.12473331171410519
#Model ACC:  0.9574973903600946
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.06it/s]
#Eval for threshold 0.20: DER 6.56%, MS 1.18%, FA 5.10%, SC 0.28%
#
#Eval for threshold 0.30: DER 5.03%, MS 1.64%, FA 3.06%, SC 0.33%
#
#Eval for threshold 0.35: DER 4.70%, MS 1.93%, FA 2.43%, SC 0.34%
#
#Eval for threshold 0.40: DER 4.56%, MS 2.22%, FA 2.00%, SC 0.34%
#
#Eval for threshold 0.45: DER 4.51%, MS 2.50%, FA 1.67%, SC 0.34%
#
#Eval for threshold 0.50: DER 4.51%, MS 2.80%, FA 1.36%, SC 0.35%
#
#Eval for threshold 0.55: DER 4.68%, MS 3.25%, FA 1.11%, SC 0.33%
#
#Eval for threshold 0.60: DER 4.94%, MS 3.73%, FA 0.91%, SC 0.30%
#
#Eval for threshold 0.70: DER 5.85%, MS 5.01%, FA 0.61%, SC 0.23%
#
#Eval for threshold 0.80: DER 7.71%, MS 7.12%, FA 0.45%, SC 0.14%
#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp/epoch-13.pt
#  cat logs/run_infer2_11.log
# Model DER:  0.13845499120702162
#Model ACC:  0.9521192470634031
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.06it/s]
#Eval for threshold 0.20: DER 7.51%, MS 1.29%, FA 5.72%, SC 0.49%
#
#Eval for threshold 0.30: DER 6.03%, MS 1.87%, FA 3.60%, SC 0.56%
#
#Eval for threshold 0.35: DER 5.76%, MS 2.17%, FA 3.02%, SC 0.58%
#
#Eval for threshold 0.40: DER 5.55%, MS 2.44%, FA 2.53%, SC 0.58%
#
#Eval for threshold 0.45: DER 5.37%, MS 2.76%, FA 2.04%, SC 0.57%
#
#Eval for threshold 0.50: DER 5.40%, MS 3.13%, FA 1.70%, SC 0.57%
#
#Eval for threshold 0.55: DER 5.54%, MS 3.58%, FA 1.43%, SC 0.54%
#
#Eval for threshold 0.60: DER 5.83%, MS 4.10%, FA 1.24%, SC 0.49%
#
#Eval for threshold 0.70: DER 6.58%, MS 5.32%, FA 0.86%, SC 0.40%
#
#Eval for threshold 0.80: DER 8.34%, MS 7.50%, FA 0.58%, SC 0.26%
#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_debug/epoch-13.pt
# cat logs/run_infer2_12.log
# Model DER:  0.13688159942508435
#Model ACC:  0.9526624309249481
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.06it/s]
#Eval for threshold 0.20: DER 7.24%, MS 1.31%, FA 5.56%, SC 0.37%
#
#Eval for threshold 0.30: DER 5.88%, MS 1.81%, FA 3.59%, SC 0.48%
#
#Eval for threshold 0.35: DER 5.60%, MS 2.10%, FA 2.98%, SC 0.53%
#
#Eval for threshold 0.40: DER 5.37%, MS 2.37%, FA 2.47%, SC 0.54%
#
#Eval for threshold 0.45: DER 5.24%, MS 2.66%, FA 2.03%, SC 0.55%
#
#Eval for threshold 0.50: DER 5.25%, MS 3.04%, FA 1.65%, SC 0.56%
#
#Eval for threshold 0.55: DER 5.35%, MS 3.44%, FA 1.38%, SC 0.53%
#
#Eval for threshold 0.60: DER 5.62%, MS 3.96%, FA 1.18%, SC 0.48%
#
#Eval for threshold 0.70: DER 6.24%, MS 5.06%, FA 0.81%, SC 0.36%
#
#Eval for threshold 0.80: DER 7.84%, MS 7.03%, FA 0.56%, SC 0.24%
#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad_phase1/exp/epoch-13.pt
#cat logs/run_infer2_13.log
# Model DER:  0.13654559694833976
#Model ACC:  0.9525115100009076
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.06it/s]
#Eval for threshold 0.20: DER 7.41%, MS 1.37%, FA 5.56%, SC 0.48%
#
#Eval for threshold 0.30: DER 5.88%, MS 1.88%, FA 3.48%, SC 0.52%
#
#Eval for threshold 0.35: DER 5.59%, MS 2.11%, FA 2.91%, SC 0.57%
#
#Eval for threshold 0.40: DER 5.36%, MS 2.35%, FA 2.44%, SC 0.57%
#
#Eval for threshold 0.45: DER 5.23%, MS 2.64%, FA 2.04%, SC 0.56%
#
#Eval for threshold 0.50: DER 5.18%, MS 2.95%, FA 1.66%, SC 0.57%
#
#Eval for threshold 0.55: DER 5.28%, MS 3.36%, FA 1.36%, SC 0.55%
#
#Eval for threshold 0.60: DER 5.51%, MS 3.83%, FA 1.15%, SC 0.52%
#
#Eval for threshold 0.70: DER 6.20%, MS 4.94%, FA 0.84%, SC 0.42%
#
#Eval for threshold 0.80: DER 8.01%, MS 7.14%, FA 0.60%, SC 0.26%

#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_2gpu_accelerate2/epoch-20.pt
# no freeze speech encoder,(no grad_clip max_norm=2.0,)
# cat logs/run_infer2_14.log
## Model DER:  0.13233529551826373
#Model ACC:  0.95436932630956
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.04it/s]
#Eval for threshold 0.20: DER 7.42%, MS 1.17%, FA 5.86%, SC 0.39%
#
#Eval for threshold 0.30: DER 5.74%, MS 1.63%, FA 3.68%, SC 0.43%
#
#Eval for threshold 0.35: DER 5.32%, MS 1.88%, FA 3.01%, SC 0.43%
#
#Eval for threshold 0.40: DER 5.07%, MS 2.14%, FA 2.49%, SC 0.44%
#
#Eval for threshold 0.45: DER 4.96%, MS 2.46%, FA 2.05%, SC 0.45%
#
#Eval for threshold 0.50: DER 5.01%, MS 2.84%, FA 1.70%, SC 0.46%
#
#Eval for threshold 0.55: DER 5.16%, MS 3.30%, FA 1.43%, SC 0.43%
#
#Eval for threshold 0.60: DER 5.33%, MS 3.74%, FA 1.20%, SC 0.38%
#
#Eval for threshold 0.70: DER 6.04%, MS 4.92%, FA 0.81%, SC 0.30%
#
#Eval for threshold 0.80: DER 7.73%, MS 6.96%, FA 0.55%, SC 0.22%

#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_2gpu_accelerate2/epoch-13.pt
# no freeze speech encoder,(no grad_clip max_norm=2.0,)
# cat logs/run_infer2_15.log
# Model DER:  0.13234007247069315
#Model ACC:  0.9544413723949803
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.04it/s]
#Eval for threshold 0.20: DER 7.34%, MS 1.18%, FA 5.79%, SC 0.37%
#
#Eval for threshold 0.30: DER 5.70%, MS 1.65%, FA 3.62%, SC 0.42%
#
#Eval for threshold 0.35: DER 5.31%, MS 1.87%, FA 3.00%, SC 0.43%
#
#Eval for threshold 0.40: DER 5.07%, MS 2.17%, FA 2.46%, SC 0.43%
#
#Eval for threshold 0.45: DER 4.97%, MS 2.48%, FA 2.05%, SC 0.44%
#
#Eval for threshold 0.50: DER 4.97%, MS 2.85%, FA 1.68%, SC 0.45%
#
#Eval for threshold 0.55: DER 5.12%, MS 3.31%, FA 1.39%, SC 0.42%
#
#Eval for threshold 0.60: DER 5.27%, MS 3.74%, FA 1.14%, SC 0.39%
#
#Eval for threshold 0.70: DER 6.05%, MS 4.93%, FA 0.80%, SC 0.31%
#
#Eval for threshold 0.80: DER 7.66%, MS 6.91%, FA 0.53%, SC 0.22%
#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_2gpu_accelerate2/epoch-7.pt
# no freeze speech encoder,(no grad_clip max_norm=2.0,)
# cat logs/run_infer2_16.log
# Model DER:  0.1310577452039561
#Model ACC:  0.954490837923867
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:24<00:00,  1.03it/s]
#Eval for threshold 0.20: DER 8.69%, MS 0.88%, FA 7.45%, SC 0.35%
#
#Eval for threshold 0.30: DER 6.18%, MS 1.39%, FA 4.34%, SC 0.45%
#
#Eval for threshold 0.35: DER 5.46%, MS 1.61%, FA 3.37%, SC 0.48%
#
#Eval for threshold 0.40: DER 4.99%, MS 1.91%, FA 2.57%, SC 0.50%
#
#Eval for threshold 0.45: DER 4.87%, MS 2.30%, FA 2.04%, SC 0.53%
#
#Eval for threshold 0.50: DER 4.88%, MS 2.74%, FA 1.62%, SC 0.52%
#
#Eval for threshold 0.55: DER 5.01%, MS 3.25%, FA 1.28%, SC 0.48%
#
#Eval for threshold 0.60: DER 5.32%, MS 3.84%, FA 1.06%, SC 0.43%
#
#Eval for threshold 0.70: DER 6.21%, MS 5.22%, FA 0.66%, SC 0.33%
#
#Eval for threshold 0.80: DER 8.38%, MS 7.76%, FA 0.44%, SC 0.18%

#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_2gpu_accelerate3/epoch-7.pt
# freeze speech encoder before num_update equal to 4000,(no grad_clip max_norm=2.0,)
# cat logs/run_infer2_17.log
# Model DER:  0.14129172945757196
#Model ACC:  0.950446912409946
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:24<00:00,  1.04it/s]
#Eval for threshold 0.20: DER 9.35%, MS 1.00%, FA 7.84%, SC 0.51%
#
#Eval for threshold 0.30: DER 6.89%, MS 1.49%, FA 4.80%, SC 0.60%
#
#Eval for threshold 0.35: DER 6.11%, MS 1.76%, FA 3.65%, SC 0.70%
#
#Eval for threshold 0.40: DER 5.74%, MS 2.13%, FA 2.90%, SC 0.71%
#
#Eval for threshold 0.45: DER 5.53%, MS 2.56%, FA 2.25%, SC 0.72%
#
#Eval for threshold 0.50: DER 5.50%, MS 3.04%, FA 1.75%, SC 0.71%
#
#Eval for threshold 0.55: DER 5.65%, MS 3.70%, FA 1.35%, SC 0.60%
#
#Eval for threshold 0.60: DER 6.07%, MS 4.52%, FA 1.05%, SC 0.51%
#
#Eval for threshold 0.70: DER 7.45%, MS 6.41%, FA 0.69%, SC 0.35%
#
#Eval for threshold 0.80: DER 9.93%, MS 9.28%, FA 0.47%, SC 0.19%

#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_2gpu_accelerate3/epoch-13.pt
# freeze speech encoder before num_update equal to 4000,(no grad_clip max_norm=2.0,)
# cat logs/run_infer2_18.log
# Model DER:  0.13495051939871022
#Model ACC:  0.9534829243536964
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.04it/s]
#Eval for threshold 0.20: DER 7.18%, MS 1.23%, FA 5.57%, SC 0.38%
#
#Eval for threshold 0.30: DER 5.70%, MS 1.78%, FA 3.51%, SC 0.41%
#
#Eval for threshold 0.35: DER 5.36%, MS 2.01%, FA 2.89%, SC 0.45%
#
#Eval for threshold 0.40: DER 5.19%, MS 2.30%, FA 2.41%, SC 0.47%
#
#Eval for threshold 0.45: DER 5.13%, MS 2.64%, FA 2.02%, SC 0.47%
#
#Eval for threshold 0.50: DER 5.23%, MS 3.07%, FA 1.68%, SC 0.48%
#
#Eval for threshold 0.55: DER 5.38%, MS 3.52%, FA 1.40%, SC 0.46%
#
#Eval for threshold 0.60: DER 5.61%, MS 4.02%, FA 1.16%, SC 0.43%
#
#Eval for threshold 0.70: DER 6.67%, MS 5.53%, FA 0.83%, SC 0.30%
#
#Eval for threshold 0.80: DER 8.59%, MS 7.80%, FA 0.56%, SC 0.23%
#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_2gpu_accelerate3/epoch-20.pt
# freeze speech encoder before num_update equal to 4000 ,(no grad_clip max_norm=2.0,)
#cat logs/run_infer2_19.log
# Model DER:  0.13538745731653556
#Model ACC:  0.9531456417394888
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.04it/s]
#Eval for threshold 0.20: DER 7.34%, MS 1.25%, FA 5.69%, SC 0.41%
#
#Eval for threshold 0.30: DER 5.79%, MS 1.78%, FA 3.55%, SC 0.46%
#
#Eval for threshold 0.35: DER 5.47%, MS 2.04%, FA 2.95%, SC 0.48%
#
#Eval for threshold 0.40: DER 5.26%, MS 2.35%, FA 2.42%, SC 0.48%
#
#Eval for threshold 0.45: DER 5.20%, MS 2.69%, FA 2.00%, SC 0.50%
#
#Eval for threshold 0.50: DER 5.29%, MS 3.09%, FA 1.69%, SC 0.50%
#
#Eval for threshold 0.55: DER 5.41%, MS 3.53%, FA 1.40%, SC 0.49%
#
#Eval for threshold 0.60: DER 5.71%, MS 4.09%, FA 1.17%, SC 0.45%
#
#Eval for threshold 0.70: DER 6.67%, MS 5.51%, FA 0.84%, SC 0.32%
#
#Eval for threshold 0.80: DER 8.62%, MS 7.81%, FA 0.57%, SC 0.24%

#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/ts_vad_ddp_phase1_two_gpus/epoch-20.pt
# add grad clip max_norm=2.0, freeze speech encoder before num_update equal to 4000
# cat logs/run_infer2_20.log
# Model DER:  0.13460497844686437
#Model ACC:  0.9535534805787533
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:28<00:00,  1.14s/it]
#Eval for threshold 0.20: DER 7.57%, MS 1.22%, FA 5.98%, SC 0.37%
#
#Eval for threshold 0.30: DER 5.91%, MS 1.67%, FA 3.82%, SC 0.42%
#
#Eval for threshold 0.35: DER 5.50%, MS 1.89%, FA 3.17%, SC 0.45%
#
#Eval for threshold 0.40: DER 5.17%, MS 2.14%, FA 2.58%, SC 0.45%
#
#Eval for threshold 0.45: DER 4.96%, MS 2.43%, FA 2.07%, SC 0.46%
#
#Eval for threshold 0.50: DER 4.96%, MS 2.77%, FA 1.71%, SC 0.48%
#
#Eval for threshold 0.55: DER 5.10%, MS 3.20%, FA 1.43%, SC 0.47%
#
#Eval for threshold 0.60: DER 5.38%, MS 3.70%, FA 1.23%, SC 0.45%
#
#Eval for threshold 0.70: DER 6.17%, MS 4.97%, FA 0.83%, SC 0.37%
#
#Eval for threshold 0.80: DER 7.96%, MS 7.16%, FA 0.56%, SC 0.25%
#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/ts_vad_ddp_phase1_two_gpus/epoch-13.pt
# add grad clip max_norm=2.0, freeze speech encoder before num_update equal to 4000
# cat logs/run_infer2_21.log
# Model DER:  0.13458737070855356
#Model ACC:  0.9537309346698694
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 7.58%, MS 1.24%, FA 5.99%, SC 0.35%
#
#Eval for threshold 0.30: DER 5.92%, MS 1.67%, FA 3.84%, SC 0.42%
#
#Eval for threshold 0.35: DER 5.45%, MS 1.89%, FA 3.14%, SC 0.42%
#
#Eval for threshold 0.40: DER 5.19%, MS 2.14%, FA 2.64%, SC 0.41%
#
#Eval for threshold 0.45: DER 4.97%, MS 2.44%, FA 2.10%, SC 0.43%
#
#Eval for threshold 0.50: DER 4.93%, MS 2.77%, FA 1.71%, SC 0.45%
#
#Eval for threshold 0.55: DER 5.11%, MS 3.23%, FA 1.44%, SC 0.43%
#
#Eval for threshold 0.60: DER 5.29%, MS 3.67%, FA 1.21%, SC 0.42%
#
#Eval for threshold 0.70: DER 6.14%, MS 4.97%, FA 0.82%, SC 0.35%
#
#Eval for threshold 0.80: DER 7.99%, MS 7.22%, FA 0.53%, SC 0.24%

#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/ts_vad_ddp_phase1_two_gpus/epoch-7.pt
# add grad clip max_norm=2.0, freeze speech encoder before num_update equal to 4000
# cat logs/run_infer2_22.log
# Model DER:  0.13305524517756867
#Model ACC:  0.9536688589027751
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.06it/s]
#Eval for threshold 0.20: DER 9.20%, MS 0.87%, FA 7.94%, SC 0.39%
#
#Eval for threshold 0.30: DER 6.70%, MS 1.34%, FA 4.91%, SC 0.46%
#
#Eval for threshold 0.35: DER 6.00%, MS 1.57%, FA 3.93%, SC 0.50%
#
#Eval for threshold 0.40: DER 5.50%, MS 1.85%, FA 3.09%, SC 0.56%
#
#Eval for threshold 0.45: DER 5.19%, MS 2.17%, FA 2.41%, SC 0.61%
#
#Eval for threshold 0.50: DER 5.11%, MS 2.57%, FA 1.92%, SC 0.62%
#
#Eval for threshold 0.55: DER 5.21%, MS 3.10%, FA 1.53%, SC 0.57%
#
#Eval for threshold 0.60: DER 5.49%, MS 3.75%, FA 1.25%, SC 0.49%
#
#Eval for threshold 0.70: DER 6.45%, MS 5.32%, FA 0.76%, SC 0.37%
#
#Eval for threshold 0.80: DER 8.56%, MS 7.86%, FA 0.48%, SC 0.23%

#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/ts_vad_ddp_phase1_two_gpus/best-valid-loss.pt
# add grad clip max_norm=2.0, freeze speech encoder before num_update equal to 4000
# cat logs/run_infer2_23.log
# Model DER:  0.12844753852368793
#Model ACC:  0.9553976369730277
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 7.80%, MS 1.00%, FA 6.45%, SC 0.36%
#
#Eval for threshold 0.30: DER 5.91%, MS 1.53%, FA 3.97%, SC 0.42%
#
#Eval for threshold 0.35: DER 5.40%, MS 1.81%, FA 3.11%, SC 0.48%
#
#Eval for threshold 0.40: DER 5.14%, MS 2.11%, FA 2.53%, SC 0.49%
#
#Eval for threshold 0.45: DER 4.99%, MS 2.41%, FA 2.04%, SC 0.53%
#
#Eval for threshold 0.50: DER 4.94%, MS 2.77%, FA 1.66%, SC 0.51%
#
#Eval for threshold 0.55: DER 5.00%, MS 3.17%, FA 1.35%, SC 0.48%
#
#Eval for threshold 0.60: DER 5.21%, MS 3.70%, FA 1.08%, SC 0.43%
#
#Eval for threshold 0.70: DER 6.18%, MS 5.16%, FA 0.71%, SC 0.31%
#
#Eval for threshold 0.80: DER 8.12%, MS 7.45%, FA 0.46%, SC 0.20%

#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/ts_vad_ddp_phase1_two_gpus/best-valid-der.pt
# add grad clip max_norm=2.0, freeze speech encoder before num_update equal to 4000
# cat logs/run_infer2_24.log
# Model DER:  0.12871112659402167
#Model ACC:  0.955327438509156
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 7.81%, MS 1.01%, FA 6.45%, SC 0.35%
#
#Eval for threshold 0.30: DER 5.92%, MS 1.53%, FA 3.96%, SC 0.43%
#
#Eval for threshold 0.35: DER 5.44%, MS 1.81%, FA 3.15%, SC 0.48%
#
#Eval for threshold 0.40: DER 5.15%, MS 2.10%, FA 2.54%, SC 0.51%
#
#Eval for threshold 0.45: DER 4.99%, MS 2.42%, FA 2.05%, SC 0.53%
#
#Eval for threshold 0.50: DER 4.96%, MS 2.78%, FA 1.67%, SC 0.51%
#
#Eval for threshold 0.55: DER 5.01%, MS 3.18%, FA 1.34%, SC 0.49%
#
#Eval for threshold 0.60: DER 5.24%, MS 3.73%, FA 1.07%, SC 0.43%
#
#Eval for threshold 0.70: DER 6.13%, MS 5.11%, FA 0.72%, SC 0.31%
#
#Eval for threshold 0.80: DER 8.18%, MS 7.52%, FA 0.46%, SC 0.20%

#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/ts_vad_ddp_phase1_two_gpus_unfreeze/best-valid-der.pt
# add grad clip max_norm=2.0, no freeze speech encoder
# cat logs/run_infer2_25.log
# Model DER:  0.13490954777494413
#Model ACC:  0.9531129266200791
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 7.98%, MS 1.04%, FA 6.55%, SC 0.38%
#
#Eval for threshold 0.30: DER 6.04%, MS 1.52%, FA 4.03%, SC 0.49%
#
#Eval for threshold 0.35: DER 5.60%, MS 1.82%, FA 3.27%, SC 0.51%
#
#Eval for threshold 0.40: DER 5.35%, MS 2.10%, FA 2.72%, SC 0.54%
#
#Eval for threshold 0.45: DER 5.18%, MS 2.44%, FA 2.18%, SC 0.57%
#
#Eval for threshold 0.50: DER 5.07%, MS 2.80%, FA 1.74%, SC 0.53%
#
#Eval for threshold 0.55: DER 5.17%, MS 3.25%, FA 1.42%, SC 0.49%
#
#Eval for threshold 0.60: DER 5.46%, MS 3.83%, FA 1.19%, SC 0.44%
#
#Eval for threshold 0.70: DER 6.46%, MS 5.26%, FA 0.89%, SC 0.31%
#
#Eval for threshold 0.80: DER 8.47%, MS 7.68%, FA 0.57%, SC 0.22%

#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/ts_vad_ddp_phase1_two_gpus_unfreeze/best-valid-loss.pt
# add grad clip max_norm=2.0, no freeze speech encoder
# cat logs/run_infer2_26.log
# Model DER:  0.1348472032510829
#Model ACC:  0.9531007787737867
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 8.00%, MS 1.03%, FA 6.59%, SC 0.39%
#
#Eval for threshold 0.30: DER 6.06%, MS 1.51%, FA 4.05%, SC 0.50%
#
#Eval for threshold 0.35: DER 5.61%, MS 1.81%, FA 3.28%, SC 0.52%
#
#Eval for threshold 0.40: DER 5.33%, MS 2.08%, FA 2.71%, SC 0.55%
#
#Eval for threshold 0.45: DER 5.16%, MS 2.41%, FA 2.19%, SC 0.56%
#
#Eval for threshold 0.50: DER 5.10%, MS 2.81%, FA 1.76%, SC 0.53%
#
#Eval for threshold 0.55: DER 5.18%, MS 3.26%, FA 1.43%, SC 0.49%
#
#Eval for threshold 0.60: DER 5.44%, MS 3.82%, FA 1.19%, SC 0.44%
#
#Eval for threshold 0.70: DER 6.47%, MS 5.26%, FA 0.89%, SC 0.32%
#
#Eval for threshold 0.80: DER 8.46%, MS 7.66%, FA 0.58%, SC 0.22%
#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/ts_vad_ddp_phase1_two_gpus_unfreeze/epoch-20.pt
# add grad clip max_norm=2.0, no freeze speech encoder
# cat logs/run_infer2_27.log
# Model DER:  0.14040790419224092
#Model ACC:  0.951070467982344
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 7.80%, MS 1.19%, FA 6.14%, SC 0.48%
#
#Eval for threshold 0.30: DER 6.16%, MS 1.68%, FA 3.92%, SC 0.55%
#
#Eval for threshold 0.35: DER 5.77%, MS 1.97%, FA 3.19%, SC 0.61%
#
#Eval for threshold 0.40: DER 5.52%, MS 2.27%, FA 2.61%, SC 0.64%
#
#Eval for threshold 0.45: DER 5.36%, MS 2.60%, FA 2.11%, SC 0.66%
#
#Eval for threshold 0.50: DER 5.43%, MS 3.05%, FA 1.76%, SC 0.62%
#
#Eval for threshold 0.55: DER 5.57%, MS 3.48%, FA 1.47%, SC 0.61%
#
#Eval for threshold 0.60: DER 5.79%, MS 3.98%, FA 1.23%, SC 0.58%
#
#Eval for threshold 0.70: DER 6.69%, MS 5.37%, FA 0.89%, SC 0.43%
#
#Eval for threshold 0.80: DER 8.55%, MS 7.66%, FA 0.61%, SC 0.28%
#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/ts_vad_ddp_phase1_two_gpus_unfreeze/epoch-13.pt
# add grad clip max_norm=2.0, no freeze speech encoder
# cat logs/run_infer2_28.log
# Model DER:  0.13933993925609003
#Model ACC:  0.9515641824311105
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.06it/s]
#Eval for threshold 0.20: DER 7.68%, MS 1.20%, FA 6.03%, SC 0.45%
#
#Eval for threshold 0.30: DER 6.00%, MS 1.66%, FA 3.80%, SC 0.54%
#
#Eval for threshold 0.35: DER 5.64%, MS 1.92%, FA 3.11%, SC 0.61%
#
#Eval for threshold 0.40: DER 5.39%, MS 2.24%, FA 2.51%, SC 0.64%
#
#Eval for threshold 0.45: DER 5.28%, MS 2.61%, FA 2.04%, SC 0.63%
#
#Eval for threshold 0.50: DER 5.30%, MS 3.02%, FA 1.71%, SC 0.58%
#
#Eval for threshold 0.55: DER 5.46%, MS 3.48%, FA 1.44%, SC 0.54%
#
#Eval for threshold 0.60: DER 5.68%, MS 3.94%, FA 1.22%, SC 0.52%
#
#Eval for threshold 0.70: DER 6.52%, MS 5.26%, FA 0.85%, SC 0.42%
#
#Eval for threshold 0.80: DER 8.45%, MS 7.55%, FA 0.60%, SC 0.30%
#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/ts_vad_ddp_phase1_two_gpus_unfreeze/epoch-7.pt
# add grad clip max_norm=2.0, no freeze speech encoder
# cat logs/run_infer2_29.log
# Model DER:  0.1420959208193695
#Model ACC:  0.9499303248849541
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 9.50%, MS 0.97%, FA 8.00%, SC 0.53%
#
#Eval for threshold 0.30: DER 7.09%, MS 1.48%, FA 4.93%, SC 0.68%
#
#Eval for threshold 0.35: DER 6.41%, MS 1.76%, FA 3.84%, SC 0.82%
#
#Eval for threshold 0.40: DER 6.00%, MS 2.10%, FA 3.00%, SC 0.89%
#
#Eval for threshold 0.45: DER 5.80%, MS 2.53%, FA 2.34%, SC 0.93%
#
#Eval for threshold 0.50: DER 5.83%, MS 3.07%, FA 1.87%, SC 0.90%
#
#Eval for threshold 0.55: DER 6.01%, MS 3.72%, FA 1.46%, SC 0.83%
#
#Eval for threshold 0.60: DER 6.34%, MS 4.42%, FA 1.17%, SC 0.75%
#
#Eval for threshold 0.70: DER 7.54%, MS 6.24%, FA 0.75%, SC 0.55%
#
#Eval for threshold 0.80: DER 9.99%, MS 9.17%, FA 0.49%, SC 0.33%

#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_2gpu_accelerate2/best-valid-der.pt
# no freeze speech encoder and (no grad_clip max_norm=2.0,)
# cat logs/run_infer2_30.log
# Model DER:  0.13243536345296233
#Model ACC:  0.9544051843753449
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.06it/s]
#Eval for threshold 0.20: DER 7.34%, MS 1.19%, FA 5.77%, SC 0.37%
#
#Eval for threshold 0.30: DER 5.67%, MS 1.64%, FA 3.61%, SC 0.42%
#
#Eval for threshold 0.35: DER 5.32%, MS 1.86%, FA 3.03%, SC 0.44%
#
#Eval for threshold 0.40: DER 5.10%, MS 2.15%, FA 2.50%, SC 0.46%
#
#Eval for threshold 0.45: DER 4.93%, MS 2.45%, FA 2.02%, SC 0.46%
#
#Eval for threshold 0.50: DER 4.99%, MS 2.87%, FA 1.68%, SC 0.45%
#
#Eval for threshold 0.55: DER 5.08%, MS 3.28%, FA 1.37%, SC 0.43%
#
#Eval for threshold 0.60: DER 5.27%, MS 3.73%, FA 1.14%, SC 0.39%
#
#Eval for threshold 0.70: DER 6.01%, MS 4.90%, FA 0.79%, SC 0.32%
#
#Eval for threshold 0.80: DER 7.67%, MS 6.92%, FA 0.53%, SC 0.22%

model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad1/exp_2gpu_accelerate2/best-valid-loss.pt
# no freeze speech encoder and (no grad_clip max_norm=2.0,)
# cat logs/run_infer2_31.log
# Model DER:  0.13257381926036974
#Model ACC:  0.9543903547466307
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 7.31%, MS 1.18%, FA 5.77%, SC 0.36%
#
#Eval for threshold 0.30: DER 5.70%, MS 1.65%, FA 3.63%, SC 0.42%
#
#Eval for threshold 0.35: DER 5.33%, MS 1.88%, FA 3.01%, SC 0.44%
#
#Eval for threshold 0.40: DER 5.12%, MS 2.16%, FA 2.53%, SC 0.44%
#
#Eval for threshold 0.45: DER 4.93%, MS 2.44%, FA 2.04%, SC 0.45%
#
#Eval for threshold 0.50: DER 4.98%, MS 2.84%, FA 1.69%, SC 0.45%
#
#Eval for threshold 0.55: DER 5.12%, MS 3.28%, FA 1.40%, SC 0.43%
#
#Eval for threshold 0.60: DER 5.27%, MS 3.75%, FA 1.15%, SC 0.38%
#
#Eval for threshold 0.70: DER 6.03%, MS 4.91%, FA 0.80%, SC 0.31%
#
#Eval for threshold 0.80: DER 7.68%, MS 6.95%, FA 0.52%, SC 0.21%


for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path

 python3 ts_vad/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name
done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
exp_dir=ts_vad/exp2
rs_len=4
segment_shift=1
label_rate=25
min_silence=0.32
min_speech=0.0
#infer_sets="Eval Test"
infer_sets="Eval"
rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
sctk_tool_path="./SCTK-2.4.12"
collar=0.25
results_path=$exp_dir/
#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7/best-valid-loss.pt
# cat logs/run_infer2_1.log          Eval for threshold 0.50: DER 8.97%, MS 4.24%, FA 3.57%, SC 1.16%
#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7/checkpoint-85500.pt
# cat logs/run_infer2_2.log          Eval for threshold 0.55: DER 9.21%, MS 5.06%, FA 2.85%, SC 1.30%

#model_file=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7/best-valid-loss.pt
# cat logs/run_infer2_3.log          Eval for threshold 0.55: DER 8.66%, MS 5.16%, FA 2.53%, SC 0.97%
use_averaged_model=true
#avg=3
#epoch=30
#exp_dir_=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7/
#                                    Eval for threshold 0.55: DER 8.71%, MS 5.18%, FA 2.50%, SC 1.04%

avg=3
epoch=30
exp_dir_=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7_1/
#cat cat logs/run_infer2_8.log       Eval for threshold 0.55: DER 8.84%, MS 5.19%, FA 2.46%, SC 1.20%

for name in $infer_sets;do
    results_path=$exp_dir/${name}_avg
    mkdir -p $results_path

 python3 ts_vad/infer.py \
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --use-averaged-model $use_averaged_model\
    --avg $avg\
    --epoch $epoch\
    --exp-dir $exp_dir_
done
fi