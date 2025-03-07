#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

import torch

#model_name='M' # ~b3-b4 size
#train_type='ft_mix'
#dataset='vb2+vox2+cnc'

model_name="b2"
train_type='ft_lm'
dataset="vox2"
#model = torch.hub.load('/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/redimnet/redimnet','ReDimNet',
hubcofig_path_dir="/mntnfs/lee_data1/maduo/codebase/speaker_diarization/egs/magicdata-ramc/ts_vad2/redimnet/"
model=torch.hub.load(hubcofig_path_dir, 'ReDimNet',
                       model_name=model_name,
                       train_type=train_type,
                       dataset=dataset,source="local")

if __name__ == "__main__":
    input= torch.randn(1,96000)
    model.eval()
    y = model(input)
    print(f"y shape: {y.shape}")
    y_frame = model.get_frame_level_feat(input)
    print(f"y_frame shape: {y_frame.shape}") # 6s audio, output:torch.Size([1, 72, 401])
