# speaker_diarization

# install
step0: install kaldi, you can refer https://github.com/kaldi-asr/kaldi/blob/master/tools/INSTALL
it is used to prepared data.

step1: install python package
```
conda create -n speaker_diarization python=3.9 -y
conda activate speaker_diarization
## for hltsz cluster
## python=3.9 pytorch=2.1.1
conda install pytorch=2.1.1  torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia -c https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/linux-64/ -y
or
(recommend in sribd cluster)
pip --timeout=1000 install torch==2.1.2 torchaudio  --force-reinstall  --no-cache-dir --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```
