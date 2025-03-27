# speaker_diarization

# install
step0: install kaldi, you can refer https://github.com/kaldi-asr/kaldi/blob/master/tools/INSTALL
it is used to prepared data.

step1: install python package
```
conda create -n speaker_diarization python=3.9 -y
conda activate speaker_diarization
# for data prepared
pip install git+https://github.com/lhotse-speech/lhotse
# for network
pip install torch==2.1.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e .

install flac command
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install libflac
```
