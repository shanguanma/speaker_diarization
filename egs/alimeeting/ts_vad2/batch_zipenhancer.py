import os
import soundfile as sf
import io
import torch
import numpy as np
import time
import logging
from modelscope.utils.config import Config
from modelscope.models.audio.ans.zipenhancer import ZipEnhancer, mag_pha_istft, mag_pha_stft, AttrDict
# from models.zipenhancer import ZipEnhancer, mag_pha_istft, mag_pha_stft, AttrDict
from modelscope.utils.file_utils import get_modelscope_cache_dir
from modelscope.fileio import File
from modelscope.utils.audio.audio_utils import audio_norm
logging.getLogger().setLevel(logging.INFO)
# ans = pipeline(
#     Tasks.acoustic_noise_suppression,
#     model='damo/speech_zipenhancer_ans_multiloss_16k_base')

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def build_model():

    MS_CACHE_HOME = get_modelscope_cache_dir()
    config_file = os.path.join(MS_CACHE_HOME, 'hub/iic/speech_zipenhancer_ans_multiloss_16k_base/configuration.json')
    model_file = os.path.join(MS_CACHE_HOME, 'hub/iic/speech_zipenhancer_ans_multiloss_16k_base/pytorch_model.bin')
    kwargs = Config.from_file(config_file)['model']
    h = dict(
        num_tsconformers=kwargs['num_tsconformers'],
        dense_channel=kwargs['dense_channel'],
        former_conf=kwargs['former_conf'],
        batch_first=kwargs['batch_first'],
        model_num_spks=kwargs['model_num_spks'],
    )
    h = AttrDict(h)
    model = ZipEnhancer(h)
    model.load_state_dict(torch.load(model_file, map_location='cpu')['generator'])
    model.eval()
    return model
def select_device():
    if torch.cuda.is_available():
        msg = 'Using gpu for inference.'
        logging.info(f'{msg}')
        device = torch.device('cuda:0')
    else:
        msg = 'No cuda device is detected. Using cpu.'
        logging.info(f'{msg}')
        device = torch.device('cpu')
    return device

def prepared_single_wav():
    audio_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/speech_with_noise1.wav'
    output_path = 'torch_output.wav'

    if audio_path.startswith("http"):
        file_bytes = File.read(audio_path)
        wav, fs = sf.read(io.BytesIO(file_bytes))
    else:
        wav, fs = sf.read(audio_path)
    return wav


def prepared_single_input(wav: np.array,device):
    n_fft = 400
    hop_size = 100
    win_size = 400
    is_verbose = False

    # wav = audio_norm(wav).astype(np.float32)
    wav = wav.astype(np.float32)
    noisy_wav = torch.from_numpy(np.reshape(wav, [1, wav.shape[0]])).to(device)

    norm_factor = torch.sqrt(noisy_wav.shape[1] / torch.sum(noisy_wav ** 2.0))
    if is_verbose:
        print(f"norm_factor {norm_factor}" )

    noisy_audio = (noisy_wav * norm_factor)

    noisy_mag, noisy_pha, _ = mag_pha_stft(
        noisy_audio,
        n_fft,
        hop_size,
        win_size,
        compress_factor=0.3,
        center=True)
    return noisy_mag, noisy_pha, norm_factor

def collate_data(frames):
    max_len = max(frame.size(0) for frame in frames)
    out = frames[0].new_zeros((len(frames), max_len))
    for i, v in enumerate(frames):
        out[i, : v.size(0)] = v
    return out
def prepared_batch_input(device):
    n_fft = 400
    hop_size = 100
    win_size = 400
    is_verbose = False
    audio_path = "/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/target_audio/R8001_M8004_MS801/all.wav"
    wav, fs = sf.read(audio_path)
    print(f"wav shape: {wav.shape}")
    wav = wav.astype(np.float32)
    chunk_size=64000
    shift_len=int(16000*0.8)
    length=len(wav)
    wav_segs = []
    for start in range(0,length,shift_len):
        end = (start + chunk_size)if start + chunk_size < length else length
        if end - start > 0:
            wav_segs.append(torch.from_numpy(wav[start:end]))
    
    wav_segs = collate_data(wav_segs)#(B,T) # torch.tensor
    print(f"wav_segs type: {type(wav_segs)}, its shape:{wav_segs.shape}")
    noisy_mags = []
    noisy_phas = []
    norm_factors = []
    for i in range(0,64,8): 
        wav_segss = wav_segs[i:i+8]
        print(f"wav_segss type: {type(wav_segss)}, its shape:{wav_segss.shape}!!!")
        wav_segss = wav_segss.to(device)
        norm_factor = torch.sqrt(wav_segss.shape[1] / torch.sum(wav_segss ** 2.0,dim=1)) # (B,)
        noisy_audio = (wav_segss * norm_factor.unsqueeze(1)) #(B,T)
        noisy_mag, noisy_pha, _ = mag_pha_stft(
            noisy_audio,
            n_fft,
            hop_size,
            win_size,
            compress_factor=0.3,
            center=True)
        noisy_mags.append(noisy_mag)
        noisy_phas.append(noisy_pha)
        norm_factors.append(noisy_audio)
    return noisy_mags, noisy_phas, norm_factors

def model_forward(noisy_mags, noisy_phas,norm_factors,model):
    n_fft = 400
    hop_size = 100
    win_size = 400
    is_verbose = False
    for noisy_mag, noisy_pha, norm_factor in zip(noisy_mags, noisy_phas,norm_factors):
        print(f"noisy_mag: {noisy_mag.shape}, noisy_pha: {noisy_pha.shape}, norm_factor: {norm_factor.shape}")
        amp_g, pha_g, _, _, _ = model(noisy_mag, noisy_pha)

        enhanced_wav = mag_pha_istft(
            amp_g,
            pha_g,
            n_fft,
            hop_size,
            win_size,
            compress_factor=0.3,
            center=True)

        enhanced_wav = enhanced_wav / norm_factor.unsqueeze(1)
        enhanced_wav = to_numpy(enhanced_wav)
        print(f"enhaced_wav: {enhanced_wav.shape}")
def load_multi_audio():
    n_fft = 400
    hop_size = 100
    win_size = 400
    is_verbose = False
    audio_path = "/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/target_audio/R8001_M8004_MS801/all.wav"
    wav, fs = sf.read(audio_path,always_2d=True,dtype="float32")

    print(f"wav shape: {wav.shape}")
    wav = wav[:, 0]
    print(f"wav shape: {wav.shape}")

def test_single_sample():
    model = build_model()
    device = select_device()
    model = model.to(device)
    wav = prepared_single_wav()
    noisy_mag, noisy_pha,norm_factor = prepared_single_input(wav,device)
    start = time.time()
    model_forward(noisy_mag, noisy_pha,norm_factor,model)
    print(time.time()-start)

def test_batch():
    model = build_model()
    device = select_device()
    model = model.to(device)
    noisy_mags, noisy_phas,norm_factors = prepared_batch_input(device)
    start = time.time()
    model_forward(noisy_mags, noisy_phas,norm_factors,model)
    print(time.time()-start)
if __name__ == "__main__":
    #test_single_sample()
    #test_batch()
    load_multi_audio()
